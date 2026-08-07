"""RAG Test Runner: web UI + API on the server :8095. Stdlib only.

Run lifecycle: POST /api/run -> 202 {run_id} (409 if one is active) ->
background thread executes -> GET /api/runs/<id> polls status.
Prior lane mode and generation model are ALWAYS restored afterwards."""
import json
import os
import threading
import traceback
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dockerctl
import suites

HERE = Path(__file__).resolve().parent
RUNS_DIR = Path("/data/runs")
# Optional remote gold-set URL (env GOLD_URL); empty or unreachable falls back to the bundled file
GOLD_URL = os.environ.get("GOLD_URL", "")
RUN_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_gold():
    try:
        gold = suites.http_get(GOLD_URL, timeout=10)
        return gold, "wiki"
    except Exception:
        return json.loads((HERE / "gold_set.json").read_text(encoding="utf-8")), "bundled"


def save_run(run):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    target = RUNS_DIR / f"{run['run_id']}.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=1))
    os.replace(tmp, target)


def execute(run):
    """Full run: snapshot regime -> optionally change it -> suite -> restore.
    Every scored arm is preceded by a readiness gate for the exact model it is
    about to score, because a cold load absorbed into a scored latency corrupts the run."""
    prior_mode = suites.current_mode()
    want_mode = run.get("mode") or prior_mode
    run["mode"] = want_mode
    run["model"] = run.get("model") or suites.BASELINE_MODEL
    run["gold_source"] = "none"
    # rag-uploader ignores request-level model overrides (probed 2026-07-30):
    # scoring any non-baseline model requires the env-restart swap path.
    # Derived server-side only, because a client-sent swap_mode could otherwise be
    # used to skip the env-restart swap the corpus lane actually requires.
    swap_mode = "env" if run["model"] != suites.BASELINE_MODEL else "request"
    run["swap_mode"] = swap_mode
    swapped = False
    mode_warnings = []
    try:
        run["phase"] = "load_gold"
        gold, source = load_gold()
        run["gold_source"] = source
        if run["suite"] == "latency":
            gold = gold[: run.get("n", 5)]
        if want_mode != prior_mode:
            run["phase"] = "set_mode"
            mode_warnings.extend(suites.set_mode(want_mode) or [])
        if run["suite"] == "model_compare":
            base = dict(run, model=suites.BASELINE_MODEL)
            run["phase"] = "gate_baseline"
            suites.readiness_gate(
                suites.BASELINE_MODEL if swap_mode == "request" else None)
            run["phase"] = "suite_baseline"
            suites.run_suite(base, gold)
            run["baseline"] = {k: base[k] for k in
                               ("model", "fact_recall", "refusals",
                                "median_latency_s", "max_latency_s")}
            run["baseline"]["questions"] = base.get("questions")
            run["phase"] = "swap_model"
            suites.swap_model(run["model"], swap_mode)   # gates internally
            swapped = True
        elif run["model"] != suites.BASELINE_MODEL:
            # single-arm suite on a non-baseline model still needs the swap
            run["phase"] = "swap_model"
            suites.swap_model(run["model"], swap_mode)   # gates internally
            swapped = True
        else:
            run["phase"] = "gate"
            suites.readiness_gate(
                run["model"] if swap_mode == "request" else None)
        run["phase"] = "suite"
        suites.run_suite(run, gold)
        run["status"] = "done"
        run["phase"] = "done"
    except Exception as exc:
        run["status"] = "error"
        run["error"] = f"{type(exc).__name__}: {exc} (phase: {run.get('phase')})"
        run["trace"] = traceback.format_exc()[-1500:]
    finally:
        # model restore and mode restore are independent recoveries, so one
        # failing must not suppress the other's attempt
        errors = []
        if swapped and swap_mode == "env":
            try:
                suites.swap_model(suites.BASELINE_MODEL, "env")
            except Exception as exc:
                errors.append(f"model restore: {exc}")
        try:
            if suites.current_mode() != prior_mode:
                mode_warnings.extend(suites.set_mode(prior_mode) or [])
                if prior_mode == "rag":
                    suites.readiness_gate(None)
        except Exception as exc:
            errors.append(f"mode restore: {exc}")
        if errors:
            run["restore_error"] = "; ".join(errors)
        if mode_warnings:
            run["mode_warnings"] = mode_warnings
        try:
            run["finished"] = now()
            save_run(run)
        finally:
            RUN_LOCK.release()


EXECUTE = execute      # test seam


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, (HERE / "ui.html").read_bytes(), "text/html")
        elif self.path == "/api/status":
            resident = {}
            try:
                resident = suites.http_get(f"{suites.OLLAMA_URL}/api/ps")
            except Exception:
                pass
            vram = ""
            try:
                vram = dockerctl.exec_in(
                    "ollama", ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                               "--format=csv,noheader"])
            except Exception:
                pass
            self._send(200, {
                "mode": suites.current_mode(),
                "resident": resident.get("models", []),
                "reranker_up": suites.probe(suites.RERANKER_HEALTH),
                "vram": vram.strip(),
                "run_active": RUN_LOCK.locked(),
            })
        elif self.path == "/api/models":
            try:
                tags = suites.http_get(f"{suites.OLLAMA_URL}/api/tags")
                self._send(200, {"models": [m["name"] for m in tags.get("models", [])]})
            except Exception as exc:
                self._send(502, {"error": str(exc)})
        elif self.path == "/api/runs":
            runs = sorted(RUNS_DIR.glob("*.json"), reverse=True)
            self._send(200, {"runs": [json.loads(p.read_text())
                                      | {"questions": None} for p in runs[:50]]})
        elif self.path.startswith("/api/runs/") and self.path.endswith("/markdown"):
            run_id = self.path[len("/api/runs/"):-len("/markdown")]
            if "/" in run_id or "\\" in run_id:
                return self._send(404, {"error": "no such run"})
            p = RUNS_DIR / f"{run_id}.json"
            if p.exists():
                run = json.loads(p.read_text())
                md = suites.markdown_summary(run, run.get("baseline"))
                self._send(200, md.encode(), "text/plain")
            else:
                self._send(404, {"error": "no such run"})
        elif self.path.startswith("/api/runs/"):
            p = RUNS_DIR / f"{self.path.rsplit('/', 1)[1]}.json"
            if p.exists():
                self._send(200, json.loads(p.read_text()))
            else:
                self._send(404, {"error": "no such run"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        if self.path == "/api/mode":
            mode = body.get("mode")
            if mode not in ("rag", "code"):
                return self._send(400, {"error": "mode must be rag|code"})
            if not RUN_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "run active"})
            try:
                suites.set_mode(mode)
                if mode == "rag":
                    suites.readiness_gate(None)
                return self._send(200, {"mode": suites.current_mode()})
            finally:
                RUN_LOCK.release()
        if self.path == "/api/run":
            if body.get("suite") not in ("latency", "eval", "model_compare"):
                return self._send(400, {"error": "suite must be latency|eval|model_compare"})
            if body.get("mode") not in (None, "", "rag", "code"):
                return self._send(400, {"error": "mode must be rag|code"})
            n = body.get("n")
            if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n <= 0):
                return self._send(400, {"error": "n must be a positive integer"})
            if not RUN_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "run already active"})
            try:
                run = {"run_id": uuid.uuid4().hex[:12], "started": now(),
                       "status": "running", **body}
                save_run(run)
                threading.Thread(target=EXECUTE, args=(run,), daemon=True).start()
            except Exception as exc:
                RUN_LOCK.release()
                return self._send(500, {"error": f"failed to start run: {exc}"})
            return self._send(202, {"run_id": run["run_id"]})
        self._send(404, {"error": "not found"})


def make_server(port=8095):
    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


if __name__ == "__main__":
    make_server().serve_forever()
