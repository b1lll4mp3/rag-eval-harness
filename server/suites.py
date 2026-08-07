"""Suite execution. Lane discipline: record the regime on every result.
A number without its contention regime is uninterpretable."""
import json
import os
import statistics
import time
import urllib.request

import dockerctl
from scoring import score_facts, is_refusal

# --- Config ----------------------------------------------------------------
# Container names, endpoints, and models for YOUR stack. Everything here can
# be overridden with environment variables so the deployed image never needs
# editing. The two container names must match `docker ps` exactly: dockerctl
# addresses containers by name over the socket.
VLLM = os.environ.get("VLLM_CONTAINER", "vllm")
RAG_UPLOADER = os.environ.get("RAG_CONTAINER", "rag-uploader")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
RAG_URL = os.environ.get("RAG_URL", "http://127.0.0.1:8090/v1/chat/completions")
RERANKER_HEALTH = os.environ.get("RERANKER_HEALTH", "http://127.0.0.1:8090/health/reranker")
VLLM_MODELS = os.environ.get("VLLM_MODELS_URL", "http://127.0.0.1:8010/v1/models")
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "qwen2.5:7b-instruct-q4_K_M")
# The model name the RAG endpoint expects on requests (its own alias, not an
# Ollama tag).
RAG_REQUEST_MODEL = os.environ.get("RAG_REQUEST_MODEL", "my-rag")
QUERY_TIMEOUT = int(os.environ.get("QUERY_TIMEOUT", "300"))


def http_get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def http_post(url, payload, timeout=QUERY_TIMEOUT):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def probe(url):
    try:
        http_get(url)
        return True
    except Exception:
        return False


def current_mode():
    return "code" if probe(VLLM_MODELS) else "rag"


def set_mode(mode):
    """Replicates the host's GPU lane-toggle script via the Docker socket.
    Returns a list of warning strings (empty when everything unloaded clean)."""
    warnings = []
    if mode == "code":
        # ANY resident model on the GPU blocks vLLM (~13 GB) from coming up,
        # not just the baseline, since a model_compare run may have left a
        # candidate model resident. Unload everything Ollama is holding; in
        # code mode Ollama reloads what it needs with CPU placement on demand.
        try:
            resident = http_get(f"{OLLAMA_URL}/api/ps").get("models", [])
        except Exception as exc:
            resident = []
            warnings.append(f"list resident models failed: {exc}")
        for m in resident:
            name = m.get("name")
            if not name:
                continue
            try:
                http_post(f"{OLLAMA_URL}/api/generate",
                          {"model": name, "keep_alive": 0}, timeout=60)
            except Exception as exc:
                warnings.append(f"unload {name} failed: {exc}")
        if not dockerctl.container_running(VLLM):
            dockerctl.start(VLLM)
        elif not probe(VLLM_MODELS):
            # container reports running but the models endpoint is dead:
            # wedged in place (same failure shape as the known Ollama NVML
            # wedge). Restart once.
            dockerctl.stop(VLLM)
            dockerctl.start(VLLM)
        deadline = time.time() + 300
        while time.time() < deadline:            # measured ~40 s to ready
            if probe(VLLM_MODELS):
                return warnings
            time.sleep(5)
        raise RuntimeError("vllm did not come up within 300 s")
    if mode == "rag":
        if dockerctl.container_running(VLLM):
            dockerctl.stop(VLLM)
        # the baseline model may still be loaded with CPU placement from code
        # mode. Ollama keeps the existing instance until unloaded, so queries
        # would stay on CPU and can blow the timeout cap. Force an unload; the
        # readiness gate's warm-up then reloads it onto the freed GPU.
        try:
            http_post(f"{OLLAMA_URL}/api/generate",
                      {"model": BASELINE_MODEL, "keep_alive": 0}, timeout=60)
        except Exception as exc:
            warnings.append(f"unload {BASELINE_MODEL} failed: {exc}")
        return warnings
    raise ValueError(f"unknown mode {mode!r}")


def readiness_gate(model=None, post=None):
    """After any mode toggle or model swap: one throwaway warm-up query must
    return 200 before scored queries start (cold model = timeout/OOM = corrupt
    run, or silent cold-load pollution of the first scored latency)."""
    post = post or http_post
    body = {"model": model or RAG_REQUEST_MODEL,
            "messages": [{"role": "user", "content": "warm-up ping"}],
            "stream": False}
    deadline = time.time() + 300
    last = None
    while time.time() < deadline:
        try:
            post(RAG_URL, body, timeout=QUERY_TIMEOUT)
            return
        except Exception as exc:
            last = exc
            time.sleep(5)
    raise RuntimeError(f"readiness gate failed after 300 s: {last}")


def swap_model(model, swap_mode, post=None):
    """'request' passes the model per-query, but the backend may still
    cold-load its weights. Warm up so the load is never absorbed into a
    scored latency."""
    if swap_mode == "request":
        readiness_gate(model, post=post)
        return
    dockerctl.restart_with_env(RAG_UPLOADER, {"OLLAMA_MODEL": model})
    readiness_gate(model, post=post)


def ask(question, model, swap_mode, post):
    body = {"model": RAG_REQUEST_MODEL,
            "messages": [{"role": "user", "content": question}], "stream": False}
    if swap_mode == "request":
        body["model"] = model
    t0 = time.perf_counter()
    data = post(RAG_URL, body, timeout=QUERY_TIMEOUT)
    latency = time.perf_counter() - t0
    return data["choices"][0]["message"]["content"], latency


def run_suite(run, gold, post=None):
    post = post or http_post
    model = run.get("model") or BASELINE_MODEL
    swap_mode = run.get("swap_mode", "request")
    questions = []
    scores, refusals = [], 0
    for entry in gold:
        answer, latency = ask(entry["question"], model, swap_mode, post)
        score, matched, missed = score_facts(entry.get("key_facts", []), answer)
        refused = is_refusal(answer)
        refusals += bool(refused)
        if score is not None:
            scores.append(score)
        questions.append({"id": entry["id"], "question": entry["question"],
                          "answer": answer, "fact_recall": score,
                          "missed": missed, "refused": refused,
                          "latency_s": round(latency, 2)})
    lat = [q["latency_s"] for q in questions]
    run.update({
        "questions": questions,
        "gold_count": len(gold),
        "fact_recall": round(sum(scores) / len(scores), 3) if scores else None,
        "refusals": refusals,
        "median_latency_s": round(statistics.median(lat), 2) if lat else None,
        "max_latency_s": max(lat) if lat else None,
    })
    return run


def markdown_summary(run, baseline=None):
    fr = f"{run['fact_recall']:.3f}" if run.get("fact_recall") is not None else "n/a"
    lines = [
        f"### RAG run {run.get('started', '')}: {run.get('suite', '?')}, "
        f"{run.get('mode', '?')} mode, model `{run.get('model', '?')}`",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| fact_recall | **{fr}** ({run.get('gold_count', '?')} questions, "
        f"gold source: {run.get('gold_source', '?')}) |",
        f"| refusals | {run.get('refusals', '?')} |",
        f"| latency | {run.get('median_latency_s', 'n/a')} s median, "
        f"{run.get('max_latency_s', 'n/a')} s max |",
    ]
    if baseline:
        if run.get("fact_recall") is not None and baseline.get("fact_recall") is not None:
            delta_fr = f"{run['fact_recall'] - baseline['fact_recall']:+.3f} fact_recall"
        else:
            delta_fr = "fact_recall n/a"
        if run.get("median_latency_s") is not None and baseline.get("median_latency_s") is not None:
            delta_lat = f"{run['median_latency_s'] - baseline['median_latency_s']:+.2f} s median"
        else:
            delta_lat = "latency n/a"
        lines.append(f"| vs baseline `{baseline.get('model', '?')}` | {delta_fr}, {delta_lat} |")
    lines.append("")
    lines.append(f"*{run.get('mode', '?')} mode: numbers are only comparable to runs "
                 "measured under the same lane split and contention regime.*")
    return "\n".join(lines)
