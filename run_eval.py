#!/usr/bin/env python3
"""RAG eval harness for the server's SOTA RAG stack.

Runs eval/gold_set.json against rag-uploader and scores every answer two
independent ways:

  fact_recall  deterministic substring/boundary match on `key_facts`.
               No LLM involved, so it cannot hallucinate or drift between runs.
               This is the signal to trust.

  correctness  LLM judge (Ollama qwen2.5:7b-instruct) grading 0.0/0.5/1.0 on
               whether the answer asserts the same thing as ground_truth.
               Noisy in absolute terms, so use it to compare runs, not to grade.

Plus a `refusal` flag, which separates "retrieval found nothing" from
"retrieval found the wrong thing".

The judge is self-tested before the run (known-right and known-wrong pairs).
If it fails its own sanity check the run ABORTS rather than emitting numbers
nobody should trust.

Stdlib only. Runs on the laptop; talks HTTP to the server. Adds no containers and
uses no VRAM beyond the models already resident.

Usage:
    python eval/run_eval.py                    # full run
    python eval/run_eval.py --only 5,25        # just those gold ids
    python eval/run_eval.py --no-judge         # deterministic scoring only
    python eval/run_eval.py --self-test-only   # verify plumbing, no eval
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    "rag_url": "http://localhost:8090/v1/chat/completions",
    "rag_model": "my-rag",
    "judge_url": "http://localhost:11434/api/chat",
    "judge_model": "qwen2.5:7b-instruct-q4_K_M",
}

REFUSAL_PATTERNS = [
    r"\bi (?:do not|don't) know\b",
    r"\bno (?:relevant )?(?:context|information|documents?)\b",
    r"\bnot (?:enough|sufficient) (?:context|information)\b",
    r"\bcannot (?:find|answer|determine)\b",
    r"\bunable to (?:find|answer|determine)\b",
    r"\bthe (?:provided )?context does not\b",
]

JUDGE_SYSTEM = (
    "You grade whether a candidate answer agrees with a reference answer. "
    "You are strict about facts and lenient about wording.\n"
    "Reply with ONLY a JSON object: {\"score\": <0, 0.5, or 1>, \"reason\": \"<one short sentence>\"}\n"
    "score 1   = states the same fact(s) as the reference. Extra correct detail is fine.\n"
    "score 0.5 = partially right: some required facts present, others missing.\n"
    "score 0   = contradicts the reference, or is missing the point entirely, "
    "or refuses to answer.\n"
    "Different wording, units, or formatting is NOT a reason to lower the score. "
    "Contradicting a number or an identifier IS."
)

# (reference, candidate, expected_score). The judge must get these right or we abort.
JUDGE_SELF_TEST = [
    ("The GPU is an RTX 5070 Ti with 16 GB VRAM.",
     "the server has an NVIDIA RTX 5070 Ti, 16GB of video memory.", 1.0),
    ("The GPU is an RTX 5070 Ti with 16 GB VRAM.",
     "the server has an RTX 4090 with 24 GB of VRAM.", 0.0),
    ("The context window is 32768 tokens.",
     "The context window is 80,000 tokens.", 0.0),
    ("Port 8090.",
     "It runs on port 8090.", 1.0),
    ("The reranker is Qwen3-Reranker-0.6B on port 8787.",
     "I don't know, there is no context about that.", 0.0),
]


# --------------------------------------------------------------------------- http

def post_json(url, payload, timeout=180, retries=2):
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            # 4xx is a request problem (wrong model name, bad path), so retrying
            # just triples the wait before showing the same error.
            if 400 <= exc.code < 500:
                raise RuntimeError(f"POST {url} -> HTTP {exc.code} {exc.reason}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {retries + 1} tries: {last}")


def ask_rag(question, cfg):
    """Query the system under test. Returns (answer, contexts, elapsed_seconds)."""
    t0 = time.perf_counter()
    data = post_json(cfg["rag_url"], {
        "model": cfg["rag_model"],
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        # temperature 0 so runs are comparable. The answer path samples otherwise,
        # and three same-state runs spread 0.577-0.712 before this was pinned
        "temperature": 0,
    }, timeout=cfg["timeout"])
    elapsed = time.perf_counter() - t0
    answer = data["choices"][0]["message"]["content"]
    # Tier 2: if rag-uploader ever starts returning retrieved chunks, pick them up
    # automatically so faithfulness/context metrics become possible with no client change.
    contexts = data.get("contexts") or data["choices"][0].get("contexts") or []
    return answer, contexts, elapsed


def judge(reference, candidate, cfg):
    """Ask the judge model to grade. Returns (score, reason)."""
    user = (f"REFERENCE ANSWER:\n{reference}\n\n"
            f"CANDIDATE ANSWER:\n{candidate}\n\n"
            "Grade the candidate. JSON only.")
    try:
        data = post_json(cfg["judge_url"], {
            "model": cfg["judge_model"],
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 200},
        }, timeout=cfg["timeout"])
    except RuntimeError as exc:
        # Never let a judge outage take down a run that has already spent real
        # minutes querying the RAG stack. The deterministic scores still stand.
        return None, f"judge unreachable: {exc}"

    raw = data.get("message", {}).get("content", "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None, f"unparseable judge output: {raw[:120]!r}"
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, f"unparseable judge output: {raw[:120]!r}"

    try:
        score = float(parsed.get("score"))
    except (TypeError, ValueError):
        return None, f"judge returned no usable score: {raw[:120]!r}"

    # snap to the allowed grid
    score = min([0.0, 0.5, 1.0], key=lambda allowed: abs(allowed - score))
    return score, str(parsed.get("reason", ""))[:200]


# --------------------------------------------------------------------- scoring

def normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower())


def fact_present(fact, normalized_answer):
    """Substring match, with word boundaries for compact tokens.

    Boundaries stop '5' from matching inside '8083' and '768' inside '7680'.
    Only \\w is excluded (not '.') so a trailing sentence period does not break
    a match on something like '192.0.2.10'.
    """
    fact = normalize(fact)
    if re.fullmatch(r"[\w.:@/-]+", fact):
        return re.search(rf"(?<!\w){re.escape(fact)}(?!\w)", normalized_answer) is not None
    return fact in normalized_answer


def score_facts(key_facts, answer):
    """Returns (score_or_None, matched_list, missed_list)."""
    if not key_facts:
        return None, [], []
    norm = normalize(answer)
    matched, missed = [], []
    for alternatives in key_facts:
        hit = next((a for a in alternatives if fact_present(a, norm)), None)
        (matched if hit else missed).append(hit or alternatives[0])
    return len(matched) / len(key_facts), matched, missed


def is_refusal(answer):
    norm = normalize(answer)
    if len(norm.strip()) < 15:
        return True
    return any(re.search(p, norm) for p in REFUSAL_PATTERNS)


# ------------------------------------------------------------------ self-test

def run_self_test(cfg):
    print("judge self-test:", flush=True)
    failures = []
    for reference, candidate, expected in JUDGE_SELF_TEST:
        score, reason = judge(reference, candidate, cfg)
        ok = score is not None and abs(score - expected) < 0.01
        print(f"  [{'PASS' if ok else 'FAIL'}] expected {expected} got {score}"
              f"  on {candidate[:52]!r}", flush=True)
        if not ok:
            failures.append((candidate, expected, score, reason))
    if failures:
        print(f"\nJUDGE SELF-TEST FAILED ({len(failures)}/{len(JUDGE_SELF_TEST)}).", flush=True)
        for cand, exp, got, why in failures:
            print(f"  expected {exp}, got {got}: {why}", flush=True)
        return False
    print(f"  all {len(JUDGE_SELF_TEST)} passed\n", flush=True)
    return True


# --------------------------------------------------------------------- report

def corpus_state(base=os.environ.get("RAG_BASE", "http://localhost:8090")):
    """Snapshot what the RAG is actually serving, for the results file.

    Best-effort by design: a probe failure must never abort a scoring run, so every field
    degrades to None and the run continues. An unknown corpus size is worth strictly more than
    a crashed eval, and strictly more than a score that silently pretends the question was
    never worth asking.
    """
    # search_rpc_client_override records only what THIS process was told. The authoritative
    # setting lives in rag-uploader's own environment on the server and is not exposed over HTTP, so
    # a null here means "not overridden from the client", NOT "dense" or "hybrid". Recording it
    # under an honest name beats recording a confident wrong value.
    out = {"documents": None, "chunks": None,
           "search_rpc_client_override": os.environ.get("RAG_SEARCH_RPC")}
    try:
        with urllib.request.urlopen(f"{base}/api/indexed", timeout=20) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        if isinstance(d, dict) and d.get("error"):
            out["error"] = d["error"]          # empty-because-broken, not empty-because-empty
            return out
        docs = d.get("documents", d) if isinstance(d, dict) else d
        out["documents"] = len(docs)
        chunks = sum(x.get("chunk_count", 0) for x in docs if isinstance(x, dict))
        out["chunks"] = chunks or None
    except Exception as e:
        out["error"] = f"{e.__class__.__name__}: {e}"
    return out


def previous_run(results_dir, exclude):
    runs = sorted(p for p in results_dir.glob("eval-*.json") if p != exclude)
    if not runs:
        return None
    try:
        return json.loads(runs[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def fmt(value, spec=".3f"):
    return "n/a" if value is None else format(value, spec)


def build_markdown(payload, prior):
    s = payload["summary"]
    lines = [
        "# RAG Eval Report",
        "",
        f"*Run {payload['run_id']}, {payload['started_utc']}*",
        "",
        f"- System under test: `{payload['config']['rag_model']}` @ `{payload['config']['rag_url']}`",
        f"- Judge: `{payload['config']['judge_model']}` @ `{payload['config']['judge_url']}`",
        f"- Questions: **{s['total']}**",
        "",
        "## Headline",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| fact_recall (deterministic) | **{fmt(s['fact_recall'])}** "
        f"({s['fact_recall_scored']} scored, {s['judge_only']} judge-only) |",
    ]
    if s["correctness"] is not None:
        lines.append(f"| correctness (LLM judge) | **{fmt(s['correctness'])}** |")
    lines += [
        f"| refusals | {s['refusals']} |",
        f"| errors | {s['errors']} |",
        f"| median latency | {s['median_latency_s']:.1f} s |",
        "",
    ]

    if prior:
        ps = prior["summary"]
        lines += [f"### Change vs previous run (`{prior['run_id']}`)", ""]
        if ps["total"] != s["total"]:
            lines += [f"> Different question counts ({ps['total']} vs {s['total']}). "
                      "These deltas are not comparable. Re-run the same selection.", ""]
        for name in ("fact_recall", "correctness"):
            before, after = ps.get(name), s.get(name)
            if before is None or after is None:
                lines.append(f"- {name}: {fmt(before)} -> {fmt(after)}")
            else:
                lines.append(f"- {name}: {before:.3f} -> {after:.3f} (**{after - before:+.3f}**)")
        lines.append("")

    disagree = [r for r in payload["results"]
                if r.get("fact_recall") is not None and r.get("correctness") is not None
                and abs(r["fact_recall"] - r["correctness"]) > 0.5]
    if disagree:
        lines += [
            "## Needs human review: the two scorers disagree",
            "",
            "Deterministic and judge scores differ by more than 0.5. Usually means either the "
            "`key_facts` list is wrong, or the judge is being fooled.",
            "",
            "| id | question | fact_recall | correctness | judge reason |",
            "|---|---|---|---|---|",
        ]
        for r in disagree:
            lines.append(f"| {r['id']} | {r['question'][:60]} | {r['fact_recall']:.2f} | "
                         f"{r['correctness']:.2f} | {r.get('judge_reason', '')[:70]} |")
        lines.append("")

    lines += [
        "## Per-question",
        "",
        "| id | fact_recall | correctness | refused | missed facts | question |",
        "|---|---|---|---|---|---|",
    ]
    for r in payload["results"]:
        fr = "n/a" if r.get("fact_recall") is None else f"{r['fact_recall']:.2f}"
        co = "n/a" if r.get("correctness") is None else f"{r['correctness']:.2f}"
        missed = ", ".join(r.get("missed_facts", []))[:48] or "n/a"
        flag = "yes" if r.get("refused") else ""
        lines.append(f"| {r['id']} | {fr} | {co} | {flag} | {missed} | {r['question'][:56]} |")

    failures = [r for r in payload["results"]
                if r.get("error") or r.get("refused")
                or (r.get("fact_recall") is not None and r["fact_recall"] < 1.0)]
    if failures:
        lines += ["", "## Failures in detail", ""]
        for r in failures:
            lines += [f"### id {r['id']}: {r['question']}", ""]
            if r.get("error"):
                lines += [f"**ERROR:** `{r['error']}`", ""]
                continue
            lines += [
                f"- **expected:** {r['ground_truth']}",
                f"- **got:** {r['answer'][:500]}",
                f"- **missed facts:** {', '.join(r.get('missed_facts', [])) or 'none'}",
                f"- **judge:** {r.get('judge_reason', 'n/a')}",
                f"- **source doc:** `{r.get('source', '?')}`",
                "",
            ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Score the server's RAG stack against the gold set.")
    ap.add_argument("--gold", default=str(HERE / "gold_set.example.json"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--rag-url", default=os.environ.get("RAG_URL", DEFAULTS["rag_url"]))
    ap.add_argument("--rag-model", default=os.environ.get("RAG_MODEL", DEFAULTS["rag_model"]))
    ap.add_argument("--judge-url", default=os.environ.get("JUDGE_URL", DEFAULTS["judge_url"]))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", DEFAULTS["judge_model"]))
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--only", help="comma-separated gold ids to run")
    ap.add_argument("--no-judge", action="store_true", help="deterministic scoring only")
    ap.add_argument("--self-test-only", action="store_true", help="check the judge, then exit")
    args = ap.parse_args()

    cfg = {
        "rag_url": args.rag_url, "rag_model": args.rag_model,
        "judge_url": args.judge_url, "judge_model": args.judge_model,
        "timeout": args.timeout,
    }

    if not args.no_judge:
        if not run_self_test(cfg):
            print("Refusing to produce scores from a judge that fails its own sanity check.",
                  file=sys.stderr)
            return 2
    if args.self_test_only:
        return 0

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        gold = [g for g in gold if g["id"] in wanted]
    if not gold:
        print("no questions selected", file=sys.stderr)
        return 2

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []

    print(f"running {len(gold)} questions against {cfg['rag_model']}\n", flush=True)
    for i, item in enumerate(gold, 1):
        row = {k: item[k] for k in ("id", "question", "ground_truth", "source")}
        row["notes"] = item.get("notes")
        try:
            answer, contexts, elapsed = ask_rag(item["question"], cfg)
        except RuntimeError as exc:
            row.update(error=str(exc), fact_recall=None, correctness=None, refused=None)
            results.append(row)
            print(f"[{i}/{len(gold)}] id {item['id']:>2}  ERROR  {exc}", flush=True)
            continue

        fr, matched, missed = score_facts(item.get("key_facts", []), answer)
        refused = is_refusal(answer)
        row.update(answer=answer, contexts=contexts, latency_s=round(elapsed, 2),
                   fact_recall=fr, matched_facts=matched, missed_facts=missed,
                   refused=refused, error=None)

        if args.no_judge:
            row.update(correctness=None, judge_reason=None)
        else:
            score, reason = judge(item["ground_truth"], answer, cfg)
            row.update(correctness=score, judge_reason=reason)

        results.append(row)
        fr_s = " n/a " if fr is None else f"{fr:.2f} "
        co_s = " n/a " if row.get("correctness") is None else f"{row['correctness']:.2f} "
        print(f"[{i}/{len(gold)}] id {item['id']:>2}  fact {fr_s} judge {co_s}"
              f" {elapsed:5.1f}s{'  REFUSED' if refused else ''}"
              f"{'  missed: ' + ', '.join(missed) if missed else ''}", flush=True)

    scored = [r["fact_recall"] for r in results if r.get("fact_recall") is not None]
    judged = [r["correctness"] for r in results if r.get("correctness") is not None]
    lat = sorted(r["latency_s"] for r in results if r.get("latency_s") is not None)

    summary = {
        "total": len(results),
        # None, not 0.0, when nothing could be scored. An outage must not be
        # indistinguishable from the stack answering every question wrong.
        "fact_recall": (sum(scored) / len(scored)) if scored else None,
        "fact_recall_scored": len(scored),
        "judge_only": sum(1 for r in results
                          if r.get("fact_recall") is None and not r.get("error")),
        "correctness": (sum(judged) / len(judged)) if judged else None,
        # Load-bearing, not presentation. A refusal and a confident wrong answer both score 0.00
        # on fact_recall, and that is not a gap to close, because a recall metric structurally cannot
        # represent the difference. This count is the ONLY carrier of it. Question 5 is the worked
        # case: it used to answer "80,000 tokens" with total confidence and now refuses, which is a
        # large improvement that is invisible in the headline number. If this is ever folded into
        # the summary while tidying output, the distinction disappears silently.
        "refusals": sum(1 for r in results if r.get("refused")),
        "errors": sum(1 for r in results if r.get("error")),
        "median_latency_s": lat[len(lat) // 2] if lat else 0.0,
    }

    payload = {
        "run_id": run_id,
        "started_utc": started,
        "config": {k: v for k, v in cfg.items() if k != "timeout"},
        # What state was measured. Without this a score is a number with no subject: on
        # 2026-08-03 five runs inside seven minutes produced 0.615/0.635/0.635/0.673/0.712 and
        # nothing on disk recorded which retrieval mode or corpus size each one saw, so the
        # spread could not afterwards be split into "config change" versus "corpus change"
        # versus "noise", and the highest of them became the quoted baseline by default.
        # A result that does not carry its own inputs cannot be compared to anything later.
        "corpus": corpus_state(),
        "summary": summary,
        "results": results,
    }

    json_path = results_dir / f"eval-{run_id}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    prior = previous_run(results_dir, json_path)
    md_path = results_dir / f"eval-{run_id}.md"
    md_path.write_text(build_markdown(payload, prior), encoding="utf-8")

    print(f"\nfact_recall  {fmt(summary['fact_recall'])}   "
          f"correctness  {fmt(summary['correctness'])}   "
          f"refusals {summary['refusals']}   errors {summary['errors']}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
