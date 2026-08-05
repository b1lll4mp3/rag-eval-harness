#!/usr/bin/env python3
"""Client-side stopwatch benchmark for an OpenAI-compatible lane.

Measures what one person at a keyboard actually feels. The engine's own
throughput numbers are never trusted — only its token counts, timed here.

  ttft         request sent -> first content token
  decode tok/s (completion_tokens - 1) / (last token time - first token time)
  prefill      measured separately, with a UNIQUE prompt per run so vLLM's
               automatic prefix caching cannot serve a cache hit and make
               prefill look instant. (A naive repeated-prompt probe reported
               53 ms for 3.6k tokens on 2026-07-21 — that was a cache hit,
               not prefill.)

Baseline measured 2026-07-21 on Qwen2.5-Coder-7B-Instruct-AWQ @ localhost:8010:
decode 153 tok/s, ttft 22-37 ms. The 5070 Ti's 896 GB/s over ~4.7 GB of weights
puts the memory-bandwidth ceiling near 190 tok/s, so that is ~81% of roof.

Stdlib only.

Usage:
    python eval/bench_lane.py                       # the vLLM coder lane
    python eval/bench_lane.py --repeats 3 --no-prefill
    python eval/bench_lane.py --base http://localhost:8090/v1 --model my-rag
"""

import argparse
import json
import os
import random
import statistics
import string
import sys
import time
import urllib.request

DEFAULT_BASE = os.environ.get("BENCH_BASE", "http://localhost:8010/v1")
DEFAULT_MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")

TASKS = {
    "code": "Write a Python function that merges two sorted lists into one sorted list. "
            "Include a docstring and handle empty inputs.",
    "math": "A train leaves at 14:20 travelling 96 km/h and another leaves the same "
            "station at 15:05 travelling 128 km/h on the same track. When does the "
            "second catch the first? Show your working step by step.",
}


def unique_long_prompt(approx_tokens=3000):
    """A long prompt that is different on every call.

    Random letter strings defeat prefix caching; without this the prefill
    measurement is meaningless. They also tokenize badly on purpose — a
    gibberish "word" costs roughly 2.6 tokens, not the ~1.3 of real English —
    so this uses that ratio. The real prompt_tokens is always read back from
    the usage block rather than trusted from this estimate.
    """
    words = []
    for _ in range(max(1, int(approx_tokens / 2.6))):
        n = random.randint(3, 9)
        words.append("".join(random.choices(string.ascii_lowercase, k=n)))
    return ("Below is a block of log tokens from a storage system.\n"
            + " ".join(words)
            + "\n\nHow many distinct words are in the block above? Answer in one sentence.")


def stream_once(base, model, prompt, max_tokens, timeout=300):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})

    t_send = time.perf_counter()
    t_first = t_last = None
    chunks = 0
    text = []
    usage = None

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunks += 1
                    text.append(piece)

    if t_first is None:
        raise RuntimeError("no content tokens returned")

    completion_tokens = (usage or {}).get("completion_tokens") or chunks
    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    window = t_last - t_first
    return {
        "ttft": t_first - t_send,
        "decode_tps": (completion_tokens - 1) / window if window > 0 else 0.0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "text": "".join(text),
    }


def degenerate(text, completion_tokens=None):
    """A fast-but-garbage answer must not be allowed to inflate tok/s.

    The short-output rule only applies when the model was actually given room
    to write. The prefill probe caps output at 64 tokens and asks for a
    one-sentence answer, so a correct reply there is legitimately short.
    """
    words = text.split()
    if completion_tokens is not None and completion_tokens < 40:
        return False
    if len(words) < 20:
        return True
    return max(words.count(w) for w in set(words)) / len(words) > 0.30


def run(label, make_prompt, base, model, max_tokens, warmup, repeats):
    print(f"\n=== {label} ===", flush=True)
    for i in range(warmup):
        stream_once(base, model, make_prompt(), max_tokens)
        print(f"  warmup {i + 1} discarded", flush=True)

    ttfts, tpss, prompt_toks = [], [], []
    for i in range(repeats):
        r = stream_once(base, model, make_prompt(), max_tokens)
        flag = "  DEGENERATE" if degenerate(r["text"], r["completion_tokens"]) else ""
        print(f"  run {i + 1}: ttft {r['ttft'] * 1000:8.1f} ms | decode {r['decode_tps']:6.1f} tok/s"
              f" | {r['completion_tokens']} out / {r['prompt_tokens']} in{flag}", flush=True)
        ttfts.append(r["ttft"])
        tpss.append(r["decode_tps"])
        prompt_toks.append(r["prompt_tokens"])

    med_ttft, med_tps = statistics.median(ttfts), statistics.median(tpss)
    print(f"  MEDIAN: ttft {med_ttft * 1000:.1f} ms | decode {med_tps:.1f} tok/s "
          f"(spread {min(tpss):.1f}-{max(tpss):.1f})", flush=True)
    return {"decode_tps": med_tps, "ttft": med_ttft,
            "prompt_tokens": statistics.median(prompt_toks) if prompt_toks else 0}


def main():
    ap = argparse.ArgumentParser(description="Stopwatch bench for an OpenAI-compatible lane.")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--prefill-tokens", type=int, default=3000)
    ap.add_argument("--no-prefill", action="store_true", help="skip the long-prompt probe")
    ap.add_argument("--json", help="also write results to this path")
    args = ap.parse_args()

    print(f"lane: {args.model} @ {args.base}")
    results = {}
    for name, prompt in TASKS.items():
        results[name] = run(name, lambda p=prompt: p, args.base, args.model,
                            args.max_tokens, args.warmup, args.repeats)

    if not args.no_prefill:
        results["prefill"] = run(
            f"prefill (~{args.prefill_tokens} unique tokens per run)",
            lambda: unique_long_prompt(args.prefill_tokens),
            args.base, args.model, 64, 1, max(3, args.repeats // 2))
        p = results["prefill"]
        if p["prompt_tokens"]:
            print(f"  prefill rate: ~{p['prompt_tokens'] / p['ttft']:,.0f} prompt tok/s "
                  f"({p['prompt_tokens']:.0f} tokens in {p['ttft'] * 1000:.0f} ms)")

    print("\n--- summary (median) ---")
    for name, r in results.items():
        print(f"{name:10s} decode {r['decode_tps']:6.1f} tok/s   ttft {r['ttft'] * 1000:8.1f} ms")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"base": args.base, "model": args.model, "results": results}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
