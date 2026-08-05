#!/usr/bin/env python3
"""Corpus health check for the server's RAG — and the fix for silent ingestion failure.

The problem this exists for: on 2026-07-21 the RAG had been answering homelab
questions from a June snapshot because `notes/current-infra-status.md`
was never indexed. Two other files sat in `processed/` with zero chunks.
Nothing anywhere surfaced either fact.

Checks, in order of how badly they bite:
  GHOST    in processed/ but no chunks indexed  -> ingestion failed SILENTLY
  MISSING  local doc never ingested at all      -> RAG cannot answer about it
  STALE    local file edited after it was indexed -> RAG serves the old version
  STUCK    sitting in pending/ or error/

Read-only by default. `--sync` uploads MISSING files only.

Safety: this tool never deletes and never overwrites. Re-uploading an already
indexed file is deliberately NOT automatic, because the ingestion template has a
known "Delete Previous Vectors" bug — re-ingesting duplicates chunks rather than
replacing them. STALE files are therefore reported, not fixed; use
--sync-stale only when you have accepted that duplicate risk.

Exit 0 clean, 1 problems found (so it can drive a monitor), 2 could not reach.

Usage:
    python eval/ingest_health.py                 # report
    python eval/ingest_health.py --sync          # upload MISSING docs
    python eval/ingest_health.py --json out.json # machine-readable snapshot
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE = os.environ.get("RAG_BASE", "http://localhost:8090")
# Corpus scope: which local directories hold the documents the RAG should serve.
WATCH_DIRS = os.environ.get("WATCH_DIRS", "docs notes").split()


def get(path, timeout=20):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def upload(path, timeout=120):
    name = Path(path).name
    boundary = "----claude" + uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{name}"\r\nContent-Type: text/markdown\r\n\r\n').encode()
    body += Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{BASE}/api/upload", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def local_docs(root):
    found = {}
    for d in WATCH_DIRS:
        base = Path(root) / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.md")):
            if p.name.endswith("-test-plan.md"):
                continue
            rel = p.as_posix()
            # The doc-brain's own process docs (its spec, meta-area notes) are not homelab
            # knowledge -- keep them out of the factual RAG corpus.
            if "/meta/" in rel or "/doc-brain/" in rel:
                continue
            found[p.name] = p
    return found


def main():
    ap = argparse.ArgumentParser(description="Check (and optionally repair) RAG corpus coverage.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--sync", action="store_true", help="upload MISSING docs")
    ap.add_argument("--sync-stale", action="store_true",
                    help="also re-upload STALE docs (WARNING: duplicates chunks)")
    ap.add_argument("--json", help="write the full snapshot here")
    args = ap.parse_args()

    try:
        indexed = get("/api/indexed")
        files = get("/api/files")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"cannot reach rag-uploader at {BASE}: {exc}", file=sys.stderr)
        return 2

    by_source = {d["source"]: d for d in indexed.get("documents", [])}
    processed = {f if isinstance(f, str) else f.get("name", "") for f in files.get("processed", [])}
    pending = [f if isinstance(f, str) else f.get("name", "") for f in files.get("pending", [])]
    errored = [f if isinstance(f, str) else f.get("name", "") for f in files.get("error", [])]
    local = local_docs(args.root)

    ghosts = sorted(n for n in processed if n and n not in by_source)
    missing = sorted(n for n in local if n not in by_source)
    stale = []
    for name, doc in by_source.items():
        path = local.get(name)
        if not path:
            continue
        indexed_at = datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00"))
        edited_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if edited_at > indexed_at:
            stale.append((name, indexed_at, edited_at))

    print(f"corpus: {indexed.get('total', len(by_source))} indexed docs, "
          f"{sum(d['chunk_count'] for d in by_source.values())} chunks, "
          f"{len(processed)} in processed/\n")

    if ghosts:
        print(f"GHOST ({len(ghosts)}) — processed but zero chunks. Ingestion failed silently:")
        for n in ghosts:
            print(f"  {n}")
    if missing:
        print(f"\nMISSING ({len(missing)}) — local doc never ingested:")
        for n in missing:
            print(f"  {n}")
    if stale:
        print(f"\nSTALE ({len(stale)}) — edited after indexing, RAG serves the old text:")
        for n, i, e in stale:
            print(f"  {n}  indexed {i:%Y-%m-%d}, edited {e:%Y-%m-%d}")
    if pending:
        print(f"\nSTUCK in pending/ ({len(pending)}): {', '.join(pending)}")
    if errored:
        print(f"\nERROR dir ({len(errored)}): {', '.join(errored)}")
    if not any([ghosts, missing, stale, pending, errored]):
        print("all clean")

    uploaded = []
    if args.sync or args.sync_stale:
        targets = [local[n] for n in missing] if (args.sync or args.sync_stale) else []
        if args.sync_stale:
            print("\n!! --sync-stale re-uploads already-indexed files. The ingestion template's "
                  "'Delete Previous Vectors' bug means chunks DUPLICATE rather than replace.")
            targets += [local[n] for n, _, _ in stale]
        for path in targets:
            try:
                res = upload(path)
                uploaded.append(res.get("filename", path.name))
                print(f"  uploaded {res.get('filename')} ({res.get('size')} bytes)")
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"  FAILED {path.name}: {exc}", file=sys.stderr)
        if uploaded:
            print(f"\n{len(uploaded)} queued. Ingestion takes ~2-4 min. Re-run to confirm "
                  "they leave MISSING and do not appear as GHOST.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base": BASE,
            "indexed": {k: v["chunk_count"] for k, v in by_source.items()},
            "ghosts": ghosts, "missing": missing,
            "stale": [n for n, _, _ in stale],
            "pending": pending, "errored": errored, "uploaded": uploaded,
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if (ghosts or missing or stale or errored) else 0


if __name__ == "__main__":
    sys.exit(main())
