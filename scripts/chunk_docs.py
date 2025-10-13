"""Chunk plain text files into JSONL records for later RAG indexing.

Usage examples:
  python scripts/chunk_docs.py --in docs/sample_corpus.txt --out data/processed/corpus.jsonl --chunk-size 400
  python scripts/chunk_docs.py --indir docs/ --out data/processed/corpus.jsonl --chunk-size 300

The script is CPU-only and uses simple whitespace tokenization (words) to split text into chunks.
Each line in the output JSONL will be a dict with keys: id, source, subject, chunk_idx, text.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def log_event(event_type: str, source: str, details: dict = None):
    """Log an event to the telemetry system (simulation)."""
    event = {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": source,
        "timestamp": int(datetime.now(timezone.utc).timestamp()),
        "details": details or {},
    }
    events_file = Path("data/raw/events.jsonl")
    events_file.parent.mkdir(parents=True, exist_ok=True)
    with open(events_file, "a", encoding="utf-8") as f:
        json.dump(event, f)
        f.write("\n")


def chunk_text(text: str, n_words: int = 400) -> Iterator[str]:
    words = text.split()
    for i in range(0, len(words), n_words):
        yield " ".join(words[i : i + n_words])


def process_file(
    path: Path, out_handle, subject: str | None, source_label: str, chunk_size: int
):
    # Read file with fallback encodings to handle Windows-created files
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        data = path.read_bytes()
        text = None
        for enc in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except Exception:
                continue
        if text is None:
            # As a last resort, replace invalid bytes
            text = data.decode("utf-8", errors="replace")
    chunk_count = 0
    for idx, chunk in enumerate(chunk_text(text, chunk_size)):
        rec = {
            "id": str(uuid.uuid4()),
            "source": source_label,
            "file": str(path.name),
            "subject": subject or "",
            "chunk_idx": idx,
            "text": chunk,
        }
        out_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        chunk_count += 1

    # Log telemetry event
    log_event(
        "CORPUS_CHUNKED",
        "DataPrep",
        {
            "file": str(path),
            "chunks_created": chunk_count,
            "chunk_size": chunk_size,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--in", dest="infile", help="Single text file to chunk")
    group.add_argument("--indir", dest="indir", help="Directory of text files to chunk")
    p.add_argument("--out", dest="outfile", required=True, help="Output JSONL path")
    p.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=400,
        help="Chunk size in words",
    )
    p.add_argument(
        "--subject", dest="subject", help="Optional subject label for chunks"
    )
    p.add_argument("--source", dest="source", default="textfile", help="Source label")
    args = p.parse_args()

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.infile:
        infile = Path(args.infile)
        if not infile.exists():
            raise SystemExit(f"Input file not found: {infile}")
        with out_path.open("w", encoding="utf-8") as out:
            process_file(infile, out, args.subject, args.source, args.chunk_size)
    else:
        indir = Path(args.indir)
        if not indir.exists():
            raise SystemExit(f"Input directory not found: {indir}")
        txt_files = sorted(indir.glob("*.txt"))
        if not txt_files:
            raise SystemExit(f"No .txt files found in {indir}")
        with out_path.open("w", encoding="utf-8") as out:
            for f in txt_files:
                process_file(f, out, args.subject, args.source, args.chunk_size)


if __name__ == "__main__":
    main()
