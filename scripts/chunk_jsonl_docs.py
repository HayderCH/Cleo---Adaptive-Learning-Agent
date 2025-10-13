"""Chunk JSONL content into smaller pieces for RAG indexing.

Usage examples:
  python scripts/chunk_jsonl_docs.py --in data/raw/serp_scrapes.jsonl --out data/processed/corpus.jsonl --chunk-size 400

This script extracts the 'text' field from JSONL entries and chunks the actual content.
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
    """Split text into chunks of approximately n_words."""
    words = text.split()
    for i in range(0, len(words), n_words):
        yield " ".join(words[i : i + n_words])


def process_jsonl_file(jsonl_path: Path, out_handle, chunk_size: int):
    """Process JSONL file and chunk the text content."""
    chunk_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract the actual text content
            text_content = entry.get("text", "").strip()
            if not text_content:
                continue

            subject = entry.get("subject", "unknown")
            url = entry.get("url", "")

            # Chunk the text content
            for chunk_idx, chunk in enumerate(chunk_text(text_content, chunk_size)):
                rec = {
                    "id": str(uuid.uuid4()),
                    "source": "serp_scrapes",
                    "subject": subject,
                    "url": url,
                    "chunk_idx": chunk_idx,
                    "text": chunk,
                    "original_entry_id": entry.get("id", str(line_num)),
                }
                out_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                chunk_count += 1

    # Log telemetry event
    log_event(
        "JSONL_CORPUS_CHUNKED",
        "DataPrep",
        {
            "file": str(jsonl_path),
            "chunks_created": chunk_count,
            "chunk_size": chunk_size,
        },
    )

    return chunk_count


def main() -> None:
    p = argparse.ArgumentParser(description="Chunk JSONL content for RAG")
    p.add_argument("--in", dest="infile", required=True, help="Input JSONL file")
    p.add_argument("--out", dest="outfile", required=True, help="Output JSONL path")
    p.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=400,
        help="Chunk size in words",
    )

    args = p.parse_args()

    input_path = Path(args.infile)
    output_path = Path(args.outfile)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_handle:
        chunk_count = process_jsonl_file(input_path, out_handle, args.chunk_size)

    print(f"Created {chunk_count} chunks in {output_path}")


if __name__ == "__main__":
    main()
