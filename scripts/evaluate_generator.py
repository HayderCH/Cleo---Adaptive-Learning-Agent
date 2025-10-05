from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

from services.qgen import evaluator, generator
from services.qgen.schemas import GenerateRequest

DEFAULT_REQUESTS: List[GenerateRequest] = [
    GenerateRequest(
        subject="algebra",
        focus_concepts=["linear equations"],
        difficulty_target=0.4,
        bloom_target="remember",
        item_type="mcq",
        learner_snapshot={},
    ),
    GenerateRequest(
        subject="calculus",
        focus_concepts=["derivatives"],
        difficulty_target=0.6,
        bloom_target="understand",
        item_type="mcq",
        learner_snapshot={},
    ),
]


def _load_requests(path: Path) -> List[GenerateRequest]:
    raw: Iterable[dict]
    if path.suffix.lower() == ".jsonl":
        raw = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            raw = data
        else:
            raise ValueError("Input file must contain a list of requests")
    return [GenerateRequest(**item) for item in raw]


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate question generator quality gate performance"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to JSON/JSONL file of GenerateRequest objects",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the summary report",
        default=None,
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "template", "transformer"],
        default="auto",
    )
    parser.add_argument(
        "--emit-quality-report",
        action="store_true",
        help="Send summary to telemetry service",
    )
    parser.add_argument(
        "--telemetry-base",
        default=os.getenv("TELEMETRY_BASE", "http://127.0.0.1:8000"),
        help="Telemetry service base URL",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag to include in the quality report",
    )
    args = parser.parse_args()

    if args.backend == "auto":
        generator.set_backend_override(None)
    else:
        generator.set_backend_override(args.backend)

    requests = DEFAULT_REQUESTS
    if args.input:
        requests = _load_requests(args.input)

    summary = evaluator.evaluate_requests(requests)
    report = evaluator.make_quality_report(summary, tag=args.tag)

    print(json.dumps(report, indent=2))

    if args.output:
        _write_output(args.output, report)

    if args.emit_quality_report:
        response = evaluator.emit_quality_report(summary, args.telemetry_base)
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
