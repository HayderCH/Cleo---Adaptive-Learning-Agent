from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

from services.qgen import generator, quality_gate
from services.qgen.schemas import GenerateRequest, Question

EvaluationSink = Callable[[dict], None]


@dataclass
class EvaluationSample:
    request: GenerateRequest
    question: Question
    passed: bool
    reasons: List[str]
    latency_ms: float

    def to_report(self) -> dict:
        return {
            "request": self.request.model_dump(),
            "question": self.question.model_dump(),
            "passed": self.passed,
            "reasons": list(self.reasons),
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class EvaluationSummary:
    samples: List[EvaluationSample] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def passed(self) -> int:
        return sum(1 for sample in self.samples if sample.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        if not self.total:
            return 0.0
        return round(self.passed / self.total, 4)

    def to_report(self, max_failures: int = 10) -> dict:
        failures = [s.to_report() for s in self.samples if not s.passed]
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "failures": failures[:max_failures],
        }


def evaluate_requests(
    requests: Iterable[GenerateRequest],
    emit_report: bool = False,
    sink: Optional[EvaluationSink] = None,
    report_tag: str | None = None,
) -> EvaluationSummary:
    samples: List[EvaluationSample] = []
    for req in requests:
        start = time.perf_counter()
        question = generator.generate(req)
        latency_ms = (time.perf_counter() - start) * 1000.0
        ok, reasons = quality_gate.validate(question)
        samples.append(
            EvaluationSample(
                request=req,
                question=question,
                passed=ok,
                reasons=reasons,
                latency_ms=latency_ms,
            )
        )
    summary = EvaluationSummary(samples=samples)
    if emit_report and sink is not None:
        payload = make_quality_report(summary, tag=report_tag)
        sink(payload)
    return summary


def make_quality_report(
    summary: EvaluationSummary,
    tag: str | None = None,
) -> dict:
    report = summary.to_report()
    if tag:
        report["tag"] = tag
    return {
        "type": "QUALITY_REPORT",
        "source": "QualityGate",
        "payload": report,
    }


def emit_quality_report(
    summary: EvaluationSummary,
    telemetry_base: str,
) -> dict:
    import requests

    payload = make_quality_report(summary)
    response = requests.post(
        f"{telemetry_base.rstrip('/')}/events",
        json=payload,
        timeout=5,
    )
    response.raise_for_status()
    return response.json()
