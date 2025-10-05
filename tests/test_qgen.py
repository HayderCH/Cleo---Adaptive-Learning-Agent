from __future__ import annotations

import math
from typing import Iterable

import pytest

from services.qgen import evaluator, generator, quality_gate
from services.qgen.schemas import GenerateRequest, Question, QuestionMeta
from scoring import mcq


@pytest.fixture(autouse=True)
def force_template_backend() -> Iterable[None]:
    generator.set_backend_override("template")
    yield
    generator.set_backend_override(None)


def build_valid_question() -> Question:
    meta = QuestionMeta(
        concepts=["linear equations"],
        difficulty_est=0.5,
        bloom="remember",
        source="unit-test",
        version="v-test",
    )
    return Question(
        stem="Define the standard slope-intercept form of a linear equation.",
        choices=[
            "The slope-intercept form is y = mx + b.",
            "Quadratic form x = [-b ± √(b² - 4ac)] / 2a.",
            "An exponential curve written as y = a · b^x.",
            "A polynomial with arbitrary degree and coefficients.",
        ],
        correct_index=0,
        meta=meta,
    )


def test_generator_produces_fact_based_choices() -> None:
    req = GenerateRequest(
        subject="algebra",
        focus_concepts=["linear equations"],
        difficulty_target=0.4,
        bloom_target="remember",
        item_type="mcq",
        learner_snapshot={},
    )
    question = generator.generate(req)

    assert question.choices is not None
    assert len(question.choices) == 4
    assert question.correct_index is not None

    correct_choice = question.choices[question.correct_index]
    assert "slope-intercept form" in correct_choice

    ok, reasons = quality_gate.validate(question)
    assert ok, f"Quality gate should pass generated question: {reasons}"


def test_quality_gate_accepts_valid_question() -> None:
    q = build_valid_question()
    ok, reasons = quality_gate.validate(q)
    assert ok
    assert not reasons


def test_quality_gate_rejects_duplicate_choices() -> None:
    q = build_valid_question()
    q.choices[2] = q.choices[0]
    ok, reasons = quality_gate.validate(q)
    assert not ok
    assert "duplicate" in " ".join(reasons).lower()


def test_mcq_scoring_returns_correctness_and_latency() -> None:
    correctness, latency_ms = mcq.score(2, 2, 0.0, 1.5)
    assert correctness == 1
    assert math.isclose(latency_ms, 1500.0, rel_tol=1e-6)

    correctness, latency_ms = mcq.score(1, 3, 2.0, 2.2)
    assert correctness == 0
    assert math.isclose(latency_ms, 200.0, rel_tol=1e-6)


def test_evaluator_summary_and_report() -> None:
    requests = [
        GenerateRequest(
            subject="algebra",
            focus_concepts=["linear equations"],
            difficulty_target=0.4,
            bloom_target="remember",
            item_type="mcq",
            learner_snapshot={},
        ),
        GenerateRequest(
            subject="machine learning",
            focus_concepts=["gradient descent"],
            difficulty_target=0.6,
            bloom_target="understand",
            item_type="mcq",
            learner_snapshot={},
        ),
    ]

    summary = evaluator.evaluate_requests(requests)
    assert summary.total == 2
    assert summary.passed >= 1
    assert summary.failed == summary.total - summary.passed
    report = evaluator.make_quality_report(summary, tag="unit-test")
    assert report["type"] == "QUALITY_REPORT"
    assert report["payload"]["total"] == 2
    for failure in report["payload"].get("failures", []):
        assert "question" in failure
        assert "passed" in failure
