import math

from services.ui.app import (
    _aggregate_self_report,
    _build_emotion_state_from_buckets,
    _flatten_predictions,
)


def test_flatten_predictions_handles_nested_structure():
    raw = [
        [
            {"label": "Anger", "score": 0.66},
            {"label": "Neutral", "score": 0.12},
        ],
        [{"label": "Fear", "score": 0.51}],
    ]

    flattened = _flatten_predictions(raw)

    assert flattened == [
        {"label": "Anger", "score": 0.66},
        {"label": "Neutral", "score": 0.12},
        {"label": "Fear", "score": 0.51},
    ]


def test_aggregate_self_report_max_per_bucket():
    predictions = [
        {"label": "Frustration", "score": 0.42},
        {"label": "Anger", "score": 0.71},
        {"label": "Fear", "score": 0.33},
        {"label": "Boredom", "score": 0.64},
        {"label": "Fear", "score": 0.85},
    ]

    aggregated = _aggregate_self_report(predictions)

    assert math.isclose(aggregated["frustration"], 0.71, rel_tol=1e-6)
    assert math.isclose(aggregated["stress"], 0.85, rel_tol=1e-6)
    assert math.isclose(aggregated["demotivation"], 0.64, rel_tol=1e-6)


def test_build_emotion_state_from_buckets_rounds():
    buckets = {
        "frustration": 0.12345,
        "demotivation": 0.98765,
        "stress": 0.54321,
    }

    state = _build_emotion_state_from_buckets(buckets)

    assert state == {
        "frustration_prob": 0.123,
        "demotivation_prob": 0.988,
        "stress_prob": 0.543,
    }
