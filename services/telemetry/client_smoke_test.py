import json
import os
import time
import uuid
from typing import Any, Dict

import requests
import pytest
from requests.exceptions import RequestException


BASE = os.environ.get("TELEMETRY_BASE", "http://127.0.0.1:8000")
TIMEOUT = 5


def pretty(resp: requests.Response) -> None:
    try:
        print(resp.status_code, json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.status_code, resp.text)


def _assure_service(base: str) -> None:
    try:
        requests.get(f"{base}/healthz", timeout=TIMEOUT)
    except RequestException as exc:
        pytest.skip(f"Telemetry service unavailable at {base}: {exc}")


@pytest.fixture(scope="module")
def telemetry_base() -> str:
    _assure_service(BASE)
    return BASE


def test_health(telemetry_base: str) -> None:
    r = requests.get(f"{telemetry_base}/healthz", timeout=TIMEOUT)
    pretty(r)
    assert r.status_code == 200


def test_event(telemetry_base: str) -> None:
    payload: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "type": "RESPONSE_RAW",
        "source": "UI",
        "timestamp": int(time.time()),
        "payload": {
            "user_id": "u1",
            "question_id": "q1",
            "segment_id": "s1",
            "correct": True,
            "user_answer": "example",
            "confidence": 80,
            "latency_ms": 1200,
        },
    }
    try:
        r = requests.post(
            f"{telemetry_base}/events",
            json=payload,
            timeout=TIMEOUT,
        )
    except RequestException as exc:
        pytest.skip(f"/events unreachable: {exc}")
    pretty(r)
    assert r.status_code < 500


def test_sample(telemetry_base: str) -> None:
    sample: Dict[str, Any] = {
        "segment": {"segment_id": "s1", "text": "Intro to fractions"},
        "question": {
            "question_id": "q1",
            "bloom": "Understand",
            "stem": "What is 1/2 + 1/2?",
            "answer_key": "1",
        },
        "response": {
            "user_id": "u1",
            "correct": True,
            "user_answer": "1",
            "confidence": 90,
            "latency_ms": 900,
        },
        "affect": {"labels": ["Neutral"], "source": "self_report"},
    }
    try:
        r = requests.post(
            f"{telemetry_base}/ingest_sample",
            json=sample,
            timeout=TIMEOUT,
        )
    except RequestException as exc:
        pytest.skip(f"/ingest_sample unreachable: {exc}")
    pretty(r)
    assert r.status_code < 500


if __name__ == "__main__":
    test_health()
    test_event()
    test_sample()
