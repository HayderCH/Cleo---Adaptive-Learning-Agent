import json
import os
import time
import uuid
from typing import Any, Dict

import requests


BASE = os.environ.get("TELEMETRY_BASE", "http://127.0.0.1:8000")
TIMEOUT = 5


def pretty(resp: requests.Response) -> None:
    try:
        print(resp.status_code, json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.status_code, resp.text)


def test_health() -> None:
    r = requests.get(f"{BASE}/healthz", timeout=TIMEOUT)
    pretty(r)


def test_event() -> None:
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
    r = requests.post(f"{BASE}/events", json=payload, timeout=TIMEOUT)
    pretty(r)


def test_sample() -> None:
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
    r = requests.post(f"{BASE}/ingest_sample", json=sample, timeout=TIMEOUT)
    pretty(r)


if __name__ == "__main__":
    test_health()
    test_event()
    test_sample()
