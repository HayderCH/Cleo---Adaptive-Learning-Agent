from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# Contracts (aligned with docs/contracts.md and data/schema.json)
class EventEnvelope(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal[
        "RESPONSE_RAW",
        "DIAGNOSTIC_EVENT",
        "STRATEGY_UPDATE",
        "QUESTION_PLAN",
        "GENERATED_QUESTIONS",
        "REVIEW_TASK",
        "QUALITY_REPORT",
    ]
    source: Literal[
        "UI",
        "Pedagogical",
        "Attention",
        "Emotional",
        "Memory",
        "Coach",
        "QualityGate",
    ]
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp())
    )
    payload: Dict[str, Any]


class Segment(BaseModel):
    segment_id: str
    topic: Optional[str] = None
    text: str
    concepts: Optional[List[str]] = None


class Question(BaseModel):
    question_id: str
    bloom: Literal[
        "Remember",
        "Understand",
        "Apply",
        "Analyze",
        "Evaluate",
        "Create",
    ]
    type: Optional[str] = None
    stem: str
    answer_key: str
    distractors: Optional[List[str]] = None


class Response(BaseModel):
    user_id: str
    correct: bool
    user_answer: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=100)
    latency_ms: int
    hint_used: Optional[bool] = None


class Affect(BaseModel):
    labels: Optional[
        List[
            Literal[
                "Frustration",
                "Demotivation",
                "Stress",
                "Neutral",
            ]
        ]
    ] = None
    valence: Optional[float] = Field(default=None, ge=-1, le=1)
    arousal: Optional[float] = Field(default=None, ge=0, le=1)
    source: Optional[Literal["self_report", "text_model", "interaction_model"]] = None


class DerivedSignals(BaseModel):
    latency_z: Optional[float] = None
    error_streak: Optional[int] = None
    novelty_score: Optional[float] = None


class Intervention(BaseModel):
    recommended: Optional[
        Literal[
            "Break",
            "Encouragement",
            "Clarify",
            "EasierTask",
            "ConfidenceProbe",
            "None",
        ]
    ] = None
    accepted: Optional[bool] = None


class Consent(BaseModel):
    store_free_text: Optional[bool] = None
    store_audio: Optional[bool] = None


class Timestamps(BaseModel):
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class Sample(BaseModel):
    sample_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    segment: Segment
    question: Question
    response: Response
    affect: Optional[Affect] = None
    derived_signals: Optional[DerivedSignals] = None
    intervention: Optional[Intervention] = None
    consent: Optional[Consent] = None
    timestamps: Optional[Timestamps] = None


app = FastAPI(title="Telemetry Service", version="0.1.0")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/events")
def post_event(ev: EventEnvelope) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = DATA_RAW / f"events_{ts}.jsonl"
    _append_jsonl(out, ev.model_dump())
    return {"status": "queued", "file": str(out), "id": ev.id}


@app.post("/ingest_sample")
def ingest_sample(sample: Sample) -> Dict[str, Any]:
    """Accept a full schema-compliant sample and append to processed JSONL."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = DATA_PROCESSED / f"samples_{ts}.jsonl"
    _append_jsonl(out, sample.model_dump())
    return {
        "status": "stored",
        "file": str(out),
        "sample_id": sample.sample_id,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.telemetry.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
