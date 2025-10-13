from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


BloomLevel = Literal[
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
]
ItemType = Literal["mcq", "open"]


class GenerateRequest(BaseModel):
    subject: str
    focus_concepts: List[str] = Field(default_factory=list)
    difficulty_target: float = Field(ge=0.0, le=1.0, default=0.5)
    bloom_target: BloomLevel = "understand"
    item_type: ItemType = "mcq"
    learner_snapshot: dict = Field(default_factory=dict)
    policy_overrides: Optional[dict] = None


class QuestionMeta(BaseModel):
    concepts: List[str]
    difficulty_est: float = Field(ge=0.0, le=1.0)
    bloom: BloomLevel
    source: str = "template"
    version: str = "v0"
    origin: Optional[str] = None
    subject: str = ""
    # RAG pipeline information
    retrieved_chunks: Optional[List[dict]] = None


class Question(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stem: str
    # MCQ fields
    choices: Optional[List[str]] = None
    correct_index: Optional[int] = None
    answer_key: Optional[str] = None
    # Open-ended optional fields could be added later
    answer_expl: Optional[str] = None
    meta: QuestionMeta
