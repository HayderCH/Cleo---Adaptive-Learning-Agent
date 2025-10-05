from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AttentionSignals:
    latency_z: Optional[float] = None
    error_streak: int = 0
    sample_size: int = 0


@dataclass
class Strategy:
    subject: str
    focus_concepts: list[str]
    difficulty_target: float  # 0..1
    bloom_target: str  # keep string for simplicity
    item_type: str = "mcq"


class BandController:
    def __init__(self, step: float = 0.1, hysteresis: float = 0.05):
        self.step = step
        self.hysteresis = hysteresis

    def update(
        self,
        difficulty: float,
        correct: bool,
        attention: AttentionSignals | None = None,
    ) -> float:
        result = difficulty
        if attention and attention.error_streak >= 2:
            result = difficulty - (self.step + 0.05)
        elif attention and (attention.latency_z or 0.0) > 1.25:
            result = difficulty - self.step
        elif correct and (attention and (attention.latency_z or 0.0) < -0.75):
            result = difficulty + (self.step + 0.05)
        elif correct:
            result = difficulty + self.step
        elif not correct:
            result = difficulty - self.step

        upper = 1.0 - self.hysteresis
        lower = 0.0 + self.hysteresis
        result = max(lower, min(upper, result))
        return round(result, 2)


def mix_bloom(
    history: List[str],
    attention: AttentionSignals | None = None,
    last_correct: Optional[bool] = None,
) -> str:
    order = ["remember", "understand", "apply", "analyze"]
    if attention and attention.error_streak >= 2:
        return "remember"
    if attention and (attention.latency_z or 0.0) > 1.25:
        return "understand"

    if attention and (attention.latency_z or 0.0) < -1.0:
        return "analyze"
    if last_correct is True and attention and attention.sample_size > 5:
        return "apply"

    if not history:
        return order[1]
    last = history[-1]
    try:
        idx = order.index(last)
        return order[(idx + 1) % len(order)]
    except ValueError:
        return order[1]
