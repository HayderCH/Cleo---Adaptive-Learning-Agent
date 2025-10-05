from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AttentionStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    error_streak: int = 0

    def update(
        self,
        latency_ms: float,
        correct: bool | None,
    ) -> Dict[str, float | int]:
        correct_flag = True if correct is None else bool(correct)
        count_prev = self.count
        mean_prev = self.mean
        delta = latency_ms - mean_prev
        count_new = count_prev + 1
        mean_new = mean_prev + (delta / count_new)
        delta2 = latency_ms - mean_new
        m2_new = self.m2 + delta * delta2

        if count_new > 1 and m2_new > 0.0:
            variance = m2_new / (count_new - 1)
            std = variance**0.5
            latency_z = (latency_ms - mean_new) / std if std > 1e-6 else 0.0
        else:
            latency_z = 0.0

        error_streak = 0 if correct_flag else self.error_streak + 1

        self.count = count_new
        self.mean = mean_new
        self.m2 = m2_new
        self.error_streak = error_streak

        return {
            "latency_z": round(latency_z, 2),
            "error_streak": error_streak,
            "sample_size": count_new,
        }

    def snapshot(self) -> Dict[str, float | int]:
        return {
            "count": self.count,
            "mean_latency_ms": round(self.mean, 2),
            "error_streak": self.error_streak,
        }


@dataclass
class AttentionAgent:
    state: Dict[str, AttentionStats] = field(default_factory=dict)

    def update(
        self,
        user_id: str,
        latency_ms: float,
        correct: bool | None,
    ) -> Dict[str, float | int]:
        stats = self.state.setdefault(user_id, AttentionStats())
        return stats.update(latency_ms=float(latency_ms), correct=correct)

    def get_snapshot(self, user_id: str) -> Optional[Dict[str, float | int]]:
        stats = self.state.get(user_id)
        if not stats:
            return None
        return stats.snapshot()

    def reset_user(self, user_id: str) -> None:
        self.state.pop(user_id, None)


__all__ = ["AttentionAgent", "AttentionStats"]
