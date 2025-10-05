from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class EmotionState:
    frustration: float = 0.2
    demotivation: float = 0.2
    stress: float = 0.2

    def blend(self, raw: Dict[str, float], alpha: float) -> Dict[str, float]:
        self.frustration = _clamp(
            (1 - alpha) * self.frustration + alpha * raw.get("frustration", 0.0)
        )
        self.demotivation = _clamp(
            (1 - alpha) * self.demotivation + alpha * raw.get("demotivation", 0.0)
        )
        self.stress = _clamp((1 - alpha) * self.stress + alpha * raw.get("stress", 0.0))
        return self.snapshot()

    def snapshot(self) -> Dict[str, float]:
        return {
            "frustration_prob": round(self.frustration, 3),
            "demotivation_prob": round(self.demotivation, 3),
            "stress_prob": round(self.stress, 3),
        }


@dataclass
class EmotionAgent:
    alpha: float = 0.45
    state: Dict[str, EmotionState] = field(default_factory=dict)

    def update(
        self,
        user_id: str,
        *,
        correctness: bool | None,
        confidence: float | None,
        latency_ms: float | None,
        latency_z: float | None,
        error_streak: int | None,
    ) -> Dict[str, float]:
        raw_scores = self._compute_raw_scores(
            correctness=correctness,
            confidence=confidence,
            latency_ms=latency_ms,
            latency_z=latency_z,
            error_streak=error_streak,
        )
        state = self.state.setdefault(user_id, EmotionState())
        return state.blend(raw_scores, self.alpha)

    def get_snapshot(self, user_id: str) -> Optional[Dict[str, float]]:
        state = self.state.get(user_id)
        if not state:
            return None
        return state.snapshot()

    def reset_user(self, user_id: str) -> None:
        self.state.pop(user_id, None)

    def integrate_text(
        self,
        user_id: str,
        emotion_state: Dict[str, float],
        *,
        weight: float | None = None,
    ) -> Dict[str, float]:
        if not emotion_state:
            return self.state.setdefault(user_id, EmotionState()).snapshot()

        alpha = self._resolve_alpha(weight)

        raw = {
            "frustration": float(
                emotion_state.get("frustration_prob")
                or emotion_state.get("frustration")
                or 0.0
            ),
            "demotivation": float(
                emotion_state.get("demotivation_prob")
                or emotion_state.get("demotivation")
                or 0.0
            ),
            "stress": float(
                emotion_state.get("stress_prob") or emotion_state.get("stress") or 0.0
            ),
        }
        state = self.state.setdefault(user_id, EmotionState())
        return state.blend(raw, alpha)

    def _resolve_alpha(self, weight: float | None) -> float:
        if weight is None:
            return _clamp(self.alpha, 0.0, 1.0)
        try:
            value = float(weight)
        except (TypeError, ValueError):
            value = self.alpha
        return _clamp(value, 0.0, 1.0)

    def _compute_raw_scores(
        self,
        *,
        correctness: bool | None,
        confidence: float | None,
        latency_ms: float | None,
        latency_z: float | None,
        error_streak: int | None,
    ) -> Dict[str, float]:
        lat_z = latency_z or 0.0
        streak = error_streak or 0
        latency = float(latency_ms or 0.0)
        conf = float(confidence) if confidence is not None else None

        frustration = 0.1
        demotivation = 0.1
        stress = 0.1

        if correctness is False:
            frustration += 0.4
            demotivation += 0.25
        else:
            demotivation -= 0.05
            frustration -= 0.05

        if lat_z > 1.0:
            frustration += 0.15
            stress += 0.2
        elif lat_z < -0.75:
            frustration -= 0.05

        if streak >= 2:
            frustration += 0.15
            demotivation += 0.1
        if streak >= 3:
            stress += 0.1

        if latency > 1800:
            stress += 0.2
        elif latency < 700:
            stress -= 0.05

        if conf is not None:
            if conf < 40:
                demotivation += 0.25
                stress += 0.15
            elif conf < 60:
                demotivation += 0.1
            elif conf > 80 and correctness is True:
                demotivation -= 0.05
                stress -= 0.05

        frustration = _clamp(frustration)
        demotivation = _clamp(demotivation)
        stress = _clamp(stress)

        return {
            "frustration": frustration,
            "demotivation": demotivation,
            "stress": stress,
        }


__all__ = ["EmotionAgent", "EmotionState"]
