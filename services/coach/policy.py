from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from services.coach.memory import ReviewItem

try:  # pragma: no cover - runtime fallback when strategy module unavailable
    from services.coach.strategy import AttentionSignals
except ImportError:  # Fallback when project root missing from sys.path

    @dataclass
    class AttentionSignals:  # type: ignore[override]
        latency_z: Optional[float] = None
        error_streak: int = 0
        sample_size: int = 0


try:  # pragma: no cover - optional component
    from services.emotion.interpreter import EmotionInterpreter
except ImportError:  # pragma: no cover - when emotion module unavailable
    EmotionInterpreter = None  # type: ignore[assignment]


@dataclass
class CoachPolicyState:
    """Mutable state we carry between successive policy decisions."""

    last_adjust_index: int = -999
    last_bloom: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, object]],
    ) -> "CoachPolicyState":
        if not data:
            return cls()
        return cls(
            last_adjust_index=int(data.get("last_adjust_index", -999)),
            last_bloom=cls._normalize_bloom(data.get("last_bloom")),
        )

    @staticmethod
    def _normalize_bloom(value: object) -> Optional[str]:
        if isinstance(value, str) and value:
            return value.lower()
        return None


@dataclass
class StrategyDecision:
    difficulty: float
    bloom: str
    item_type: str
    focus_concepts: List[str]
    mode: str
    band_status: str
    rationale: Dict[str, object]
    interventions: List[str] = field(default_factory=list)
    emotion_notes: List[str] = field(default_factory=list)


class CoachPolicy:
    """Encapsulates policy-driven strategy selection for the coach."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = config_path or (self.root / "configs" / "agents.yaml")
        self.config = self._load_config(self.config_path)

        policy_cfg = self.config.get("policy", {})
        self.band_low, self.band_high = self._pair(
            policy_cfg.get("target_success_band", [0.7, 0.85])
        )
        bloom_weights = policy_cfg.get("bloom_target", {})
        self.bloom_weights = self._normalize_weights(bloom_weights)
        hysteresis = policy_cfg.get("hysteresis", {})
        self.min_consistent_events = int(hysteresis.get("min_consistent_events", 2))
        self.plan_cooldown = int(hysteresis.get("plan_update_cooldown_questions", 3))

        difficulty_cfg = self.config.get("difficulty", {})
        self.difficulty_min = float(difficulty_cfg.get("min", 0.2))
        self.difficulty_max = float(difficulty_cfg.get("max", 0.9))
        self.adjust_step = float(difficulty_cfg.get("step", 0.1))

        attention_cfg = self.config.get("attention", {})
        self.latency_threshold = float(attention_cfg.get("latency_z_threshold", 1.0))
        self.error_streak_threshold = int(
            attention_cfg.get("error_streak_threshold", 2)
        )

        self.emotion_interpreter = self._init_emotion_interpreter()

    @staticmethod
    def _init_emotion_interpreter():
        if EmotionInterpreter is None:  # pragma: no cover - defensive
            return None
        try:
            return EmotionInterpreter.from_config()
        except (OSError, yaml.YAMLError):
            return None

    def decide_next(
        self,
        *,
        subject: str,
        base_concepts: Sequence[str],
        base_difficulty: float,
        learner_state: Dict,
        history: Sequence[Dict[str, object]],
        attention: Optional[AttentionSignals],
        emotion: Optional[Dict[str, float]],
        review_item: Optional[ReviewItem],
        preferred_item_type: str,
        last_difficulty: Optional[float],
        question_index: int,
        state: Optional[CoachPolicyState] = None,
    ) -> Tuple[StrategyDecision, CoachPolicyState]:
        state = state or CoachPolicyState()
        mastery = learner_state.get("mastery", {})

        mode = "review" if review_item else "learn"
        focus_concepts = self._select_focus_concepts(
            base_concepts,
            mastery,
            review_item,
        )

        rolling_acc, acc_samples = self._rolling_accuracy(history)
        band_status = self._band_status(rolling_acc, len(acc_samples))
        direction = self._difficulty_direction(band_status)
        direction = self._apply_attention(direction, attention)

        if (
            direction != 0
            and (question_index - state.last_adjust_index) < self.plan_cooldown
        ):
            direction = 0
        elif direction != 0:
            state.last_adjust_index = question_index

        starting_difficulty = (
            last_difficulty if last_difficulty is not None else base_difficulty
        )
        difficulty = self._clamp(
            starting_difficulty + direction * self.adjust_step,
            self.difficulty_min,
            self.difficulty_max,
        )
        if mode == "review":
            difficulty = min(difficulty, base_difficulty)

        bloom = self._choose_bloom(history, attention, state.last_bloom, mode)
        state.last_bloom = bloom

        item_type = self._choose_item_type(
            preferred_item_type,
            mastery,
            focus_concepts,
            band_status,
            mode,
        )

        emotion_snapshot = emotion or {}
        emotion_notes: List[str] = []
        interventions: List[str] = []

        if self.emotion_interpreter:
            attention_dict: Optional[Dict[str, float | int]] = None
            if attention:
                attention_dict = {
                    "latency_z": getattr(attention, "latency_z", None),
                    "error_streak": getattr(attention, "error_streak", None),
                }
            interventions = self.emotion_interpreter.suggest_interventions(
                emotion_snapshot or None,
                attention_dict,
            )

        frustration = float(emotion_snapshot.get("frustration_prob", 0.0) or 0.0)
        demotivation = float(emotion_snapshot.get("demotivation_prob", 0.0) or 0.0)
        stress = float(emotion_snapshot.get("stress_prob", 0.0) or 0.0)

        emotion_adjust = 0.0
        if frustration >= 0.6 or stress >= 0.65:
            emotion_adjust -= self.adjust_step
            emotion_notes.append(
                "Easing difficulty due to high frustration/stress levels"
            )

        if demotivation >= 0.6 and mode != "review":
            if item_type == "mcq":
                item_type = "open"
                emotion_notes.append("Switching to open response to re-engage learner")
            else:
                emotion_notes.append("Keeping varied format to support motivation")

        if stress >= 0.6 and bloom not in {"remember", "understand"}:
            bloom = "remember"
            emotion_notes.append("Reducing Bloom level in response to stress signal")

        if emotion_adjust != 0.0:
            difficulty = self._clamp(
                difficulty + emotion_adjust,
                self.difficulty_min,
                self.difficulty_max,
            )

        difficulty = round(difficulty, 2)

        rationale: Dict[str, object] = {
            "subject": subject,
            "rolling_accuracy": rolling_acc,
            "accuracy_samples": len(acc_samples),
            "band_status": band_status,
            "focus_concepts": focus_concepts,
            "mode": mode,
            "attention": {
                "latency_z": getattr(attention, "latency_z", None),
                "error_streak": getattr(attention, "error_streak", None),
            },
            "mastery_snapshot": {
                concept: float(mastery.get(concept, 0.5)) for concept in focus_concepts
            },
        }
        if emotion_snapshot or emotion_notes or interventions:
            rationale["emotion"] = {
                "snapshot": emotion_snapshot,
                "notes": emotion_notes,
                "interventions": interventions,
            }

        decision = StrategyDecision(
            difficulty=difficulty,
            bloom=bloom,
            item_type=item_type,
            focus_concepts=focus_concepts,
            mode=mode,
            band_status=band_status,
            rationale=rationale,
            interventions=interventions,
            emotion_notes=emotion_notes,
        )
        return decision, state

    # --- internal helpers -------------------------------------------------

    @staticmethod
    def _pair(values: Sequence[float]) -> Tuple[float, float]:
        if not values:
            return 0.7, 0.85
        if len(values) == 1:
            return float(values[0]), float(values[0]) + 0.1
        return float(values[0]), float(values[1])

    @staticmethod
    def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
        if not raw:
            return {
                "remember": 0.25,
                "understand": 0.25,
                "apply": 0.25,
                "analyze": 0.25,
            }
        normalized = {k.lower(): float(v) for k, v in raw.items()}
        total = sum(normalized.values())
        if total <= 0:
            count = len(normalized) or 1
            return {
                k: 1 / count
                for k in normalized or {"remember": None, "understand": None}
            }
        return {k: v / total for k, v in normalized.items()}

    def _load_config(self, path: Path) -> Dict[str, object]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    @staticmethod
    def _select_focus_concepts(
        base_concepts: Sequence[str],
        mastery: Dict[str, float],
        review_item: Optional[ReviewItem],
    ) -> List[str]:
        if review_item:
            return [review_item.concept]

        pool: List[str] = []
        seen = set()
        for concept in base_concepts:
            key = concept.strip()
            if key and key.lower() not in seen:
                pool.append(key)
                seen.add(key.lower())

        if not pool:
            pool = list(mastery.keys())

        if not pool:
            return []

        def mastery_score(concept: str) -> float:
            return float(mastery.get(concept.lower(), mastery.get(concept, 0.5)))

        pool.sort(key=mastery_score)
        return pool[:2]

    def _rolling_accuracy(
        self,
        history: Sequence[Dict[str, object]],
        window: int = 6,
    ) -> Tuple[Optional[float], List[bool]]:
        samples: List[bool] = []
        for entry in reversed(history):
            value = entry.get("correct")
            if value is None:
                continue
            samples.append(bool(value))
            if len(samples) >= window:
                break
        if not samples:
            return None, []
        accuracy = sum(1 for v in samples if v) / len(samples)
        return accuracy, samples

    def _band_status(
        self,
        accuracy: Optional[float],
        sample_count: int,
    ) -> str:
        if accuracy is None or sample_count < self.min_consistent_events:
            return "insufficient"
        if accuracy < self.band_low:
            return "below"
        if accuracy > self.band_high:
            return "above"
        return "in_band"

    @staticmethod
    def _difficulty_direction(band_status: str) -> int:
        if band_status == "below":
            return -1
        if band_status == "above":
            return 1
        return 0

    def _apply_attention(
        self,
        direction: int,
        attention: Optional[AttentionSignals],
    ) -> int:
        if not attention:
            return direction
        if attention.error_streak >= self.error_streak_threshold:
            return min(direction, -1)
        latency = attention.latency_z or 0.0
        if latency > self.latency_threshold:
            return min(direction, -1)
        if latency < -self.latency_threshold:
            return max(direction, 1)
        return direction

    def _choose_bloom(
        self,
        history: Sequence[Dict[str, object]],
        attention: Optional[AttentionSignals],
        last_bloom: Optional[str],
        mode: str,
    ) -> str:
        if attention and attention.error_streak >= self.error_streak_threshold:
            return "remember"
        if attention and (attention.latency_z or 0.0) > self.latency_threshold:
            return "understand"
        if attention and (attention.latency_z or 0.0) < -self.latency_threshold:
            return "analyze"

        if mode == "review" and last_bloom:
            return last_bloom

        counts = Counter(
            (entry.get("bloom") or "").lower()
            for entry in history
            if entry.get("bloom")
        )
        total = sum(counts.values()) or 1

        best_level = None
        best_gap = float("-inf")
        for level, target_weight in self.bloom_weights.items():
            actual = counts.get(level, 0) / total
            gap = target_weight - actual
            if gap > best_gap:
                best_gap = gap
                best_level = level

        if best_level:
            return best_level

        return last_bloom or next(iter(self.bloom_weights.keys()))

    @staticmethod
    def _choose_item_type(
        preferred: str,
        mastery: Dict[str, float],
        focus_concepts: Sequence[str],
        band_status: str,
        mode: str,
    ) -> str:
        preferred = preferred or "mcq"
        if mode == "review":
            return "mcq"

        if band_status == "above" and focus_concepts:
            avg_mastery = CoachPolicy._average_mastery(mastery, focus_concepts)
            if avg_mastery >= 0.75:
                return "open"
        return preferred

    @staticmethod
    def _average_mastery(
        mastery: Dict[str, float],
        concepts: Sequence[str],
    ) -> float:
        if not concepts:
            return 0.0
        scores = []
        for concept in concepts:
            key = concept.lower()
            scores.append(float(mastery.get(key, mastery.get(concept, 0.5))))
        return sum(scores) / len(scores)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


_DEFAULT_POLICY = CoachPolicy()


def decide_next_strategy(
    *,
    subject: str,
    base_concepts: Sequence[str],
    base_difficulty: float,
    learner_state: Dict,
    history: Sequence[Dict[str, object]],
    attention: Optional[AttentionSignals],
    emotion: Optional[Dict[str, float]] = None,
    review_item: Optional[ReviewItem],
    preferred_item_type: str,
    last_difficulty: Optional[float],
    question_index: int,
    policy_state: Optional[Dict[str, object]] = None,
) -> Tuple[StrategyDecision, Dict[str, object]]:
    state_obj = CoachPolicyState.from_dict(policy_state)
    decision, state_obj = _DEFAULT_POLICY.decide_next(
        subject=subject,
        base_concepts=base_concepts,
        base_difficulty=base_difficulty,
        learner_state=learner_state,
        history=history,
        attention=attention,
        emotion=emotion,
        review_item=review_item,
        preferred_item_type=preferred_item_type,
        last_difficulty=last_difficulty,
        question_index=question_index,
        state=state_obj,
    )
    return decision, state_obj.to_dict()
