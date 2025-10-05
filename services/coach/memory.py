from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

DEFAULT_BASE_INTERVAL_HOURS = 12.0
DEFAULT_STABILITY_INIT = 1.0
DEFAULT_SUCCESS_BOOST = 1.3
DEFAULT_FAILURE_DECAY = 0.7
DEFAULT_REVIEW_THRESHOLD = 0.65
DEFAULT_MIN_INTERVAL_HOURS = 1.0
MAX_STABILITY = 10.0
MIN_STABILITY = 0.25


@dataclass
class ReviewItem:
    concept: str
    next_due_ts: float
    stability: float

    def due_at(self) -> datetime:
        return datetime.fromtimestamp(self.next_due_ts, tz=timezone.utc)

    @property
    def is_overdue(self) -> bool:
        return datetime.now(timezone.utc).timestamp() >= self.next_due_ts


class SpacedReviewScheduler:
    """Simple spaced review scheduler stored inside the learner model."""

    def __init__(
        self,
        base_interval_hours: float = DEFAULT_BASE_INTERVAL_HOURS,
        stability_init: float = DEFAULT_STABILITY_INIT,
        success_boost: float = DEFAULT_SUCCESS_BOOST,
        failure_decay: float = DEFAULT_FAILURE_DECAY,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
        min_interval_hours: float = DEFAULT_MIN_INTERVAL_HOURS,
    ) -> None:
        self.base_interval_hours = base_interval_hours
        self.stability_init = stability_init
        self.success_boost = success_boost
        self.failure_decay = failure_decay
        self.review_threshold = review_threshold
        self.min_interval_hours = max(0.001, min_interval_hours)

    def _ensure_memory(self, learner_model: Dict) -> Dict:
        return learner_model.setdefault("memory", {"concepts": {}})

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def update(  # noqa: C901
        self,
        learner_model: Dict,
        concept: str,
        correctness: Optional[bool],
        mastery_score: Optional[float],
        *,
        now: Optional[datetime] = None,
    ) -> None:
        concept_key = concept.lower()
        memory_section = self._ensure_memory(learner_model)
        concepts = memory_section.setdefault("concepts", {})
        state = concepts.setdefault(
            concept_key,
            {
                "stability": self.stability_init,
                "next_due_ts": 0.0,
                "last_seen_ts": None,
                "review_count": 0,
                "last_correct": None,
            },
        )

        now_dt = now or self._now()
        now_ts = now_dt.timestamp()
        stability = float(state.get("stability", self.stability_init))

        if correctness is True:
            stability = min(
                MAX_STABILITY,
                max(MIN_STABILITY, stability * self.success_boost),
            )
        elif correctness is False:
            stability = max(
                MIN_STABILITY,
                stability * self.failure_decay,
            )
        else:
            # keep stability as-is if we could not score the item
            stability = max(MIN_STABILITY, min(MAX_STABILITY, stability))

        interval_hours = max(
            self.min_interval_hours,
            self.base_interval_hours * stability,
        )
        if correctness is False:
            interval_hours = max(
                self.min_interval_hours,
                interval_hours * self.failure_decay,
            )

        interval = timedelta(hours=interval_hours)
        next_due_ts = (now_dt + interval).timestamp()

        state.update(
            {
                "stability": round(stability, 4),
                "next_due_ts": next_due_ts,
                "last_seen_ts": now_ts,
                "review_count": int(state.get("review_count", 0)) + 1,
                "last_correct": (None if correctness is None else bool(correctness)),
                "last_mastery": (
                    None if mastery_score is None else float(mastery_score)
                ),
            }
        )

        if correctness is False:
            # ensure we prompt for review soon even if mastery is low
            state["priority"] = "needs_reteach"
        elif mastery_score is not None and mastery_score >= self.review_threshold:
            state["priority"] = "review"
        else:
            state["priority"] = "learn"

    def get_due_reviews(
        self,
        learner_model: Dict,
        *,
        now: Optional[datetime] = None,
        include_future: bool = False,
    ) -> List[ReviewItem]:
        memory_section = learner_model.get("memory", {})
        concepts: Dict[str, Dict] = memory_section.get("concepts", {})
        now_ts = (now or self._now()).timestamp()
        due_items: List[ReviewItem] = []
        for concept, state in concepts.items():
            next_due_ts = float(state.get("next_due_ts", 0.0) or 0.0)
            stability = float(state.get("stability", self.stability_init))
            last_mastery = state.get("last_mastery")
            if last_mastery is not None and last_mastery < self.review_threshold:
                continue
            if not include_future and next_due_ts > now_ts:
                continue
            due_items.append(
                ReviewItem(
                    concept=concept,
                    next_due_ts=next_due_ts,
                    stability=stability,
                )
            )
        due_items.sort(key=lambda item: item.next_due_ts)
        return due_items

    def peek_next_review(
        self,
        learner_model: Dict,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[ReviewItem]:
        due_items = self.get_due_reviews(
            learner_model,
            now=now,
            include_future=True,
        )
        return due_items[0] if due_items else None
