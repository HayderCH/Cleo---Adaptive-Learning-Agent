from datetime import datetime, timedelta, timezone

from services.coach.memory import SpacedReviewScheduler


def test_scheduler_skips_low_mastery_concepts() -> None:
    scheduler = SpacedReviewScheduler(
        base_interval_hours=1.0,
        review_threshold=0.65,
    )
    learner_state = {"mastery": {"fractions": 0.4}}

    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    scheduler.update(learner_state, "Fractions", True, 0.4, now=now)

    due = scheduler.get_due_reviews(learner_state, now=now)
    assert due == []


def test_scheduler_returns_due_reviews_when_overdue() -> None:
    scheduler = SpacedReviewScheduler(
        base_interval_hours=0.001,
        review_threshold=0.2,
        stability_init=1.0,
        min_interval_hours=0.001,
    )
    learner_state = {"mastery": {"fractions": 0.8}}

    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    scheduler.update(learner_state, "fractions", True, 0.8, now=now)

    later = now + timedelta(hours=0.01)
    due = scheduler.get_due_reviews(learner_state, now=later)
    assert len(due) == 1
    assert due[0].concept == "fractions"
    assert due[0].is_overdue
