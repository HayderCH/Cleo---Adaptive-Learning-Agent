from __future__ import annotations

from services.attention import AttentionAgent


def test_attention_agent_updates_error_streak_and_z_score() -> None:
    agent = AttentionAgent()

    first = agent.update(user_id="learner", latency_ms=1000.0, correct=True)
    assert first["error_streak"] == 0
    assert first["sample_size"] == 1
    assert first["latency_z"] == 0.0

    second = agent.update(user_id="learner", latency_ms=1300.0, correct=False)
    assert second["error_streak"] == 1
    assert second["sample_size"] == 2

    snapshot = agent.get_snapshot("learner")
    assert snapshot is not None
    assert snapshot["count"] == 2
    assert snapshot["error_streak"] == 1


def test_attention_agent_reset_user() -> None:
    agent = AttentionAgent()
    agent.update(user_id="learner", latency_ms=900.0, correct=True)
    assert agent.get_snapshot("learner") is not None

    agent.reset_user("learner")
    assert agent.get_snapshot("learner") is None
