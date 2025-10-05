from __future__ import annotations

from services.emotion import EmotionAgent


def test_emotion_agent_increases_frustration_on_errors() -> None:
    agent = EmotionAgent(alpha=0.5)

    state1 = agent.update(
        user_id="learner",
        correctness=True,
        confidence=80,
        latency_ms=900,
        latency_z=0.1,
        error_streak=0,
    )
    assert state1["frustration_prob"] < 0.5

    state2 = agent.update(
        user_id="learner",
        correctness=False,
        confidence=30,
        latency_ms=2000,
        latency_z=1.5,
        error_streak=3,
    )
    assert state2["frustration_prob"] >= state1["frustration_prob"]
    assert state2["stress_prob"] > 0.4


def test_emotion_agent_reset_user() -> None:
    agent = EmotionAgent()
    agent.update(
        user_id="learner",
        correctness=False,
        confidence=20,
        latency_ms=2500,
        latency_z=2.0,
        error_streak=4,
    )
    assert agent.get_snapshot("learner") is not None
    agent.reset_user("learner")
    assert agent.get_snapshot("learner") is None
