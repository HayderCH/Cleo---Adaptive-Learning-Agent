import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import argparse

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCHEMA_PATH = DATA_DIR / "schema.json"

TOPICS = [
    "Optimization",
    "Neural Networks",
    "Probability",
]
CONCEPTS_BY_TOPIC = {
    "Optimization": ["Gradient Descent", "Momentum", "Adam optimizer", "Learning Rate"],
    "Neural Networks": [
        "Activation Functions",
        "Overfitting",
        "Regularization",
        "Dropout",
    ],
    "Probability": ["Bayes Theorem", "Random Variables", "Expectation", "Variance"],
}
BLOOMS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def rand_text(n_words: int) -> str:
    words = (
        "the of and to in is are that it for with as on be by at from this using or into such if when then "
        "model data learning gradient network probability function loss training inference regularization"
    ).split()
    return " ".join(random.choice(words) for _ in range(n_words)).capitalize() + "."


def gen_segment(topic: str) -> dict:
    seg_id = f"seg_{uuid.uuid4().hex[:6]}"
    concepts = random.sample(CONCEPTS_BY_TOPIC[topic], k=random.randint(1, 3))
    return {
        "segment_id": seg_id,
        "topic": topic,
        "text": rand_text(random.randint(40, 90)),
        "concepts": concepts,
    }


def gen_question(segment_id: str) -> dict:
    qid = f"q_{uuid.uuid4().hex[:6]}"
    bloom = random.choice(BLOOMS)
    stem = f"{rand_text(random.randint(6, 14))}"
    answer = rand_text(random.randint(6, 12))
    distractors = [
        rand_text(random.randint(6, 12)) for _ in range(random.randint(2, 3))
    ]
    return {
        "question_id": qid,
        "bloom": bloom,
        "type": "MCQ",
        "stem": stem,
        "answer_key": answer,
        "distractors": distractors,
    }


def gen_response(user_id: str, correct_prob: float) -> dict:
    correct = random.random() < correct_prob
    latency = int(random.gauss(6000, 2000))
    latency = max(500, min(30000, latency))
    confidence = max(0, min(100, int(random.gauss(70 if correct else 55, 20))))
    return {
        "user_id": user_id,
        "correct": correct,
        "user_answer": rand_text(random.randint(4, 10)),
        "confidence": confidence,
        "latency_ms": latency,
        "hint_used": random.random() < 0.2,
    }


def gen_affect() -> dict:
    labels_all = ["Frustration", "Demotivation", "Stress", "Neutral"]
    # 70% neutral, 30% one of negative
    if random.random() < 0.7:
        labels = ["Neutral"]
        valence = random.uniform(0.1, 0.6)
        arousal = random.uniform(0.1, 0.6)
    else:
        label = random.choice(labels_all[:-1])
        labels = [label]
        valence = random.uniform(-1.0, 0.2)
        arousal = random.uniform(0.3, 0.9)
    source = random.choice(["self_report", "text_model", "interaction_model"])
    return {
        "labels": labels,
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "source": source,
    }


def derive_signals(resp_history):
    if not resp_history:
        return {
            "latency_z": 0.0,
            "error_streak": 0,
            "novelty_score": round(random.random(), 3),
        }
    latencies = [r["response"]["latency_ms"] for r in resp_history]
    mean = sum(latencies) / len(latencies)
    std = max(1.0, (sum((x - mean) ** 2 for x in latencies) / len(latencies)) ** 0.5)
    lat = latencies[-1]
    latency_z = (lat - mean) / std
    # error streak
    streak = 0
    for r in reversed(resp_history):
        if r["response"]["correct"]:
            break
        streak += 1
    return {
        "latency_z": round(latency_z, 3),
        "error_streak": streak,
        "novelty_score": round(random.random(), 3),
    }


def generate_dataset(
    n_users: int = 3, segments_per_topic: int = 4, questions_per_segment: int = 3
):
    now = datetime.utcnow()
    samples = []
    raw_events = []

    for topic in TOPICS:
        for _ in range(segments_per_topic):
            seg = gen_segment(topic)
            questions = [
                gen_question(seg["segment_id"]) for _ in range(questions_per_segment)
            ]

            for u in range(n_users):
                user_id = f"anon_{u+1}"
                correct_prob = random.uniform(0.5, 0.85)
                resp_history = []
                for q in questions:
                    response = gen_response(user_id, correct_prob)
                    affect = gen_affect()
                    derived = derive_signals(resp_history + [{"response": response}])

                    sample = {
                        "sample_id": str(uuid.uuid4()),
                        "segment": seg,
                        "question": q,
                        "response": response,
                        "affect": affect,
                        "derived_signals": derived,
                        "intervention": {
                            "recommended": random.choice(
                                [
                                    "None",
                                    "Break",
                                    "Encouragement",
                                    "Clarify",
                                    "EasierTask",
                                    "ConfidenceProbe",
                                ]
                            ),
                            "accepted": random.random() < 0.3,
                        },
                        "consent": {
                            "store_free_text": random.random() < 0.5,
                            "store_audio": False,
                        },
                        "timestamps": {
                            "created_at": (
                                now - timedelta(minutes=random.randint(0, 10000))
                            ).isoformat(),
                            "updated_at": now.isoformat(),
                        },
                    }
                    samples.append(sample)

                    # raw event envelope example
                    raw_events.append(
                        {
                            "id": str(uuid.uuid4()),
                            "type": "RESPONSE_RAW",
                            "source": "UI",
                            "timestamp": int(now.timestamp()),
                            "payload": {
                                "question_id": q["question_id"],
                                "user_answer": sample["response"]["user_answer"],
                                "confidence_reported": sample["response"]["confidence"],
                                "latency_ms": sample["response"]["latency_ms"],
                                "correctness": sample["response"]["correct"],
                            },
                        }
                    )

                    resp_history.append({"response": response})

    return samples, raw_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ALC dataset")
    parser.add_argument(
        "--users", type=int, default=3, help="Number of users to simulate per segment"
    )
    parser.add_argument(
        "--segments-per-topic", type=int, default=4, help="Segments per topic"
    )
    parser.add_argument(
        "--questions-per-segment", type=int, default=3, help="Questions per segment"
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    samples, events = generate_dataset(
        n_users=args.users,
        segments_per_topic=args.segments_per_topic,
        questions_per_segment=args.questions_per_segment,
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_processed = PROCESSED_DIR / f"samples_{ts}.jsonl"
    out_events = RAW_DIR / f"events_{ts}.jsonl"

    with open(out_processed, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(out_events, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {out_processed}")
    print(f"Wrote {len(events)} events to {out_events}")
