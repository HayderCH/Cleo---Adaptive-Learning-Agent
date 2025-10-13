#!/usr/bin/env python3
"""Analyze telemetry data to understand performance issues."""

import json
from pathlib import Path
from collections import defaultdict


def analyze_telemetry():
    """Analyze telemetry data for performance insights."""
    DATA_RAW = Path("data/raw")
    print("🔍 Analyzing telemetry data for performance insights...")

    # Find recent event files
    event_files = list(DATA_RAW.glob("events_*.jsonl"))
    event_files.sort(reverse=True)

    questions_answers = []
    event_counts = defaultdict(int)

    for event_file in event_files[:5]:  # Check last 5 days
        print(f"Processing {event_file.name}...")
        with open(event_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    event_type = event.get("type")
                    event_counts[event_type] += 1

                    if event_type == "RESPONSE_SCORED":
                        payload = event.get("payload", {})
                        questions_answers.append(
                            {
                                "question": payload.get("question_stem", ""),
                                "expected": payload.get("expected_answer", ""),
                                "provided": payload.get("user_answer", ""),
                                "correct": payload.get("correctness", False),
                                "latency": payload.get("latency_ms", 0),
                            }
                        )
                except json.JSONDecodeError:
                    continue

    print(f"\n📊 Event Summary:")
    for event_type, count in sorted(event_counts.items()):
        print(f"  {event_type}: {count}")

    print(f"\n📝 Q&A Analysis: {len(questions_answers)} pairs found")

    if questions_answers:
        # Show some examples
        for i, qa in enumerate(questions_answers[:3]):
            print(f"\n--- Question {i+1} ---")
            print(
                f'Q: {qa["question"][:80]}...'
                if len(qa["question"]) > 80
                else f'Q: {qa["question"]}'
            )
            print(
                f'Expected: {qa["expected"][:40]}...'
                if len(qa["expected"]) > 40
                else f'Expected: {qa["expected"]}'
            )
            print(
                f'Provided: {qa["provided"][:40]}...'
                if qa["provided"] and len(qa["provided"]) > 40
                else f'Provided: {qa["provided"] or "(empty)"}'
            )
            print(f'Correct: {qa["correct"]}')
            print(f'Latency: {qa["latency"]/1000:.1f}s')

        # Analyze correctness patterns
        correct_count = sum(1 for qa in questions_answers if qa["correct"])
        total_count = len(questions_answers)
        accuracy = correct_count / total_count if total_count > 0 else 0

        print(f"\n📊 Overall Accuracy: {accuracy:.1%} ({correct_count}/{total_count})")

        # Check for patterns in incorrect answers
        incorrect = [qa for qa in questions_answers if not qa["correct"]]
        if incorrect:
            print(f"\n❌ Incorrect answers analysis:")
            empty_answers = sum(1 for qa in incorrect if not qa["provided"])
            print(
                f"  Empty answers: {empty_answers}/{len(incorrect)} ({empty_answers/len(incorrect):.1%})"
            )

            # Analyze latency patterns
            latencies = [qa["latency"] / 1000 for qa in questions_answers]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                slow_count = sum(1 for l in latencies if l > 30)
                print(f"\n⏱️  Latency Analysis:")
                print(f"  Average: {avg_latency:.1f}s")
                print(f"  Max: {max_latency:.1f}s")
                print(
                    f"  >30s count: {slow_count}/{len(latencies)} ({slow_count/len(latencies):.1%})"
                )


if __name__ == "__main__":
    analyze_telemetry()
