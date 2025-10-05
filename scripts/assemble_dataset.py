from __future__ import annotations

import argparse
import json
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


RAW_DIR_DEFAULT = Path("data/raw")
OUT_DIR_DEFAULT = Path("data/processed")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_events(raw_dir: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for event in _iter_jsonl(path):
            etype = event.get("type") or event.get("event_type")
            if etype:
                event["type"] = etype
            events.append(event)
    events.sort(key=lambda ev: (ev.get("timestamp", 0), ev.get("id", "")))
    return events


@dataclass
class Attempt:
    question: Dict[str, Any]
    question_event: Dict[str, Any]
    raw: Optional[Dict[str, Any]] = None
    raw_event: Optional[Dict[str, Any]] = None
    scored: Optional[Dict[str, Any]] = None
    scored_event: Optional[Dict[str, Any]] = None
    created_ts: Optional[int] = None
    updated_ts: Optional[int] = None
    user_id: Optional[str] = None

    @property
    def question_id(self) -> Optional[str]:
        return self.question.get("question_id")


def _find_attempt(
    by_user: Dict[str, Attempt],
    by_question: Dict[str, Attempt],
    payload: Dict[str, Any],
    fallback_order: List[Attempt],
) -> Optional[Attempt]:
    user_id = payload.get("user_id")
    if user_id and user_id in by_user:
        return by_user[user_id]
    q_id = payload.get("question_id")
    if q_id and q_id in by_question:
        return by_question[q_id]
    # fall back to the most recent attempt that matches question id
    if q_id:
        for attempt in reversed(fallback_order):
            if attempt.question_id == q_id:
                return attempt
    if user_id:
        for attempt in reversed(fallback_order):
            if attempt.user_id == user_id:
                return attempt
    return None


def _collect_attempts(events: Iterable[Dict[str, Any]]) -> List[Attempt]:
    attempts: List[Attempt] = []
    active_by_user: Dict[str, Attempt] = {}
    active_by_question: Dict[str, Attempt] = {}

    for event in events:
        etype = event.get("type")
        payload = event.get("payload", {})
        timestamp = event.get("timestamp")
        user_id = payload.get("user_id")

        if etype == "QUESTION_SHOWN":
            attempt = Attempt(
                question=payload,
                question_event=event,
                created_ts=timestamp,
                user_id=user_id,
            )
            attempts.append(attempt)
            if user_id:
                active_by_user[user_id] = attempt
            q_id = payload.get("question_id")
            if q_id:
                active_by_question[q_id] = attempt
        elif etype == "RESPONSE_RAW":
            attempt = _find_attempt(
                active_by_user, active_by_question, payload, attempts
            )
            if not attempt:
                continue
            attempt.raw = payload
            attempt.raw_event = event
            attempt.updated_ts = timestamp
            if not attempt.user_id and user_id:
                attempt.user_id = user_id
                active_by_user[user_id] = attempt
        elif etype == "RESPONSE_SCORED":
            attempt = _find_attempt(
                active_by_user, active_by_question, payload, attempts
            )
            if not attempt:
                continue
            attempt.scored = payload
            attempt.scored_event = event
            attempt.updated_ts = timestamp or attempt.updated_ts
            if not attempt.user_id and user_id:
                attempt.user_id = user_id
            if attempt.user_id:
                active_by_user.pop(attempt.user_id, None)
            q_id = payload.get("question_id")
            if q_id:
                active_by_question.pop(q_id, None)

    return attempts


def _format_bloom(value: Optional[str]) -> str:
    if not value:
        return "Understand"
    formatted = value.strip().replace("_", " ")
    return formatted.capitalize()


def _ensure_list(values: Optional[Iterable[Any]]) -> List[str]:
    if not values:
        return []
    return [str(v) for v in values]


def _ts_to_iso(ts: Optional[int]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def _affect_from_level(level: Optional[str]) -> Optional[Dict[str, Any]]:
    if not level:
        return None
    level = str(level).lower().strip()
    mapping: Dict[str, Tuple[str, float, float]] = {
        "low": ("Demotivation", -0.4, 0.35),
        "mid": ("Neutral", 0.0, 0.45),
        "high": ("Stress", -0.6, 0.75),
    }
    label, valence, arousal = mapping.get(level, ("Neutral", 0.0, 0.45))
    return {
        "labels": [label],
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "source": "self_report",
    }


def _recommend_intervention(
    correct: Optional[bool],
    confidence: Optional[float],
    affect_level: Optional[str],
) -> Optional[str]:
    if correct is False:
        if confidence is not None and confidence >= 70:
            return "Clarify"
        if affect_level and affect_level.lower() == "high":
            return "Break"
        return "EasierTask"
    if confidence is not None and confidence < 40:
        return "ConfidenceProbe"
    return "None"


def _confidence_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("confidence", "confidence_reported", "confidence_pct"):
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                return None
    return None


def _latency_from_payload(
    raw: Optional[Dict[str, Any]], scored: Optional[Dict[str, Any]]
) -> Optional[int]:
    if scored and "latency_ms" in scored:
        try:
            return int(round(float(scored["latency_ms"])))
        except (TypeError, ValueError):
            return None
    if raw:
        if "latency_ms" in raw:
            try:
                return int(round(float(raw["latency_ms"])))
            except (TypeError, ValueError):
                return None
        t_start = raw.get("t_start")
        t_end = raw.get("t_end")
        if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)):
            latency = max(0.0, (float(t_end) - float(t_start)) * 1000.0)
            return int(round(latency))
    return None


def _extract_question(attempt: Attempt) -> Optional[Dict[str, Any]]:
    payload = attempt.question
    if not payload:
        return None
    q_id = payload.get("question_id")
    stem = payload.get("stem")
    if not (q_id and stem):
        return None
    answer_key = payload.get("answer_key")
    distractors = payload.get("distractors")
    choices = payload.get("choices")
    correct_index = payload.get("correct_index")
    if not answer_key and isinstance(choices, list) and correct_index is not None:
        try:
            answer_key = choices[int(correct_index)]
            distractors = [
                str(choice)
                for idx, choice in enumerate(choices)
                if idx != int(correct_index)
            ]
        except (ValueError, IndexError):
            answer_key = None
    return {
        "question_id": q_id,
        "bloom": _format_bloom(payload.get("bloom")),
        "type": str(payload.get("item_type", "mcq")).upper(),
        "stem": stem,
        "answer_key": answer_key or "",
        "distractors": _ensure_list(distractors),
    }


def _build_segment(payload: Dict[str, Any], question_id: str) -> Dict[str, Any]:
    concepts = payload.get("concepts") or payload.get("focus_concepts")
    return {
        "segment_id": payload.get("segment_id")
        or payload.get("segment", "seg_" + question_id[:6]),
        "topic": payload.get("subject"),
        "text": payload.get("segment_text") or payload.get("subject", ""),
        "concepts": _ensure_list(concepts),
    }


def _attempt_to_sample_dict(attempt: Attempt) -> Optional[Dict[str, Any]]:
    if not attempt.raw:
        return None
    question_dict = _extract_question(attempt)
    if not question_dict:
        return None
    raw_payload = attempt.raw
    scored_payload = attempt.scored or {}
    user_id = raw_payload.get("user_id") or attempt.user_id or "unknown"
    confidence = _confidence_from_payload(raw_payload)
    affect_level = raw_payload.get("affect") or raw_payload.get("affect_level")
    latency_ms = _latency_from_payload(raw_payload, scored_payload)
    correctness = scored_payload.get("correctness")
    if correctness is None:
        correctness = raw_payload.get("correctness")
    user_answer = raw_payload.get("user_answer")
    if user_answer is None and raw_payload.get("user_answer_index") is not None:
        choices = attempt.question.get("choices", [])
        idx = raw_payload.get("user_answer_index")
        try:
            user_answer = choices[int(idx)]
        except (TypeError, ValueError, IndexError):
            user_answer = str(raw_payload.get("user_answer_index"))

    response_dict = {
        "user_id": user_id,
        "correct": bool(correctness) if correctness is not None else False,
        "user_answer": None if user_answer is None else str(user_answer),
        "confidence": confidence,
        "latency_ms": latency_ms or 0,
        "hint_used": raw_payload.get("hint_used", False),
    }

    question_id = question_dict["question_id"]
    segment_dict = _build_segment(attempt.question, question_id)

    affect_dict = _affect_from_level(affect_level)
    intervention_recommendation = _recommend_intervention(
        response_dict["correct"], confidence, affect_level
    )

    timestamps_dict = {
        "created_at": _ts_to_iso(attempt.created_ts),
        "updated_at": _ts_to_iso(
            attempt.updated_ts
            or (attempt.scored_event or attempt.raw_event or {}).get("timestamp")
        ),
    }

    sample_dict: Dict[str, Any] = {
        "sample_id": scored_payload.get("event_id")
        or raw_payload.get("event_id")
        or str(uuid.uuid4()),
        "segment": segment_dict,
        "question": question_dict,
        "response": response_dict,
        "timestamps": timestamps_dict,
    }

    if affect_dict:
        sample_dict["affect"] = affect_dict

    if intervention_recommendation:
        sample_dict["intervention"] = {
            "recommended": intervention_recommendation,
            "accepted": False,
        }

    sample_dict["derived_signals"] = {}

    return sample_dict


def _finalize_samples(attempts: List[Attempt]) -> List[Dict[str, Any]]:
    attempts_sorted = sorted(
        attempts,
        key=lambda att: (att.created_ts if att.created_ts is not None else math.inf),
    )
    error_streak_by_user: Dict[str, int] = defaultdict(int)
    seen_concepts_by_user: Dict[str, set[str]] = defaultdict(set)
    latencies_by_user: Dict[str, List[int]] = defaultdict(list)
    temp_records: List[Tuple[Dict[str, Any], str, Optional[int]]] = []

    for attempt in attempts_sorted:
        sample_dict = _attempt_to_sample_dict(attempt)
        if not sample_dict:
            continue
        user = sample_dict["response"]["user_id"]
        latency = sample_dict["response"].get("latency_ms")
        if latency is not None:
            latencies_by_user[user].append(latency)

        concepts = sample_dict["segment"].get("concepts") or []
        concepts_normalized = {str(c).lower() for c in concepts}
        seen = seen_concepts_by_user[user]
        if concepts:
            unseen = [c for c in concepts_normalized if c not in seen]
            novelty = len(unseen) / len(concepts_normalized)
            seen.update(concepts_normalized)
        else:
            novelty = None

        correct = sample_dict["response"].get("correct")
        previous_streak = error_streak_by_user[user]
        if correct:
            streak = 0
        elif correct is False:
            streak = previous_streak + 1
        else:
            streak = previous_streak
        error_streak_by_user[user] = streak

        derived = sample_dict.setdefault("derived_signals", {})
        derived["error_streak"] = streak
        derived["novelty_score"] = round(novelty, 3) if novelty is not None else None
        derived.setdefault("latency_z", None)

        temp_records.append((sample_dict, user, latency))

    # compute latency z-scores per user
    latency_stats: Dict[str, Tuple[float, float]] = {}
    for user, values in latencies_by_user.items():
        if len(values) >= 2:
            mu = mean(values)
            sigma = pstdev(values)
            latency_stats[user] = (mu, sigma)
        elif values:
            latency_stats[user] = (values[0], 0.0)

    samples: List[Dict[str, Any]] = []
    for sample_dict, user, latency in temp_records:
        mu_sigma = latency_stats.get(user)
        if mu_sigma and latency is not None:
            mu, sigma = mu_sigma
            if sigma and sigma > 0:
                z = (latency - mu) / sigma
            else:
                z = 0.0
            sample_dict.setdefault("derived_signals", {})["latency_z"] = round(z, 3)
        else:
            sample_dict.setdefault("derived_signals", {})["latency_z"] = None

        # validate via pydantic
        samples.append(sample_dict)

    return samples


def assemble(raw_dir: Path, out_path: Optional[Path] = None) -> Path:
    events = _load_events(raw_dir)
    attempts = _collect_attempts(events)
    samples = _finalize_samples(attempts)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if out_path is None:
        resolved_out = OUT_DIR_DEFAULT / f"samples_{ts}.jsonl"
    else:
        placeholder_path = str(out_path).replace("<timestamp>", ts)
        resolved_out = Path(placeholder_path)

    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    with resolved_out.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        f"Assembled {len(samples)} samples from {len(events)} events "
        f"into {resolved_out}"
    )
    skipped = sum(1 for att in attempts if not att.raw or not att.question)
    if skipped:
        print(
            "Skipped {count} incomplete attempts lacking question or response "
            "data.".format(count=skipped)
        )
    return resolved_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble learning samples from raw telemetry events."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR_DEFAULT,
        help="Directory containing raw telemetry JSONL files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional output JSONL path. Defaults to "
            "data/processed/samples_<timestamp>.jsonl."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assemble(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
