from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

import requests
import streamlit as st
import yaml

from services.coach import learner_model
from services.coach.memory import SpacedReviewScheduler
from services.coach.policy import decide_next_strategy
from services.emotion import (
    EmotionTransformer,
    aggregate_buckets as _emotion_aggregate_buckets,
    build_emotion_state as _emotion_build_state,
    flatten_predictions as _emotion_flatten_predictions,
)
from services.emotion.advice_agent import get_emotional_advice_agent
from services.emotion.interpreter import EmotionInterpreter
from services.qgen import generator, quality_gate
from services.qgen.schemas import GenerateRequest
from scoring import mcq, open_text

try:  # pragma: no cover - optional dependency
    from services.coach.strategy import AttentionSignals
except ImportError:  # pragma: no cover - fallback when strategy unavailable

    @dataclass
    class AttentionSignals:  # type: ignore[override]
        latency_z: Optional[float] = None
        error_streak: int = 0
        sample_size: int = 0


TELEMETRY_BASE = os.getenv("TELEMETRY_BASE", "http://127.0.0.1:8000")
SCHEDULER = SpacedReviewScheduler()


@st.cache_resource(show_spinner=False)
def get_emotion_interpreter() -> EmotionInterpreter | None:
    try:
        return EmotionInterpreter.from_config()
    except (OSError, yaml.YAMLError, ValueError):
        return None


@st.cache_resource(show_spinner=False)
def get_emotion_transformer() -> EmotionTransformer | None:
    try:
        return EmotionTransformer.from_config()
    except (RuntimeError, OSError, yaml.YAMLError, ValueError):
        return None


def _flatten_predictions(raw: object) -> list[dict]:
    flattened = _emotion_flatten_predictions(raw)
    result: list[dict] = []
    for item in flattened:
        label = str(item.get("label", ""))
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        result.append({"label": label, "score": score})
    return result


def _aggregate_self_report(predictions: list[dict]) -> dict[str, float]:
    return _emotion_aggregate_buckets(predictions)


def _build_emotion_state_from_buckets(buckets: Mapping[str, float]) -> dict[str, float]:
    return _emotion_build_state(buckets)


def post_event(name: str, payload: dict) -> dict | None:
    try:
        response = requests.post(
            f"{TELEMETRY_BASE}/events",
            json={"type": name, "source": "UI", "payload": payload},
            timeout=3,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def build_segment_text(subject: str, concepts: Sequence[str]) -> str:
    if concepts:
        concept_str = ", ".join(concepts)
        return f"Study segment covering {concept_str} within {subject}."
    return f"Study segment focused on {subject}."


def _suggest_self_report_interventions(
    emotion_state: dict[str, float],
    attention_state: Optional[Mapping[str, float | int]],
) -> list[str]:
    interpreter = get_emotion_interpreter()
    if not interpreter:
        return []
    attention_payload = None
    if attention_state:
        attention_payload = {
            "latency_z": attention_state.get("latency_z"),
            "error_streak": attention_state.get("error_streak"),
        }
    return interpreter.suggest_interventions(emotion_state, attention_payload)


def _prepare_backend_selector() -> tuple[list[str], list[Optional[str]], int]:
    available = generator.list_available_backends()
    override = generator.get_backend_override()
    current_backend = generator.current_backend_name()

    labels = [f"Config default ({current_backend.title()})"]
    values: list[Optional[str]] = [None]
    for name in available:
        labels.append(name.replace("_", " ").title())
        values.append(name)

    if override is None:
        default_index = 0
    else:
        try:
            default_index = available.index(override) + 1
        except ValueError:
            default_index = 0

    return labels, values, default_index


def init_loop_state(user_id: str, base_difficulty: float) -> None:
    learner_state = learner_model.load(user_id)
    st.session_state.loop = {
        "user_id": user_id,
        "history": [],
        "current_q": None,
        "current_difficulty": round(base_difficulty, 2),
        "base_difficulty": round(base_difficulty, 2),
        "t_start": None,
        "pending_meta": None,
        "pending_review": None,
        "policy_state": None,
        "learner_model": learner_state,
        "emotion_state": {},
        "emotion_interventions": [],
        "emotion_notes": [],
        "attention_state": None,
        "bloom_history": [],
        "last_result": None,
    }


def ensure_loop_state(user_id: str, base_difficulty: float) -> None:
    loop = st.session_state.get("loop")
    if loop is None:
        init_loop_state(user_id, base_difficulty)
        return
    if loop.get("user_id") != user_id:
        init_loop_state(user_id, base_difficulty)


def build_attention(loop: dict) -> Optional[AttentionSignals]:
    attention = loop.get("attention_state") or {}
    if attention:
        return AttentionSignals(
            latency_z=attention.get("latency_z"),
            error_streak=int(attention.get("error_streak") or 0),
            sample_size=int(attention.get("sample_size") or 0),
        )

    history = loop.get("history") or []
    for entry in reversed(history):
        if entry.get("latency_z") is not None or entry.get("error_streak") is not None:
            return AttentionSignals(
                latency_z=entry.get("latency_z"),
                error_streak=int(entry.get("error_streak") or 0),
                sample_size=int(entry.get("sample_size") or 0),
            )
    return None


def main() -> None:
    st.set_page_config(
        page_title="Adaptive Learning Companion",
        layout="wide",
    )
    st.title("Adaptive Learning Companion")

    with st.sidebar:
        st.header("Session setup")
        subject = st.text_input("Subject", value="Mathematics")
        base_difficulty = st.slider(
            "Base difficulty",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )
        concepts_input = st.text_input(
            "Focus concepts (comma separated)",
            value="Fractions, Ratios",
        )
        backend_labels, backend_values, backend_default = _prepare_backend_selector()
        backend_choice_label = st.selectbox(
            "Question generator backend",
            backend_labels,
            index=backend_default,
        )
        backend_choice_index = backend_labels.index(backend_choice_label)
        selected_backend_value = backend_values[backend_choice_index]
        previous_choice = st.session_state.get("qgen_backend_choice")
        if previous_choice != backend_choice_index:
            generator.set_backend_override(selected_backend_value)
            st.session_state.qgen_backend_choice = backend_choice_index

        bloom_choice = st.selectbox(
            "Bloom target",
            [
                "remember",
                "understand",
                "apply",
                "analyze",
                "evaluate",
                "create",
            ],
            index=1,
        )
        item_type_label = st.selectbox(
            "Question type",
            ["Multiple choice", "Open response"],
            index=0,
        )

    item_type_choice = "open" if item_type_label == "Open response" else "mcq"
    user_id = st.text_input("User ID", value="demo")
    focus_concepts = [
        concept.strip() for concept in concepts_input.split(",") if concept.strip()
    ]

    ensure_loop_state(user_id, base_difficulty)
    loop = st.session_state.loop
    loop["base_difficulty"] = round(base_difficulty, 2)
    if not loop["history"] and loop["current_q"] is None:
        loop["current_difficulty"] = round(base_difficulty, 2)

    cols = st.columns([1, 1])
    disable_next = loop["current_q"] is not None
    next_label = (
        "Start session"
        if not loop["history"] and loop["current_q"] is None
        else "Next question"
    )
    gen_clicked = cols[0].button(next_label, disabled=disable_next)
    reset_clicked = cols[1].button("Reset session")

    if reset_clicked:
        init_loop_state(user_id, base_difficulty)
        loop = st.session_state.loop
        st.success("Session reset.")

    if disable_next:
        st.info("Submit your answer to unlock the next question.")

    if gen_clicked and loop["current_q"] is None:
        attention_obj = build_attention(loop)
        due_reviews = SCHEDULER.get_due_reviews(loop["learner_model"])
        review_item = due_reviews[0] if due_reviews else None

        decision, policy_state = decide_next_strategy(
            subject=subject,
            base_concepts=focus_concepts,
            base_difficulty=loop["base_difficulty"],
            learner_state=loop["learner_model"],
            history=loop["history"],
            attention=attention_obj,
            emotion=loop.get("emotion_state"),
            review_item=review_item,
            preferred_item_type=item_type_choice,
            last_difficulty=loop.get("current_difficulty"),
            question_index=len(loop["history"]),
            policy_state=loop.get("policy_state"),
        )

        target_concepts = list(decision.focus_concepts or focus_concepts)
        if not target_concepts:
            target_concepts = [subject]

        loop["policy_state"] = policy_state
        loop["emotion_interventions"] = list(decision.interventions or [])
        loop["emotion_notes"] = list(decision.emotion_notes or [])
        loop["pending_review"] = (
            {
                "concept": review_item.concept,
                "next_due_ts": review_item.next_due_ts,
            }
            if review_item
            else None
        )

        req = GenerateRequest(
            subject=subject,
            focus_concepts=target_concepts,
            difficulty_target=decision.difficulty,
            bloom_target=decision.bloom,
            item_type=decision.item_type,
            learner_snapshot=loop["learner_model"],
        )

        post_event(
            "STRATEGY_UPDATE",
            {
                "user_id": user_id,
                "subject": subject,
                "focus_concepts": target_concepts,
                "difficulty_target": decision.difficulty,
                "bloom_target": decision.bloom,
                "item_type": decision.item_type,
                "mode": decision.mode,
                "band_status": decision.band_status,
                "rationale": decision.rationale,
            },
        )

        q = generator.generate(req)
        effective_backend = q.meta.source or generator.current_backend_name()
        expected_backend = (
            selected_backend_value
            if selected_backend_value is not None
            else generator.current_backend_name()
        )
        backend_fallback = (
            expected_backend == "transformer" and effective_backend != "transformer"
        )
        ok, reasons = quality_gate.validate(q)
        if not ok:
            st.error("Quality gate failed: " + "; ".join(reasons))
        else:
            choices = list(q.choices or [])
            correct_index = q.correct_index
            answer_key = getattr(q, "answer_key", None)
            distractors: list[str] = []
            if choices:
                if correct_index is None and answer_key in choices:
                    correct_index = choices.index(answer_key)
                if correct_index is not None and 0 <= correct_index < len(choices):
                    answer_key = choices[correct_index]
                    distractors = [
                        choice
                        for idx, choice in enumerate(choices)
                        if idx != correct_index
                    ]
            segment_text = build_segment_text(subject, target_concepts)

            loop["current_q"] = q
            loop["t_start"] = time.time()
            loop["pending_meta"] = {
                "difficulty": decision.difficulty,
                "bloom": decision.bloom,
                "backend": effective_backend,
                "backend_expected": expected_backend,
                "backend_fallback": backend_fallback,
                "concepts": q.meta.concepts,
                "choices": choices,
                "correct_index": correct_index,
                "answer_key": answer_key,
                "distractors": distractors,
                "stem": q.stem,
                "segment_text": segment_text,
                "subject": subject,
                "item_type": req.item_type,
                "answer_expl": q.answer_expl,
                "question_id": q.id,
                "difficulty_est": q.meta.difficulty_est,
                "review_mode": decision.mode == "review",
                "review_concept": (
                    review_item.concept if decision.mode == "review" else None
                ),
                "review_next_due": (review_item.next_due_ts if review_item else None),
                "target_concepts": target_concepts,
                "band_status": decision.band_status,
                "policy_rationale": decision.rationale,
                "mode": decision.mode,
                "emotion_interventions": list(decision.interventions or []),
                "emotion_notes": list(decision.emotion_notes or []),
            }
            loop["current_difficulty"] = decision.difficulty
            loop["bloom_history"].append(decision.bloom)
            loop["last_result"] = None
            post_event(
                "QUESTION_SHOWN",
                {
                    "question_id": q.id,
                    "subject": subject,
                    "segment_text": segment_text,
                    "focus_concepts": target_concepts,
                    "difficulty_target": decision.difficulty,
                    "difficulty_est": q.meta.difficulty_est,
                    "bloom": decision.bloom,
                    "item_type": req.item_type,
                    "stem": q.stem,
                    "choices": choices,
                    "correct_index": correct_index,
                    "answer_key": answer_key,
                    "distractors": distractors,
                    "answer_expl": q.answer_expl,
                    "user_id": user_id,
                    "mode": decision.mode,
                    "band_status": decision.band_status,
                    "strategy_notes": decision.rationale,
                    "emotion_interventions": list(decision.interventions or []),
                    "emotion_notes": list(decision.emotion_notes or []),
                    "generator_backend": effective_backend,
                    "backend_fallback": backend_fallback,
                },
            )
            if backend_fallback:
                st.warning(
                    "Transformer backend unavailable; question served by "
                    "template fallback."
                )

    q = loop.get("current_q")
    if q is not None:
        st.subheader("Question")
        meta = loop.get("pending_meta") or {}
        if meta.get("review_mode"):
            review_concept = meta.get("review_concept")
            st.info(f"Spaced review: {review_concept}")
        st.write(meta.get("stem", q.stem))
        display_diff = float(meta.get("difficulty", loop["current_difficulty"]))
        display_bloom = meta.get("bloom", bloom_choice)
        st.caption(
            f"Target difficulty: {display_diff:.2f} • "
            f"Bloom: {display_bloom.title()}"
        )
        backend_name = meta.get("backend") or q.meta.source
        backend_label = backend_name.replace("_", " ").title()
        st.caption("Generator backend: " f"{backend_label} • Version: {q.meta.version}")
        if meta.get("backend_fallback"):
            st.info(
                "This item was served via the template fallback while the "
                "transformer backend is unavailable."
            )

        # RAG Pipeline Visualization
        if q.meta.retrieved_chunks and backend_name == "transformer":
            with st.expander(
                "🔍 View RAG Pipeline (Retrieval-Augmented Generation)", expanded=False
            ):
                st.markdown("### How this question was generated:")

                # Step 1: Input Analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📝 Step 1: Input Analysis**")
                    st.markdown(f"**Subject:** {subject}")
                    st.markdown(
                        f"**Focus Concepts:** {', '.join(meta.get('target_concepts', focus_concepts))}"
                    )
                    st.markdown(f"**Difficulty:** {display_diff:.2f}")
                    st.markdown(f"**Bloom Level:** {display_bloom.title()}")

                with col2:
                    st.markdown("**🔎 Step 2: Context Retrieval**")
                    st.markdown(
                        f"**Corpus Size:** {len(q.meta.retrieved_chunks)} relevant chunks found"
                    )
                    st.markdown(
                        "**Search Strategy:** Keyword matching with randomization"
                    )

                # Step 3: Retrieved Context
                st.markdown("**📚 Step 3: Retrieved Context**")
                for i, chunk in enumerate(q.meta.retrieved_chunks, 1):
                    with st.container():
                        st.markdown(f"**Chunk {i}:**")
                        # Get the text content from the chunk
                        chunk_text = chunk.get("text", str(chunk))
                        # Highlight focus concepts in the chunk
                        current_target_concepts = meta.get(
                            "target_concepts", focus_concepts
                        )
                        for concept in current_target_concepts:
                            chunk_text = chunk_text.replace(concept, f"**{concept}**")
                        st.markdown(chunk_text)
                        # Show additional metadata if available
                        if "source" in chunk:
                            st.caption(f"Source: {chunk['source']}")
                        st.markdown("---")

                # Step 4: Question Generation
                st.markdown("**🤖 Step 4: Question Generation**")
                st.markdown(
                    "The retrieved context was combined with the focus concepts and fed to the Phi-3.5 language model to generate this question."
                )

                # Visual flow diagram
                st.markdown("**🔄 RAG Pipeline Flow:**")
                st.markdown(
                    """
                ```
                Input Request → Context Retrieval → Relevant Chunks → LLM Generation → Final Question
                     ↓              ↓                    ↓              ↓              ↓
                [Subject +      [Search Corpus]    [Filtered Text]   [Phi-3.5]     [MCQ/Open]
                 Concepts]       [~1000 chunks]     [Top 3 chunks]   [3.8B params]  [JSON output]
                ```
                """
                )

        with st.form("answer_form", clear_on_submit=False):
            answer_text = None
            chosen_index = None
            choices_for_render = meta.get("choices", list(q.choices or []))
            if choices_for_render:
                choice_label = st.radio(
                    "Select your answer",
                    choices_for_render,
                    index=0,
                )
                chosen_index = choices_for_render.index(choice_label)
            else:
                answer_text = st.text_area(
                    "Your answer",
                    placeholder="Type your answer here...",
                )

            affect_level = st.select_slider(
                "Affect (optional)",
                ["low", "mid", "high"],
                value="mid",
            )
            confidence = st.slider(
                "Confidence (%)",
                min_value=0,
                max_value=100,
                value=75,
                step=5,
            )
            submitted = st.form_submit_button("Submit Answer")

        if submitted:
            t_start = loop.get("t_start") or time.time()
            t_end = time.time()
            meta = loop.get("pending_meta") or {}
            difficulty_used = float(meta.get("difficulty", loop["current_difficulty"]))
            bloom_used = meta.get("bloom", bloom_choice)
            concepts_used = meta.get("target_concepts", q.meta.concepts)
            item_type = meta.get("item_type", "mcq")
            choices_meta = list(meta.get("choices", []) or [])
            correct_index_meta = meta.get(
                "correct_index",
                q.correct_index,
            )
            answer_key_meta = meta.get("answer_key") or getattr(
                q,
                "answer_key",
                "",
            )

            if choices_meta and chosen_index is not None:
                user_answer_text = choices_meta[chosen_index]
                user_answer_index = chosen_index
            else:
                user_answer_text = answer_text or ""
                user_answer_index = None

            post_event(
                "RESPONSE_RAW",
                {
                    "question_id": q.id,
                    "user_answer": user_answer_text,
                    "user_answer_index": user_answer_index,
                    "affect": affect_level,
                    "confidence": confidence,
                    "t_start": t_start,
                    "t_end": t_end,
                    "user_id": user_id,
                    "difficulty": difficulty_used,
                    "bloom": bloom_used,
                    "item_type": item_type,
                    "mode": meta.get("mode"),
                    "band_status": meta.get("band_status"),
                },
            )

            correctness = None
            latency_ms = max(0.0, (t_end - t_start) * 1000.0)
            score_metric = None
            correct_answer_text = answer_key_meta or None
            scored_response: dict | None = None
            if item_type == "mcq" and choices_meta and correct_index_meta is not None:
                idx_chosen = chosen_index if chosen_index is not None else -1
                correctness, latency_ms = mcq.score(
                    int(correct_index_meta),
                    idx_chosen,
                    t_start,
                    t_end,
                )
                if 0 <= int(correct_index_meta) < len(choices_meta):
                    correct_answer_text = choices_meta[int(correct_index_meta)]
                scored_payload = {
                    "question_id": q.id,
                    "correctness": correctness,
                    "score": correctness,
                    "latency_ms": latency_ms,
                    "user_id": user_id,
                    "difficulty": difficulty_used,
                    "bloom": bloom_used,
                    "correct_answer": correct_answer_text,
                    "item_type": item_type,
                    "mode": meta.get("mode"),
                    "band_status": meta.get("band_status"),
                }
                scored_response = post_event("RESPONSE_SCORED", scored_payload)
                if correctness:
                    st.success(f"Correct! ({latency_ms:.0f} ms)")
                else:
                    st.error(
                        "Incorrect. Correct: "
                        f"{correct_answer_text} ({latency_ms:.0f} ms)"
                    )
            elif item_type != "mcq":
                answer_expl_meta = meta.get("answer_expl") or getattr(
                    q,
                    "answer_expl",
                    "",
                )
                stem_for_eval = meta.get("stem", q.stem)
                (
                    correctness,
                    latency_ms,
                    score_metric,
                    evaluator_rationale,
                ) = open_text.score(
                    answer_key_meta,
                    user_answer_text,
                    t_start,
                    t_end,
                    stem=stem_for_eval,
                    explanation=answer_expl_meta,
                )
                scored_payload = {
                    "question_id": q.id,
                    "correctness": correctness,
                    "score": score_metric,
                    "latency_ms": latency_ms,
                    "user_id": user_id,
                    "difficulty": difficulty_used,
                    "bloom": bloom_used,
                    "correct_answer": correct_answer_text,
                    "item_type": item_type,
                    "similarity": score_metric,
                    "mode": meta.get("mode"),
                    "band_status": meta.get("band_status"),
                }
                if evaluator_rationale:
                    scored_payload["evaluator_rationale"] = evaluator_rationale
                scored_response = post_event("RESPONSE_SCORED", scored_payload)
                if correctness:
                    st.success(
                        "Great answer! Score "
                        f"{(score_metric or 0.0):.2f} ({latency_ms:.0f} ms)"
                    )
                    if evaluator_rationale:
                        st.caption(evaluator_rationale)
                else:
                    st.warning(
                        "Recorded. Score "
                        f"{(score_metric or 0.0):.2f} ({latency_ms:.0f} ms)"
                    )
                    if correct_answer_text:
                        st.info(f"Reference answer: {correct_answer_text}")
                    if evaluator_rationale:
                        st.caption(evaluator_rationale)
            else:
                st.info(f"Answer recorded. ({latency_ms:.0f} ms)")
            attention_signals = None
            emotion_snapshot: dict | None = None
            if scored_response:
                raw_signals = scored_response.get("derived_signals") or {}
                attention_payload = raw_signals.get("attention")
                if not attention_payload and raw_signals:
                    attention_payload = raw_signals

                def _as_float(value: object) -> float | None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None

                def _as_int(value: object) -> int | None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return None

                if attention_payload:
                    attention_signals = {
                        "latency_z": _as_float(attention_payload.get("latency_z")),
                        "error_streak": _as_int(attention_payload.get("error_streak")),
                        "sample_size": _as_int(attention_payload.get("sample_size")),
                    }

                emotion_payload = raw_signals.get("emotion")
                if isinstance(emotion_payload, dict):
                    loop["emotion_state"] = emotion_payload
                    emotion_snapshot = emotion_payload

            if attention_signals is not None:
                loop["attention_state"] = attention_signals

            answer_display = user_answer_text
            correct_choice = correct_answer_text

            history_entry = {
                "question_id": q.id,
                "difficulty": round(difficulty_used, 2),
                "bloom": bloom_used,
                "correct": (bool(correctness) if correctness is not None else None),
                "latency_ms": round(latency_ms, 1),
                "user_answer": answer_display,
                "correct_answer": correct_choice,
                "concepts": ", ".join(concepts_used),
                "confidence": confidence,
                "item_type": item_type,
                "backend": meta.get("backend"),
                "backend_expected": meta.get("backend_expected"),
                "backend_fallback": bool(meta.get("backend_fallback")),
                "similarity": (
                    round(score_metric, 2) if score_metric is not None else None
                ),
                "mode": meta.get("mode"),
                "band_status": meta.get("band_status"),
                "policy_rationale": meta.get("policy_rationale"),
                "emotion_interventions": list(
                    meta.get("emotion_interventions")
                    or loop.get("emotion_interventions")
                    or []
                ),
                "emotion_notes": list(
                    meta.get("emotion_notes") or loop.get("emotion_notes") or []
                ),
            }
            if attention_signals:
                if attention_signals.get("latency_z") is not None:
                    history_entry["latency_z"] = round(
                        float(attention_signals["latency_z"]),
                        2,
                    )
                if attention_signals.get("error_streak") is not None:
                    history_entry["error_streak"] = int(
                        attention_signals["error_streak"]
                    )
                if attention_signals.get("sample_size") is not None:
                    history_entry["sample_size"] = int(attention_signals["sample_size"])
            if emotion_snapshot:
                history_entry["emotion"] = emotion_snapshot
                history_entry["emotion_summary"] = (
                    f"F {emotion_snapshot.get('frustration_prob', 0):.2f}, "
                    f"D {emotion_snapshot.get('demotivation_prob', 0):.2f}, "
                    f"S {emotion_snapshot.get('stress_prob', 0):.2f}"
                )
            loop["history"].append(history_entry)
            loop["last_result"] = {
                "stem": meta.get("stem", q.stem),
                **history_entry,
            }

            if correctness is not None:
                lm = loop["learner_model"]
                mastery = lm.setdefault("mastery", {})
                delta = 0.1 if correctness else -0.1
                for concept in concepts_used:
                    key = concept.lower()
                    current = mastery.get(key, 0.5)
                    updated = max(0.0, min(1.0, round(current + delta, 2)))
                    mastery[key] = updated
                now_utc = datetime.now(timezone.utc)
                for concept in concepts_used:
                    key = concept.lower()
                    mastery_score = mastery.get(key)
                    SCHEDULER.update(
                        lm,
                        concept,
                        correctness,
                        mastery_score,
                        now=now_utc,
                    )
                learner_model.save(user_id, lm)

            with st.expander("Why?/Explanation"):
                st.write(meta.get("answer_expl", q.answer_expl) or "")

            loop["current_q"] = None
            loop["t_start"] = None
            loop["pending_meta"] = None
            loop["pending_review"] = None

    last = loop.get("last_result")
    if isinstance(last, dict):
        status = "Pending"
        if last.get("correct") is True:
            status = "✅ Correct"
        elif last.get("correct") is False:
            status = "❌ Incorrect"
        st.markdown("### Last result")
        st.write(status)
        st.write(last["stem"])
        st.write(
            "Difficulty "
            f"{last['difficulty']:.2f} | "
            f"Bloom {last['bloom'].title()}"
        )
        st.write(f"Latency: {last['latency_ms']:.1f} ms")
        if last.get("mode") == "review":
            st.write("Mode: Spaced review")
        elif last.get("mode") == "learn":
            st.write("Mode: Learning")
        if last.get("band_status"):
            st.write(f"Band status: {last['band_status']}")
        if last.get("item_type"):
            st.write(f"Item type: {last['item_type']}")
        if last.get("similarity") is not None:
            st.write(f"Similarity: {last['similarity']:.2f}")
        if last.get("latency_z") is not None:
            st.write(f"Latency z-score: {last['latency_z']:.2f}")
        if last.get("error_streak") is not None:
            st.write(f"Error streak: {last['error_streak']}")
        if last.get("user_answer"):
            st.write(f"Your answer: {last['user_answer']}")
        if last.get("correct_answer"):
            st.write(f"Correct answer: {last['correct_answer']}")
        if last.get("emotion_summary"):
            st.write(f"Emotion probs: {last['emotion_summary']}")
        interventions = last.get("emotion_interventions") or []
        if interventions:
            st.markdown("**Support actions suggested:**")
            for action in interventions:
                st.write(f"- {action}")
        notes = last.get("emotion_notes") or []
        if notes:
            st.markdown("**Emotion notes:**")
            for note in notes:
                st.write(f"- {note}")
        rationale = last.get("policy_rationale")
        if isinstance(rationale, dict):
            with st.expander("Policy rationale"):
                st.json(rationale)

    # Prompt for self-report if emotion exceeds threshold
    show_self_report = False
    emo_state = loop.get("emotion_state")
    if emo_state:
        st.markdown("### Current affect estimate")
        cols = st.columns(3)
        cols[0].metric(
            "Frustration",
            f"{emo_state.get('frustration_prob', 0):.2f}",
        )
        cols[1].metric(
            "Demotivation",
            f"{emo_state.get('demotivation_prob', 0):.2f}",
        )
        cols[2].metric("Stress", f"{emo_state.get('stress_prob', 0):.2f}")
        if (
            emo_state.get("frustration_prob", 0) > 0.6
            or emo_state.get("demotivation_prob", 0) > 0.6
            or emo_state.get("stress_prob", 0) > 0.6
        ):
            show_self_report = True

    # Always show emotion analysis section
    st.markdown("### Emotion Analysis")
    if show_self_report:
        st.warning("We noticed signs of frustration or stress. " "How are you feeling?")

    user_feeling = st.text_area(
        "Describe your feelings or thoughts",
        key="self_report_text",
    )
    analyze_clicked = st.button(
        "Analyze my feelings",
        key="analyze_feeling_btn",
    )
    if analyze_clicked and user_feeling.strip():
        transformer = get_emotion_transformer()
        if transformer is None:
            message = (
                "Emotion transformer backend unavailable. Install "
                "torch and transformers, then restart the app."
            )
            st.session_state["emo_transformer_error"] = message
            st.error(message)
            st.stop()

        try:
            analysis = transformer.analyze(user_feeling)
        except RuntimeError as exc:
            message = (
                "Failed to run emotion transformer backend. Ensure "
                "torch/transformers are installed and accessible."
            )
            st.session_state["emo_transformer_error"] = message
            st.error(message)
            st.write("Backend detail:", str(exc))
            st.stop()

        st.session_state.pop("emo_transformer_error", None)

        derived_state = analysis.state
        summary_str = analysis.summary()
        loop["emotion_state"] = derived_state
        loop["emotion_summary"] = summary_str

        # Generate personalized emotional advice
        try:
            advice_agent = get_emotional_advice_agent()
            emotional_advice = advice_agent.generate_advice(
                user_feeling,
                {
                    "flattened": analysis.flattened,
                    "dominant_bucket": analysis.dominant_bucket(),
                    "state": derived_state,
                },
                derived_state,
            )

            # Show the emotional advice prominently
            st.success("💙 **Your Emotional Support Guide**")
            st.write(emotional_advice)

        except Exception as e:
            st.warning(
                "Could not generate personalized advice, but your emotions were analyzed successfully."
            )
            st.write(f"Debug: Advice generation failed: {e}")

        # Keep debug info in expandable section for developer
        with st.expander("🔍 Technical Details (Debug Info)"):
            st.write("**Emotion analysis:**", analysis.flattened)
            st.write("**Bucketized probabilities:**", derived_state)

            dominant_label = analysis.top_label()
            if dominant_label:
                label_name, label_score = dominant_label
                st.write(
                    f"**Dominant emotion label:** {label_name.title()} ({label_score:.2f})"
                )

            dominant_bucket = analysis.dominant_bucket()
            bucket_note = None
            if dominant_bucket:
                bucket_name, bucket_value = dominant_bucket
                st.write(
                    f"**Dominant affect bucket:** {bucket_name.title()} ({bucket_value:.2f})"
                )
                bucket_note = (
                    f"Self-report dominant affect: {bucket_name} "
                    f"({bucket_value:.2f})"
                )

        interventions = (
            _suggest_self_report_interventions(
                derived_state,
                loop.get("attention_state"),
            )
            or []
        )

        loop["emotion_interventions"] = interventions
        notes_list: list[str] = []
        if bucket_note:
            notes_list.append(bucket_note)
        if interventions:
            notes_list.append("Interventions suggested: " + "; ".join(interventions))
        loop["emotion_notes"] = notes_list

        last_result = loop.get("last_result")
        if isinstance(last_result, dict):
            last_result["emotion_summary"] = summary_str
            last_result["emotion_interventions"] = list(interventions)
            last_result["emotion_notes"] = list(notes_list)
            last_result["emotion"] = derived_state

        if loop.get("history"):
            latest = loop["history"][-1]
            latest["emotion_summary"] = summary_str
            latest["emotion_interventions"] = list(interventions)
            latest["emotion_notes"] = list(notes_list)
            latest["emotion"] = derived_state

        if interventions:
            st.markdown("**Suggested support actions:**")
            for action in interventions:
                st.write(f"- {action}")

        if notes_list:
            st.markdown("**Analysis notes:**")
            for note in notes_list:
                st.write(f"- {note}")

        post_event(
            "EMOTION_SELF_REPORT",
            {
                "user_id": loop["user_id"],
                "text": user_feeling,
                "analysis": analysis.raw,
                "flattened": analysis.flattened,
                "aggregated": derived_state,
                "interventions": interventions,
            },
        )

    if loop["history"]:
        st.markdown("### Session history")
        history_display = []
        for row in loop["history"]:
            display_row = row.copy()
            rationale = display_row.get("policy_rationale")
            if isinstance(rationale, dict):
                display_row["policy_summary"] = rationale.get("band_status")
                display_row["policy_rationale"] = json.dumps(rationale)
            emotion = display_row.get("emotion")
            if isinstance(emotion, dict):
                display_row["emotion"] = json.dumps(emotion)
            interventions_row = display_row.get("emotion_interventions")
            if isinstance(interventions_row, (list, tuple, set)):
                interventions_str = "; ".join(str(item) for item in interventions_row)
                display_row["emotion_interventions"] = interventions_str
            elif interventions_row is None:
                display_row["emotion_interventions"] = ""
            else:
                display_row["emotion_interventions"] = str(interventions_row)
            notes = display_row.get("emotion_notes")
            if isinstance(notes, (list, tuple, set)):
                notes_str = "; ".join(str(item) for item in notes)
                display_row["emotion_notes"] = notes_str
            elif notes is None:
                display_row["emotion_notes"] = ""
            else:
                display_row["emotion_notes"] = str(notes)
            history_display.append(display_row)
        st.dataframe(history_display, use_container_width=True)

    mastery = loop["learner_model"].get("mastery", {})
    if mastery:
        st.markdown("### Learner mastery snapshot")
        mastery_rows = [
            {"concept": concept, "score": value} for concept, value in mastery.items()
        ]
        st.table(mastery_rows)

    upcoming = SCHEDULER.get_due_reviews(
        loop["learner_model"],
        include_future=True,
    )
    if upcoming:
        st.markdown("### Review schedule")
        now_ts = datetime.now(timezone.utc).timestamp()
        review_rows = []
        for item in upcoming[:6]:
            due_dt = datetime.fromtimestamp(item.next_due_ts, tz=timezone.utc)
            status = "Due" if item.next_due_ts <= now_ts else "Scheduled"
            review_rows.append(
                {
                    "concept": item.concept,
                    "due": due_dt.strftime("%Y-%m-%d %H:%M UTC"),
                    "status": status,
                }
            )
        st.table(review_rows)


if __name__ == "__main__":
    main()
