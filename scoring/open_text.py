from __future__ import annotations

import difflib
import json
import logging
import os
from typing import Optional, Tuple

from services.qgen.backends.transformer import (
    TransformerConfig,
    TransformerQuestionGenerator,
)
from services.qgen.generator import _read_config


logger = logging.getLogger(__name__)

_SCORER_BACKEND = os.environ.get("ALC_OPEN_SCORER", "transformer").strip().lower()
_TRANSFORMER_SCORER: Optional[TransformerQuestionGenerator] = None
_SEMANTIC_MODEL = None


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _ensure_semantic_model():
    """Load semantic model once and cache it."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer

            _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not available for semantic scoring"
            )
    return _SEMANTIC_MODEL


def semantic_similarity(expected: str, provided: str) -> float:
    """Better semantic similarity using sentence transformers."""
    try:
        model = _ensure_semantic_model()

        # Encode both answers
        expected_embedding = model.encode(expected, convert_to_tensor=True)
        provided_embedding = model.encode(provided, convert_to_tensor=True)

        # Calculate cosine similarity
        from sentence_transformers import util

        similarity_score = util.cos_sim(expected_embedding, provided_embedding).item()
        return max(0.0, min(1.0, similarity_score))
    except ImportError:
        # Fallback to basic similarity if sentence-transformers not available
        return similarity(expected, provided)


def similarity(expected: str, provided: str) -> float:
    if not expected and not provided:
        return 1.0
    return difflib.SequenceMatcher(
        None,
        _normalize(expected or ""),
        _normalize(provided or ""),
    ).ratio()


def _ensure_transformer_scorer() -> TransformerQuestionGenerator:
    global _TRANSFORMER_SCORER
    if _TRANSFORMER_SCORER is None:
        config_data = _read_config().get("transformer") or {}
        if not config_data:
            raise RuntimeError("Transformer scoring requested but config is empty.")
        config = TransformerConfig(**config_data)
        # Keep responses concise for evaluation
        config.max_new_tokens = min(config.max_new_tokens, 192)
        _TRANSFORMER_SCORER = TransformerQuestionGenerator(config)
    return _TRANSFORMER_SCORER


def _build_eval_prompt(
    stem: str,
    canonical: str,
    explanation: Optional[str],
    learner_answer: str,
) -> str:
    system_prompt = (
        "You are an instructional coach grading a student's open response. "
        "Return strict JSON with fields score (float 0-1), correct (bool), "
        "and rationale (string). Consider both the canonical answer and any "
        "explanation. Award partial credit for responses that capture key "
        "ideas even if phrasing differs."
    )
    payload = {
        "stem": stem,
        "canonical_answer": canonical,
        "reference_explanation": explanation or "",
        "learner_answer": learner_answer,
    }
    return (
        f"{system_prompt}\n\n"
        "Evaluate the learner response and output JSON only.\n"
        f"Request: {json.dumps(payload, ensure_ascii=False)}\n"
        "JSON:"
    )


def _score_with_transformer(
    expected_answer: str,
    learner_answer: str,
    stem: str,
    explanation: Optional[str],
    threshold: float,
) -> Tuple[int, float, float, Optional[str]]:
    scorer = _ensure_transformer_scorer()
    prompt = _build_eval_prompt(
        stem,
        expected_answer,
        explanation,
        learner_answer,
    )
    outputs = scorer._run_pipeline(prompt)
    if not outputs:
        raise RuntimeError("Transformer scorer returned no output.")
    payload = scorer._parse_json(outputs[0].get("generated_text", ""))
    score_value = float(payload.get("score", 0.0))
    score_value = max(0.0, min(1.0, score_value))
    correct_field = payload.get("correct")
    if isinstance(correct_field, str):
        correct_field = correct_field.strip().lower() in {"true", "1", "yes"}
    elif not isinstance(correct_field, bool):
        correct_field = score_value >= threshold
    rationale = payload.get("rationale")
    if isinstance(rationale, str):
        rationale_text: Optional[str] = rationale.strip() or None
    else:
        rationale_text = None
    correctness = 1 if score_value >= threshold else 0
    if bool(correct_field) != bool(correctness):
        logger.debug(
            "Transformer scorer disagreement resolved by threshold. " "payload=%s",
            payload,
        )
    return correctness, score_value, score_value, rationale_text


def score(
    expected_answer: str,
    learner_answer: str,
    t_start: float,
    t_end: float,
    threshold: float = 0.75,
    stem: str | None = None,
    explanation: str | None = None,
) -> Tuple[int, float, float, Optional[str]]:
    """Return correctness flag, latency milliseconds, score, and rationale."""

    latency_ms = max(0.0, (t_end - t_start) * 1000.0)

    if _SCORER_BACKEND == "transformer":
        try:
            result = _score_with_transformer(
                expected_answer or "",
                learner_answer or "",
                stem or "",
                explanation,
                threshold,
            )
            correctness, score_value, metric, rationale = result
            return correctness, latency_ms, metric, rationale
        except Exception as exc:
            logger.warning(
                "Transformer scorer failed; falling back to heuristic: %s",
                exc,
                exc_info=True,
            )

    # Fall back to semantic similarity scoring
    sim = semantic_similarity(expected_answer or "", learner_answer or "")
    correctness = 1 if sim >= threshold else 0
    return correctness, latency_ms, sim, None
