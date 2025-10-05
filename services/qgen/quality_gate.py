from __future__ import annotations
from typing import List, Tuple
from services.qgen.schemas import Question


def _has_single_correct_mcq(q: Question) -> bool:
    if q.choices is None:
        return False
    if q.correct_index is None:
        return False
    return 0 <= q.correct_index < len(q.choices)


def _no_duplicate_choices(q: Question) -> bool:
    if not q.choices:
        return True
    lower = [c.strip().lower() for c in q.choices]
    return len(lower) == len(set(lower))


def _length_ok(text: str, min_len: int = 10, max_len: int = 400) -> bool:
    s = (text or "").strip()
    return min_len <= len(s) <= max_len


def validate(q: Question) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    # stem
    if not _length_ok(q.stem, 5, 500):
        reasons.append("stem length out of bounds")
    # mcq checks
    if q.choices is not None:
        if len(q.choices) < 3 or len(q.choices) > 6:
            reasons.append("mcq choices count must be 3..6")
        if not _has_single_correct_mcq(q):
            reasons.append("mcq correct_index invalid")
        if not _no_duplicate_choices(q):
            reasons.append("mcq duplicate choices")
        answer_text = q.answer_key
        if (not answer_text) and _has_single_correct_mcq(q):
            answer_text = q.choices[q.correct_index]
        if not _length_ok(answer_text or "", 1, 400):
            if (answer_text or "").strip():
                reasons.append("answer key too short")
            else:
                reasons.append("answer key missing")
    else:
        answer_text = q.answer_key or ""
        if not _length_ok(answer_text, 1, 400):
            if answer_text.strip():
                reasons.append("open answer too short")
            else:
                reasons.append("open item requires canonical answer")
    # basic banned terms
    banned = {"password", "ssn", "credit card"}
    if any(b in (q.stem or "").lower() for b in banned):
        reasons.append("banned term in stem")
    ok = len(reasons) == 0
    return ok, reasons
