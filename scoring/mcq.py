from __future__ import annotations
from typing import Tuple


def score(
    correct_index: int, chosen_index: int, t_start: float, t_end: float
) -> Tuple[int, float]:
    correctness = 1 if correct_index == chosen_index else 0
    latency_ms = max(0.0, (t_end - t_start) * 1000.0)
    return correctness, latency_ms
