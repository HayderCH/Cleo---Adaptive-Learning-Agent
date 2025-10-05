from __future__ import annotations
import json
from pathlib import Path
from typing import Dict


BASE = Path("data/processed/lm")


def _path(user_id: str) -> Path:
    BASE.mkdir(parents=True, exist_ok=True)
    return BASE / f"{user_id}.json"


def load(user_id: str) -> Dict:
    p = _path(user_id)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = {}

    data.setdefault("mastery", {})
    data.setdefault("memory", {"concepts": {}})
    return data  # concept -> 0..1


def save(user_id: str, lm: Dict) -> None:
    p = _path(user_id)
    p.write_text(
        json.dumps(lm, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
