from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "emotion.yaml"


@dataclass
class EmotionInterpreter:
    config: Dict[str, object]

    @classmethod
    def from_config(cls, path: Path = CONFIG_PATH) -> "EmotionInterpreter":
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        else:
            config = {}
        return cls(config=config)

    def suggest_interventions(
        self,
        emotion: Optional[Dict[str, float]],
        attention: Optional[Dict[str, float | int]] = None,
    ) -> List[str]:
        if not emotion:
            return []

        frustration = float(emotion.get("frustration_prob", 0.0) or 0.0)
        demotivation = float(emotion.get("demotivation_prob", 0.0) or 0.0)
        stress = float(emotion.get("stress_prob", 0.0) or 0.0)

        latency_z = float((attention or {}).get("latency_z", 0.0) or 0.0)
        error_streak = int((attention or {}).get("error_streak", 0) or 0)

        policy_cfg: Dict[str, object] = {}
        if isinstance(self.config, dict):
            policy_section = self.config.get("policy")
            if isinstance(policy_section, dict):
                cfg = policy_section.get("interventions")
                if isinstance(cfg, dict):
                    policy_cfg = cfg

        advice: List[str] = []

        if self._trigger_high_frustration(
            frustration,
            latency_z,
            error_streak,
        ):
            advice.extend(self._actions_for(policy_cfg, "high_frustration"))

        if demotivation >= 0.6:
            advice.extend(self._actions_for(policy_cfg, "demotivation"))

        if stress >= 0.6:
            advice.extend(self._actions_for(policy_cfg, "stress_overload"))

        return advice

    @staticmethod
    def _trigger_high_frustration(
        frustration: float,
        latency_z: float,
        error_streak: int,
    ) -> bool:
        if frustration >= 0.6:
            return True
        if latency_z > 1.2 and error_streak >= 2:
            return True
        return False

    @staticmethod
    def _actions_for(
        policy_cfg: Optional[Dict[str, object]],
        key: str,
    ) -> List[str]:
        if not isinstance(policy_cfg, dict):
            return []
        node = policy_cfg.get(key)
        if not isinstance(node, dict):
            return []
        actions = node.get("actions", [])
        if not isinstance(actions, list):
            return []
        return [str(action) for action in actions]


__all__ = ["EmotionInterpreter"]
