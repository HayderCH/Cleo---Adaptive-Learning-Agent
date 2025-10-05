from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import yaml

DEFAULT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_BUCKET_MAPPING: Dict[str, set[str]] = {
    "frustration": {
        "annoyance",
        "anger",
        "disgust",
        "frustration",
    },
    "demotivation": {
        "boredom",
        "disappointment",
        "sadness",
        "tiredness",
    },
    "stress": {
        "fear",
        "nervousness",
        "anxiety",
        "stress",
        "worry",
    },
}
MODEL_ALIASES = {
    "goemotions-finetune": DEFAULT_MODEL_NAME,
    "emotion-english-distilroberta-base": DEFAULT_MODEL_NAME,
}
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "emotion.yaml"


@dataclass
class EmotionTransformerConfig:
    model_name: str = DEFAULT_MODEL_NAME
    top_k: Optional[int] = None
    return_all_scores: bool = True
    device: str = "auto"
    trust_remote_code: bool = False
    bucket_mapping: Dict[str, set[str]] = field(
        default_factory=lambda: {k: set(v) for k, v in DEFAULT_BUCKET_MAPPING.items()}
    )
    fusion_weight: float = 0.6

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmotionTransformerConfig":
        model_name = str(
            data.get("huggingface_model")
            or data.get("text_emotion_model")
            or DEFAULT_MODEL_NAME
        )
        model_name = MODEL_ALIASES.get(model_name, model_name)

        top_k = data.get("top_k")
        top_k_value: Optional[int]
        if isinstance(top_k, int):
            top_k_value = top_k
        else:
            top_k_value = None

        return_all_scores = bool(data.get("multi_label", True))
        device = str(data.get("device", "auto")).lower()
        trust_remote_code = bool(data.get("trust_remote_code", False))

        bucket_mapping_data = data.get("bucket_mapping")
        bucket_mapping: Dict[str, set[str]]
        if isinstance(bucket_mapping_data, Mapping):
            bucket_mapping = {}
            for bucket, labels in bucket_mapping_data.items():
                if not isinstance(labels, Iterable):
                    continue
                bucket_mapping[str(bucket)] = {
                    str(label).lower() for label in labels if str(label).strip()
                }
            if not bucket_mapping:
                bucket_mapping = {
                    key: set(values) for key, values in DEFAULT_BUCKET_MAPPING.items()
                }
        else:
            bucket_mapping = {
                key: set(values) for key, values in DEFAULT_BUCKET_MAPPING.items()
            }

        fusion_weight = data.get("fusion_weights", {}).get("text")
        if isinstance(fusion_weight, (int, float)):
            fusion_weight_value = float(fusion_weight)
        else:
            fusion_weight_value = 0.6

        return cls(
            model_name=model_name,
            top_k=top_k_value,
            return_all_scores=return_all_scores,
            device=device,
            trust_remote_code=trust_remote_code,
            bucket_mapping=bucket_mapping,
            fusion_weight=fusion_weight_value,
        )

    @classmethod
    def from_config_file(
        cls,
        path: Path = CONFIG_PATH,
    ) -> "EmotionTransformerConfig":
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        else:
            config = {}
        model_cfg = config.get("model", {})
        if not isinstance(model_cfg, Mapping):
            model_cfg = {}
        return cls.from_dict(model_cfg)


@dataclass
class EmotionAnalysis:
    raw: Sequence[Any]
    flattened: List[Dict[str, Any]]
    buckets: Dict[str, float]
    state: Dict[str, float]

    def summary(self) -> str:
        return (
            f"F {self.state.get('frustration_prob', 0.0):.2f}, "
            f"D {self.state.get('demotivation_prob', 0.0):.2f}, "
            f"S {self.state.get('stress_prob', 0.0):.2f}"
        )

    def top_label(self) -> Optional[Tuple[str, float]]:
        if not self.flattened:
            return None
        winner = max(
            self.flattened,
            key=lambda item: float(item.get("score", 0.0) or 0.0),
        )
        label = str(winner.get("label", "")).strip()
        score = float(winner.get("score", 0.0) or 0.0)
        if not label:
            return None
        return label, score

    def dominant_bucket(self) -> Optional[Tuple[str, float]]:
        if not self.buckets:
            return None
        bucket, value = max(
            self.buckets.items(),
            key=lambda item: float(item[1] or 0.0),
        )
        return bucket, float(value)


@dataclass
class EmotionTransformer:
    config: EmotionTransformerConfig = field(default_factory=EmotionTransformerConfig)
    pipeline_factory: Optional[Callable[..., Any]] = None
    _pipeline: Optional[Callable[[str], Any]] = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def from_config(
        cls,
        path: Path = CONFIG_PATH,
        *,
        pipeline_factory: Optional[Callable[..., Any]] = None,
    ) -> "EmotionTransformer":
        cfg = EmotionTransformerConfig.from_config_file(path)
        return cls(config=cfg, pipeline_factory=pipeline_factory)

    def analyze(self, text: str) -> EmotionAnalysis:
        pipeline = self._ensure_pipeline()
        cleaned = text.strip()
        if not cleaned:
            buckets = {key: 0.0 for key in self.config.bucket_mapping}
            state = build_emotion_state(buckets)
            return EmotionAnalysis(
                raw=[],
                flattened=[],
                buckets=buckets,
                state=state,
            )

        raw = pipeline(cleaned)
        flattened = flatten_predictions(raw)
        buckets = aggregate_buckets(flattened, self.config.bucket_mapping)
        state = build_emotion_state(buckets)
        return EmotionAnalysis(
            raw=raw,
            flattened=flattened,
            buckets=buckets,
            state=state,
        )

    def _ensure_pipeline(self) -> Callable[[str], Any]:
        if self._pipeline is not None:
            return self._pipeline

        loader = self.pipeline_factory
        if loader is None:
            try:
                from transformers import pipeline  # type: ignore
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError(
                    "Transformers library is required for emotion analysis."
                ) from exc
            loader = pipeline

        device = self._resolve_device()
        try:
            os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
            pipeline_instance = loader(
                "text-classification",
                model=self.config.model_name,
                top_k=self.config.top_k,
                device=device,
                return_all_scores=self.config.return_all_scores,
                trust_remote_code=self.config.trust_remote_code,
                truncation=True,
            )
        except Exception as exc:  # pragma: no cover - relies on external deps
            raise RuntimeError(
                "Failed to initialise Hugging Face emotion pipeline."
            ) from exc

        if not callable(pipeline_instance):  # pragma: no cover - defensive
            raise RuntimeError("Unexpected pipeline object returned.")

        self._pipeline = pipeline_instance
        return self._pipeline

    def _resolve_device(self) -> int:
        device = self.config.device
        if device == "cpu":
            return -1
        if device == "cuda":
            return 0
        if device == "auto":
            try:
                import torch  # type: ignore

                return 0 if torch.cuda.is_available() else -1
            except Exception:  # pragma: no cover - torch optional for CPU
                return -1
        return -1


def flatten_predictions(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    if raw and isinstance(raw[0], list):
        flattened: List[Dict[str, Any]] = []
        for row in raw:
            if isinstance(row, list):
                flattened.extend(item for item in row if isinstance(item, Mapping))
        return [
            {
                "label": str(item.get("label", "")),
                "score": float(item.get("score", 0.0) or 0.0),
            }
            for item in flattened
        ]
    return [
        {
            "label": str(item.get("label", "")),
            "score": float(item.get("score", 0.0) or 0.0),
        }
        for item in raw
        if isinstance(item, Mapping)
    ]


def aggregate_buckets(
    predictions: Iterable[Mapping[str, Any]],
    bucket_mapping: Optional[Mapping[str, Iterable[str]]] = None,
) -> Dict[str, float]:
    mapping = _normalise_mapping(bucket_mapping)
    scores = {bucket: 0.0 for bucket in mapping}
    for entry in predictions:
        label = str(entry.get("label", "")).lower()
        try:
            score = float(entry.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if not label:
            continue
        for bucket, labels in mapping.items():
            if label in labels:
                scores[bucket] = max(scores[bucket], score)
    return scores


def build_emotion_state(
    buckets: Mapping[str, float],
    *,
    decimals: int = 3,
) -> Dict[str, float]:
    return {
        f"{bucket}_prob": round(
            max(0.0, min(1.0, float(value or 0.0))),
            decimals,
        )
        for bucket, value in buckets.items()
    }


def _normalise_mapping(
    bucket_mapping: Optional[Mapping[str, Iterable[str]]],
) -> Dict[str, set[str]]:
    if not bucket_mapping:
        return {key: set(values) for key, values in DEFAULT_BUCKET_MAPPING.items()}
    normalised: Dict[str, set[str]] = {}
    for bucket, labels in bucket_mapping.items():
        entries = {str(label).lower() for label in labels if str(label).strip()}
        normalised[str(bucket)] = entries
    return normalised


__all__ = [
    "EmotionTransformerConfig",
    "EmotionTransformer",
    "EmotionAnalysis",
    "flatten_predictions",
    "aggregate_buckets",
    "build_emotion_state",
]
