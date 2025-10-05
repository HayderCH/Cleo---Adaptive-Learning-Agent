from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from services.qgen.backends import (
    BaseQuestionGenerator,
    TemplateQuestionGenerator,
    TransformerConfig,
    TransformerQuestionGenerator,
)
from services.qgen.schemas import GenerateRequest, Question

logger = logging.getLogger(__name__)

CONFIG_ENV_VAR = "ALC_QGEN_CONFIG"
DEFAULT_BACKEND = "template"
_AVAILABLE_BACKENDS: tuple[str, ...] = ("template", "transformer")
_backend_override: Optional[str] = None
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "qgen.yaml"


def _read_config() -> Dict[str, Any]:
    path_override = os.environ.get(CONFIG_ENV_VAR)
    config_path = Path(path_override) if path_override else CONFIG_PATH
    if not config_path.exists():
        logger.warning(
            "Question generator config not found at %s. Using defaults.",
            config_path,
        )
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _normalise_backend_name(name: str | None) -> Optional[str]:
    if name is None:
        return None
    lowered = str(name).strip().lower()
    if lowered not in _AVAILABLE_BACKENDS:
        raise ValueError(f"Unsupported question generator backend: {name}")
    return lowered


@lru_cache(maxsize=1)
def _load_backend() -> BaseQuestionGenerator:
    config = _read_config()
    backend_name = (
        _backend_override or str(config.get("backend", DEFAULT_BACKEND)).lower()
    )
    backend_name = _normalise_backend_name(backend_name) or DEFAULT_BACKEND

    if backend_name == "template":
        return TemplateQuestionGenerator()

    if backend_name == "transformer":
        transformer_cfg_raw = dict(config.get("transformer") or {})
        if "torch_dtype" in transformer_cfg_raw and "dtype" not in transformer_cfg_raw:
            transformer_cfg_raw["dtype"] = transformer_cfg_raw.pop("torch_dtype")
        transformer_cfg = TransformerConfig(**transformer_cfg_raw)
        try:
            return TransformerQuestionGenerator(transformer_cfg)
        except Exception as exc:
            if transformer_cfg.fallback_to_template:
                logger.warning(
                    "Transformer backend unavailable; using template: %s",
                    exc,
                )
                return TemplateQuestionGenerator()
            raise

    raise ValueError(f"Unsupported question generator backend: {backend_name}")


def reload_backend() -> None:
    _load_backend.cache_clear()


def set_backend_override(name: str | None) -> None:
    """Force a backend at runtime; pass ``None`` to revert to config."""

    global _backend_override
    normalized = _normalise_backend_name(name)
    if normalized is None:
        _backend_override = None
    else:
        _backend_override = normalized
    reload_backend()


def get_backend_override() -> Optional[str]:
    return _backend_override


def list_available_backends() -> List[str]:
    return list(_AVAILABLE_BACKENDS)


def current_backend_name() -> str:
    config = _read_config()
    name = _backend_override or str(config.get("backend", DEFAULT_BACKEND))
    try:
        return _normalise_backend_name(name) or DEFAULT_BACKEND
    except ValueError:
        return DEFAULT_BACKEND


def generate(request: GenerateRequest) -> Question:
    backend = _load_backend()
    config_backend = current_backend_name()
    if config_backend == "transformer" and isinstance(
        backend, TemplateQuestionGenerator
    ):
        logger.info(
            "Transformer backend previously fell back to template; " "retrying load."
        )
        reload_backend()
        backend = _load_backend()
    try:
        return backend.generate(request)
    except Exception as exc:
        config = _read_config()
        should_fallback = bool(
            (config.get("transformer") or {}).get(
                "fallback_to_template",
                True,
            )
        )
        if should_fallback and not isinstance(
            backend,
            TemplateQuestionGenerator,
        ):
            logger.warning(
                "Transformer generation failed. Using template fallback: %s",
                exc,
            )
            template_backend = TemplateQuestionGenerator()
            return template_backend.generate(request)
        raise
