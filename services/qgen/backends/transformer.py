from __future__ import annotations

import gc
import inspect
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from services.qgen.backends.base import BaseQuestionGenerator
from services.qgen.schemas import GenerateRequest, Question, QuestionMeta

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    model_name: str
    device: str | int | None = None
    device_map: str | None = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    json_system_prompt: str | None = None
    fallback_to_template: bool = True
    dtype: str | None = "float16"
    trust_remote_code: bool = False
    max_attempts: int = 2

    def __post_init__(self) -> None:
        # Backwards compatibility for older configs
        legacy = getattr(
            self,
            "torch_dtype",
            None,
        )  # type: ignore[attr-defined]
        if legacy and not self.dtype:
            self.dtype = legacy


class TransformerQuestionGenerator(BaseQuestionGenerator):
    """Generates questions using a Hugging Face causal language model."""

    _pipeline_lock = threading.Lock()
    _pipeline: Any | None | bool = None
    _device_map_enabled: bool | None = None

    def __init__(self, config: TransformerConfig) -> None:
        if not config.model_name:
            raise ValueError("Transformer model_name must be provided in config.")
        self.config = config
        self._logger = logger.getChild(self.__class__.__name__)

    def _empty_cuda_cache(self) -> None:
        try:  # pragma: no cover - hardware specific
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def _reset_pipeline(
        self,
        *,
        device_map_enabled: Optional[bool] = None,
    ) -> None:
        with self._pipeline_lock:
            self._pipeline = None
            if device_map_enabled is not None:
                self._device_map_enabled = device_map_enabled
        gc.collect()
        self._empty_cuda_cache()

    @staticmethod
    def _is_meta_device_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return isinstance(exc, RuntimeError) and "expected device meta" in message

    def _materialise_model(
        self,
        model: Any,
        *,
        target_device: str | int | None,
    ) -> Tuple[Any, str | int | None]:
        if target_device is None:
            return model, None
        try:
            import torch
        except ImportError:  # pragma: no cover - torch missing
            return model, target_device

        resolved_device = target_device
        try:
            if isinstance(target_device, int):
                device = torch.device(f"cuda:{target_device}")
                resolved_device = f"cuda:{target_device}"
            else:
                device = torch.device(str(target_device))
                resolved_device = str(target_device)
        except (TypeError, ValueError):
            device = torch.device("cpu")
            resolved_device = "cpu"

        try:
            model.to(device)
        except Exception as exc:  # pragma: no cover - hardware specific
            self._logger.warning(
                "Unable to place transformer model on %s. Falling back to CPU.",
                target_device,
            )
            self._logger.debug("Model placement failure: %s", exc, exc_info=True)
            resolved_device = "cpu"
            model.to(torch.device("cpu"))
        return model, resolved_device

    def _resolve_dtype(self) -> Any | None:
        if not self.config.dtype:
            return None
        try:  # pragma: no cover - simple mapping
            import torch

            return getattr(torch, self.config.dtype)
        except (ImportError, AttributeError) as exc:  # pragma: no cover
            raise ValueError(f"Unsupported torch dtype '{self.config.dtype}'.") from exc

    def _resolve_device(self) -> str | int | None:
        if self.config.device not in (None, "auto"):
            return self.config.device
        try:  # pragma: no cover - simple hardware inspection
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:  # pragma: no cover - torch missing
            self._logger.debug(
                "Torch not available while resolving device; defaulting to CPU"
            )
        return "cpu"

    def _ensure_pipeline(
        self,
        *,
        use_device_map: Optional[bool] = None,
        forced_device: str | int | None = None,
    ) -> Any:
        if self._pipeline is False:
            raise RuntimeError("Transformer backend previously failed to initialize.")
        with self._pipeline_lock:
            if self._pipeline is None:
                try:  # pragma: no cover - heavy import
                    from transformers import (
                        AutoModelForCausalLM,
                        AutoTokenizer,
                        pipeline,
                    )
                except ModuleNotFoundError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "The transformers package is required for the "
                        "transformer question generator. Install it with 'pip "
                        "install transformers'."
                    ) from exc

                try:
                    dtype = self._resolve_dtype()
                    if use_device_map is None:
                        if self._device_map_enabled is None:
                            self._device_map_enabled = bool(self.config.device_map)
                        device_map_enabled = self._device_map_enabled
                    else:
                        device_map_enabled = bool(use_device_map)
                        self._device_map_enabled = device_map_enabled

                    logger.info(
                        "Loading transformer question generator model %s",
                        self.config.model_name,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model_name,
                        trust_remote_code=self.config.trust_remote_code,
                    )

                    model_kwargs: Dict[str, Any] = {
                        "trust_remote_code": self.config.trust_remote_code,
                    }
                    model_signature = inspect.signature(
                        AutoModelForCausalLM.from_pretrained
                    )
                    if dtype is not None:
                        if "dtype" in model_signature.parameters:
                            model_kwargs["dtype"] = dtype
                        else:
                            model_kwargs["torch_dtype"] = dtype
                    resolved_device: str | int | None = None
                    if device_map_enabled and self.config.device_map:
                        model_kwargs["device_map"] = self.config.device_map
                    else:
                        if "low_cpu_mem_usage" in model_signature.parameters:
                            model_kwargs["low_cpu_mem_usage"] = False
                        if "offload_state_dict" in model_signature.parameters:
                            model_kwargs["offload_state_dict"] = False
                        resolved_device = forced_device
                        if resolved_device is None:
                            resolved_device = self._resolve_device()
                        if resolved_device is None:
                            resolved_device = "cpu"

                    model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_name,
                        **model_kwargs,
                    )

                    materialised_device: str | int | None = None
                    if not device_map_enabled:
                        model, materialised_device = self._materialise_model(
                            model,
                            target_device=resolved_device,
                        )

                    pipe_kwargs: Dict[str, Any] = {}
                    if dtype is not None:
                        pipeline_signature = inspect.signature(pipeline)
                        if "dtype" in pipeline_signature.parameters:
                            pipe_kwargs["dtype"] = dtype
                        elif "torch_dtype" in pipeline_signature.parameters:
                            pipe_kwargs["torch_dtype"] = dtype
                    if device_map_enabled and self.config.device_map:
                        pipe_kwargs["device_map"] = self.config.device_map
                    else:
                        if materialised_device is not None:
                            pipe_kwargs["device"] = materialised_device
                        elif resolved_device is not None:
                            pipe_kwargs["device"] = resolved_device
                        else:
                            pipe_kwargs["device"] = "cpu"
                    pipe_kwargs["trust_remote_code"] = self.config.trust_remote_code

                    self._pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        **pipe_kwargs,
                    )
                except Exception as exc:  # pragma: no cover - init failures
                    self._pipeline = False
                    self._logger.exception(
                        "Unable to initialize transformer model %s",
                        self.config.model_name,
                    )
                    raise RuntimeError(
                        "Transformer backend failed during model load."
                    ) from exc
            if self._pipeline is False:
                raise RuntimeError(
                    "Transformer backend previously failed to initialize."
                )
            return self._pipeline
        return self._pipeline

    def _run_pipeline(self, prompt: str) -> List[Dict[str, Any]]:
        def _invoke(gen: Any) -> List[Dict[str, Any]]:
            return gen(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                num_return_sequences=1,
                return_full_text=False,
            )

        generator = self._ensure_pipeline()
        try:
            return _invoke(generator)
        except Exception as exc:  # pragma: no cover - pipeline errors
            self._logger.exception("Generation failure via transformer pipeline")
            if self._is_meta_device_error(exc):
                self._logger.warning(
                    "Detected device_map/meta-device mismatch. "
                    "Reinitializing pipeline without device_map."
                )
                self._reset_pipeline(device_map_enabled=False)
                try:
                    generator = self._ensure_pipeline(use_device_map=False)
                    return _invoke(generator)
                except Exception as retry_exc:
                    if self._is_meta_device_error(retry_exc):
                        self._logger.error(
                            "Transformer still reported meta-device tensors after GPU reload; "
                            "falling back to CPU execution.",
                        )
                        self._reset_pipeline(device_map_enabled=False)
                        generator = self._ensure_pipeline(
                            use_device_map=False,
                            forced_device="cpu",
                        )
                        try:
                            return _invoke(generator)
                        except (
                            Exception
                        ) as cpu_exc:  # pragma: no cover - pipeline errors
                            raise RuntimeError(
                                "Transformer generation failed after CPU fallback."
                            ) from cpu_exc
                    raise RuntimeError(
                        "Transformer generation failed after retry."
                    ) from retry_exc
            raise RuntimeError("Transformer generation failed.") from exc

    def _build_prompt(self, request: GenerateRequest) -> str:
        base_prompt = (
            self.config.json_system_prompt
            or "You are a question author. Return valid JSON."
        )

        if request.item_type == "mcq":
            extra = (
                "Produce multiple choice output with 4 choices max. "
                "Populate 'choices' as a list of plain strings, set "
                "'correct_index' to the position of the correct choice, "
                "and include concise 'answer_key' and 'answer_expl' strings."
            )
        else:
            extra = (
                "Produce an open response item. Provide a short 'answer_key' "
                "containing the canonical response, and an 'answer_expl' of "
                "1-2 sentences. Ensure the stem is between 15 and 120 "
                "characters."
            )

        system_prompt = f"{base_prompt}\n\n{extra}"
        payload = {
            "subject": request.subject,
            "focus_concepts": request.focus_concepts,
            "difficulty_target": request.difficulty_target,
            "bloom_target": request.bloom_target,
            "item_type": request.item_type,
        }
        return (
            f"{system_prompt}\n\n"
            "Follow the schema exactly and ensure choices are relevant.\n"
            "Request: "
            f"{json.dumps(payload)}\n"
            "JSON:"
        )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object detected in model output.")
        end = text.rfind("}")
        snippet = text[start : end + 1] if end > start else text[start:]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            repaired = self._repair_json(snippet)
            if repaired is not None:
                return repaired
            raise

    def _repair_json(self, snippet: str) -> Optional[Dict[str, Any]]:
        candidate = snippet
        while candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                candidate = candidate[: exc.pos].rstrip()
                if not candidate:
                    break
                while candidate and candidate[-1] not in "}]":
                    candidate = candidate[:-1].rstrip()
                if not candidate:
                    break
                open_braces = candidate.count("{")
                close_braces = candidate.count("}")
                if close_braces < open_braces:
                    candidate += "}" * (open_braces - close_braces)
        return None

    def _extract_text(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        if isinstance(value, Mapping):
            for key in ("text", "choice", "answer", "value"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            values = [self._extract_text(item) for item in value.values()]
            return next((item for item in values if item), None)
        return str(value).strip() or None

    def _normalise_choices(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Optional[List[str]], Optional[int]]:
        raw_choices = payload.get("choices")
        if not isinstance(raw_choices, list):
            return None, payload.get("correct_index")

        processed: List[str] = []
        explicit_index = payload.get("correct_index")
        if isinstance(explicit_index, int):
            inferred_index: Optional[int] = explicit_index
        else:
            inferred_index = None

        for idx, item in enumerate(raw_choices):
            if isinstance(item, Mapping):
                label = self._extract_text(item)
                processed.append(label or "")
                if inferred_index is None:
                    index_value = item.get("correct_index")
                    if isinstance(index_value, int) and index_value >= 0:
                        inferred_index = idx
                    elif str(item.get("is_correct", "")).lower() in {
                        "true",
                        "1",
                        "yes",
                    }:
                        inferred_index = idx
            else:
                processed.append(self._extract_text(item) or "")

        if not processed:
            return None, inferred_index

        if inferred_index is not None and not (0 <= inferred_index < len(processed)):
            inferred_index = None

        return processed, inferred_index

    def _build_question_from_payload(
        self,
        payload: Dict[str, Any],
    ) -> Question:
        meta_data = payload.get("meta") or {}
        concepts = meta_data.get("concepts") or []
        if not isinstance(concepts, list):
            concepts = [str(concepts)]

        choices, correct_index = self._normalise_choices(payload)
        if choices is None:
            choices = payload.get("choices")
        if correct_index is None:
            correct_index = payload.get("correct_index")

        bloom = str(meta_data.get("bloom", "understand")).lower()
        origin_label = meta_data.get("source")
        origin_clean: Optional[str] = None
        if origin_label is not None:
            origin_clean = str(origin_label).strip() or None

        answer_key = payload.get("answer_key")
        if isinstance(answer_key, Mapping):
            answer_key = self._extract_text(answer_key)

        answer_expl = payload.get("answer_expl")
        if isinstance(answer_expl, Mapping):
            answer_expl = self._extract_text(answer_expl)
        elif isinstance(answer_expl, Iterable) and not isinstance(
            answer_expl,
            str,
        ):
            texts = filter(
                None,
                (self._extract_text(item) for item in answer_expl),
            )
            answer_expl = " ".join(texts)

        if correct_index is None and choices:
            key_text = self._extract_text(payload.get("answer_key"))
            if key_text:
                for idx, option in enumerate(choices):
                    if option and option.strip().lower() == key_text.lower():
                        correct_index = idx
                        break

        if correct_index is None and choices:
            correct_index = 0

        if answer_key is None and choices and correct_index is not None:
            try:
                answer_key = choices[correct_index]
            except (IndexError, TypeError):
                answer_key = None

        question = Question(
            stem=str(payload.get("stem", "")),
            choices=choices,
            correct_index=correct_index,
            answer_key=answer_key,
            answer_expl=answer_expl,
            meta=QuestionMeta(
                concepts=[str(c) for c in concepts],
                difficulty_est=float(meta_data.get("difficulty_est", 0.5)),
                bloom=bloom if bloom else "understand",
                source="transformer",
                version=str(meta_data.get("version", "v0")),
                origin=origin_clean,
            ),
        )
        return question

    def generate(self, request: GenerateRequest) -> Question:
        prompt = self._build_prompt(request)
        self._logger.debug("Transformer generator prompt: %s", prompt)
        attempts = max(1, int(getattr(self.config, "max_attempts", 1)))
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                outputs = self._run_pipeline(prompt)
            except Exception as exc:  # pragma: no cover - pipeline errors
                last_error = exc
                continue

            if not outputs:
                self._logger.warning(
                    "Transformer pipeline returned no output (attempt %s)",
                    attempt + 1,
                )
                last_error = RuntimeError("Transformer backend returned no output.")
                continue

            candidate = outputs[0].get("generated_text", "")
            try:
                payload = self._parse_json(candidate)
                question = self._build_question_from_payload(payload)
            except Exception as exc:  # pragma: no cover - parsing resilience
                self._logger.warning(
                    "Failed to parse transformer output on attempt %s: %s",
                    attempt + 1,
                    candidate,
                    exc_info=True,
                )
                last_error = RuntimeError("Failed to parse transformer output.")
                last_error.__cause__ = exc  # type: ignore[attr-defined]
                continue

            if request.item_type == "mcq":
                choices = question.choices or []
                filled_choices = [c for c in choices if c and c.strip()]
                if len(filled_choices) < 2:
                    self._logger.warning(
                        "Transformer produced insufficient choices on "
                        "attempt %s: %s",
                        attempt + 1,
                        candidate,
                    )
                    last_error = RuntimeError(
                        "Transformer output missing choices for MCQ request."
                    )
                    continue
            elif request.item_type == "open":
                raw_choices = question.choices or []
                if raw_choices:
                    self._logger.info(
                        "Transformer supplied choices for open response; "
                        "removing MCQ fields and keeping canonical answer."
                    )
                    canonical = question.answer_key
                    if not canonical:
                        for option in raw_choices:
                            if option and option.strip():
                                canonical = option.strip()
                                break
                    if canonical:
                        canonical = canonical.strip()
                    question.answer_key = canonical
                question.choices = None
                question.correct_index = None

                # For open response, require a non-empty stem and answer_key
                stem_ok = bool(question.stem and question.stem.strip())
                answer_key_ok = bool(
                    question.answer_key and question.answer_key.strip()
                )
                if not (stem_ok and answer_key_ok):
                    self._logger.warning(
                        "Transformer produced invalid open response on "
                        "attempt %s: %s",
                        attempt + 1,
                        candidate,
                    )
                    last_error = RuntimeError(
                        "Transformer output missing stem or answer_key "
                        "for open response."
                    )
                    continue
                if not question.answer_key:
                    self._logger.warning(
                        "Open response lacks canonical answer even after "
                        "normalisation (attempt %s): %s",
                        attempt + 1,
                        candidate,
                    )
                    last_error = RuntimeError(
                        "Transformer output missing answer_key for open " "response."
                    )
                    continue

            question.meta.difficulty_est = request.difficulty_target
            question.meta.bloom = request.bloom_target
            if request.focus_concepts:
                question.meta.concepts = request.focus_concepts
            return question

        if last_error is not None:
            raise last_error
        raise RuntimeError("Transformer generation failed.")
