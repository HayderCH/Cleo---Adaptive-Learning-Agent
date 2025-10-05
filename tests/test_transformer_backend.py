from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from services.qgen.backends.transformer import (
    TransformerConfig,
    TransformerQuestionGenerator,
)

META_ERROR = RuntimeError("Tensor on device cuda:0 is not on the expected device meta!")


def _successful_pipeline() -> Any:
    def _runner(*_: Any, **__: Any) -> List[dict[str, Any]]:
        return [
            {
                "generated_text": '{"stem": "s", "choices": ["a"],'
                ' "correct_index": 0}',
            }
        ]

    return _runner


def _failing_pipeline() -> Any:
    def _runner(*_: Any, **__: Any) -> List[dict[str, Any]]:
        raise META_ERROR

    return _runner


@pytest.mark.parametrize(
    "failure_pattern,expected_calls",
    [
        (
            [True, False],
            [(None, None), (False, None)],
        ),
        (
            [True, True, False],
            [(None, None), (False, None), (False, "cpu")],
        ),
    ],
)
def test_run_pipeline_handles_meta_device_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure_pattern: List[bool],
    expected_calls: List[Tuple[Any, Any]],
) -> None:
    config = TransformerConfig(model_name="dummy-model")
    backend = TransformerQuestionGenerator(config)
    call_history: List[Tuple[Any, Any]] = []
    pattern_iter = iter(failure_pattern)

    def fake_ensure_pipeline(
        self: TransformerQuestionGenerator,
        *,
        use_device_map: Any = None,
        forced_device: Any = None,
    ) -> Any:
        call_history.append((use_device_map, forced_device))
        should_fail = next(pattern_iter)
        return _failing_pipeline() if should_fail else _successful_pipeline()

    monkeypatch.setattr(
        TransformerQuestionGenerator,
        "_ensure_pipeline",
        fake_ensure_pipeline,
    )

    result = backend._run_pipeline("prompt")

    assert result[0]["generated_text"].startswith("{")
    assert call_history == expected_calls
    assert backend._device_map_enabled is False
