from __future__ import annotations

import abc
from typing import Protocol

from services.qgen.schemas import GenerateRequest, Question


class BaseQuestionGenerator(abc.ABC):
    """Abstract strategy for producing assessment questions."""

    @abc.abstractmethod
    def generate(self, request: GenerateRequest) -> Question:
        """Produce a question for the given request."""


class SupportsGenerate(Protocol):
    def generate(  # pragma: no cover - Protocol stub
        self,
        request: GenerateRequest,
    ) -> Question: ...
