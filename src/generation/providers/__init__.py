"""Answer-provider selection with one opt-in commercial provider."""

from __future__ import annotations

from src.common.config import LOCAL_ANSWER_PROVIDER, OPENAI_ANSWER_PROVIDER, Settings
from src.generation.providers.base import AnswerProvider
from src.generation.providers.deterministic import DeterministicAnswerProvider
from src.generation.providers.openai_provider import OpenAIAnswerProvider


def build_answer_provider(settings: Settings) -> AnswerProvider:
    if settings.answer_provider == LOCAL_ANSWER_PROVIDER:
        return DeterministicAnswerProvider()
    if settings.answer_provider == OPENAI_ANSWER_PROVIDER:
        return OpenAIAnswerProvider(
            api_key=settings.openai_api_key,
            model_name=settings.answer_model,
            timeout_seconds=settings.answer_timeout_seconds,
            max_retries=settings.answer_max_retries,
        )
    raise ValueError(f"Unsupported answer provider: {settings.answer_provider!r}")


__all__ = [
    "AnswerProvider",
    "DeterministicAnswerProvider",
    "OpenAIAnswerProvider",
    "build_answer_provider",
]
