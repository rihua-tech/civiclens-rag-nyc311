"""Provider contract and safe provider error taxonomy."""

from __future__ import annotations

from typing import Protocol, Sequence

from src.generation.schemas import EvidenceItem, ProviderResult


class AnswerProvider(Protocol):
    provider_name: str
    model_name: str

    def generate(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
    ) -> ProviderResult: ...


class AnswerProviderError(RuntimeError):
    """Base error with a non-secret diagnostic code."""

    code = "provider_failure"


class MissingCredentialError(AnswerProviderError):
    code = "missing_credentials"


class ProviderTimeoutError(AnswerProviderError):
    code = "provider_timeout"


class ProviderUnavailableError(AnswerProviderError):
    code = "provider_unavailable"


class ProviderConfigurationError(AnswerProviderError):
    code = "provider_configuration"


class ProviderResponseError(AnswerProviderError):
    code = "malformed_provider_response"

