"""Lazy LangChain Core wrapper over native CivicLens retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from src.common.config import Settings
from src.retrieval.retrieve_context import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    DIAGNOSTIC_FIELDS,
    RESULT_METADATA_FIELDS,
    retrieve_context,
    result_display_score,
)


class LangChainAdapterUnavailableError(RuntimeError):
    """Raised when the explicitly requested optional adapter is unavailable."""


SAFE_RESULT_FIELDS = tuple(
    field for field in RESULT_METADATA_FIELDS if field != "chunk_text"
) + DIAGNOSTIC_FIELDS + ("retrieval_mode", "rank")


def _document_metadata(result: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        field: result.get(field)
        for field in SAFE_RESULT_FIELDS
        if field in result
    }
    metadata["chunk_id"] = str(result["chunk_id"])
    metadata["score"] = result_display_score(result)
    if metadata.get("heading_path") is not None:
        metadata["heading_path"] = list(metadata["heading_path"])
    return metadata


def create_langchain_retriever(
    *,
    settings: Settings | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    retrieval_callable: Callable[..., list[dict[str, Any]]] = retrieve_context,
):
    """Create a LangChain ``BaseRetriever`` without changing CivicLens RAG."""

    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from pydantic import ConfigDict, Field
    except ImportError as exc:
        raise LangChainAdapterUnavailableError(
            "LangChain compatibility requires requirements-langchain.txt"
        ) from exc

    class CivicLensLangChainRetriever(BaseRetriever):
        """Thin document mapping over the existing CivicLens retrieval boundary."""

        civic_settings: Settings | None = Field(default=None, exclude=True, repr=False)
        civic_top_k: int = top_k
        civic_min_similarity: float = min_similarity
        civic_retrieval: Callable[..., list[dict[str, Any]]] = Field(
            default=retrieval_callable,
            exclude=True,
            repr=False,
        )

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Any,
        ) -> list[Any]:
            del run_manager
            results = self.civic_retrieval(
                query,
                top_k=self.civic_top_k,
                min_similarity=self.civic_min_similarity,
                settings=self.civic_settings,
            )
            return [
                Document(
                    id=str(result["chunk_id"]),
                    page_content=str(result["chunk_text"]),
                    metadata=_document_metadata(result),
                )
                for result in results
            ]

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Any,
        ) -> list[Any]:
            return await asyncio.to_thread(
                self._get_relevant_documents,
                query,
                run_manager=run_manager,
            )

    return CivicLensLangChainRetriever()
