"""Local Streamlit UI for CivicLens RAG."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.simple_analytics import (
    answer_analytics_question,
    looks_like_analytics_question,
)
from src.generation.answer_question import answer_question


PAGE_TITLE = "CivicLens RAG \u2014 NYC 311 Operations Copilot"
BACKEND_NOT_READY_MESSAGE = (
    "The local PostgreSQL/pgvector backend is not ready. Start Docker with "
    "`docker compose up -d`, then run ingestion, chunking, and embedding commands."
)
EXAMPLE_QUESTIONS = (
    "What is the local retrieval and cited answer flow?",
    "What is the no-answer rule?",
    "Which borough has the highest complaint volume?",
    "What are the top complaint types?",
    "Which agencies handle the most requests?",
    "What is the backlog summary?",
)
CHUNK_PREVIEW_MAX_CHARS = 1200


def route_question(question: str) -> dict[str, Any]:
    """Route simple sample analytics questions before using RAG retrieval."""

    analytics_response = answer_analytics_question(question)
    if analytics_response["mode"] == "analytics":
        return analytics_response
    if looks_like_analytics_question(question):
        return analytics_response

    try:
        rag_response = answer_question(question)
    except Exception as exc:  # pragma: no cover - exercised through Streamlit runtime
        return {
            "answer": BACKEND_NOT_READY_MESSAGE,
            "sources": [],
            "confidence_note": "Local backend unavailable.",
            "retrieved_chunks": [],
            "sample_rows": [],
            "mode": "backend_error",
            "error_detail": f"{type(exc).__name__}: {exc}",
        }

    rag_response["mode"] = "rag"
    rag_response.setdefault("sample_rows", [])
    return rag_response


def render_sources(sources: list[dict[str, Any]]) -> None:
    st.subheader("Source Citations")
    if not sources:
        st.write("No sources returned.")
        return

    for index, source in enumerate(sources, start=1):
        source_name = source.get("source_name", "Unknown source")
        source_path = source.get("source_path", "Unknown path")
        chunk_id = source.get("chunk_id", "n/a")
        st.markdown(f"{index}. `{source_name}` - `{source_path}` - chunk `{chunk_id}`")


def truncate_chunk_text(text: str, max_chars: int = CHUNK_PREVIEW_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_similarity_score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def render_chunk_preview(chunks: list[dict[str, Any]], mode: str) -> None:
    st.subheader("Retrieved Chunk Preview")
    if mode == "analytics":
        st.write("Analytics answers use predefined sample CSV outputs, so no vector chunks were retrieved.")
        return
    if not chunks:
        st.write("No retrieved chunks to preview.")
        return

    for index, chunk in enumerate(chunks, start=1):
        source_name = chunk.get("source_name", "Unknown source")
        source_path = chunk.get("source_path", "Unknown path")
        chunk_id = chunk.get("chunk_id", "n/a")
        similarity_score = format_similarity_score(chunk.get("similarity_score"))
        label = (
            f"{chunk.get('rank', '?')}. {source_name} "
            f"(score={similarity_score})"
        )
        with st.expander(label, expanded=False):
            st.text(
                "\n".join(
                    (
                        f"source_name: {source_name}",
                        f"source_path: {source_path}",
                        f"chunk_id: {chunk_id}",
                        f"similarity_score: {similarity_score}",
                    )
                )
            )
            st.text_area(
                "Chunk text (plain text preview)",
                value=truncate_chunk_text(str(chunk.get("chunk_text", ""))),
                height=180,
                disabled=True,
                key=f"chunk_preview_{index}_{chunk_id}",
            )


def render_sample_rows(rows: list[dict[str, Any]]) -> None:
    if rows:
        st.subheader("Sample Analytics Rows")
        st.dataframe(rows, hide_index=True, use_container_width=True)


def render_response(response: dict[str, Any]) -> None:
    st.subheader("Answer")
    if response.get("mode") == "backend_error":
        st.warning(response["answer"])
        with st.expander("Local debugging detail"):
            st.code(response.get("error_detail", "No exception detail returned."))
    else:
        st.write(response["answer"])

    confidence_note = response.get("confidence_note")
    if confidence_note:
        st.caption(confidence_note)

    render_sources(response.get("sources", []))
    render_sample_rows(response.get("sample_rows", []))
    render_chunk_preview(response.get("retrieved_chunks", []), response.get("mode", "rag"))


def main() -> None:
    st.set_page_config(page_title="CivicLens RAG", layout="wide")
    st.title(PAGE_TITLE)
    st.caption(
        "Local AI Data Engineering / Hybrid RAG project for cited NYC 311 documentation answers "
        "and small predefined analytics summaries."
    )
    st.info("Local in-progress app only. This is not deployed, production-ready, or connected to live NYC 311 data.")

    selected_example = st.selectbox(
        "Example question suggestions",
        ("",) + EXAMPLE_QUESTIONS,
        index=0,
        format_func=lambda value: "Choose an example..." if value == "" else value,
    )
    question = st.text_input(
        "Ask a question about NYC 311 documentation, fields, runbooks, or sample analytics:",
    )
    submitted_question = question.strip() or selected_example.strip()

    if st.button("Ask", type="primary", disabled=not submitted_question):
        with st.spinner("Routing question locally..."):
            response = route_question(submitted_question)
        render_response(response)


if __name__ == "__main__":
    main()
