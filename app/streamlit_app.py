"""Local Streamlit UI for CivicLens RAG."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api_client import APIClientError, ask_question  # noqa: E402


__all__ = ["PAGE_TITLE", "route_question"]


PAGE_TITLE = "CivicLens RAG \u2014 NYC 311 Operations Copilot"
EXAMPLE_QUESTIONS = (
    "What is the local retrieval and cited answer flow?",
    "What is the no-answer rule?",
    "Which borough has the highest complaint volume?",
    "What are the top complaint types?",
    "Which agencies handle the most requests?",
    "What is the backlog summary?",
)
def route_question(question: str, *, top_k: int = 5) -> dict[str, Any]:
    """Send a question through the public FastAPI boundary."""

    try:
        return ask_question(question, top_k=top_k)
    except APIClientError as exc:
        return {
            "answer": exc.user_message,
            "mode": "backend_error",
            "error_code": exc.code,
        }


def render_sources(sources: list[dict[str, Any]]) -> None:
    st.subheader("Source Citations")
    if not sources:
        st.write("No sources returned.")
        return

    for fallback_number, source in enumerate(sources, start=1):
        source_name = source.get("source_name", "Unknown source")
        source_path = source.get("source_path", "Unknown path")
        chunk_id = source.get("chunk_id", "n/a")
        citation_number = source.get("citation_number", fallback_number)
        st.markdown(
            f"{citation_number}. `{source_name}` - `{source_path}` - chunk `{chunk_id}`"
        )


def render_response(response: dict[str, Any]) -> None:
    st.subheader("Answer")
    if response.get("mode") == "backend_error":
        st.warning(response["answer"])
        return

    st.write(response["answer"])

    route = response.get("route")
    status = response.get("status")
    if route or status:
        st.caption(f"Route: {route or 'unknown'} | Status: {status or 'unknown'}")

    confidence_note = response.get("confidence_note")
    if confidence_note:
        st.caption(confidence_note)

    query_id = response.get("query_id")
    if query_id:
        st.caption(f"Query ID: {query_id}")

    render_sources(response.get("sources", []))


def main() -> None:
    st.set_page_config(page_title="CivicLens RAG", layout="wide")
    st.title(PAGE_TITLE)
    st.caption(
        "Local AI Data Engineering / Hybrid RAG project for cited NYC 311 documentation answers "
        "and small predefined analytics summaries."
    )
    st.info(
        "Non-production portfolio demo using curated documentation and sample "
        "analytics; it is not connected to live NYC 311 operational data."
    )

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
        with st.spinner("Contacting the local CivicLens API..."):
            response = route_question(submitted_question)
        render_response(response)


if __name__ == "__main__":
    main()
