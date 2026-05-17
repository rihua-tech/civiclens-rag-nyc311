"""Streamlit UI for CivicLens RAG.

This is a starter placeholder. Implement the UI after the retrieval and generation modules are ready.
"""

import streamlit as st


st.set_page_config(page_title="CivicLens RAG", page_icon="🏙️", layout="wide")

st.title("CivicLens RAG — NYC 311 Operations Copilot")
st.caption("AI Data Engineering / Hybrid RAG project for cited NYC 311 operational answers.")

question = st.text_input("Ask a question about NYC 311 documentation, fields, runbooks, or analytics:")

if question:
    st.info("RAG pipeline is not implemented yet. This placeholder will later show cited answers.")
