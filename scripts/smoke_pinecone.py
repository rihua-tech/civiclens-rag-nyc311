"""Explicit, non-CI Pinecone sync/query/PostgreSQL-hydration smoke test."""

from __future__ import annotations

import argparse
from dataclasses import replace

from src.common.config import PINECONE_VECTOR_STORE, Settings
from src.embeddings.embed_chunks import DEFAULT_INPUT_PATH, load_chunks, synchronize_chunks
from src.embeddings.providers import create_embedding_provider
from src.retrieval.retrieve_context import hydrate_vector_matches
from src.vectorstores.models import VectorIdentity
from src.vectorstores.pinecone_store import PineconeVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Opt-in Pinecone smoke test; writes a small explicit namespace and "
            "canonical PostgreSQL metadata."
        )
    )
    parser.add_argument(
        "--namespace-prefix",
        required=True,
        help="Explicit disposable namespace prefix (a corpus hash is appended).",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--confirm-live-write",
        action="store_true",
        help="Required acknowledgement that this command writes to Pinecone.",
    )
    args = parser.parse_args()
    if not args.confirm_live_write:
        parser.error("--confirm-live-write is required")
    if not 1 <= args.limit <= 10:
        parser.error("--limit must be between 1 and 10")

    settings = Settings.from_env()
    if settings.vector_store_provider != PINECONE_VECTOR_STORE:
        parser.error("VECTOR_STORE_PROVIDER must be explicitly set to pinecone")
    settings = replace(
        settings,
        pinecone_namespace_prefix=args.namespace_prefix.strip(),
    )
    provider = create_embedding_provider(settings)
    chunks = load_chunks(DEFAULT_INPUT_PATH)[: args.limit]
    identities = [VectorIdentity.from_chunk(chunk) for chunk in chunks]
    store = PineconeVectorStore(settings, provider.spec, identities)

    result = synchronize_chunks(
        chunks,
        settings,
        provider=provider,
        vector_store=store,
    )
    query_vector = provider.embed(str(chunks[0]["chunk_text"]))
    matches = store.query(
        query_vector,
        candidate_limit=len(chunks),
        min_similarity=-1.0,
    )
    hydrated = hydrate_vector_matches(matches, settings)
    if not hydrated:
        raise RuntimeError("Pinecone smoke query returned no hydrated current chunks")

    print(
        "Pinecone smoke succeeded: "
        f"synced={result.records_written}, "
        f"hydrated={len(hydrated)}, "
        f"target={result.target}."
    )
    print("The smoke namespace is not deleted automatically; remove it explicitly.")


if __name__ == "__main__":
    main()
