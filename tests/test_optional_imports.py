from __future__ import annotations

import subprocess
import sys


def test_native_imports_do_not_load_optional_issue18_packages():
    script = """
import sys
import api.main
import src.embeddings.embed_chunks
import src.retrieval.retrieve_context
import src.vectorstores.pinecone_store
import src.integrations.langchain_retriever
assert 'pinecone' not in sys.modules
assert 'langchain_core' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_native_mode_and_explicit_errors_work_when_optional_imports_are_blocked():
    script = """
import builtins
from dataclasses import replace

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'pinecone', 'langchain_core'}:
        raise ImportError('blocked optional package')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import

from src.common.config import Settings
from src.embeddings.providers import EmbeddingSpec
from src.integrations.langchain_retriever import (
    LangChainAdapterUnavailableError,
    create_langchain_retriever,
)
from src.vectorstores.factory import create_vector_store
from src.vectorstores.models import VectorIdentity, VectorStoreConfigurationError

settings = Settings(
    database_url='postgresql://local/test',
    embedding_model='fake',
    use_openai_embeddings=False,
    use_openai_answers=False,
    openai_api_key='',
    embedding_provider='fake',
    embedding_dimension=1536,
)
identity = VectorIdentity('chunk', 'doc', 'chunk-hash', 'doc-hash', 'config-hash')
spec = EmbeddingSpec('fake', 'fake', 1536)
assert create_vector_store(settings, spec, [identity]).provider_name == 'pgvector'

try:
    create_langchain_retriever()
except LangChainAdapterUnavailableError:
    pass
else:
    raise AssertionError('missing LangChain package did not fail explicitly')

pinecone_settings = replace(
    settings,
    vector_store_provider='pinecone',
    pinecone_api_key='secret',
    pinecone_index_name='index',
    pinecone_index_host='index.svc.pinecone.io',
)
pinecone_store = create_vector_store(pinecone_settings, spec, [identity])
try:
    pinecone_store.prepare_sync()
except VectorStoreConfigurationError:
    pass
else:
    raise AssertionError('missing Pinecone package did not fail explicitly')
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
