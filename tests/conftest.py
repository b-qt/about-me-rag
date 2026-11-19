import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch, tmp_path):
    """Tests never download models or call an LLM."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("ABOUTME_IGNORE_SECRETS", "1")  # don't read a local .streamlit/secrets.toml
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "LLM_PROVIDER", "OLLAMA_HOST",
                 "OLLAMA_ENDPOINT", "OLLAMA_API_KEY", "OLLAMA_MODEL", "GROQ_API_KEY", "OPENAI_COMPAT_API_KEY",
                 "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    from aboutme import llm

    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)  # no real backoff waits in tests


@pytest.fixture(scope="session")
def retriever(tmp_path_factory):
    import os

    os.environ["EMBEDDING_BACKEND"] = "hash"
    from aboutme.embeddings import HashEmbedder
    from aboutme.index import ensure_index
    from aboutme.retrieval import Retriever

    collection, _, embedder = ensure_index(HashEmbedder(), tmp_path_factory.mktemp("index"))
    return Retriever(collection, embedder)
