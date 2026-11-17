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
    for name in ("GEMINI_API_KEY", "LLM_PROVIDER", "OLLAMA_HOST", "OLLAMA_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def retriever(tmp_path_factory):
    import os

    os.environ["EMBEDDING_BACKEND"] = "hash"
    from aboutme.embeddings import HashEmbedder
    from aboutme.index import ensure_index
    from aboutme.retrieval import Retriever

    collection, _, embedder = ensure_index(HashEmbedder(), tmp_path_factory.mktemp("index"))
    return Retriever(collection, embedder)
