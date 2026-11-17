"""Build and load the ChromaDB index.

The index is rebuilt from scratch whenever the contents of knowledge/ (or the embedding
backend, or the chunker) change, detected with a fingerprint stored next to the index.
IDs are content hashes, so rebuilding never creates duplicates.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb

from .chunking import CHUNKER_VERSION, SemanticSplitter, chunk_documents
from .config import COLLECTION_NAME, KNOWLEDGE_DIR, index_dir
from .embeddings import Embedder, get_embedder
from .loaders import load_documents

MANIFEST = "manifest.json"


@dataclass
class IndexStats:
    fingerprint: str
    documents: int
    chunks: int
    by_type: dict
    embedder: str
    built_at: float
    rebuilt: bool = False


def knowledge_fingerprint(embedder_name: str, knowledge_dir: Path = KNOWLEDGE_DIR) -> str:
    digest = hashlib.sha256(f"{embedder_name}|chunker-{CHUNKER_VERSION}".encode())
    for path in sorted(p for p in knowledge_dir.rglob("*") if p.is_file()):
        digest.update(path.relative_to(knowledge_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _client(path: Path) -> chromadb.ClientAPI:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))


def read_manifest(path: Path) -> IndexStats | None:
    try:
        return IndexStats(**json.loads((path / MANIFEST).read_text()))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None


def build_index(embedder: Embedder, path: Path | None = None,
                knowledge_dir: Path = KNOWLEDGE_DIR) -> IndexStats:
    path = path or index_dir()
    docs = load_documents(knowledge_dir)
    chunks = chunk_documents(docs, SemanticSplitter(embedder))
    if not chunks:
        raise RuntimeError(f"No content found in {knowledge_dir}")

    client = _client(path)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
    )
    vectors = embedder.embed([c.text for c in chunks])
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.meta for c in chunks],
        embeddings=vectors.tolist(),
    )
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.meta["doc_type"]] = by_type.get(c.meta["doc_type"], 0) + 1
    stats = IndexStats(
        fingerprint=knowledge_fingerprint(embedder.name, knowledge_dir), documents=len(docs),
        chunks=len(chunks), by_type=by_type, embedder=embedder.name, built_at=time.time(), rebuilt=True,
    )
    (path / MANIFEST).write_text(json.dumps(asdict(stats), indent=2))
    return stats


def ensure_index(embedder: Embedder | None = None, path: Path | None = None,
                 knowledge_dir: Path = KNOWLEDGE_DIR) -> tuple[chromadb.Collection, IndexStats, Embedder]:
    """Return the collection, rebuilding it first if knowledge/ changed."""
    embedder = embedder or get_embedder()
    path = path or index_dir()
    stats = read_manifest(path)
    current = knowledge_fingerprint(embedder.name, knowledge_dir)
    if stats is None or stats.fingerprint != current:
        stats = build_index(embedder, path, knowledge_dir)
    else:
        stats.rebuilt = False
    try:
        collection = _client(path).get_collection(COLLECTION_NAME)
    except Exception:  # manifest present but collection missing: rebuild
        stats = build_index(embedder, path, knowledge_dir)
        collection = _client(path).get_collection(COLLECTION_NAME)
    return collection, stats, embedder
