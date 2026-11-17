"""Sentence embeddings.

`onnx` (default) runs all-MiniLM-L6-v2 through ONNX Runtime via ChromaDB's built-in
embedding function: the same model the project used with sentence-transformers, without
installing PyTorch. The model (~80 MB) downloads once on first use.

`hash` is a deterministic bag-of-words embedder with no download. It exists only so the
test suite and CI can run offline; never use it for the public app.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import setting

TOKEN = re.compile(r"[a-z0-9áéíóúñç+#]+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OnnxMiniLM:
    name = "onnx-all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self._fn(texts), dtype=np.float32)
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


class HashEmbedder:
    name = "hash-bow-512"
    dims = 512

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = TOKEN.findall(text.lower())
            for gram in tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]:
                bucket = int(hashlib.md5(gram.encode()).hexdigest(), 16) % self.dims
                out[row, bucket] += 1.0
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


def get_embedder() -> Embedder:
    backend = (setting("EMBEDDING_BACKEND") or "onnx").lower()
    if backend == "hash":
        return HashEmbedder()
    if backend == "onnx":
        return OnnxMiniLM()
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}' (use 'onnx' or 'hash')")
