"""Split documents into retrievable chunks.

1. Split on markdown headings, so a chunk never mixes two sections.
2. Sections longer than `max_chars` are split semantically: consecutive sentences/bullets
   are embedded and a new chunk starts where the cosine distance to the next unit is in
   the top (100 - percentile)% for that section (the original SemanticSplitter idea).
3. Fragments shorter than `min_chars` are merged into their neighbour.
4. Every chunk gets a context header (type, title, org, period, section) so that a
   bullet like "increased efficiency by 25%" is still tied to GIMAC-BEAC once retrieved.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .embeddings import Embedder
from .loaders import TYPE_LABELS, Document

CHUNKER_VERSION = "3"
HEADING = re.compile(r"^(#{1,6})\s+(.+)$", flags=re.MULTILINE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str  # header + body: what gets embedded and shown to the model
    body: str
    meta: dict


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs; text before the first heading gets heading ''."""
    sections: list[tuple[str, str]] = []
    matches = list(HEADING.finditer(text))
    if not matches:
        return [("", text.strip())]
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()].strip()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(2).strip(), text[match.end():end].strip()))
    return [(h, b) for h, b in sections if b]


def split_units(text: str) -> list[str]:
    """Bullets and lines first, then sentences within long lines."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        units.extend(s.strip() for s in SENTENCE_END.split(line) if s.strip())
    return units


class SemanticSplitter:
    def __init__(self, embedder: Embedder, percentile: float = 80.0,
                 max_chars: int = 900, min_chars: int = 180) -> None:
        self.embedder = embedder
        self.percentile = percentile
        self.max_chars = max_chars
        self.min_chars = min_chars

    def split(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        units = split_units(text)
        if len(units) < 2:
            return [text]
        vectors = self.embedder.embed(units)
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = np.percentile(distances, self.percentile)

        groups: list[list[str]] = [[units[0]]]
        for unit, distance in zip(units[1:], distances):
            current_len = sum(len(u) + 1 for u in groups[-1])
            if distance > threshold or current_len + len(unit) > self.max_chars:
                groups.append([unit])
            else:
                groups[-1].append(unit)
        return self._merge_small(["\n".join(g) for g in groups])

    def _merge_small(self, pieces: list[str]) -> list[str]:
        merged: list[str] = []
        for piece in pieces:
            if merged and (len(piece) < self.min_chars or len(merged[-1]) < self.min_chars) \
                    and len(merged[-1]) + len(piece) <= self.max_chars * 1.3:
                merged[-1] = f"{merged[-1]}\n{piece}"
            else:
                merged.append(piece)
        return merged


def context_header(doc: Document, section: str) -> str:
    parts = [TYPE_LABELS[doc.doc_type], str(doc.title)]
    if doc.meta.get("org") and doc.meta["org"] not in doc.title:
        parts.append(doc.meta["org"])
    if doc.meta.get("period"):
        parts.append(doc.meta["period"])
    header = " | ".join(parts)
    if section and section != doc.title:
        header += f" | {section}"
    return f"[{header}]"


def chunk_documents(docs: list[Document], splitter: SemanticSplitter) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        position = 0
        for section, body in split_sections(doc.text):
            for piece in splitter.split(body):
                header = context_header(doc, section)
                digest = hashlib.sha1(f"{doc.doc_id}|{position}|{piece}".encode()).hexdigest()[:16]
                meta = {
                    "doc_id": doc.doc_id, "doc_type": doc.doc_type, "title": doc.title,
                    "section": section, "position": position,
                    **{k: v for k, v in doc.meta.items() if k in {"org", "role", "period", "timeline_id", "url", "project"}},
                }
                chunks.append(Chunk(chunk_id=digest, text=f"{header}\n{piece}", body=piece, meta=meta))
                position += 1
    return chunks
