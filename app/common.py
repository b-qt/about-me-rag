"""Shared, cached resources for the Streamlit pages."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from aboutme.config import PROJECTS_CSV, TIMELINE_CSV, setting  # noqa: E402,F401
from aboutme.embeddings import Embedder, get_embedder  # noqa: E402
from aboutme.index import IndexStats, ensure_index, knowledge_fingerprint  # noqa: E402
from aboutme.retrieval import Retriever  # noqa: E402
from aboutme.timeline import load_projects, load_timeline  # noqa: E402

TYPE_COLORS = {"work": "#2F6FDE", "education": "#14917A"}


@st.cache_resource(show_spinner="Loading the embedding model…")
def embedder() -> Embedder:
    return get_embedder()


@st.cache_resource(show_spinner="Indexing the knowledge base…", max_entries=1)
def _retriever_for(fingerprint: str) -> tuple[Retriever, IndexStats]:
    collection, stats, emb = ensure_index(embedder())
    return Retriever(collection, emb), stats


def retriever() -> tuple[Retriever, IndexStats]:
    """Cached per knowledge/ fingerprint, so editing a file triggers a rebuild on the next run."""
    return _retriever_for(knowledge_fingerprint(embedder().name))


@st.cache_data
def _timeline(mtime: float) -> pd.DataFrame:
    return load_timeline(TIMELINE_CSV)


@st.cache_data
def _projects(mtime: float) -> pd.DataFrame:
    return load_projects(PROJECTS_CSV)


def timeline() -> pd.DataFrame:
    return _timeline(TIMELINE_CSV.stat().st_mtime)


def projects() -> pd.DataFrame:
    return _projects(PROJECTS_CSV.stat().st_mtime)


def subject() -> str:
    return setting("SUBJECT_NAME") or "Brenda"


def ask_in_chat(question: str) -> None:
    st.session_state["pending_prompt"] = question
    st.switch_page("app/chat.py")
