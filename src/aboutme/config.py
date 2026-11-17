"""Paths and settings.

Settings are read from environment variables first, then from Streamlit secrets
(.streamlit/secrets.toml locally, the Secrets box on Streamlit Community Cloud).
Paths are resolved from this file, so scripts work from any working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = ROOT / "knowledge"
TIMELINE_CSV = KNOWLEDGE_DIR / "timeline.csv"
PROJECTS_CSV = KNOWLEDGE_DIR / "projects.csv"
COLLECTION_NAME = "about_me"

DEFAULTS: dict[str, str] = {
    "SUBJECT_NAME": "Brenda",
    "SUBJECT_FULL_NAME": "Quinsy Brenda Tebid",
    "LLM_PROVIDER": "auto",  # auto | gemini | ollama
    "GEMINI_MODEL": "gemini-3.6-flash",
    "OLLAMA_HOST": "http://localhost:11434",
    "OLLAMA_MODEL": "llama3.2:1b",
    "EMBEDDING_BACKEND": "onnx",  # onnx (all-MiniLM-L6-v2) | hash (offline tests only)
    "MAX_QUESTIONS_PER_SESSION": "30",
    "MAX_QUESTION_CHARS": "500",
}


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env support for local runs (real environment variables win)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def setting(name: str, default: str | None = None) -> str | None:
    """Return a setting from the environment, then Streamlit secrets, then DEFAULTS."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file, or Streamlit not running
        pass
    return default if default is not None else DEFAULTS.get(name)


def int_setting(name: str) -> int:
    return int(setting(name) or DEFAULTS[name])


def index_dir() -> Path:
    return Path(setting("INDEX_DIR") or ROOT / ".index")
