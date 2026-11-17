"""One question in, one grounded answer out. Used by the Streamlit app and scripts/ask.py."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import llm
from .config import setting
from .prompts import choose_mode, order_and_label, system_prompt, user_prompt
from .retrieval import Hit, Retriever


@dataclass
class Answer:
    text: str | None
    hits: list[Hit]
    mode: str
    model: str | None = None
    error: str | None = None


def answer(question: str, retriever: Retriever, timeline: pd.DataFrame, *, k: int = 6,
           mode: str = "auto", include_targets: bool | None = None) -> Answer:
    is_fit = include_targets is not False and retriever.is_fit_question(question)
    mode = choose_mode(question, mode, is_fit=is_fit)
    if mode == "fit":
        evidence, postings = retriever.search_fit(question, k=k)
        hits = evidence + postings
    else:
        hits = retriever.search(question, k=k, include_targets=include_targets)
    hits = [h for _, h in order_and_label(hits)]
    system = system_prompt(setting("SUBJECT_NAME"), setting("SUBJECT_FULL_NAME"), mode)
    try:
        reply = llm.generate(system, user_prompt(question, hits, timeline))
    except llm.LLMError as err:
        return Answer(text=None, hits=hits, mode=mode, error=str(err))
    return Answer(text=reply.text, hits=hits, mode=mode, model=f"{reply.provider} · {reply.model}")

