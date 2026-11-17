"""Prompt construction for the Mind Map V2 (Where / How / Who / Why) assistant."""
from __future__ import annotations

import pandas as pd

from .retrieval import Hit

MINDMAP_HINTS = (
    "career", "evolution", "growth", "grow", "journey", "experience", "roles", "summary", "summar",
    "map", "progression", "transformation", "overview", "background", "path", "trajectory",
    "where how who why", "mind map",
)

SYSTEM_PROMPT = """You are the career assistant for {full_name} ("{name}"). You answer questions from recruiters, interviewers and {name} herself.

Rules:
1. Use ONLY the TIMELINE, HER RECORD and JOB POSTINGS sections of the message. Cite the block that supports each claim, like [E2] or [J1]; cite the timeline as [T].
2. If they do not contain the answer, say: "That isn't covered in the documents I was given." Then, if useful, say what is covered. Never fill gaps with general knowledge or guesses.
3. HER RECORD [E#] and the TIMELINE are what {name} has actually studied and done. JOB POSTINGS [J#] are vacancies she is considering: they are NOT past or current jobs. Never say she worked at a posting's company or did its duties, and never treat one of her own roles as a posting.
4. Use only numbers, dates and names that appear in the TIMELINE or CONTEXT. Never invent metrics, team sizes or salaries.
5. When you rely on a timeline row whose status is needs_confirmation, mention the uncertainty in a few words.
6. Speak about {name} in the third person, warmly and professionally. Never output placeholders such as [Company]."""

FORMAT_MINDMAP = """Format: a Mind Map. For each relevant role or degree, oldest first, write a heading "### Org: Title (period)" followed by four bullets:
- **Where:** place, organisation and setting
- **How:** methods, tools and what she built
- **Who:** the people and teams she worked with or for
- **Why:** the purpose and the measurable result
Leave out a bullet rather than invent it. If the question is about growth or career evolution, end with a short **Growth** paragraph comparing the earliest and the latest role using only cited facts."""

FORMAT_DIRECT = "Format: answer directly in 1–5 sentences or a short list. No Mind Map headings unless asked."

FORMAT_FIT = """Format: a role-fit assessment.
- The ONLY candidate positions are the ones under JOB POSTINGS [J#]. Her roles in HER RECORD and the TIMELINE (including her current role) are evidence, never candidates.
- Start with "**Best fit:** <Org>: <posting title> [J#]" and one sentence explaining why.
- Then, for each posting from best to weakest fit, a heading "### <Org>: <posting title> [J#]" with:
  - **Matches:** a requirement from the posting, then the evidence from her record that meets it, e.g. "SQL data modeling [J2] ↔ built SQL data models at Emerging Engineering [E1]"
  - **Gaps:** requirements with no evidence in her record (say "none found" if there are none)
  - **Fit:** Strong, Partial or Weak
- Do not use the Where/How/Who/Why format here."""


MODE_LABELS = {"mindmap": "Mind Map", "direct": "Direct", "fit": "Role fit"}
FORMATS = {"mindmap": FORMAT_MINDMAP, "direct": FORMAT_DIRECT, "fit": FORMAT_FIT}


def choose_mode(question: str, mode: str, is_fit: bool = False) -> str:
    """An explicit choice wins; in auto, fit questions beat Mind Map keywords such as "experience"."""
    if mode in FORMATS:
        return mode
    if is_fit:
        return "fit"
    q = question.lower()
    return "mindmap" if any(h in q for h in MINDMAP_HINTS) else "direct"


def system_prompt(name: str, full_name: str, mode: str) -> str:
    return SYSTEM_PROMPT.format(name=name, full_name=full_name) + "\n\n" + FORMATS[mode]


def timeline_block(timeline: pd.DataFrame) -> str:
    lines = ["| id | type | org | title | period | where | key result | status |", "|---|---|---|---|---|---|---|---|"]
    for _, r in timeline.iterrows():
        status = "needs_confirmation" if r["needs_confirmation"] else (r["status"] or "verified")
        lines.append(
            f"| {r['id']} | {r['type']} | {r['org']} | {r['title']} | {r['period']} | {r['where']} | "
            f"{r['key_metric'] or '-'} | {status} |"
        )
    return "\n".join(lines)


def is_posting(hit: Hit) -> bool:
    return hit.meta["doc_type"] == "target_role"


def order_and_label(hits: list[Hit]) -> list[tuple[str, Hit]]:
    """Her record first as E1, E2…, then job postings as J1, J2…. The UI uses the same labels."""
    record = [h for h in hits if not is_posting(h)]
    jobs = [h for h in hits if is_posting(h)]
    return [(f"E{i}", h) for i, h in enumerate(record, 1)] + [(f"J{i}", h) for i, h in enumerate(jobs, 1)]


def _blocks(labelled: list[tuple[str, Hit]]) -> str:
    return "\n\n".join(f"[{label}] (source: {hit.meta['doc_id']})\n{hit.text}" for label, hit in labelled)


def user_prompt(question: str, hits: list[Hit], timeline: pd.DataFrame) -> str:
    labelled = order_and_label(hits)
    record = [x for x in labelled if x[0].startswith("E")]
    jobs = [x for x in labelled if x[0].startswith("J")]
    parts = [
        f"TIMELINE [T] (her real roles and degrees):\n{timeline_block(timeline)}",
        "HER RECORD (what she has actually done):\n" + (_blocks(record) or "(no matching passages)"),
    ]
    if jobs:
        names = "\n".join(f"- [{label}] {hit.meta['title'].replace('Target role: ', '')}" for label, hit in jobs)
        parts.append(
            "JOB POSTINGS (vacancies she is considering; NOT her experience). The candidates are:\n"
            f"{names}\n\n{_blocks(jobs)}"
        )
    parts.append(f"QUESTION: {question}")
    return "\n\n".join(parts)
