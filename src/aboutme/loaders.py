"""Read knowledge/ into Documents. The folder a file sits in decides its type,
so a job posting can never be mistaken for a resume because of its filename."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import KNOWLEDGE_DIR
from .timeline import format_period, load_projects, load_timeline

FOLDER_TYPES = {
    "profile": "profile",
    "experience": "experience",
    "education": "education",
    "projects": "project",
    "targets": "target_role",
}
TYPE_LABELS = {
    "profile": "PROFILE",
    "experience": "EXPERIENCE",
    "education": "EDUCATION",
    "project": "SIDE PROJECT",
    "target_role": "TARGET ROLE (job posting, NOT past work)",
}
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class Document:
    doc_id: str  # path relative to knowledge/, e.g. "experience/undp.md"
    doc_type: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)


def split_front_matter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---\n"):
        end = raw.find("\n---", 4)
        if end != -1:
            meta = yaml.safe_load(raw[4:end]) or {}
            return meta, raw[end + 4:].lstrip("\n")
    return {}, raw


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    return "\n\n".join(re.sub(r"[ \t]+", " ", p) for p in pages)


def first_heading(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def load_documents(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[Document]:
    timeline = load_timeline(knowledge_dir / "timeline.csv").set_index("id")
    docs: list[Document] = []
    for folder, doc_type in FOLDER_TYPES.items():
        for path in sorted((knowledge_dir / folder).rglob("*")):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or path.name.startswith(("_", ".")):
                continue
            raw = read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
            meta, body = split_front_matter(raw)
            meta = {k: str(v).strip() for k, v in meta.items() if v is not None}
            row_id = meta.get("timeline_id")
            if row_id:
                if row_id not in timeline.index:
                    raise ValueError(f"{path}: timeline_id '{row_id}' is not in timeline.csv")
                row = timeline.loc[row_id]
                meta.update(org=row["org"], role=row["title"], period=row["period"])
            title = meta.pop("title", None) or first_heading(body) or path.stem.replace("-", " ").title()
            docs.append(Document(
                doc_id=path.relative_to(knowledge_dir).as_posix(),
                doc_type=doc_type, title=title, text=body.strip(), meta=meta,
            ))
    docs.extend(project_row_documents(knowledge_dir))
    return docs


def project_row_documents(knowledge_dir: Path) -> list[Document]:
    """One short document per projects.csv row, so every repo is searchable."""
    projects = load_projects(knowledge_dir / "projects.csv")
    docs = []
    for _, row in projects.iterrows():
        lines = [
            f"Side project: {row['name']}",
            f"Category: {row['category']}. Main language/tool: {row['language']}.",
            f"Stack: {row['stack'].replace(';', ', ')}.",
            row["summary"],
        ]
        if row["year"]:
            lines.insert(1, f"Year: {row['year']}.")
        if row["url"]:
            lines.append(f"Repository: {row['url']}")
        docs.append(Document(
            doc_id=f"projects.csv#{row['name']}", doc_type="project",
            title=f"Side project: {row['name']}", text="\n".join(lines),
            meta={"project": row["name"], "url": row["url"]},
        ))
    return docs


def target_orgs(docs: list[Document]) -> set[str]:
    return {d.meta["org"].lower() for d in docs if d.doc_type == "target_role" and d.meta.get("org")}
