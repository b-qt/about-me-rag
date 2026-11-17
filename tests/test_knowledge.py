"""Checks on knowledge/: valid timeline, consistent links, and nothing private."""
import re
from pathlib import Path

from aboutme.config import KNOWLEDGE_DIR
from aboutme.loaders import load_documents
from aboutme.timeline import DATE_PATTERN, REQUIRED_COLUMNS, compute_kpis, load_projects, load_timeline

PRIVATE_PATTERNS = {
    "phone number": re.compile(r"\(\+\d{1,3}\)|\+\d{2}[\s\d]{8,}"),
    "email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "salary or stipend": re.compile(r"\b(salary|stipend|gross annual|SGD|EUR)\b|€\s?\d", re.I),
    "signature block": re.compile(r"_{6,}"),
}


def test_timeline_columns_dates_and_ids():
    df = load_timeline()
    assert list(df.columns[: len(REQUIRED_COLUMNS)]) == REQUIRED_COLUMNS
    assert df["id"].is_unique
    assert set(df["type"]) <= {"work", "education"}
    for col in ("start", "end"):
        bad = [v for v in df[col] if v and not DATE_PATTERN.match(v)]
        assert not bad, f"{col} must be YYYY or YYYY-MM: {bad}"
    assert (df.loc[df["type"] == "work", "start"] != "").all(), "every work row needs a start date"
    assert df.loc[df["type"] == "work", "level_num"].between(1, 4).all()


def test_every_experience_file_links_to_a_timeline_row():
    ids = set(load_timeline()["id"])
    docs = [d for d in load_documents() if d.doc_type in {"experience", "education"}]
    assert docs
    for doc in docs:
        assert doc.meta.get("timeline_id") in ids, doc.doc_id


def test_kpis():
    kpis = compute_kpis(load_timeline(), load_projects())
    assert kpis.organisations >= 4 and kpis.countries == 2 and kpis.degrees == 2
    assert kpis.quantified_results >= 1  # GIMAC-BEAC +25%


def test_no_private_details_in_knowledge():
    problems = []
    for path in Path(KNOWLEDGE_DIR).rglob("*"):
        if path.suffix.lower() not in {".md", ".txt", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{path.relative_to(KNOWLEDGE_DIR)}: {label}")
    assert not problems, "Private details found (the public app could repeat them):\n" + "\n".join(problems)


def test_targets_are_labelled_not_past_work():
    for doc in load_documents():
        if doc.doc_type == "target_role":
            assert "NOT past work" in doc.text, doc.doc_id


def _timeline_copy(tmp_path, extra_lines: str) -> Path:
    from aboutme.config import TIMELINE_CSV

    path = tmp_path / "timeline.csv"
    path.write_text(TIMELINE_CSV.read_text(encoding="utf-8").rstrip("\n") + "\n" + extra_lines, encoding="utf-8")
    return path


def test_duplicate_id_gives_a_clear_error(tmp_path):
    import pytest

    from aboutme.config import TIMELINE_CSV

    copied_row = TIMELINE_CSV.read_text(encoding="utf-8").splitlines()[3]  # a row copied to start a new one
    with pytest.raises(ValueError, match=r"id 'edu-harbour-space' is used more than once"):
        load_timeline(_timeline_copy(tmp_path, copied_row + "\n"))


def test_empty_rows_and_whitespace_are_ignored(tmp_path):
    df = load_timeline(_timeline_copy(tmp_path, ",,,,,,,,,,,,,,,,,\n"))
    assert len(df) == len(load_timeline())


def test_bad_month_gives_a_clear_error(tmp_path):
    import pytest

    bad = "x-test,work,Org,Role,2024-13,,City,Spain,remote,2,SQL,,,,,,verified,test\n"
    with pytest.raises(ValueError, match=r"start '2024-13' must look like"):
        load_timeline(_timeline_copy(tmp_path, bad))
