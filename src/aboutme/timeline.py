"""Load the structured career data (timeline.csv, projects.csv) and derive growth metrics."""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .config import PROJECTS_CSV, TIMELINE_CSV

REQUIRED_COLUMNS = [
    "id", "type", "org", "title", "start", "end", "city", "country", "work_mode", "level",
    "stack", "where", "how", "who", "why", "key_metric", "status", "source",
]
DATE_PATTERN = re.compile(r"^\d{4}(-(0[1-9]|1[0-2]))?$")


def parse_start(value: str) -> date | None:
    if not value:
        return None
    year, _, month = value.partition("-")
    return date(int(year), int(month or 1), 1)


def parse_end(value: str) -> date | None:
    if not value:
        return None
    year, _, month = value.partition("-")
    m = int(month or 12)
    return date(int(year), m, calendar.monthrange(int(year), m)[1])


def format_period(start: str, end: str) -> str:
    if not start and end:
        return f"completed {end}"
    if start and start == end:
        return start
    return f"{start or '?'} – {end or 'present'}"


def split_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def load_timeline(path: Path = TIMELINE_CSV, today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"timeline.csv is missing columns: {sorted(missing)}")
    df = df.apply(lambda col: col.str.strip())
    df["csv_line"] = df.index + 2  # +1 for the header, +1 because spreadsheets count from 1
    df = df[(df[REQUIRED_COLUMNS] != "").any(axis=1)]  # ignore fully empty rows like ",,,,"
    validate_timeline(df, path)
    df["start_date"] = df["start"].map(parse_start)
    df["end_date"] = df["end"].map(parse_end)
    df["ongoing"] = (df["end"] == "") & (df["start"] != "")
    df.loc[df["ongoing"], "end_date"] = today
    df["period"] = [format_period(s, e) for s, e in zip(df["start"], df["end"])]
    df["level_num"] = pd.to_numeric(df["level"], errors="coerce")
    df["needs_confirmation"] = df["status"].str.startswith("needs_confirmation")
    df["sort_date"] = df["start_date"].fillna(df["end_date"])
    return df.sort_values("sort_date").reset_index(drop=True)


def validate_timeline(df: pd.DataFrame, path: Path) -> None:
    """Fail early with a message that says which line of the CSV to fix."""
    problems: list[str] = []
    for _, row in df[df["id"] == ""].iterrows():
        problems.append(f"line {row['csv_line']}: the id is empty")
    dupes = df[(df["id"] != "") & df["id"].duplicated(keep=False)]
    for row_id, group in dupes.groupby("id"):
        lines = ", ".join(str(n) for n in group["csv_line"])
        problems.append(f"lines {lines}: id '{row_id}' is used more than once; every row needs its own id")
    for _, row in df.iterrows():
        if row["type"] not in {"work", "education"}:
            problems.append(f"line {row['csv_line']}: type must be 'work' or 'education', not '{row['type']}'")
        for col in ("start", "end"):
            if row[col] and not DATE_PATTERN.match(row[col]):
                problems.append(f"line {row['csv_line']}: {col} '{row[col]}' must look like 2023 or 2023-04")
        if row["type"] == "work" and not row["start"]:
            problems.append(f"line {row['csv_line']}: work rows need a start date")
    if problems:
        raise ValueError(f"Please fix {path.name}:\n- " + "\n- ".join(problems))


def load_projects(path: Path = PROJECTS_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@dataclass(frozen=True)
class Kpis:
    years_since_first_role: float
    organisations: int
    countries: int
    roles: int
    degrees: int
    tools: int
    quantified_results: int
    side_projects: int
    current_title: str


def compute_kpis(timeline: pd.DataFrame, projects: pd.DataFrame, today: date | None = None) -> Kpis:
    today = today or date.today()
    work = timeline[timeline["type"] == "work"]
    first_start = work["start_date"].dropna().min()
    years = (today - first_start).days / 365.25 if first_start else 0.0
    ongoing = work[work["ongoing"]].sort_values("start_date")
    current = ongoing.iloc[-1] if len(ongoing) else work.iloc[-1]
    tools = {t.lower() for stack in timeline["stack"] for t in split_list(stack)}
    return Kpis(
        years_since_first_role=round(years, 1),
        organisations=work["org"].nunique(),
        countries=timeline["country"].replace("", pd.NA).dropna().nunique(),
        roles=len(work),
        degrees=int((timeline["type"] == "education").sum()),
        tools=len(tools),
        quantified_results=int(work["key_metric"].str.contains(r"\d", regex=True).sum()),
        side_projects=len(projects),
        current_title=f"{current['title']} · {current['org']}",
    )


def toolset_growth(timeline: pd.DataFrame) -> pd.DataFrame:
    """Cumulative count of distinct tools, by the year each role or degree began (or ended, for degrees)."""
    seen: set[str] = set()
    rows = []
    for _, row in timeline.sort_values("sort_date").iterrows():
        new = [t for t in split_list(row["stack"]) if t.lower() not in seen]
        seen.update(t.lower() for t in new)
        rows.append({
            "date": row["sort_date"], "label": f"{row['org']}", "new_tools": ", ".join(new),
            "new_count": len(new), "cumulative": len(seen),
        })
    return pd.DataFrame(rows)


def seniority_steps(timeline: pd.DataFrame) -> pd.DataFrame:
    work = timeline[(timeline["type"] == "work") & timeline["level_num"].notna()].copy()
    return work.sort_values("start_date")[["start_date", "end_date", "org", "title", "level_num", "ongoing"]]
