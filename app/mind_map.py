"""Mind Map V2: every role and degree as Where / How / Who / Why, from knowledge/timeline.csv."""
from __future__ import annotations

import textwrap

import streamlit as st

from app.common import TYPE_COLORS, ask_in_chat, subject, timeline

name = subject()
career = timeline()
FACETS = [("where", "🗺️ Where"), ("how", "🛠️ How"), ("who", "👥 Who"), ("why", "🎯 Why")]


def esc(text: str, width: int = 34) -> str:
    wrapped = "\n".join(textwrap.wrap(text, width)) if text else "—"
    return wrapped.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def overview_dot() -> str:
    lines = [
        "digraph G {", 'rankdir=TB; bgcolor="transparent"; nodesep=0.2; ranksep=0.5;',
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#94a3b8"];',
        'edge [color="#94a3b8" arrowsize=0.6];',
        f'root [label="{esc(name)}" shape=ellipse fillcolor="#1e293b" fontcolor="white" fontsize=14];',
    ]
    for _, r in career.iterrows():
        flag = " *" if r["needs_confirmation"] else ""
        lines.append(
            f'"{r["id"]}" [label="{esc(r["org"], 18)}\\n{esc(r["title"], 18)}\\n{r["period"]}{flag}" '
            f'fillcolor="{TYPE_COLORS[r["type"]]}" fontcolor="white"];'
        )
        lines.append(f'root -> "{r["id"]}";')
    lines.append("}")
    return "\n".join(lines)


def detail_dot(row) -> str:
    lines = [
        "digraph G {", 'rankdir=LR; bgcolor="transparent"; nodesep=0.3;',
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#94a3b8" fillcolor="#f1f5f9" fontcolor="#0f172a"];',
        'edge [color="#94a3b8" arrowsize=0.6];',
        f'role [label="{esc(row["org"], 26)}\\n{esc(row["title"], 26)}\\n{row["period"]}" '
        f'fillcolor="{TYPE_COLORS[row["type"]]}" fontcolor="white" fontsize=13];',
    ]
    for key, title in FACETS:
        clean_title = title.split(" ", 1)[1]
        lines.append(f'{key} [label="{clean_title.upper()}\\n{esc(row[key])}"];')
        lines.append(f"role -> {key};")
    if row["key_metric"]:
        lines.append(f'result [label="RESULT\\n{esc(row["key_metric"])}" fillcolor="#fef3c7"];')
        lines.append("why -> result;")
    lines.append("}")
    return "\n".join(lines)


st.title(f"{name}'s Mind Map")
st.caption("Where, How, Who and Why for every role and degree, oldest first. Pick one to zoom in.")

options = ["Overview"] + [f"{r['org']}: {r['title']} ({r['period']})" for _, r in career.iterrows()]
choice = st.selectbox("Focus", options, label_visibility="collapsed")

if choice == "Overview":
    st.graphviz_chart(overview_dot(), width="stretch")
    st.caption("Blue: work · Green: education · * needs confirmation")
else:
    row = career.iloc[options.index(choice) - 1]
    st.graphviz_chart(detail_dot(row), width="stretch")
    cols = st.columns(4)
    for col, (key, title) in zip(cols, FACETS):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.write(row[key] or ":gray[Not recorded yet]")
    if row["key_metric"]:
        st.success(f"**Result:** {row['key_metric']}")
    if row["needs_confirmation"]:
        st.warning(f"Needs confirmation: {row['status'].split(':', 1)[-1].strip()}")
    if st.button("Ask the assistant about this role", type="primary"):
        ask_in_chat(
            f"Using Where, How, Who and Why, describe {name}'s time at {row['org']} as {row['title']} "
            "and what it added to her career."
        )
