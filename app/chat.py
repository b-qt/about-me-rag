"""Chat page: grounded answers with visible sources."""
from __future__ import annotations

import streamlit as st

from app.common import retriever, subject, timeline
from aboutme import llm
from aboutme.assistant import answer
from aboutme.config import int_setting
from aboutme.loaders import TYPE_LABELS
from aboutme.prompts import MODE_LABELS, order_and_label

name = subject()
MAX_QUESTIONS = int_setting("MAX_QUESTIONS_PER_SESSION")
MAX_CHARS = int_setting("MAX_QUESTION_CHARS")
QUICK_PROMPTS = {
    "🗺️ Career map": f"Map {name}'s career evolution using Where, How, Who and Why.",
    "🚀 Growth & impact": f"What measurable results and transformations has {name} delivered across her roles?",
    "⏱️ Most recent role": f"What does {name} do in her most recent role, and how does it build on earlier ones?",
    "🌱 Where it started": f"What did {name} do in her earliest role, and what did it lead to?",
    "🧪 Side projects": f"Which side projects has {name} built, and what skills do they show?",
    "🎯 Role fit": f"Which of the saved job postings best fits {name}'s experience, and where are the gaps?",
}
MODES = {"Auto": "auto", "Mind Map": "mindmap", "Direct": "direct", "Role fit": "fit"}
TARGETS = {"Auto": None, "Include": True, "Exclude": False}

st.title(f"Ask about {name}'s career")
st.caption(
    "Answers come only from her resume, project notes and saved job postings. "
    "Every answer lists the passages it used."
)

try:
    rag, stats = retriever()
    career = timeline()
except Exception as err:  # surface indexing problems instead of failing silently
    st.error("The knowledge base could not be loaded.")
    st.exception(err)
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("asked", 0)

with st.sidebar:
    st.subheader("Answer style")
    mode = MODES[st.radio("Format", list(MODES), horizontal=True,
                          help="Auto picks Role fit for questions about job postings, the Where/How/Who/Why Mind Map for career questions, and a short answer otherwise.")]
    include_targets = TARGETS[st.radio("Job postings", list(TARGETS), horizontal=True,
                                       help="Saved job postings are only searched for fit questions unless you include them.")]
    top_k = st.slider("Passages per answer", 3, 10, 6)
    st.subheader("Try asking")
    for label, prompt in QUICK_PROMPTS.items():
        if st.button(label, width="stretch"):
            st.session_state["pending_prompt"] = prompt
    st.divider()
    st.caption(f"{stats.documents} documents · {stats.chunks} passages · {llm.describe()}")
    if st.button("Clear chat", width="stretch"):
        st.session_state["messages"] = []
        st.rerun()


def render_sources(hits) -> None:
    with st.expander(f"Sources ({len(hits)})"):
        for i, hit in order_and_label(hits):
            label = TYPE_LABELS.get(hit.meta["doc_type"], hit.meta["doc_type"]).split(" (")[0].title()
            section = hit.meta.get("section") or ""
            section = f" · {section}" if section and section != hit.meta["title"] else ""
            st.markdown(f"**[{i}] {hit.meta['title']}**{section}  \n:gray[{label} · `{hit.meta['doc_id']}`]")
            st.text(hit.text.split("\n", 1)[-1][:600])


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.warning(f"The language model is unavailable, so here are the matching passages instead.\n\n{message['error']}")
        if message.get("content"):
            st.markdown(message["content"])
        if message.get("model"):
            st.caption(f"{message['model']} · {MODE_LABELS[message['mode']]} answer")
        if message.get("hits"):
            render_sources(message["hits"])


if not st.session_state["messages"]:
    st.info(f"Ask anything about {name}'s studies, roles, results or side projects, or pick a question in the sidebar.")
for message in st.session_state["messages"]:
    render_message(message)

limit_reached = st.session_state["asked"] >= MAX_QUESTIONS
if limit_reached:
    st.info(f"You've reached the limit of {MAX_QUESTIONS} questions for this session. Refresh the page to start again.")
typed = st.chat_input(f"Ask about {name}'s experience…", max_chars=MAX_CHARS, disabled=limit_reached)
question = typed or st.session_state.pop("pending_prompt", None)

if question and not limit_reached:
    user_message = {"role": "user", "content": question}
    st.session_state["messages"].append(user_message)
    render_message(user_message)
    with st.spinner("Searching her documents…"):
        result = answer(question, rag, career, k=top_k, mode=mode, include_targets=include_targets)
    reply = {"role": "assistant", "content": result.text, "hits": result.hits, "mode": result.mode,
             "model": result.model, "error": result.error}
    st.session_state["messages"].append(reply)
    st.session_state["asked"] += 1
    render_message(reply)