"""Golden questions for retrieval. Run offline with the hash embedder, so they mostly
exercise the BM25 half, routing and quotas; the real MiniLM model only does better."""
import pytest

from aboutme.chunking import SemanticSplitter, chunk_documents
from aboutme.embeddings import HashEmbedder
from aboutme.loaders import load_documents


def top_docs(retriever, question, k=6, **kw):
    return [h.meta["doc_id"] for h in retriever.search(question, k=k, **kw)]


@pytest.mark.parametrize("question, expected", [
    ("What did she achieve at GIMAC-BEAC?", "experience/gimac-beac.md"),
    ("How did she use Apache Arrow to speed up reporting?", "experience/emerging-engineering.md"),
    ("What did she do for UNDP and the SIDS policy-makers?", "experience/undp.md"),
    ("What did she study at Harbour Space Institute?", "education/harbour-space.md"),
    ("Which languages does she speak? French? Spanish?", "profile/summary.md"),
])
def test_golden_questions(retriever, question, expected):
    assert expected in top_docs(retriever, question, k=4)


def test_job_postings_excluded_from_career_questions(retriever):
    docs = top_docs(retriever, "Summarise her data engineering experience and results", k=8)
    assert not any(d.startswith("targets/") for d in docs)


def test_fit_questions_mix_postings_with_real_experience(retriever):
    hits = retriever.search("Is she a good fit for the Open Earth Foundation platform data engineer job?", k=6)
    types = [h.meta["doc_type"] for h in hits]
    assert "target_role" in types
    assert sum(t in {"experience", "education", "profile"} for t in types) >= 3


def test_project_questions_return_projects(retriever):
    docs = top_docs(retriever, "Which side projects use dbt and Streamlit?", k=6)
    assert any("job-hunter-pipeline" in d for d in docs)


def test_chunks_carry_context_headers_and_stable_ids():
    docs = load_documents()
    first = chunk_documents(docs, SemanticSplitter(HashEmbedder()))
    second = chunk_documents(docs, SemanticSplitter(HashEmbedder()))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    gimac = [c for c in first if c.meta["doc_id"] == "experience/gimac-beac.md"]
    assert all("GIMAC-BEAC" in c.text.splitlines()[0] for c in gimac)
    target = [c for c in first if c.meta["doc_type"] == "target_role"]
    assert all("NOT past work" in c.text.splitlines()[0] for c in target)
