import io
import json
import urllib.error

import pytest

from aboutme import llm
from aboutme.assistant import answer
from aboutme.prompts import choose_mode, user_prompt
from aboutme.timeline import load_timeline


def test_provider_resolution(monkeypatch):
    assert llm.resolve_provider() == "ollama"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert llm.resolve_provider() == "gemini"
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert llm.resolve_provider() == "ollama"


def test_legacy_ollama_endpoint(monkeypatch):
    monkeypatch.setenv("OLLAMA_ENDPOINT", "http://host.docker.internal:11434/api/generate")
    assert llm.ollama_host() == "http://host.docker.internal:11434"


def test_mode_selection():
    assert choose_mode("Map her career evolution", "auto") == "mindmap"
    assert choose_mode("Which languages does she speak?", "auto") == "direct"
    assert choose_mode("Which languages does she speak?", "mindmap") == "mindmap"


def test_prompt_contains_timeline_and_labels(retriever):
    hits = retriever.search("Is she a fit for the Jampp role?", k=6)
    text = user_prompt("Is she a fit for the Jampp role?", hits, load_timeline())
    assert "TIMELINE [T]" in text and "GIMAC-BEAC" in text
    assert "TARGET ROLE (job posting, NOT past work)" in text


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_gemini_request_and_parsing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["headers"] = request.full_url, dict(request.header_items())
        seen["body"] = json.loads(request.data)
        return FakeResponse(json.dumps({"candidates": [{"content": {"parts": [
            {"text": "thinking…", "thought": True}, {"text": "She studied in Barcelona [1]."}]}}]}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    reply = llm.generate("system", "user")
    assert reply.text == "She studied in Barcelona [1]."
    assert "gemini-test:generateContent" in seen["url"] and "secret" not in seen["url"]
    assert seen["headers"]["X-goog-api-key"] == "secret"
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "system"


def test_gemini_missing_model_gives_clear_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {},
                                     io.BytesIO(b'{"error": {"message": "model not found"}}'))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(llm.LLMError, match="GEMINI_MODEL"):
        llm.generate("s", "u")


def test_answer_falls_back_to_sources_when_llm_is_down(retriever, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    result = answer("What did she achieve at GIMAC-BEAC?", retriever, load_timeline(), k=4)
    assert result.text is None and "Ollama" in result.error
    assert result.hits


ROLE_FIT = "Which of the saved job postings best fits Brenda's experience, and where are the gaps?"


def test_fit_questions_use_fit_mode_even_with_mindmap_words(retriever):
    assert retriever.is_fit_question(ROLE_FIT)
    assert choose_mode(ROLE_FIT, "auto", is_fit=True) == "fit"
    assert choose_mode(ROLE_FIT, "mindmap", is_fit=True) == "mindmap"  # an explicit choice still wins


def test_fit_retrieval_covers_every_posting_and_keeps_evidence_separate(retriever):
    evidence, jobs = retriever.search_fit(ROLE_FIT, k=6)
    assert evidence and all(h.meta["doc_type"] != "target_role" for h in evidence)
    assert {h.meta["doc_id"] for h in jobs} == set(retriever.postings())
    assert all(h.meta["doc_type"] == "target_role" for h in jobs)
    with_reqs = [h for h in jobs if h.meta["doc_id"] != "targets/jampp-product-bi-analyst.md"]
    assert all(("Requirement" in h.meta["section"]) or ("Qualification" in h.meta["section"]) for h in with_reqs)


def test_naming_a_company_returns_only_that_posting(retriever):
    _, jobs = retriever.search_fit("Is she a good fit for the Jampp role?", k=6)
    assert jobs and {h.meta["doc_id"] for h in jobs} == {"targets/jampp-product-bi-analyst.md"}


def test_fit_prompt_separates_her_record_from_postings(retriever):
    evidence, jobs = retriever.search_fit(ROLE_FIT, k=6)
    text = user_prompt(ROLE_FIT, evidence + jobs, load_timeline())
    record, postings = text.split("JOB POSTINGS", 1)
    assert "[E1]" in record and "[J1]" not in record
    assert "experience/" not in postings and "profile/" not in postings
    assert postings.count("source: targets/") == len(jobs)
    candidates = postings.split("\n\n", 1)[0]
    assert "Sagaci" not in candidates and "Emerging Engineering" not in candidates


def test_fit_evidence_covers_every_role_and_degree(retriever):
    from aboutme.timeline import load_timeline as lt

    evidence, _ = retriever.search_fit(ROLE_FIT, k=6)
    ids = [h.meta["doc_id"] for h in evidence]
    linked = {h.meta.get("timeline_id") for h in evidence}
    assert set(lt()["id"]) <= linked, "every timeline row should contribute evidence"
    assert len(ids) == len(set(ids)) or all(i.startswith("profile/") for i in ids if ids.count(i) > 1)
    assert any(h.meta.get("section") == "Technical skills" for h in evidence)
