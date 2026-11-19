import io
import json
import urllib.error

import pytest

from aboutme import llm
from aboutme.assistant import answer
from aboutme.prompts import choose_mode, user_prompt
from aboutme.timeline import load_timeline


def test_provider_resolution(monkeypatch):
    assert llm.resolve_providers() == ["ollama"]
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert llm.resolve_providers() == ["gemini"]
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    assert llm.resolve_providers() == ["gemini", "openai"]
    monkeypatch.setenv("LLM_PROVIDER", "groq, gemini")
    assert llm.resolve_providers() == ["openai", "gemini"]
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert llm.resolve_providers() == ["ollama"]
    monkeypatch.setenv("LLM_PROVIDER", "chatgpt")
    with pytest.raises(llm.LLMError, match="Unknown LLM_PROVIDER"):
        llm.resolve_providers()


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



class Script:
    """Fake urlopen that plays back a list of (status, body) responses and records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request.full_url, dict(request.header_items()), json.loads(request.data)))
        status, body = self.responses.pop(0)
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "err", {}, io.BytesIO(json.dumps(body).encode()))
        return FakeResponse(json.dumps(body).encode())


GEMINI_OK = {"candidates": [{"content": {"parts": [{"text": "Gemini answer"}]}}]}
GROQ_OK = {"choices": [{"message": {"content": "Groq answer"}}]}
BUSY = {"error": {"message": "The model is overloaded. Please try again later."}}


def test_gemini_retries_a_503_then_succeeds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    script = Script((503, BUSY), (503, BUSY), (200, GEMINI_OK))
    monkeypatch.setattr(llm.urllib.request, "urlopen", script)
    assert llm.generate("s", "u").text == "Gemini answer"
    assert len(script.requests) == 3


def test_gemini_fallback_model_after_persistent_503(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "main-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "backup-model")
    script = Script((503, BUSY), (503, BUSY), (503, BUSY), (200, GEMINI_OK))
    monkeypatch.setattr(llm.urllib.request, "urlopen", script)
    reply = llm.generate("s", "u")
    assert reply.model == "backup-model"
    assert "backup-model:generateContent" in script.requests[-1][0]


def test_falls_back_to_groq_when_gemini_stays_busy(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    script = Script((503, BUSY), (503, BUSY), (503, BUSY), (200, GROQ_OK))
    monkeypatch.setattr(llm.urllib.request, "urlopen", script)
    reply = llm.generate("system text", "user text")
    assert (reply.text, reply.provider, reply.model) == ("Groq answer", "groq", "llama-3.1-8b-instant")
    url, headers, body = script.requests[-1]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer groq-key"
    assert body["messages"][0] == {"role": "system", "content": "system text"}


def test_bad_key_is_not_retried_but_still_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "wrong")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    script = Script((403, {"error": {"message": "denied"}}), (200, GROQ_OK))
    monkeypatch.setattr(llm.urllib.request, "urlopen", script)
    assert llm.generate("s", "u").text == "Groq answer"
    assert len(script.requests) == 2


def test_all_providers_down_reports_each(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setattr(llm.urllib.request, "urlopen", Script(*[(503, BUSY)] * 6))
    with pytest.raises(llm.LLMError, match=r"No model could answer.*Gemini is busy.*groq is busy"):
        llm.generate("s", "u")


def test_ollama_cloud_sends_api_key(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-oss:20b")
    script = Script((200, {"message": {"content": "Ollama answer"}}))
    monkeypatch.setattr(llm.urllib.request, "urlopen", script)
    assert llm.generate("s", "u").text == "Ollama answer"
    url, headers, _ = script.requests[0]
    assert url == "https://ollama.com/api/chat" and headers["Authorization"] == "Bearer ollama-key"


def test_describe_shows_the_chain(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert llm.describe() == "gemini · gemini-3.6-flash → groq · llama-3.1-8b-instant"
