"""LLM providers: Gemini (cloud/public deployment) and Ollama (local).

LLM_PROVIDER=auto picks Gemini when GEMINI_API_KEY is set, otherwise Ollama.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import setting

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    """A failure with a message that is safe to show to app visitors."""


@dataclass(frozen=True)
class Reply:
    text: str
    provider: str
    model: str


def resolve_provider() -> str:
    provider = (setting("LLM_PROVIDER") or "auto").lower()
    if provider == "auto":
        return "gemini" if setting("GEMINI_API_KEY") else "ollama"
    if provider not in {"gemini", "ollama"}:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}' (use auto, gemini or ollama).")
    return provider


def ollama_host() -> str:
    """OLLAMA_HOST, or the base of the older OLLAMA_ENDPOINT (…/api/generate) setting."""
    legacy = setting("OLLAMA_ENDPOINT", "")
    if legacy and not setting("OLLAMA_HOST", ""):
        return re.sub(r"/api/.*$", "", legacy.rstrip("/"))
    return (setting("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")


def describe() -> str:
    try:
        provider = resolve_provider()
    except LLMError as err:
        return str(err)
    model = setting("GEMINI_MODEL") if provider == "gemini" else setting("OLLAMA_MODEL")
    return f"{provider} · {model}"


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _error_detail(err: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(err.read().decode("utf-8"))
        return body.get("error", {}).get("message") or str(body)[:300]
    except Exception:
        return err.reason or ""


def gemini(system: str, user: str, temperature: float) -> Reply:
    key = setting("GEMINI_API_KEY")
    model = setting("GEMINI_MODEL")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set. Add it to the app's secrets.")
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
    }
    try:
        data = _post_json(GEMINI_URL.format(model=model), payload, {"x-goog-api-key": key}, timeout=60)
    except urllib.error.HTTPError as err:
        detail = _error_detail(err)
        if err.code == 404:
            raise LLMError(f"Gemini model '{model}' was not found. Set GEMINI_MODEL to a current model. ({detail})")
        if err.code == 429:
            raise LLMError("The assistant is getting too many questions right now. Please try again in a minute.")
        if err.code in (401, 403):
            raise LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY in the app's secrets.")
        raise LLMError(f"Gemini returned an error ({err.code}): {detail}")
    except (urllib.error.URLError, TimeoutError) as err:
        raise LLMError(f"Could not reach Gemini: {getattr(err, 'reason', err)}")

    candidates = data.get("candidates") or []
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "no answer returned")
        raise LLMError(f"Gemini did not return an answer ({reason}).")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise LLMError(f"Gemini returned an empty answer (finish reason: {candidates[0].get('finishReason')}).")
    return Reply(text, "gemini", model)


def ollama(system: str, user: str, temperature: float) -> Reply:
    host, model = ollama_host(), setting("OLLAMA_MODEL")
    payload = {
        "model": model, "stream": False, "options": {"temperature": temperature},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    try:
        data = _post_json(f"{host}/api/chat", payload, {}, timeout=180)
    except urllib.error.HTTPError as err:
        raise LLMError(f"Ollama returned an error ({err.code}): {_error_detail(err)}. Is '{model}' pulled?")
    except (urllib.error.URLError, TimeoutError) as err:
        raise LLMError(
            f"Could not reach Ollama at {host} ({getattr(err, 'reason', err)}). "
            "Run `ollama serve` locally, or set GEMINI_API_KEY for the hosted app."
        )
    text = (data.get("message") or {}).get("content", "").strip()
    if not text:
        raise LLMError("Ollama returned an empty answer.")
    return Reply(text, "ollama", model)


def generate(system: str, user: str, temperature: float = 0.2) -> Reply:
    return gemini(system, user, temperature) if resolve_provider() == "gemini" else ollama(system, user, temperature)
