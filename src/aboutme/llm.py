"""LLM providers with retries and automatic fallback.

Providers
- gemini: Google Gemini (GEMINI_API_KEY, GEMINI_MODEL, optional GEMINI_FALLBACK_MODELS).
- openai: any OpenAI-compatible API, e.g. Groq (free tier, Llama models), OpenRouter, Together.
          OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_API_KEY (or GROQ_API_KEY), OPENAI_COMPAT_MODEL.
- ollama: Ollama on your machine, a remote Ollama behind a tunnel, or Ollama Cloud
          (OLLAMA_HOST=https://ollama.com plus OLLAMA_API_KEY).

LLM_PROVIDER is "auto" (every provider that has a key, Gemini first, then the OpenAI-compatible
one, Ollama only when nothing else is configured or OLLAMA_API_KEY is set) or an explicit
ordered list such as "gemini,openai". Each provider retries busy/overloaded errors (429, 5xx,
timeouts) with a short backoff; if it still fails, the next provider answers.
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

from .config import setting

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_OPENAI_BASE = "https://api.groq.com/openai/v1"
DEFAULT_OPENAI_MODEL = "llama-3.1-8b-instant"
PROVIDER_ALIASES = {"groq": "openai", "openrouter": "openai", "openai_compat": "openai", "openai-compatible": "openai"}
KNOWN = ("gemini", "openai", "ollama")
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
ATTEMPTS = 3          # per model
BACKOFF_SECONDS = 1.5  # 1.5 s, then 3 s (plus jitter)
_sleep = time.sleep    # replaced in tests


class LLMError(RuntimeError):
    """A failure with a message that is safe to show to app visitors."""


class TransientLLMError(LLMError):
    """Busy, rate-limited or unreachable: worth retrying or falling back."""


@dataclass(frozen=True)
class Reply:
    text: str
    provider: str
    model: str


# --- configuration ------------------------------------------------------------

def openai_key() -> str | None:
    return setting("OPENAI_COMPAT_API_KEY") or setting("GROQ_API_KEY")


def openai_base() -> str:
    return (setting("OPENAI_COMPAT_BASE_URL") or DEFAULT_OPENAI_BASE).rstrip("/")

def openai_models() -> list[str]:
    raw = setting("OPENAI_COMPAT_MODEL") or DEFAULT_OPENAI_MODEL
    models = [m.strip() for m in raw.split(",") if m.strip()]
    extra = [m.strip() for m in (setting("OPENAI_COMPAT_FALLBACK_MODELS") or "").split(",") if m.strip()]
    combined = models + [m for m in extra if m not in models]
    return combined or [DEFAULT_OPENAI_MODEL]


def openai_model() -> str:
    return openai_models()[0] or DEFAULT_OPENAI_MODEL


def openai_label() -> str:
    host = urlparse(openai_base()).hostname or ""
    for name in ("groq", "openrouter", "together", "huggingface", "mistral", "openai"):
        if name in host:
            return name
    return "openai-compatible"


def gemini_models() -> list[str]:
    extra = [m.strip() for m in (setting("GEMINI_FALLBACK_MODELS") or "").split(",") if m.strip()]
    return [setting("GEMINI_MODEL")] + [m for m in extra if m != setting("GEMINI_MODEL")]


def ollama_host() -> str:
    """OLLAMA_HOST, or the base of the older OLLAMA_ENDPOINT (…/api/generate) setting."""
    legacy = setting("OLLAMA_ENDPOINT", "")
    if legacy and not setting("OLLAMA_HOST", ""):
        return re.sub(r"/api/.*$", "", legacy.rstrip("/"))
    return (setting("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")


def resolve_providers() -> list[str]:
    raw = (setting("LLM_PROVIDER") or "auto").lower().replace(" ", "")
    if raw != "auto":
        chain = [PROVIDER_ALIASES.get(p, p) for p in raw.split(",") if p]
        unknown = [p for p in chain if p not in KNOWN]
        if unknown or not chain:
            raise LLMError(f"Unknown LLM_PROVIDER '{raw}'. Use auto, or a list of: gemini, openai (Groq etc.), ollama.")
        return list(dict.fromkeys(chain))
    chain = []
    if setting("GEMINI_API_KEY"):
        chain.append("gemini")
    if openai_key():
        chain.append("openai")
    if not chain or setting("OLLAMA_API_KEY"):
        chain.append("ollama")
    return chain


def describe() -> str:
    try:
        chain = resolve_providers()
    except LLMError as err:
        return str(err)
    names = {
        "gemini": lambda: f"gemini · {gemini_models()[0]}",
        "openai": lambda: f"{openai_label()} · {openai_model()}",
        "ollama": lambda: f"ollama · {setting('OLLAMA_MODEL')}",
    }
    return " → ".join(names[p]() for p in chain)


# --- HTTP ---------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "about-me-rag/1.0",
            **headers,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _error_detail(err: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(err.read().decode("utf-8"))
        error = body.get("error", body)
        return (error.get("message") if isinstance(error, dict) else str(error)) or str(body)[:300]
    except Exception:
        return err.reason or ""


def _post_with_retry(name: str, url: str, payload: dict, headers: dict, timeout: int,
                     model: str | None = None) -> dict:
    """POST, retrying busy/overloaded responses. Raises TransientLLMError or LLMError."""
    model = model or payload.get("model")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return _post_json(url, payload, headers, timeout)
        except urllib.error.HTTPError as err:
            detail = _error_detail(err)
            if err.code not in TRANSIENT_STATUS:
                raise _permanent_error(name, err.code, detail, model)
            failure = TransientLLMError(f"{name} is busy ({err.code}: {detail[:160]})")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            failure = TransientLLMError(f"Could not reach {name} ({getattr(err, 'reason', err)})")
        if attempt < ATTEMPTS:
            _sleep(BACKOFF_SECONDS * attempt + random.uniform(0, 0.5))
    raise failure


def _permanent_error(name: str, code: int, detail: str, model: str | None) -> LLMError:
    if code == 404:
        setting_name = {"Gemini": "GEMINI_MODEL", "Ollama": "OLLAMA_MODEL"}.get(name, "OPENAI_COMPAT_MODEL")
        return LLMError(f"{name}: model '{model}' was not found. Set {setting_name} to a current model. ({detail[:160]})")
    if code == 401:
        return LLMError(f"{name}: the API key was rejected (401). Check it in the app's secrets.")
    if code == 403:
        return LLMError(f"{name}: request forbidden (403). ({detail[:160]})")
    return LLMError(f"{name} returned an error ({code}): {detail[:200]}")


# --- providers -------------------------------------------------------------

def gemini(system: str, user: str, temperature: float) -> Reply:
    key = setting("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set.")
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
    }
    last: LLMError | None = None
    for model in gemini_models():
        try:
            data = _post_with_retry("Gemini", GEMINI_URL.format(model=model), body,
                                    {"x-goog-api-key": key}, timeout=60, model=model)
        except LLMError as err:  # busy after retries, or model retired: try the next Gemini model
            last = err
            continue
        candidates = data.get("candidates") or []
        if not candidates:
            last = LLMError(f"Gemini did not return an answer ({data.get('promptFeedback', {}).get('blockReason', 'empty')}).")
            continue
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if text:
            return Reply(text, "gemini", model)
        last = LLMError(f"Gemini returned an empty answer (finish reason: {candidates[0].get('finishReason')}).")
    raise last or LLMError("No Gemini model configured.")


def openai_compatible__OLD(system: str, user: str, temperature: float) -> Reply:
    key, model, label = openai_key(), openai_model(), openai_label()
    if not key:
        raise LLMError("OPENAI_COMPAT_API_KEY (or GROQ_API_KEY) is not set.")
    data = _post_with_retry(
        label, f"{openai_base()}/chat/completions",
        {"model": model, "temperature": temperature, "max_tokens": 2048,
         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        {"Authorization": f"Bearer {key}"}, timeout=60,
    )
    choices = data.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    if not text:
        raise LLMError(f"{label} returned an empty answer.")
    return Reply(text, label, model)

def openai_compatible(system: str, user: str, temperature: float) -> Reply:
    key, label = openai_key(), openai_label()
    if not key:
        raise LLMError("OPENAI_COMPAT_API_KEY (or GROQ_API_KEY) is not set.")
    last: LLMError | None = None
    for model in openai_models():
        try:
            data = _post_with_retry(
                label, f"{openai_base()}/chat/completions",
                {"model": model, "temperature": temperature, "max_tokens": 2048,
                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                {"Authorization": f"Bearer {key}"}, timeout=60,
            )
        except LLMError as err:
            last = err
            continue
        choices = data.get("choices") or []
        text = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
        if text:
            return Reply(text, label, model)
        last = LLMError(f"{label} returned an empty answer.")
    raise last or LLMError(f"No {label} model configured.")

def ollama(system: str, user: str, temperature: float) -> Reply:
    host, model = ollama_host(), setting("OLLAMA_MODEL")
    headers = {"Authorization": f"Bearer {setting('OLLAMA_API_KEY')}"} if setting("OLLAMA_API_KEY") else {}
    try:
        data = _post_with_retry(
            "Ollama", f"{host}/api/chat",
            {"model": model, "stream": False, "options": {"temperature": temperature},
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            headers, timeout=180,
        )
    except TransientLLMError as err:
        raise TransientLLMError(
            f"{err} at {host}. Run `ollama serve` locally, or configure a hosted provider for the public app."
        )
    text = (data.get("message") or {}).get("content", "").strip()
    if not text:
        raise LLMError("Ollama returned an empty answer.")
    return Reply(text, "ollama", model)


PROVIDERS = {"gemini": gemini, "openai": openai_compatible, "ollama": ollama}


def generate(system: str, user: str, temperature: float = 0.2) -> Reply:
    """Try each configured provider in order; the first answer wins."""
    errors = []
    for name in resolve_providers():
        try:
            return PROVIDERS[name](system, user, temperature)
        except LLMError as err:
            errors.append(str(err))
    if len(errors) == 1:
        raise LLMError(errors[0])
    raise LLMError("No model could answer right now. " + " | ".join(errors))
