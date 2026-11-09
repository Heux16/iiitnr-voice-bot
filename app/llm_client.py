"""
llm_client.py — Adapter provider Google Gemini untuk pipeline RAG.

Seluruh pipeline kini memakai Google Gemini API:
- **Chat, Rerank, Query Rewrite**: Google Gemini API via google.genai Client.
- **Embedding**: Google Gemini API (gemini-embedding-001).

API internal tetap dipertahankan:
- `call_llm()`
- `call_utility_llm()`
- `embed_texts()`
- `get_query_embedding()`
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from google import genai
from google.genai import types

from app.config import (
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    EMBED_BATCH_DELAY,
    EMBED_BATCH_SIZE,
    EMBED_MAX_RETRIES,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    MODEL_FALLBACKS,
    PROJECT_ROOT,
    TASK_TYPE_DOCUMENT,
    TASK_TYPE_QUERY,
    UTILITY_MODEL,
    UTILITY_TEMPERATURE,
)

_PLACEHOLDER_KEYS = {
    "",
    "MASUKKAN_API_KEY_GEMINI_ANDA_DI_SINI",
    "your_actual_gemini_api_key_here",
    "your_bandel_api_key_here",
    "your_api_key_here",
}

_MODEL_UNAVAILABLE_MARKERS = (
    "404", "NOT_FOUND",
    "MODEL_DISABLED",
    "429", "RESOURCE_EXHAUSTED", "RATE_LIMIT",
    "500", "502", "503", "504", "UNAVAILABLE",
)

_WORKING_MODEL: dict[str, str] = {}
_QUERY_EMBEDDING_CACHE: dict[tuple[str, str | None], list[float]] = {}
_QUERY_EMBEDDING_CACHE_LIMIT = 512

HTTP_TIMEOUT_SECONDS = 120


class LLMConfigError(RuntimeError):
    """Provider/key belum dikonfigurasi."""


class LLMCallError(RuntimeError):
    """Panggilan provider gagal setelah semua percobaan."""


# ──────────────────────────────────────────────
# Secrets & provider config
# ──────────────────────────────────────────────

def _valid_key(value: Any) -> bool:
    return bool(value and str(value).strip() not in _PLACEHOLDER_KEYS)


def _looks_like_gateway_key(value: str | None) -> bool:
    return bool(value and str(value).strip().lower().startswith(("sk-", "sk_")))


def _read_secrets_file(*names: str) -> str | None:
    """Baca key generik dari secrets.toml tanpa mencetak nilainya."""
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None

    try:
        import tomllib
        with open(secrets_path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, ValueError, ModuleNotFoundError):
        return None

    for name in names:
        value = data.get(name)
        if _valid_key(value):
            return str(value).strip()
    return None


def _read_streamlit_secret(*names: str) -> str | None:
    """Baca secret dari Streamlit bila aplikasi sedang berjalan."""
    try:
        import streamlit as st
        for name in names:
            value = st.secrets.get(name)
            if _valid_key(value):
                return str(value).strip()
    except Exception:
        pass
    return None


def resolve_llm_api_key(explicit_key: str | None = None) -> str:
    """
    Cari key untuk provider LLM.

    Prioritas:
    1. argumen eksplisit
    2. `LLM_API_KEY`, `BANDEL_API_KEY`, `OPENAI_API_KEY`
    3. `GEMINI_API_KEY`, `GOOGLE_API_KEY`
    4. secrets.toml
    """
    candidates: list[str | None] = [explicit_key]
    for env_name in (
        "LLM_API_KEY",
        "BANDEL_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        candidates.append(os.environ.get(env_name))

    candidates.append(_read_secrets_file(
        "LLM_API_KEY", "BANDEL_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"
    ))
    candidates.append(_read_streamlit_secret(
        "LLM_API_KEY", "BANDEL_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"
    ))

    for candidate in candidates:
        if _valid_key(candidate):
            return str(candidate).strip()

    raise LLMConfigError(
        "API key provider belum diatur. Isi salah satu:\n"
        "  1. GEMINI_API_KEY atau LLM_API_KEY di .streamlit/secrets.toml\n"
        "  2. environment variable GEMINI_API_KEY atau LLM_API_KEY\n"
        "  3. flag --api-key YOUR_KEY"
    )


def resolve_api_key(explicit_key: str | None = None) -> str:
    """
    Alias kompatibilitas untuk kode yang membaca API key.
    """
    return resolve_llm_api_key(explicit_key)


def resolve_embedding_api_key(explicit_key: str | None = None) -> str:
    """Cari key Google untuk embedding."""
    candidates: list[str | None] = [explicit_key]
    for env_name in (
        "GEMINI_EMBEDDING_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "EMBEDDING_API_KEY",
        "LLM_API_KEY",
    ):
        candidates.append(os.environ.get(env_name))

    candidates.append(_read_secrets_file(
        "GEMINI_EMBEDDING_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "EMBEDDING_API_KEY",
        "LLM_API_KEY",
    ))
    candidates.append(_read_streamlit_secret(
        "GEMINI_EMBEDDING_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "EMBEDDING_API_KEY",
        "LLM_API_KEY",
    ))

    for candidate in candidates:
        if _valid_key(candidate):
            return str(candidate).strip()

    raise LLMConfigError(
        "API key Gemini embedding belum diatur. Isi "
        "GEMINI_API_KEY atau GOOGLE_API_KEY dengan key Google AI Studio."
    )


def resolve_llm_base_url(explicit_url: str | None = None) -> str:
    """Cari base URL OpenAI-compatible dari argumen, env, secrets, default."""
    candidates: list[str | None] = [explicit_url]
    candidates.extend([
        os.environ.get("LLM_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        _read_secrets_file("LLM_BASE_URL", "OPENAI_BASE_URL"),
        _read_streamlit_secret("LLM_BASE_URL", "OPENAI_BASE_URL"),
        LLM_BASE_URL,
    ])

    for candidate in candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip().rstrip("/")

    raise LLMConfigError("Base URL provider chat belum diatur.")


@lru_cache(maxsize=4)
def get_client(api_key: str | None = None) -> genai.Client:
    """Gemini client yang di-cache khusus untuk Gemini API."""
    key = api_key
    if not key:
        try:
            key = resolve_embedding_api_key()
        except LLMConfigError:
            key = resolve_api_key()
    return genai.Client(api_key=key)


# ──────────────────────────────────────────────
# Chat & Generation
# ──────────────────────────────────────────────

def _is_model_unavailable(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in _MODEL_UNAVAILABLE_MARKERS)


def model_chain(primary: str) -> list[str]:
    """Urutan model utama dan cadangannya."""
    configured = [primary, *MODEL_FALLBACKS.get(primary, ())]
    working = _WORKING_MODEL.get(primary)
    if working and working in configured:
        return [working] + [model for model in configured if model != working]
    return configured


def effective_model(primary: str) -> str:
    """Model yang terakhir terbukti berhasil untuk model konfigurasi tertentu."""
    return _WORKING_MODEL.get(primary, primary)


def reset_model_cache() -> None:
    """Reset ingatan model aktif, terutama untuk test/operator."""
    _WORKING_MODEL.clear()


def _openai_error(status: int | None, body: str) -> LLMCallError:
    compact = " ".join(body.split())[:700]
    return LLMCallError(f"{status or 'HTTP'} {compact}")


def _post_json(
    path: str,
    payload: dict,
    api_key: str,
    base_url: str,
) -> dict:
    """POST JSON dengan stdlib, untuk kompatibilitas endpoint OpenAI-compatible."""
    request = Request(
        f"{base_url}/{path.lstrip('/')}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise _openai_error(error.code, body) from error
    except (URLError, TimeoutError, OSError) as error:
        raise LLMCallError(f"Provider tidak dapat dihubungi: {error}") from error

    try:
        data = json.loads(body)
    except json.JSONDecodeError as error:
        raise _openai_error(status, body) from error

    if not isinstance(data, dict):
        raise LLMCallError("Provider mengembalikan format JSON yang tidak valid.")
    if data.get("error"):
        raise _openai_error(status, json.dumps(data["error"], ensure_ascii=False))
    return data


def _extract_chat_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip() if content else ""


def _generate_once_gemini(
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    model: str,
    api_key: str | None,
    response_mime_type: str | None = None,
) -> str:
    """Panggilan generasi ke Google Gemini API via google.genai Client."""
    client = get_client(api_key)

    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    if "2.5" in model and (temperature == 0.0 or max_tokens < 1000):
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass

    cfg = types.GenerateContentConfig(**config_kwargs)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=cfg,
        )
        return (response.text or "").strip()
    except Exception as error:
        raise LLMCallError(f"Google Gemini call failed: {error}") from error


def _generate_once_openai(
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    model: str,
    api_key: str | None,
    response_mime_type: str | None,
) -> str:
    """Satu panggilan chat completions ke endpoint OpenAI-compatible."""
    key = resolve_llm_api_key(api_key)
    base_url = resolve_llm_base_url()

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_mime_type == "application/json":
        payload["response_format"] = {"type": "json_object"}

    return _extract_chat_text(_post_json("chat/completions", payload, key, base_url))


def _generate_once(
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    model: str,
    api_key: str | None,
    response_mime_type: str | None = None,
) -> str:
    from app.config import LLM_PROVIDER
    # Jalur OpenAI-compatible untuk backward compatibility dan unit testing
    if (
        LLM_PROVIDER == "openai_compatible"
        or model in ("auto", "deepseek-v4-flash", "glm-5.3-flash")
        or model.startswith(("deepseek", "glm", "gpt", "claude"))
    ):
        return _generate_once_openai(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            api_key=api_key,
            response_mime_type=response_mime_type,
        )

    return _generate_once_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        api_key=api_key,
        response_mime_type=response_mime_type,
    )


def _generate(
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    model: str,
    api_key: str | None,
    response_mime_type: str | None = None,
) -> str:
    """Generate dengan fallback model bila provider mengembalikan error model."""
    last_error: Exception | None = None

    for candidate in model_chain(model):
        for attempt in range(4):
            try:
                text = _generate_once(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=candidate,
                    api_key=api_key,
                    response_mime_type=response_mime_type,
                )
                if candidate != model:
                    _WORKING_MODEL[model] = candidate
                return text
            except Exception as error:
                last_error = error
                msg = str(error)
                rate_limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()
                if rate_limited and attempt < 3:
                    wait_time = _parse_retry_delay(error)
                    time.sleep(wait_time)
                    continue
                if not _is_model_unavailable(error):
                    raise
                break

    raise last_error or LLMCallError("Tidak ada model chat yang bisa dipanggil.")


# ──────────────────────────────────────────────
# Public generation API
# ──────────────────────────────────────────────

def call_llm(
    prompt: str,
    system_instruction: str = "",
    temperature: float = ANSWER_TEMPERATURE,
    max_tokens: int = ANSWER_MAX_TOKENS,
    model: str = LLM_MODEL,
    api_key: str | None = None,
    raise_on_error: bool = False,
) -> str:
    """Panggil provider Gemini untuk jawaban akhir."""
    if raise_on_error:
        return _generate(
            prompt, system_instruction, temperature, max_tokens, model, api_key
        )

    try:
        text = _generate(
            prompt, system_instruction, temperature, max_tokens, model, api_key
        )
        return text or "I apologize, but I could not generate an answer for this question."
    except LLMConfigError as error:
        return f"⚠️ {error}"
    except Exception as error:
        message = str(error)
        lowered = message.lower()
        if "quota" in lowered or "rate" in lowered or "429" in message:
            return "⚠️ API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."
        if any(token in lowered for token in ("api key", "api_key", "authentication", "401")):
            return "⚠️ Invalid API key. Please check secrets configuration."
        return f"⚠️ Error processing request: {message[:500]}"


def call_llm_stream(
    prompt: str,
    system_instruction: str = "",
    temperature: float = ANSWER_TEMPERATURE,
    max_tokens: int = ANSWER_MAX_TOKENS,
    model: str = LLM_MODEL,
    api_key: str | None = None,
) -> Iterator[str]:
    """Stream answer tokens from Gemini, with fallback to standard generation if needed."""
    from app.config import LLM_PROVIDER
    if (
        LLM_PROVIDER == "openai_compatible"
        or model in ("auto", "deepseek-v4-flash", "glm-5.3-flash")
        or model.startswith(("deepseek", "glm", "gpt", "claude"))
    ):
        full_text = call_llm(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            api_key=api_key,
        )
        yield full_text
        return

    try:
        client = get_client(api_key)
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if "2.5" in model and (temperature == 0.0 or max_tokens < 1000):
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass

        cfg = types.GenerateContentConfig(**config_kwargs)

        stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=cfg,
        )
        has_content = False
        for chunk in stream:
            if chunk.text:
                has_content = True
                yield chunk.text

        if not has_content:
            yield "I apologize, but I could not generate an answer for this question."
    except Exception as error:
        yield call_llm(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            api_key=api_key,
        )


def call_utility_llm(
    prompt: str,
    system_instruction: str = "",
    max_tokens: int = 1024,
    api_key: str | None = None,
    json_mode: bool = False,
) -> str:
    """Panggil provider Gemini untuk reranker/query rewriting."""
    return _generate(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=UTILITY_TEMPERATURE,
        max_tokens=max_tokens,
        model=UTILITY_MODEL,
        api_key=api_key,
        response_mime_type="application/json" if json_mode else None,
    )


def test_connection(api_key: str | None = None) -> bool:
    """Tes koneksi DAN kepatuhan model terhadap prompt."""
    probe = "LLM_HEALTH_PROBE_7F3C"
    try:
        answer = call_utility_llm(
            f"Balas tepat dengan token {probe}. Jangan tambahkan kata lain.",
            max_tokens=32,
            api_key=api_key,
        )
        return probe in answer
    except Exception:
        return False


# ──────────────────────────────────────────────
# Gemini embedding API
# ──────────────────────────────────────────────

def _parse_retry_delay(error: Exception) -> float:
    """Extract retry delay in seconds from Gemini API error response."""
    import re
    msg = str(error)
    m = re.search(r"retry\s+in\s+([0-9.]+)\s*s", msg, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)) + 2.0
        except ValueError:
            pass
    m = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"]([0-9]+)s?['\"]", msg)
    if m:
        try:
            return float(m.group(1)) + 2.0
        except ValueError:
            pass
    return 35.0


def embed_texts(
    texts: list[str],
    task_type: str | None = TASK_TYPE_DOCUMENT,
    api_key: str | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    batch_delay: float = EMBED_BATCH_DELAY,
    max_retries: int = EMBED_MAX_RETRIES,
    progress: bool = False,
) -> list[list[float]]:
    """Embed texts using Google Gemini Embedding model."""
    if not texts:
        return []

    client = get_client(api_key)

    config = types.EmbedContentConfig(task_type=task_type) if task_type else None
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) - 1) // batch_size + 1

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        if progress:
            print(f"   Embedding batch {batch_num}/{total_batches} ({len(batch)} items)...", flush=True)

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                contents = [
                    types.Content(parts=[types.Part.from_text(text=t)])
                    for t in batch
                ]
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=contents,
                    config=config,
                )
                all_embeddings.extend(e.values for e in result.embeddings)
                last_error = None
                break
            except Exception as error:
                last_error = error
                message = str(error)
                rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
                if not rate_limited or attempt == max_retries - 1:
                    break
                wait = _parse_retry_delay(error)
                if progress:
                    print(f"   >> Rate limit encountered, waiting {wait:.1f}s "
                          f"(attempt {attempt + 1}/{max_retries})...", flush=True)
                time.sleep(wait)

        if last_error is not None:
            raise LLMCallError(
                f"Failed to embed batch {batch_num}/{total_batches}: {last_error}"
            ) from last_error

        if batch_num < total_batches and batch_delay > 0:
            time.sleep(batch_delay)

    return all_embeddings


def get_query_embedding(
    query: str,
    task_type: str | None = TASK_TYPE_QUERY,
    api_key: str | None = None,
    use_cache: bool = True,
) -> list[float]:
    """Embedding query Gemini dengan cache proses."""
    cache_key = (query, task_type)
    if use_cache and cache_key in _QUERY_EMBEDDING_CACHE:
        return _QUERY_EMBEDDING_CACHE[cache_key]

    embeddings = embed_texts(
        [query], task_type=task_type, api_key=api_key, batch_delay=0.0
    )
    if not embeddings:
        raise LLMCallError("API embedding tidak mengembalikan vector apa pun.")

    if use_cache:
        if len(_QUERY_EMBEDDING_CACHE) >= _QUERY_EMBEDDING_CACHE_LIMIT:
            _QUERY_EMBEDDING_CACHE.clear()
        _QUERY_EMBEDDING_CACHE[cache_key] = embeddings[0]

    return embeddings[0]


def clear_query_embedding_cache() -> None:
    """Kosongkan cache embedding query."""
    _QUERY_EMBEDDING_CACHE.clear()
