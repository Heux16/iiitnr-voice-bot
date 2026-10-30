"""Circuit breaker to ensure invalid provider responses are not shown to users."""

from __future__ import annotations

import re

from app.citations import extract_cited_pages

_PAGE_CITATION_RE = re.compile(r"\b(?:page|halaman)\s*:?\s*\d{1,4}\b", re.IGNORECASE)
_ERROR_MARKERS = (
    "⚠️",
    "not_found",
    "resource_exhausted",
    "rate limit",
    "api key",
)


def has_page_citation(answer: str) -> bool:
    """True if the answer contains at least one page citation."""
    return bool(answer and _PAGE_CITATION_RE.search(answer))


def is_safe_answer(answer: str) -> bool:
    """
    Minimum validation before an answer is displayed.

    Circuit breaker for unsafe conditions:
    1. Provider returns an error message as answer text;
    2. Model ignores prompt and returns generic text without citation.
    """
    if not answer or not answer.strip():
        return False
    lowered = answer.lower()
    if any(marker.lower() in lowered for marker in _ERROR_MARKERS):
        return False
    return has_page_citation(answer)


def diagnose_answer(answer: str, context_pages: set[int]) -> tuple[str, str] | None:
    """Return a safe diagnostic code/message when a generated answer is blocked.

    Never include raw provider response here: error bodies and model output
    can contain user or configuration data.
    """
    if not answer or not answer.strip():
        return "empty_response", "Provider returned an empty response."

    lowered = answer.lower()
    if any(marker.lower() in lowered for marker in _ERROR_MARKERS):
        if any(token in lowered for token in (
            "quota", "rate limit", "usage limit", "batas penggunaan", "429", "resource_exhausted"
        )):
            return "provider_rate_limit", "Provider rejected request due to quota or rate limit (HTTP 429)."
        if any(token in lowered for token in ("api key", "api_key", "authentication", "401")):
            return "provider_auth", "Provider rejected authentication (API key or HTTP 401)."
        if any(token in lowered for token in ("404", "not_found", "model not found")):
            return "provider_model_or_endpoint", "Provider model or endpoint not found (HTTP 404)."
        if "model_disabled" in lowered:
            return "provider_models_unavailable", "Provider model is disabled or unavailable; fallback models also failed."
        return "provider_error", "Provider returned an error; raw details hidden."

    if not has_page_citation(answer):
        return "missing_page_citation", "Answer does not contain a citation in 'Page N' format."

    cited_pages = set(extract_cited_pages(answer))
    invalid_pages = sorted(cited_pages - context_pages)
    if invalid_pages:
        available = ", ".join(map(str, sorted(context_pages))) or "none"
        invalid = ", ".join(map(str, invalid_pages))
        return (
            "citation_outside_context",
            f"Page citation {invalid} is not in retrieval context; available pages: {available}.",
        )

    return None
