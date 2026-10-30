"""
query_rewriter.py — Rewrite follow-up questions into standalone queries before retrieval.

Problem solved:
    Retrieval operates per-query without conversation memory. When a user asks
    "what are the requirements?" after discussing leave of absence, the embedding
    of that sentence alone carries little signal — retrieving random chunks that mention
    "requirements". The chat history is passed to the final answer LLM, but by then
    the wrong context has already been retrieved.

    The solution is rewriting follow-up questions into standalone form
    ("what are the requirements for academic leave of absence?") BEFORE retrieval runs.

Cost saving:
    Rewriting every query would add an extra LLM call per turn. However, most questions
    are already standalone. Heuristics filter queries first; the LLM is only called
    when the question clearly depends on prior context.

This module is **fail-safe**: any failure falls back to the original query.
"""

from __future__ import annotations

import re

from app.llm_client import call_utility_llm

#: Indicators that a question refers to something mentioned previously.
_REFERENTIAL_MARKERS = (
    "it", "they", "them", "that", "this", "these", "those", "their", "his", "her", "its",
    "the former", "the latter", "previous", "earlier", "above",
    "how about", "what about", "then", "next", "also", "besides",
    "how long", "why", "is that so", "for example", "details", "difference",
    # Legacy Indonesian markers
    "itu", "tersebut", "tadi", "sebelumnya", "yang mana", "kalau",
    "bagaimana dengan", "gimana dengan", "terus", "lalu", "kemudian",
    "dia", "mereka", "nya", "ini", "begitu", "sama seperti",
    "berapa lama", "kenapa", "mengapa", "apakah begitu", "selain itu",
    "contohnya", "misalnya", "detailnya", "lengkapnya", "bedanya",
)

#: Questions with this many raw words or fewer almost certainly require context.
_SHORT_QUERY_WORD_LIMIT = 4

#: Number of previous messages used as rewrite context.
_HISTORY_WINDOW = 6

#: Character limit for each history message to keep prompt concise.
_HISTORY_MESSAGE_CHAR_LIMIT = 400

REWRITE_SYSTEM_PROMPT = """You are a query rewriting component in an academic document search system.

Your task: rewrite the user's latest question into a STANDALONE question that can be understood without reading conversation history.

Rules:
1. Replace reference words ("it", "they", "that", "this", "its", "their") with the referenced entity from the conversation history.
2. Maintain the original intent. DO NOT add new conditions, constraints, or topics.
3. Retain all specific terms and numbers from the original question.
4. If the latest question is already standalone, return it AS IS without modification.
5. Use English.
6. Reply ONLY with the single standalone rewritten question. No explanations, no quotation marks, no prefixes."""


def needs_rewrite(query: str, chat_history: list[dict] | None) -> bool:
    """
    Determine whether a query needs rewriting, without calling an LLM.

    True if conversation history exists AND the question appears context-dependent:
    very short, or contains referential markers.

    Args:
        query: User's latest question.
        chat_history: Prior conversation history.
    """
    if not chat_history:
        return False

    # Need at least one prior assistant message to refer to.
    if not any(msg.get("role") == "assistant" for msg in chat_history):
        return False

    stripped = query.strip()
    if not stripped:
        return False

    if len(stripped.split()) <= _SHORT_QUERY_WORD_LIMIT:
        return True

    lowered = stripped.lower()

    # Possessive suffix "-nya" in Indonesian
    if re.search(r"\w{3,}nya\b", lowered):
        return True

    return any(
        re.search(rf"\b{re.escape(marker)}\b", lowered)
        for marker in _REFERENTIAL_MARKERS
    )


def build_rewrite_prompt(query: str, chat_history: list[dict]) -> str:
    """Construct rewrite prompt from recent conversation history window."""
    recent = chat_history[-_HISTORY_WINDOW:]

    parts = ["CONVERSATION HISTORY:"]
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if len(content) > _HISTORY_MESSAGE_CHAR_LIMIT:
            content = content[:_HISTORY_MESSAGE_CHAR_LIMIT].rsplit(" ", 1)[0] + " …"
        parts.append(f"{role}: {content}")

    parts.append(f"\nLATEST USER QUESTION:\n{query}")
    parts.append("\nStandalone rewritten question:")
    return "\n".join(parts)


def _sanitize(raw: str, fallback: str) -> str:
    """
    Clean up model response and reject invalid output.
    """
    if not raw:
        return fallback

    text = raw.strip()

    # Take first non-empty line
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break

    text = re.sub(r'^["\'`]+|["\'`]+$', "", text).strip()
    text = re.sub(
        r"^(standalone question|rewritten question|query|answer|pertanyaan mandiri|hasil penulisan ulang|jawaban)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if not text:
        return fallback

    # Model answering the question directly instead of rewriting
    if len(text.split()) > 40:
        return fallback

    return text


def rewrite_query(
    query: str,
    chat_history: list[dict] | None,
    api_key: str | None = None,
    force: bool = False,
) -> str:
    """
    Rewrite follow-up question into standalone query using utility LLM.

    Fail-safe: any failure returns the original query.
    """
    if not query or not query.strip():
        return query

    if not force and not needs_rewrite(query, chat_history):
        return query

    if not chat_history:
        return query

    prompt = build_rewrite_prompt(query, chat_history)

    try:
        raw_response = call_utility_llm(
            prompt,
            system_instruction=REWRITE_SYSTEM_PROMPT,
            max_tokens=64,
            api_key=api_key,
        )
        return _sanitize(raw_response, fallback=query)
    except Exception:
        return query
