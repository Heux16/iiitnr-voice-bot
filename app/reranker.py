"""
reranker.py — Rerank retrieval candidates using LLM as a cross-encoder.

Why reranking?
    Dense retrieval and BM25 assess queries and documents separately
    (bi-encoder / bag-of-words), scoring *similarity* rather than *answerability*.
    A chunk may share topical keywords without containing the answer.
    A reranker evaluates the query and chunk together, determining whether the chunk
    actually answers the user's question.

Why LLM instead of local cross-encoder?
    Multilingual cross-encoders are hundreds of megabytes and require PyTorch.
    A single call to Gemini Flash evaluates all candidates simultaneously
    with sub-second latency on CPU.

The approach is *listwise*: all candidates are sent in one prompt, and the model
scores each 0–10.

The reranker is **fail-safe**: any failure preserves fusion order so the chatbot
continues answering.
"""

from __future__ import annotations

import json
import re
import time

from app.llm_client import call_utility_llm

#: Character limit for each candidate snippet sent to reranker.
#: Maximum chunk length in the index is ~1520 characters; 2000 ensures complete context.
SNIPPET_CHAR_LIMIT = 2000

RERANK_SYSTEM_PROMPT = """You are a relevance ranking component (reranker) in an academic document search system.

Your task: evaluate how well each document excerpt ANSWERS the user's question.

Scoring guidelines (0-10):
- 9-10 : Contains a direct and complete answer to the question.
- 6-8  : Contains partial answers or clearly relevant supporting information.
- 3-5  : Related topic, but does not directly answer the question.
- 0-2  : Irrelevant, only shares generic keywords with the question.

Important rules:
- Evaluate the ability to ANSWER the question, not mere lexical similarity. Excerpts that share keywords but discuss a different topic must receive low scores.
- If the question is completely outside the scope of academic documents, give a score of 0-2 to ALL excerpts.
- Do not fabricate information. Evaluate solely based on the provided text.

Respond ONLY with a JSON array, without explanation or markdown wrap:
[{"id": 1, "score": 8}, {"id": 2, "score": 3}]"""


def build_rerank_prompt(query: str, snippets: list[str]) -> str:
    """Build listwise prompt for the reranker."""
    parts = [f"USER QUESTION:\n{query}\n", "DOCUMENT EXCERPTS:"]
    for i, snippet in enumerate(snippets, start=1):
        parts.append(f"\n[{i}]\n{snippet}")
    parts.append(
        f"\nGive a relevance score 0-10 for each excerpt [1]-[{len(snippets)}]. "
        "Reply only with a JSON array."
    )
    return "\n".join(parts)


def _snippet_for(candidate) -> str:
    """Candidate snippet, prefixed by heading for section context."""
    heading = candidate.heading_label
    text = candidate.text.strip()
    if len(text) > SNIPPET_CHAR_LIMIT:
        text = text[:SNIPPET_CHAR_LIMIT].rsplit(" ", 1)[0] + " …"
    return f"(Page {candidate.page_label} — {heading})\n{text}" if heading else text


def parse_rerank_response(raw: str, expected: int) -> dict[int, float]:
    """
    Convert model response into a map of ``candidate id (1-based) → score``.

    Tolerant to responses wrapped in code fences or extra text.
    IDs out of range and non-numeric scores are ignored.

    Returns:
        Dict of id → score. Empty if parsing fails.
    """
    if not raw:
        return {}

    payload = raw.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", payload).strip()

    data = None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", payload, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break

    if not isinstance(data, list):
        return {}

    scores: dict[int, float] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get("id"))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if 1 <= item_id <= expected:
            scores[item_id] = max(0.0, min(10.0, score))

    return scores


#: Retries for transient errors.
_MAX_ATTEMPTS = 3
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500")


def _call_reranker(prompt: str, api_key: str | None) -> str | None:
    """Call reranker model, retrying transient errors."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return call_utility_llm(
                prompt,
                system_instruction=RERANK_SYSTEM_PROMPT,
                max_tokens=512,
                api_key=api_key,
                json_mode=True,
            )
        except Exception as e:
            message = str(e)
            transient = any(marker in message for marker in _TRANSIENT_MARKERS)
            if not transient or attempt == _MAX_ATTEMPTS:
                return None
            time.sleep(1.5 * attempt)
    return None


def rerank_candidates(
    query: str,
    candidates: list,
    api_key: str | None = None,
) -> tuple[list, bool]:
    """
    Re-score and re-order candidates based on ability to answer the query.

    Args:
        query: User question.
        candidates: List of fusion `RetrievalCandidate` objects.
        api_key: Optional Gemini API key override.

    Returns:
        Tuple ``(candidates, success)``. If ``success`` is ``False``, fusion
        order is preserved as is.
    """
    if len(candidates) <= 1:
        return candidates, True

    snippets = [_snippet_for(c) for c in candidates]
    raw = _call_reranker(build_rerank_prompt(query, snippets), api_key)
    if raw is None:
        return candidates, False

    scores = parse_rerank_response(raw, expected=len(candidates))
    if not scores:
        return candidates, False

    for position, candidate in enumerate(candidates, start=1):
        candidate.rerank_score = scores.get(position, 0.0)

    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (item[1].rerank_score, -item[0]),
        reverse=True,
    )
    return [candidate for _, candidate in ordered], True
