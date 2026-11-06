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
    # English referential pronouns and anaphora
    "it", "they", "them", "that", "this", "these", "those", "their", "his", "her", "its",
    "the former", "the latter", "previous", "earlier", "above",
    "how about", "what about", "then", "next", "also", "besides",
    "how long", "why", "is that so", "for example", "details", "difference",
    # Hindi / Hinglish referential markers
    "iska", "iski", "iske", "uska", "uski", "uske", "inka", "inki", "inke", "unka", "unki", "unke",
    "ye", "yeh", "wo", "woh", "wahan", "yahan", "idhar", "udhar",
    "aur", "aur bhi", "aur batao", "dusra", "agla", "pichla",
    "phir", "fir", "usme", "isme", "unme", "inme", "isse", "usse",
    # Chhattisgarhi referential markers
    "okhar", "ekhar", "oman", "eman", "tahar", "hamar", "aau", "baare",
    # Devanagari referential markers
    "इसका", "इसकी", "इसके", "उसका", "उसकी", "उसके", "इनका", "उनकी", "उनके",
    "यह", "वह", "ये", "वो", "वहाँ", "यहाँ", "और", "भी", "दूसरा", "अगला", "पिछला",
    "फिर", "उसमें", "इसमें", "ओखर", "एखर", "ओमन", "एमन", "अउ", "बारे",
    # Upstream Indonesian markers (retained for backward compatibility)
    "itu", "tersebut", "tadi", "sebelumnya", "yang mana", "kalau",
    "bagaimana dengan", "gimana dengan", "terus", "lalu", "kemudian",
    "dia", "mereka", "nya", "ini", "begitu", "sama seperti",
    "berapa lama", "kenapa", "mengapa", "apakah begitu", "selain itu",
    "contohnya", "misalnya", "detailnya", "lengkapnya", "bedanya",
)

#: Strong Indic and Chhattisgarhi question/predicate words in Roman script.
#: The presence of even ONE of these words strongly indicates Hinglish or Romanized Chhattisgarhi.
_STRONG_INDIC_WORDS = frozenset({
    # Interrogatives (Hindi / Chhattisgarhi)
    "kya", "kyu", "kyon", "kaise", "kaisa", "kaisi", "kaisan", "kisan", "kitna", "kitni", "kitne",
    "kahan", "kaha", "kab", "kisko", "kiska", "kiski", "kiske", "kaun", "kaunsa", "kaunsi",
    "katek", "kaabar", "kabar", "kon", "konse",
    # Dialect / auxiliary predicates
    "hawe", "heve", "laghi", "milhi", "jaahi", "aahi", "rahithan", "rahithe",
    "batao", "bataiye", "batawa", "bataw", "batado",
    "chahiye", "milega", "milegi", "milenge",
    "hoga", "hogi", "honge",
    # Dialect pronouns
    "okhar", "ekhar", "oman", "eman",
})

#: Moderate Indic connectors/pronouns in Roman script.
#: Requires at least TWO matches (or combination with question) to avoid false positives.
_MODERATE_INDIC_WORDS = frozenset({
    "hai", "hain", "tha", "thi", "the", "liye", "bhi", "aur", "iska", "iski", "iske",
    "uska", "uski", "uske", "inka", "inki", "inke", "unka", "unki", "unke",
    "wahan", "yahan", "idhar", "udhar", "agla", "pichla", "dusra",
    "karna", "karne", "karte", "wale", "wali", "wala",
    "lagta", "lagti", "lagte", "milta", "milti", "milte",
    "sakta", "sakti", "sakte", "usme", "isme", "isse", "usse",
    "aau", "tahar", "hamar", "baare",
})

#: Questions with this many raw words or fewer almost certainly require context.
_SHORT_QUERY_WORD_LIMIT = 4

#: Number of previous messages used as rewrite context.
_HISTORY_WINDOW = 6

#: Character limit for each history message to keep prompt concise.
_HISTORY_MESSAGE_CHAR_LIMIT = 400

REWRITE_SYSTEM_PROMPT = """You are an expert academic search query rewriter for the IIIT Naya Raipur (IIIT-NR) Academic RAG System.
The official institute knowledge base and academic source documents are written in English.

Your primary mission:
Produce a single, concise, search-optimized STANDALONE ENGLISH QUERY that will retrieve the most accurate document sections using dense semantic search (FAISS) and sparse lexical search (BM25).

Rules to follow:
1. Cross-Lingual Translation (Hindi, Chhattisgarhi & Hinglish):
   - Translate any Hindi (Devanagari or Romanized/Hinglish) or Chhattisgarhi (Devanagari or Romanized) query into formal academic English.
   - Chhattisgarhi dialect glossary:
     * "कतेक" / "katek" = how much / what amount / cutoff
     * "हावे" / "heve" / "hawe" = is / exists
     * "काबर" / "kabar" / "kaabar" = why / for what reason
     * "कइसे" / "kaise" / "kisan" = how / what is the procedure
     * "ओखर" / "okhar" = its / their / for that
     * "एखर" / "ekhar" = this / for this
     * "अउ" / "aau" = and / also
     * "लागही" / "laghi" = will cost / is required
     * "मिलही" / "milhi" = will be obtained / eligibility
   - Preserve technical, institutional, and entrance exam acronyms: JoSAA, CSAB, JEE Main, GATE, CCMT, CSE, DSAI, ECE, CPI, SPI, CGPA, SGPA, IIIT-NR.

2. Context & Coreference Resolution:
   - If the query contains pronouns or referential expressions ("it", "they", "that", "this", "iska", "uska", "okhar", "wahan", "agla", "the former"), replace them with the exact academic entity from the conversation history.
   - If the query is an elliptical follow-up (e.g., "And for ECE?", "What about hostel?"), expand it into a complete, standalone question incorporating the topic from the conversation history.

3. Retrieval Optimization (BM25 & FAISS):
   - Include high-signal academic keywords in the rewritten query (e.g., "opening and closing rank cutoff", "fee structure and hostel charges", "curriculum and syllabus credits", "eligibility criteria and admission procedure").
   - Do NOT answer the question.
   - Do NOT summarize the conversation.
   - Reply ONLY with the single standalone English question.
   - Never output explanations, bullet points, markdown quotes, or prefixes like "Rewritten Question:".

Few-Shot Demonstrations:
- History: [User: "Tell me about B.Tech CSE at IIIT-NR", Assistant: "B.Tech in Computer Science and Engineering focuses on..."]
  Follow-up: "Iska cutoff kitna gaya tha pichle saal?"
  Rewritten: What was the previous year JEE Main JoSAA opening and closing cutoff rank for B.Tech Computer Science and Engineering (CSE) at IIIT-NR?

- History: [User: "What are the hostel facilities?", Assistant: "IIIT-NR provides separate hostels for boys and girls with..."]
  Follow-up: "Okhar fees katek hawe aau mess charge kitna laghi?"
  Rewritten: What is the fee structure for hostel accommodation and mess charges at IIIT-NR?

- History: None
  Question: "CSE ki opening rank kya hai"
  Rewritten: What is the opening rank for B.Tech Computer Science and Engineering (CSE) admission at IIIT-NR?

- History: [User: "What is the policy for branch sliding?", Assistant: "Branch sliding is offered after the first year..."]
  Follow-up: "Uske liye minimum CPI kitna chahiye?"
  Rewritten: What is the minimum CPI requirement for branch sliding at IIIT-NR?

- History: None
  Question: "सीएसई म एडमिशन कइसे मिलही?"
  Rewritten: What is the admission procedure and eligibility criteria for B.Tech Computer Science and Engineering at IIIT-NR?

- History: None
  Question: "IIIT Naya Raipur कहां स्थित है?"
  Rewritten: Where is the campus and geographic location of IIIT Naya Raipur?

- History: [User: "Can I take academic leave?", Assistant: "Academic leave can be granted under Ordinance Section 4..."]
  Follow-up: "How long maximum?"
  Rewritten: What is the maximum duration allowed for academic leave of absence?"""


def normalize_spoken_acronyms(text: str) -> str:
    """
    Normalise common phonetic misrecognitions of institute acronyms
    when spoken in Indian English, Hindi, or Chhattisgarhi.
    """
    if not text:
        return text

    # Phonetic variations of "Triple IT / Triple I T" in Devanagari or English
    text = re.sub(
        r"(?:प्रिबल|ट्रिपल|ट्रिपिल|त्रिपल|Triple)\s*(?:I\s*I\s*T|I\s*T|आई\s*आई\s*टी|आई\s*टी|eye\s*tea|I\.T\.)",
        "IIIT",
        text,
        flags=re.IGNORECASE,
    )
    # Devanagari expansion
    text = re.sub(r"आई\s*आई\s*आई\s*टी", "IIIT", text)
    # Common Devanagari phonetic misspelling for location query
    text = re.sub(r"\bइस्थित\b", "स्थित", text)
    return re.sub(r"\s+", " ", text).strip()


def is_indic_or_hinglish(query: str) -> bool:
    """
    Check if a query contains Hindi or Chhattisgarhi in Devanagari or Romanized (Hinglish) script.
    """
    if not query:
        return False

    # Devanagari script (Hindi, Chhattisgarhi, Marathi, etc.)
    if re.search(r"[\u0900-\u097F]", query):
        return True

    # Tokenize lowercased alphanumeric words
    tokens = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    if not tokens:
        return False

    strong_matches = sum(1 for t in tokens if t in _STRONG_INDIC_WORDS)
    if strong_matches >= 1:
        return True

    moderate_matches = sum(1 for t in tokens if t in _MODERATE_INDIC_WORDS)
    return moderate_matches >= 2


def needs_rewrite(query: str, chat_history: list[dict] | None) -> bool:
    """
    Determine whether a query needs rewriting, without calling an LLM.

    True if:
    - The query contains Hindi / Chhattisgarhi (Devanagari or Romanized/Hinglish),
      which must be translated to English for document retrieval.
    - Conversation history exists AND the question appears context-dependent (short or referential).

    Args:
        query: User's latest question.
        chat_history: Prior conversation history.
    """
    cleaned = normalize_spoken_acronyms(query)
    stripped = cleaned.strip()
    if not stripped:
        return False

    # Indic (Devanagari or Hinglish / Chhattisgarhi) always needs English translation for retrieval
    if is_indic_or_hinglish(stripped):
        return True

    if not chat_history:
        return False

    # Need at least one prior assistant message to refer to.
    if not any(msg.get("role") == "assistant" for msg in chat_history):
        return False

    if len(stripped.split()) <= _SHORT_QUERY_WORD_LIMIT:
        return True

    lowered = stripped.lower()

    return any(
        re.search(rf"\b{re.escape(marker)}\b", lowered)
        for marker in _REFERENTIAL_MARKERS
    )


def build_rewrite_prompt(query: str, chat_history: list[dict] | None = None) -> str:
    """Construct rewrite prompt from recent conversation history window."""
    if not chat_history:
        return f"LATEST USER QUESTION:\n{query}\n\nStandalone rewritten question:"

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
        r"^(standalone question|rewritten question|standalone english question|rewritten query|search query|english query|query|answer|pertanyaan mandiri|hasil penulisan ulang|jawaban)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(r"^\*\*(.+?)\*\*$", r"\1", text).strip()
    text = re.sub(r'^["\'`]+|["\'`]+$', "", text).strip()

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
    Rewrite follow-up or cross-lingual question into a standalone English query using utility LLM.

    Fail-safe: any failure returns the original query.
    """
    if not query or not query.strip():
        return query

    cleaned_query = normalize_spoken_acronyms(query)

    if not force and not needs_rewrite(cleaned_query, chat_history):
        return cleaned_query

    is_indic = is_indic_or_hinglish(cleaned_query)
    if not chat_history and not is_indic and not force:
        return cleaned_query

    prompt = build_rewrite_prompt(cleaned_query, chat_history)

    try:
        raw_response = call_utility_llm(
            prompt,
            system_instruction=REWRITE_SYSTEM_PROMPT,
            max_tokens=64,
            api_key=api_key,
        )
        return _sanitize(raw_response, fallback=cleaned_query)
    except Exception:
        return cleaned_query
