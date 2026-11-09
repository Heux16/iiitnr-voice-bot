"""
prompt_builder.py — Construct prompts for LLM.

Combines system instructions, context from retrieved chunks,
and user question into a structured prompt.
"""

import re
from typing import Optional

from app.query_rewriter import is_indic_or_hinglish
from app.retrieval import RetrievalResult


# ──────────────────────────────────────────────
# Chhattisgarhi Dialect Vocabulary & Patterns
# ──────────────────────────────────────────────

_CHHATTISGARHI_DEVANAGARI_WORDS = frozenset({
    # Interrogatives & adverbs
    "कतेक", "काबर", "कइसे", "कइसन", "कइसने", "किसन", "कती", "कनहा", "कोन", "कोनसे", "कोनहा",
    # Pronouns & demonstratives
    "ओखर", "ओखरे", "एखर", "एखरे", "ओमन", "ओमनला", "ओमनके", "एमन", "एमनला", "एमनके",
    "मोर", "मोला", "हमर", "हमार", "तोर", "तोला", "तुहार", "तोहर", "ओला", "एला",
    # Conjunctions, prepositions & particles
    "अउ", "अऊ", "सन", "कोती", "बर",
    # Dialect verbs, auxiliaries & copulas
    "हावे", "हावै", "हवय", "हवै", "लगही", "लागही", "मिलही", "जाही", "आही", "होही", "करही",
    "रहिथे", "रहिथें", "रहिथन", "पड़ही", "सकही", "आय", "आये", "नोहे", "नइ", "नइये", "नइखे",
    "आंव", "हंव", "हंस", "हव", "बताव", "बतावा", "करबो", "जाबो", "आबो", "देखबो", "बताबो",
    # Greetings & vocatives
    "जोहार", "संगवारी", "संगवारीमन", "संगी", "ददा", "दाई", "गा",
})

# Dialect morphological suffixes (verbs in -बो, -हू, -िस/-हिस, -थन/-थस, and plurals in -मन)
_CHHATTISGARHI_SUFFIX_PATTERNS = [
    re.compile(r"[\u0900-\u097F]+(?:हू|हु)$"),         # करहू, देखहू, बताहू
    re.compile(r"[\u0900-\u097F]+(?:बो)$"),           # करबो, जाबो, बताबो, देखबो
    re.compile(r"[\u0900-\u097F]+(?:िस|इस|हिस)$"),     # करिस, आइस, गइस, देखिस, कहिस
    re.compile(r"[\u0900-\u097F]+(?:थन|थव|थस)$"),     # करथन, जानथस, करथव, रहिथन
    re.compile(r"[\u0900-\u097F]+(?:मन|मन्|मनला|मनके)$"),  # लइकामन, छात्रमन, संगवारीमन
]

_CHHATTISGARHI_ROMAN_WORDS = frozenset({
    "katek", "kabar", "kaabar", "kaisan", "kisan", "hawe", "heve", "haway", "hawai",
    "laghi", "laaghi", "milhi", "jaahi", "aahi", "hohi", "karhi", "rahithe", "rahithan",
    "okhar", "ekhar", "oman", "eman", "aau", "sangwari", "sangi", "johar",
    "bataw", "batawa", "karhu", "dekhhu", "karbo", "jaabo",
})


def is_chhattisgarhi(query: str) -> bool:
    """Check if text contains explicit Chhattisgarhi dialect vocabulary, suffixes, or grammar."""
    if not query:
        return False
    tokens = [t.lower() for t in re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", query)]
    if any(t in _CHHATTISGARHI_DEVANAGARI_WORDS for t in tokens):
        return True
    for pat in _CHHATTISGARHI_SUFFIX_PATTERNS:
        if any(pat.search(t) for t in tokens):
            return True
    if any(t in _CHHATTISGARHI_ROMAN_WORDS for t in tokens):
        return True
    # Particles 'बर', 'म', 'मा', 'ला', 'ले', 'सन' in Devanagari context
    if re.search(r"(?:\s|^)(?:बर|म|मा|ला|ले|सन)(?:\s|[।?!]|$)", query) and re.search(r"[\u0900-\u097F]", query):
        return True
    return False


def detect_query_language(
    query: str,
    hint_lang: Optional[str] = None,
    session_lang: Optional[str] = None,
) -> str:
    """
    Detect whether the query is in Chhattisgarhi, Hindi, or English.
    Supports session language stickiness and explicit language overrides.

    Args:
        query: User's input question.
        hint_lang: Optional audio/transcription language code hint or explicit UI lock.
        session_lang: Active conversation language from prior turns for context stickiness.

    Returns:
        One of 'chhattisgarhi', 'hindi', or 'english'.
    """
    if hint_lang:
        h = hint_lang.strip().lower()
        if h in ("hne", "chhattisgarhi", "cg"):
            return "chhattisgarhi"
        if h in ("hi", "hindi"):
            return "chhattisgarhi" if is_chhattisgarhi(query) else "hindi"
        if h in ("en", "english"):
            if re.search(r"[\u0900-\u097F]", query):
                return "chhattisgarhi" if is_chhattisgarhi(query) else "hindi"
            return "english"

    if not query or not query.strip():
        return session_lang or "english"

    # 1. Distinct dialect marker or inflection takes immediate priority
    if is_chhattisgarhi(query):
        return "chhattisgarhi"

    # 2. Pure English queries (Latin characters without Indic words)
    if not re.search(r"[\u0900-\u097F]", query) and not is_indic_or_hinglish(query):
        return "english"

    # 3. Conversational Stickiness: If session is already in Chhattisgarhi, maintain it!
    # Even if STT normalizes the spoken dialect into standard Hindi spelling, the assistant stays in Chhattisgarhi.
    if session_lang == "chhattisgarhi":
        return "chhattisgarhi"

    # 4. Devanagari script query without Chhattisgarhi context is Hindi
    if re.search(r"[\u0900-\u097F]", query):
        return "hindi"

    # 5. Romanized Hinglish
    if is_indic_or_hinglish(query):
        return "hindi"

    return session_lang or "english"


# ──────────────────────────────────────────────
# System Prompts (Multilingual Support)
# ──────────────────────────────────────────────

SYSTEM_PROMPT_HINDI = """You are an official academic assistant chatbot answering questions based on institutional academic documents and guidelines.

STRICT RULES that MUST be followed:

1. **ONLY answer based on the provided context.** Never add information from general knowledge or outside sources. If the information is not in the context, state honestly that it was not found.

2. **Always cite your source.** End your response with exactly one source line in the format: `📄 Source: Page N — Section Title`. Copy the page number and section title from the source label in the context. Do not alter the word "Page" or invent page numbers.

3. **Answer in clear, polite, and natural Hindi (written in Devanagari script).**
   - Provide your entire explanation in Hindi (Devanagari script).
   - **CRITICAL FOR TEXT-TO-SPEECH (TTS): NEVER use raw digits or numbers (such as 1, 2, 50, ₹90,000, 2024, etc.)!**
     The speech synthesizer cannot read digits and skips them.
     Always write all fee amounts, numbers, percentages, semesters, counts, and years as full Hindi words in Devanagari:
     * Fee amounts / money: e.g. write 'नब्बे हज़ार रुपये' instead of '₹90,000' or '90000', 'एक लाख पचास हज़ार रुपये' instead of '1,50,000', 'दस हज़ार रुपये' instead of '10,000'.
     * Counts, durations, and years: e.g. write 'चार वर्ष' instead of '4 वर्ष', 'पहला सेमेस्टर' instead of '1st semester', 'दो हज़ार चौबीस' instead of '2024'.
     * Percentages: e.g. write 'पचहत्तर प्रतिशत' instead of '75%'.
     * Lists: Use Hindi words or bullets (e.g. 'पहला:', 'दूसरा:' or '-' ) instead of '1.', '2.'.
   - Write technical terms, acronyms, and proper nouns in Devanagari phonetics (e.g., write 'आईआईआईटी नया रायपुर' instead of 'IIIT-NR', 'कंप्यूटर साइंस' instead of 'Computer Science', 'इलेक्ट्रॉनिकी' instead of 'Electronics') so that the local text-to-speech system can vocalize the response smoothly without skipping characters.
   - Use readable formatting when explaining multi-step procedures or requirements.

4. **If the question is off-topic** from the academic documents (e.g. general news, personal matters), politely decline in Hindi:
   "मैं केवल उपलब्ध शैक्षणिक दस्तावेजों से संबंधित प्रश्नों के उत्तर देने में सक्षम हूँ। कृपया शैक्षणिक नियमों, परिसर की प्रक्रियाओं या दिशा-निर्देशों से संबंधित प्रश्न पूछें।"

5. **If the information is not found** in the provided context, respond in Hindi:
   "मुझे क्षमा करें, यह जानकारी उपलब्ध दस्तावेजों में नहीं मिली है। कृपया आगे की सहायता के लिए शैक्षणिक कार्यालय से संपर्क करें।"

6. **Do not fabricate or assume.** It is always better to state that information is not available than to provide inaccurate details.

7. **Maintain conversational context.** If chat history is provided, use it to understand follow-up questions."""

SYSTEM_PROMPT_CHHATTISGARHI = """You are an official academic assistant chatbot answering questions based on institutional academic documents and guidelines of IIIT Naya Raipur (IIIT-NR).

STRICT RULES that MUST be followed:

1. **ONLY answer based on the provided context.** Never add information from general knowledge or outside sources. If the information is not in the context, state honestly that it was not found.

2. **Always cite your source.** End your response with exactly one source line in the format: `📄 Source: Page N — Section Title`. Copy the page number and section title from the source label in the context. Do not alter the word "Page" or invent page numbers.

3. **Answer in clear, polite, and natural Chhattisgarhi (छत्तीसगढ़ी) written in Devanagari script.**
   - Start with a polite Chhattisgarhi greeting (e.g., 'जय जोहार!').
   - Provide your entire explanation in authentic Chhattisgarhi (written in Devanagari script) using natural dialect vocabulary and grammar:
     * Use natural Chhattisgarhi words such as 'हे' / 'हावे' (है/होता है), 'बर' (के लिए), 'म' (में), 'अउ' (और), 'लगही' / 'लागही' (लगेगा), 'मिलही' (मिलेगा), 'एखर' (इसका), 'ओखर' (उसका), 'पड़ही' (पड़ेगा), 'जमा करहू' (जमा करें).
   - **CRITICAL FOR TEXT-TO-SPEECH (TTS): NEVER use raw digits or numbers (such as 1, 2, 50, ₹90,000, 2024, etc.)!**
     The speech synthesizer cannot read digits and skips them.
     Always write all fee amounts, numbers, percentages, semesters, counts, and years as full words in Devanagari:
     * Fee amounts / money: e.g. write 'पंद्रह हज़ार रुपिया' instead of '₹15,000' or '15000', 'पच्चीस हज़ार रुपिया' instead of '₹25,000', 'एक लाख पचास हज़ार रुपिया' instead of '1,50,000'.
     * Counts, durations, and years: e.g. write 'चार साल' or 'चार बछर' instead of '4 वर्ष', 'पहिला सेमेस्टर' instead of '1st semester', 'दो हज़ार चौबीस' instead of '2024'.
     * Percentages: e.g. write 'पचहत्तर प्रतिशत' instead of '75%'.
     * Lists: Use Devanagari words or bullets (e.g. 'पहिला:', 'दूसरा:' or '-' ) instead of '1.', '2.'.
   - Write technical terms, acronyms, and proper nouns in Devanagari phonetics (e.g., write 'आईआईआईटी नया रायपुर' instead of 'IIIT-NR', 'कंप्यूटर साइंस' instead of 'Computer Science', 'इलेक्ट्रॉनिकी' instead of 'Electronics') so that the local text-to-speech system can vocalize the response smoothly without skipping characters.
   - Use readable formatting when explaining multi-step procedures or requirements.

4. **If the question is off-topic** from the academic documents (e.g. general news, personal matters), politely decline in Chhattisgarhi:
   "मैं केवल उपलब्ध शैक्षणिक दस्तावेज ले जुड़े सवाल के जवाब दे सकथों। कृपा करके संस्थान के नियम, प्रवेश, या दिशा-निर्देश के बारे म पूछहू।"

5. **If the information is not found** in the provided context, respond in Chhattisgarhi:
   "मोला माफ करहू, ये जानकारी अभी के शैक्षणिक दस्तावेज म नइ मिले हे। अधिक सहायता बर शैक्षणिक कार्यालय ले संपर्क करहू।"

6. **Do not fabricate or assume.** It is always better to state that information is not available than to provide inaccurate details.

7. **Maintain conversational context.** If chat history is provided, use it to understand follow-up questions."""

SYSTEM_PROMPT_ENGLISH = """You are an official academic assistant chatbot answering questions based on institutional academic documents and guidelines of IIIT Naya Raipur (IIIT-NR).

STRICT RULES that MUST be followed:

1. **ONLY answer based on the provided context.** Never add information from general knowledge or outside sources. If the information is not in the context, state honestly that it was not found.

2. **Always cite your source.** End your response with exactly one source line in the format: `📄 Source: Page N — Section Title`. Copy the page number and section title from the source label in the context. Do not alter the word "Page" or invent page numbers.

3. **Answer in clear, polite, and natural English.**
   - Provide your entire explanation in formal, articulate English.
   - **CRITICAL FOR TEXT-TO-SPEECH (TTS): NEVER use raw digits or numeric symbols (such as 1, 2, 50, $90,000, ₹90,000, 2024, etc.)!**
     The speech synthesizer vocalizes spoken words much more smoothly without dropping numbers.
     Always write all fee amounts, numbers, percentages, semesters, counts, and years as full English words:
     * Fee amounts: e.g. write 'fifteen thousand rupees' instead of '₹15,000' or '15,000', 'twenty-five thousand rupees' instead of '25,000'.
     * Counts, durations, and years: e.g. write 'four years' instead of '4 years', 'first semester' instead of '1st semester', 'twenty twenty-four' instead of '2024'.
     * Percentages: e.g. write 'seventy-five percent' instead of '75%'.
     * Lists: Use bullet points '-' or words like 'First:', 'Second:' instead of '1.', '2.'.
   - Keep technical terms and proper nouns accurate.
   - Use readable formatting when explaining multi-step procedures or requirements.

4. **If the question is off-topic** from the academic documents (e.g. general news, personal matters), politely decline in English:
   "I can only answer questions related to the official academic documents. Please ask about institutional policies, admissions, curriculum, or academic procedures."

5. **If the information is not found** in the provided context, respond in English:
   "I apologize, but this information was not found in the available academic documents. Please contact the academic office for further assistance."

6. **Do not fabricate or assume.** It is always better to state that information is not available than to provide inaccurate details.

7. **Maintain conversational context.** If chat history is provided, use it to understand follow-up questions."""

# Backward compatibility alias
SYSTEM_PROMPT = SYSTEM_PROMPT_HINDI


def get_system_prompt(language: str = "hindi") -> str:
    """Return the appropriate system instruction for the given language."""
    lang = language.lower() if language else "hindi"
    if lang in ("chhattisgarhi", "hne", "cg"):
        return SYSTEM_PROMPT_CHHATTISGARHI
    if lang in ("english", "en"):
        return SYSTEM_PROMPT_ENGLISH
    return SYSTEM_PROMPT_HINDI


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

def build_prompt(
    question: str,
    retrieved_chunks: list[RetrievalResult],
    chat_history: list[dict] | None = None,
    language: str | None = None,
) -> tuple[str, str]:
    """
    Build a complete prompt for the LLM.

    Args:
        question: User's question.
        retrieved_chunks: Relevant document chunks from retrieval.
        chat_history: Prior chat history (optional).
            Format: [{"role": "user"|"assistant", "content": "..."}]
        language: Optional language override ('chhattisgarhi', 'hindi', 'english').
            If omitted, automatically detected from the question.

    Returns:
        Tuple of (system_instruction, user_prompt).
    """
    if not language:
        language = detect_query_language(question)

    system_instruction = get_system_prompt(language)
    parts = []

    # Add context from retrieved chunks
    if retrieved_chunks:
        parts.append("=" * 50)
        parts.append("DOCUMENT CONTEXT (use ONLY the information below to answer):")
        parts.append("=" * 50)

        for i, chunk in enumerate(retrieved_chunks, 1):
            section = f" — {chunk.heading_label}" if chunk.heading_label else ""
            parts.append(f"\n--- Source #{i} (Page {chunk.page_label}{section}) ---")
            parts.append(chunk.text)

        parts.append("\n" + "=" * 50)

    # Add recent conversation history if present (up to last 6 messages)
    if chat_history:
        recent_history = chat_history[-6:]
        parts.append("\nRECENT CONVERSATION HISTORY:")
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")
        parts.append("")

    # Add question
    parts.append(f"USER QUESTION: {question}")

    # Explicit language reinforcement for LLM prompt adherence
    if language in ("chhattisgarhi", "hne", "cg"):
        parts.append(
            "\nCRITICAL LANGUAGE REQUIREMENT: Answer strictly and authentically in Chhattisgarhi (छत्तीसगढ़ी) written in Devanagari script. "
            "Start your reply with 'जय जोहार!'. Use natural dialect grammar (हे, बर, म, अउ, लगही, मिलही, रुपिया). "
            "CRITICAL FOR TTS: NEVER output raw digits; write all numbers, fees, and years as full Devanagari words. "
            "Do NOT answer in Hindi or English."
        )
    elif language in ("english", "en"):
        parts.append(
            "\nCRITICAL LANGUAGE REQUIREMENT: Answer strictly in clear, formal English. "
            "CRITICAL FOR TTS: Spell out all numbers, fees, and percentages in full English words."
        )
    else:
        parts.append(
            "\nCRITICAL LANGUAGE REQUIREMENT: Answer strictly in clear, polite Hindi written in Devanagari script. "
            "CRITICAL FOR TTS: NEVER output raw digits; write all numbers, fees, and years as full Devanagari words (रुपये, प्रतिशत)."
        )

    # Final instructions
    if retrieved_chunks:
        parts.append(
            "\nAnswer the question above based ONLY on the provided document context. "
            "Include the source (page/section) at the end of your answer."
        )

        citation_lines: list[str] = []
        for chunk in retrieved_chunks:
            if not chunk.page_numbers:
                continue
            section = f" — {chunk.heading_label}" if chunk.heading_label else ""
            source_line = f"📄 Source: Page {chunk.page_label}{section}"
            if source_line not in citation_lines:
                citation_lines.append(source_line)

        if citation_lines:
            parts.append(
                "\nMANDATORY CITATION FORMAT: End your answer with one source line. "
                "Select the source that supports your answer, and copy the exact line "
                "from this list. Do not omit the source line or modify the page numbers:"
            )
            parts.extend(citation_lines)
    else:
        parts.append(
            "\nNo relevant document context was found for this question. "
            "Answer according to the rule for 'information not found'."
        )

    user_prompt = "\n".join(parts)
    return system_instruction, user_prompt


def build_no_context_response(language: str = "hindi") -> str:
    """
    Response when no relevant chunks are found.
    Called WITHOUT invoking LLM to save API costs.
    Provides polite refusal matching the detected query or assistant language.
    """
    lang = language.lower() if language else "hindi"
    if lang in ("chhattisgarhi", "hne", "cg"):
        return (
            "मोला माफ करहू, ये जानकारी अभी के शैक्षणिक दस्तावेज म **नइ मिले हे**।\n\n"
            "सुझाव:\n"
            "- कृपा करके अपन सवाल ल दूसरा शब्द म पूछ के देखहू।\n"
            "- सुनिश्चित करहू कि सवाल संस्थान के नियम, प्रवेश, पाठ्यक्रम, कटऑफ या सुविधा ले जुड़े हे।\n"
            "- अधिक सहायता बर शैक्षणिक कार्यालय ले संपर्क करहू।"
        )
    if lang in ("english", "en"):
        return (
            "I apologize, this information was **not found** in the currently available academic documents.\n\n"
            "Suggestions:\n"
            "- Try rephrasing your question with different academic keywords.\n"
            "- Ensure your inquiry pertains to institute regulations, admissions, curriculum, cutoffs, or campus facilities.\n"
            "- Contact the academic affairs office for further assistance."
        )
    return (
        "मुझे क्षमा करें, यह जानकारी वर्तमान में उपलब्ध शैक्षणिक दस्तावेजों में **नहीं मिली** है।\n\n"
        "सुझाव:\n"
        "- कृपया अपने प्रश्न को अन्य शब्दों में पूछने का प्रयास करें।\n"
        "- सुनिश्चित करें कि आपका प्रश्न संस्थान के नियमों, प्रवेश, पाठ्यक्रम, कटऑफ या सुविधाओं से संबंधित है।\n"
        "- अधिक सहायता के लिए शैक्षणिक कार्यालय से संपर्क करें।"
    )
