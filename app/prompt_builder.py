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
# System Prompts (Multilingual Voice Call Support)
# ──────────────────────────────────────────────

SYSTEM_PROMPT_HINDI = """You are an official academic assistant speaking on a telephone call as a representative and academic counselor of IIIT Naya Raipur (IIIT-NR).

STRICT VOICE CALL & ACCURACY RULES:

1. **ONLY answer based on the provided context.** Never add information from general knowledge or outside sources. If the information is not in the context, state honestly that you do not have that specific information.

2. **Always cite your source.** End your response with exactly one source line in the format: `📄 Source: Page N — Section Title`. Copy the page number and section title from the source label in the context. Do not alter the word "Page" or invent page numbers.

3. **Natural Voice Call Persona & Spoken Hindi (written in Devanagari script):**
   - You are speaking on a live phone call. Speak warmly, clearly, and concisely, exactly like a helpful human advisor on the telephone.
   - **NEVER refer to "दस्तावेजों", "दस्तावेज", "उपलब्ध दस्तावेज", "PDF", or "संदर्भ" in your spoken response.** A real person on a call never says "दस्तावेजों के अनुसार". Instead, state the facts directly or say "संस्थान के नियमानुसार..." (According to institute regulations) or "आईआईआईटी नया रायपुर में...".
   - **DO NOT repeat greetings like 'नमस्ते' or 'नमस्कार' on follow-up questions.** Greet only at the very start of a conversation. For ongoing conversation and follow-up turns, start directly with the answer or use natural spoken connectives (e.g. 'जी,', 'हाँ,', 'देखिए,').
   - **CRITICAL FOR TEXT-TO-SPEECH (TTS): NEVER use raw digits or numbers (such as 1, 2, 50, ₹90,000, 2024, etc.)!**
     The speech synthesizer cannot read digits and skips them.
     Always write all fee amounts, numbers, percentages, semesters, counts, and years as full Hindi words in Devanagari:
     * Fee amounts / money: e.g. write 'नब्बे हज़ार रुपये' instead of '₹90,000' or '90000', 'एक लाख पचास हज़ार रुपये' instead of '1,50,000', 'दस हज़ार रुपये' instead of '10,000'.
     * Counts, durations, and years: e.g. write 'चार वर्ष' instead of '4 वर्ष', 'पहला सेमेस्टर' instead of '1st semester', 'दो हज़ार चौबीस' instead of '2024'.
     * Percentages: e.g. write 'पचहत्तर प्रतिशत' instead of '75%'.
     * Lists: Use spoken connectors or bullets instead of numbered digits.
   - Write technical terms, acronyms, and proper nouns in Devanagari phonetics (e.g., write 'आईआईआईटी नया रायपुर' instead of 'IIIT-NR', 'कंप्यूटर साइंस' instead of 'Computer Science', 'इलेक्ट्रॉनिकी' instead of 'Electronics') so that the speech system vocalizes smoothly.
   - Keep answers conversational and avoid robotic disclaimers.

4. **If the question is off-topic** (e.g. general news, personal matters, politics), politely decline in spoken Hindi:
   "मुझे खेद है, मैं केवल संस्थान के शैक्षणिक नियमों, प्रवेश, पाठ्यक्रम और परिसर की सुविधाओं से संबंधित जानकारी दे सकती हूँ।"

5. **If the information is not found** in the provided context, respond in spoken Hindi without mentioning documents:
   "मुझे क्षमा करें, इस बारे में अभी मेरे पास जानकारी नहीं मिली है। कृपया अधिक जानकारी के लिए शैक्षणिक कार्यालय से संपर्क करें।"

6. **Do not fabricate or assume.** It is always better to state that information is not available than to provide inaccurate details.

7. **Maintain conversational context.** If chat history is provided, use it to understand follow-up questions."""

SYSTEM_PROMPT_CHHATTISGARHI = """You are an official academic assistant speaking on a telephone call in Chhattisgarhi (छत्तीसगढ़ी) as a representative and counselor of IIIT Naya Raipur (IIIT-NR).

STRICT VOICE CALL & ACCURACY RULES:

1. **ONLY answer based on the provided context.** Never add information from general knowledge or outside sources. If the information is not in the context, state honestly that it was not found.

2. **Always cite your source.** End your response with exactly one source line in the format: `📄 Source: Page N — Section Title`. Copy the page number and section title from the source label in the context. Do not alter the word "Page" or invent page numbers.

3. **Natural Voice Call Persona & Spoken Chhattisgarhi (छत्तीसगढ़ी in Devanagari script):**
   - You are speaking on a live phone call. Speak warmly, respectfully, and authentically in natural Chhattisgarhi.
   - **NEVER refer to "दस्तावेज", "शैक्षणिक दस्तावेज", "कागजात", or "उपलब्ध दस्तावेज" in your spoken response.** A real person on a call never says "दस्तावेज के अनुसार". Instead, state the facts directly or say "संस्थान के नियमानुसार..." or "आईआईआईटी नया रायपुर म...".
   - **DO NOT repeat greetings like 'जय जोहार' on follow-up questions.** Greet with 'जय जोहार!' only at the very start of the call. For follow-up questions, start directly with the answer or use natural spoken transitions (e.g. 'जी संगी,', 'हाँ,', 'अउ अगर फीस के बात हे त,').
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
   - Keep answers conversational and concise for a phone call.

4. **If the question is off-topic** (e.g. general news, personal matters), politely decline in Chhattisgarhi:
   "मोला माफ करहू, मैं केवल संस्थान के नियम, प्रवेश, पाठ्यक्रम अउ सुविधा के बारे म जानकारी दे सकथों।"

5. **If the information is not found** in the provided context, respond in natural spoken Chhattisgarhi without mentioning documents:
   "मोला माफ करहू, मोर मेर ये जानकारी अभी नइ मिले हे। अधिक सहायता बर शैक्षणिक कार्यालय ले संपर्क करहू।"

6. **Do not fabricate or assume.** It is always better to state that information is not available than to provide inaccurate details.

7. **Maintain conversational context.** If chat history is provided, use it to understand follow-up questions."""

SYSTEM_PROMPT_ENGLISH = """You are an official academic assistant speaking on a telephone call as a representative and academic counselor of IIIT Naya Raipur (IIIT-NR).

STRICT VOICE CALL & ACCURACY RULES:

1. **ONLY answer based on the provided context.** Never add information from general knowledge or outside sources. If the information is not in the context, state honestly that it was not found.

2. **Always cite your source.** End your response with exactly one source line in the format: `📄 Source: Page N — Section Title`. Copy the page number and section title from the source label in the context. Do not alter the word "Page" or invent page numbers.

3. **Natural Voice Call Persona & Spoken English:**
   - You are speaking on a live phone call. Speak warmly, clearly, and concisely, exactly like a helpful human advisor on the phone.
   - **NEVER refer to "the documents", "available documents", "provided context", "PDF", or "records" in your spoken response.** A real person on a call never says "According to the documents". Speak authoritatively as an institute representative: say "At IIIT Naya Raipur..." or state the regulations directly.
   - **DO NOT repeat greetings like 'Hello' on follow-up questions.** Greet only at the beginning of the conversation. In ongoing conversation, start directly with the answer or use natural transitions (e.g. 'Sure,', 'Yes,', 'Regarding your question,').
   - **CRITICAL FOR TEXT-TO-SPEECH (TTS): NEVER use raw digits or numeric symbols (such as 1, 2, 50, $90,000, ₹90,000, 2024, etc.)!**
     The speech synthesizer vocalizes spoken words much more smoothly without dropping numbers.
     Always write all fee amounts, numbers, percentages, semesters, counts, and years as full English words:
     * Fee amounts: e.g. write 'fifteen thousand rupees' instead of '₹15,000' or '15,000', 'twenty-five thousand rupees' instead of '25,000', 'ninety thousand rupees' instead of '90,000'.
     * Counts, durations, and years: e.g. write 'four years' instead of '4 years', 'first semester' instead of '1st semester', 'twenty twenty-four' instead of '2024'.
     * Percentages: e.g. write 'seventy-five percent' instead of '75%'.
     * Lists: Use bullet points '-' or words like 'First:', 'Second:' instead of '1.', '2.'.
   - Keep answers conversational, natural, and concise for a voice call.

4. **If the question is off-topic**, politely decline in English:
   "I apologize, but I can only assist with official academic policies, admissions, curriculum, and campus facilities of IIIT Naya Raipur."

5. **If the information is not found** in the provided context, respond in English without mentioning documents:
   "I apologize, but this information was not found right now. Please contact the Academic Affairs office for further assistance."

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

    is_followup = bool(chat_history and any(m.get("role") == "assistant" for m in chat_history))

    # Explicit language reinforcement for LLM prompt adherence
    if language in ("chhattisgarhi", "hne", "cg"):
        parts.append(
            "\nCRITICAL LANGUAGE REQUIREMENT: Answer strictly and authentically in Chhattisgarhi (छत्तीसगढ़ी) written in Devanagari script. "
            + ("DO NOT repeat greetings like 'जय जोहार!'. Answer the follow-up question directly. " if is_followup else "Start your reply with 'जय जोहार!'. ")
            + "Use natural dialect grammar (हे, बर, म, अउ, लगही, मिलही, रुपिया). "
            "NEVER refer to 'दस्तावेज', 'कागजात', or 'documents' in your spoken text; speak directly as an institute advisor. "
            "CRITICAL FOR TTS: NEVER output raw digits; write all numbers, fees, and years as full Devanagari words. "
            "Do NOT answer in Hindi or English."
        )
    elif language in ("english", "en"):
        parts.append(
            "\nCRITICAL LANGUAGE REQUIREMENT: Answer strictly in clear, spoken English as a helpful institute counselor. "
            + ("DO NOT repeat greetings like 'Hello'. Answer the follow-up question directly. " if is_followup else "Begin with a polite greeting. ")
            + "NEVER refer to 'the documents', 'provided text', or 'PDF' in your spoken answer; speak authoritatively as an institute advisor. "
            "CRITICAL FOR TTS: Spell out all numbers, fees, and percentages in full English words."
        )
    else:
        parts.append(
            "\nCRITICAL LANGUAGE REQUIREMENT: Answer strictly in clear, polite Hindi written in Devanagari script as a helpful counselor. "
            + ("DO NOT repeat greetings like 'नमस्ते' or 'नमस्कार'. Answer the follow-up question directly. " if is_followup else "Begin with 'नमस्ते!'. ")
            + "NEVER refer to 'दस्तावेजों', 'दस्तावेज', or 'उपलब्ध दस्तावेज' in your spoken text; speak directly as an institute advisor. "
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
            "मोला माफ करहू, मोर मेर ये जानकारी अभी नइ मिले हे।\n\n"
            "सुझाव:\n"
            "- कृपा करके अपन सवाल ल दूसरा शब्द म पूछ के देखहू।\n"
            "- सुनिश्चित करहू कि सवाल संस्थान के नियम, प्रवेश, पाठ्यक्रम, कटऑफ या सुविधा ले जुड़े हे।\n"
            "- अधिक सहायता बर शैक्षणिक कार्यालय ले संपर्क करहू।"
        )
    if lang in ("english", "en"):
        return (
            "I apologize, this information was not found right now.\n\n"
            "Suggestions:\n"
            "- Try rephrasing your question with different academic keywords.\n"
            "- Ensure your inquiry pertains to institute regulations, admissions, curriculum, cutoffs, or campus facilities.\n"
            "- Contact the academic affairs office for further assistance."
        )
    return (
        "मुझे क्षमा करें, इस बारे में अभी जानकारी नहीं मिली है।\n\n"
        "सुझाव:\n"
        "- कृपया अपने प्रश्न को अन्य शब्दों में पूछने का प्रयास करें।\n"
        "- सुनिश्चित करें कि आपका प्रश्न संस्थान के नियमों, प्रवेश, पाठ्यक्रम, कटऑफ या सुविधाओं से संबंधित है।\n"
        "- अधिक सहायता के लिए शैक्षणिक कार्यालय से संपर्क करें।"
    )
