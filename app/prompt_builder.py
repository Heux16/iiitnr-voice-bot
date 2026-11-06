"""
prompt_builder.py — Construct prompts for LLM with multi-language adaptability.

Supports:
- Chhattisgarhi (hne): Authentic dialect vocabulary, 'जय जोहार!' greeting, TTS-friendly number words.
- Hindi (hi): Fluent polite Hindi, 'नमस्ते!' greeting, Devanagari words for TTS.
- English (en): Clear, professional academic English with standard citations.

Combines system instructions, context from retrieved chunks,
and user question into a structured prompt.
"""

from __future__ import annotations

import re
from app.retrieval import RetrievalResult


# ──────────────────────────────────────────────
# Language Detection
# ──────────────────────────────────────────────

#: Words strongly indicating Chhattisgarhi dialect
_CHHATTISGARHI_TOKENS = frozenset({
    # Devanagari dialect-specific words
    "कतेक", "हावे", "हवय", "हेवे", "काबर", "काबरके", "कइसे", "कइसन", "ओखर", "एखर",
    "ओमन", "एमन", "अउ", "लागही", "मिलही", "जाही", "आही", "हमार", "तुंहर", "तोहार",
    "जोहार", "बतावा", "सकही", "करहू", "रहय", "रहिथे", "रहिथें", "नईखे", "अइसन", "तइसन",
    # Romanized dialect words
    "katek", "hawe", "haway", "heve", "kabar", "kaabar", "kaisan", "kisan",
    "okhar", "ekhar", "oman", "eman", "laghi", "milhi", "batawa", "tahar", "hamar", "johar"
})

#: Words strongly indicating Hinglish (Romanized Hindi)
_HINDI_STRONG_TOKENS = frozenset({
    "kya", "kyu", "kyon", "kitna", "kitni", "kitne", "kahan", "kaun", "kaunsa", "kaunsi",
    "chahiye", "milega", "milegi", "milenge", "hoga", "hogi", "honge",
    "iska", "iski", "iske", "uska", "uski", "uske", "inka", "unka", "batao", "bataiye",
    "kaisa", "kaisi", "kaise", "batado"
})


def detect_query_language(query: str) -> str:
    """
    Detect whether the user's query is:
    - 'hne' (Chhattisgarhi)
    - 'hi'  (Hindi / Hinglish)
    - 'en'  (English)
    """
    if not query or not query.strip():
        return "hi"

    lowered = query.lower()
    tokens = set(re.findall(r"\b[a-zA-Z\u0900-\u097F]+\b", lowered))

    # 1. Distinctive Chhattisgarhi words take priority
    if any(t in _CHHATTISGARHI_TOKENS for t in tokens):
        return "hne"

    # 2. Devanagari text is Hindi (if no Chhattisgarhi markers present)
    if re.search(r"[\u0900-\u097F]", query):
        return "hi"

    # 3. Hinglish (Romanized Hindi)
    if any(t in _HINDI_STRONG_TOKENS for t in tokens):
        return "hi"

    # 4. Default to English for Latin text without Indic words
    return "en"


# ──────────────────────────────────────────────
# System Prompts by Language
# ──────────────────────────────────────────────

SYSTEM_PROMPT_CHHATTISGARHI = """You are the official academic assistant chatbot for Dr. SPM IIIT Naya Raipur.
The user is speaking or asking in CHHATTISGARHI (छत्तीसगढ़ी).

MANDATORY RULES:
1. **ONLY answer based on the provided document context.** Never fabricate or assume information. If information is not in the context, state honestly that it was not found.
2. **Answer in authentic, polite, and natural Chhattisgarhi written in Devanagari script.**
   - Start with a warm Chhattisgarhi greeting: "जय जोहार!"
   - Use authentic Chhattisgarhi dialect words and grammar:
     * Use "हावे" / "हवय" (है / हैं), "ओखर" (उसका / उसकी), "एखर" (इसका / इसकी), "कतेक" (कितना / कितनी), "काबर" (क्यों), "बर" (के लिए), "अउ" (और), "लागही" (लगेगा / आवश्यक है), "मिलही" (मिलेगा / प्राप्त होगा), "जाही" (जाएगा), "करहू" (करें).
     * Example: "जय जोहार! नवा रायपुर मा बीटेक सीएसई के फीस..."
3. **CRITICAL FOR TEXT-TO-SPEECH (TTS): NEVER use raw digits or numbers (such as 1, 2, 50, ₹90,000, 2024, etc.)!**
   The speech synthesizer cannot read digits and skips them.
   Always write all fee amounts, numbers, percentages, semesters, counts, and years as full words in Devanagari:
   - Fee amounts / money: e.g. write 'नब्बे हज़ार रूपिया' instead of '₹90,000' or '90000', 'एक लाख पचास हज़ार रूपिया' instead of '1,50,000'.
   - Counts, durations, and years: e.g. write 'चार बछर' instead of '4 वर्ष', 'पहिला सेमेस्टर' instead of '1st semester', 'दो हज़ार बाईस' instead of '2022'.
   - Percentages: e.g. write 'पचहत्तर प्रतिशत' instead of '75%'.
   - Lists: Use bullets ('-') or words ('पहिला:', 'दूसरा:') instead of '1.', '2.'.
4. **Always cite your source.** End your response with exactly one source line in the format: `📄 Source: Page N — Section Title`. Copy the page number and section title from the source label in the context.
5. **If the information is not found** in the provided context, respond in Chhattisgarhi:
   "मोला क्षमा करहू, ये जानकारी उपलब्ध शैक्षणिक दस्तावेज मा नई मिले हावे। कृपया आगू सहायता बर शैक्षणिक कार्यालय ले संपर्क करहू।"
6. **If the question is off-topic**, politely decline in Chhattisgarhi:
   "मैं केवल IIIT नया रायपुर के शैक्षणिक दस्तावेज अउ नियम ले जुड़े सवाल के जवाब दे सकथंव।"
7. **Maintain conversational context** across follow-up questions."""

SYSTEM_PROMPT_HINDI = """You are the official academic assistant chatbot for Dr. SPM IIIT Naya Raipur.
The user is speaking or asking in HINDI (हिन्दी).

MANDATORY RULES:
1. **ONLY answer based on the provided document context.** Never fabricate or assume information. If information is not in the context, state honestly that it was not found.
2. **Answer in clear, polite, and natural Hindi written in Devanagari script.**
   - Start with a polite greeting: "नमस्ते!"
3. **CRITICAL FOR TEXT-TO-SPEECH (TTS): NEVER use raw digits or numbers (such as 1, 2, 50, ₹90,000, 2024, etc.)!**
   The speech synthesizer cannot read digits and skips them.
   Always write all fee amounts, numbers, percentages, semesters, counts, and years as full Hindi words in Devanagari:
   - Fee amounts / money: e.g. write 'नब्बे हज़ार रुपये' instead of '₹90,000' or '90000', 'एक लाख पचास हज़ार रुपये' instead of '1,50,000', 'दस हज़ार रुपये' instead of '10,000'.
   - Counts, durations, and years: e.g. write 'चार वर्ष' instead of '4 वर्ष', 'पहला सेमेस्टर' instead of '1st semester', 'दो हज़ार बाईस' instead of '2022'.
   - Percentages: e.g. write 'पचहत्तर प्रतिशत' instead of '75%'.
   - Lists: Use bullets ('-') or words ('पहला:', 'दूसरा:') instead of '1.', '2.'.
   - Write technical terms and acronyms in Devanagari phonetics (e.g. 'आईआईआईटी नया रायपुर', 'कंप्यूटर साइंस', 'इलेक्ट्रॉनिक्स') so TTS vocalizes smoothly.
4. **Always cite your source.** End your response with exactly one source line in the format: `📄 Source: Page N — Section Title`. Copy the page number and section title from the source label in the context.
5. **If the information is not found** in the provided context, respond in Hindi:
   "मुझे क्षमा करें, यह जानकारी उपलब्ध दस्तावेजों में नहीं मिली है। कृपया आगे की सहायता के लिए शैक्षणिक कार्यालय से संपर्क करें।"
6. **If the question is off-topic**, politely decline in Hindi:
   "मैं केवल उपलब्ध शैक्षणिक दस्तावेजों से संबंधित प्रश्नों के उत्तर देने में सक्षम हूँ। कृपया शैक्षणिक नियमों, परिसर की प्रक्रियाओं या दिशा-निर्देशों से संबंधित प्रश्न पूछें।"
7. **Maintain conversational context** across follow-up questions."""

SYSTEM_PROMPT_ENGLISH = """You are the official academic assistant chatbot for Dr. SPM IIIT Naya Raipur.
The user is speaking or asking in ENGLISH.

MANDATORY RULES:
1. **ONLY answer based on the provided document context.** Never fabricate or assume information. If information is not in the context, state honestly that it was not found.
2. **Answer in clear, concise, and professional academic English.**
3. **For smooth speech synthesis (TTS)**, write monetary amounts and years in words alongside figures where helpful (e.g. "ninety thousand rupees (₹90,000)", "two thousand twenty-two (2022)").
4. **Always cite your source.** End your response with exactly one source line in the format: `📄 Source: Page N — Section Title`. Copy the page number and section title from the source label in the context.
5. **If the information is not found** in the provided context, respond in English:
   "I apologize, but this information is not found in the currently available academic documents. Please contact the academic office for further assistance."
6. **If the question is off-topic**, politely decline in English:
   "I can only assist with questions related to the official academic regulations, curriculum, admissions, and campus guidelines of IIIT Naya Raipur."
7. **Maintain conversational context** across follow-up questions."""

#: Default backward-compatible system prompt
SYSTEM_PROMPT = SYSTEM_PROMPT_HINDI


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

def build_prompt(
    question: str,
    retrieved_chunks: list[RetrievalResult],
    chat_history: list[dict] | None = None,
    target_language: str | None = None,
) -> tuple[str, str]:
    """
    Build a complete prompt for the LLM adapting to the user's language.

    Args:
        question: User's question.
        retrieved_chunks: Relevant document chunks from retrieval.
        chat_history: Prior chat history (optional).
        target_language: Optional override for response language ('hne', 'hi', 'en').
            If None, language is automatically detected from question.

    Returns:
        Tuple of (system_instruction, user_prompt).
    """
    lang = target_language or detect_query_language(question)

    if lang == "hne":
        system_instruction = SYSTEM_PROMPT_CHHATTISGARHI
    elif lang == "en":
        system_instruction = SYSTEM_PROMPT_ENGLISH
    else:
        system_instruction = SYSTEM_PROMPT_HINDI

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


def build_no_context_response(language: str = "hi") -> str:
    """
    Response when no relevant chunks are found.
    Called WITHOUT invoking LLM to save API costs.
    Matches the user's language (Chhattisgarhi, Hindi, or English).
    """
    if language == "hne":
        return (
            "मोला क्षमा करहू, ये जानकारी वर्तमान मा उपलब्ध शैक्षणिक दस्तावेज मा **नई मिले** हावे।\n\n"
            "सुझाव:\n"
            "- कृपया अपन सवाल ला दूसर शब्द मा पूछ के देखहू।\n"
            "- देख लेहू कि सवाल संस्थान के नियम, प्रवेश, पाठ्यक्रम या सुविधा ले जुड़े हावे।\n"
            "- जादा जानकारी बर शैक्षणिक कार्यालय ले संपर्क करहू।"
        )
    elif language == "en":
        return (
            "I apologize, but that information is **not found** in the currently available academic documents.\n\n"
            "Suggestions:\n"
            "- Try rephrasing your question with different keywords.\n"
            "- Ensure your question relates to institute regulations, admissions, curriculum, cutoffs, or campus facilities.\n"
            "- Contact the academic office for details not covered in these documents."
        )
    else:
        return (
            "मुझे क्षमा करें, यह जानकारी वर्तमान में उपलब्ध शैक्षणिक दस्तावेजों में **नहीं मिली** है।\n\n"
            "सुझाव:\n"
            "- कृपया अपने प्रश्न को अन्य शब्दों में पूछने का प्रयास करें।\n"
            "- सुनिश्चित करें कि आपका प्रश्न संस्थान के नियमों, प्रवेश, पाठ्यक्रम, कटऑफ या सुविधाओं से संबंधित है।\n"
            "- अधिक सहायता के लिए शैक्षणिक कार्यालय से संपर्क करें।"
        )
