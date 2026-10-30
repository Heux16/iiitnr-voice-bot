"""
prompt_builder.py — Construct prompts for LLM.

Combines system instructions, context from retrieved chunks,
and user question into a structured prompt.
"""

from app.retrieval import RetrievalResult


# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an official academic assistant chatbot answering questions based on institutional academic documents and guidelines.

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


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

def build_prompt(
    question: str,
    retrieved_chunks: list[RetrievalResult],
    chat_history: list[dict] | None = None,
) -> tuple[str, str]:
    """
    Build a complete prompt for the LLM.

    Args:
        question: User's question.
        retrieved_chunks: Relevant document chunks from retrieval.
        chat_history: Prior chat history (optional).
            Format: [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        Tuple of (system_instruction, user_prompt).
    """
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
    return SYSTEM_PROMPT, user_prompt


def build_no_context_response() -> str:
    """
    Response when no relevant chunks are found.
    Called WITHOUT invoking LLM to save API costs.
    """
    return (
        "I apologize, but that information is **not found** "
        "in the currently available documents.\n\n"
        "Suggestions:\n"
        "- Try rephrasing your question with different keywords.\n"
        "- Ensure your question relates to the contents of the academic guidelines.\n"
        "- Contact the academic office for details not covered in these documents."
    )
