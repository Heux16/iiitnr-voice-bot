"""Answer prompt must provide exact citation format that the model can copy."""

from app.prompt_builder import (
    build_no_context_response,
    build_prompt,
    detect_query_language,
    get_system_prompt,
    is_chhattisgarhi,
)
from app.retrieval import RetrievalCandidate


def test_prompt_provides_exact_source_line_from_context():
    context = RetrievalCandidate(
        chunk_index=1,
        text="Students submit study leave applications to the department.",
        page_numbers=[39, 40, 41],
        section_title="Study Leave",
        heading_path=["ACADEMIC SERVICES", "Study Leave"],
    )

    _system_prompt, user_prompt = build_prompt(
        "What is the procedure for study leave?", [context]
    )

    assert "MANDATORY CITATION FORMAT" in user_prompt
    assert "📄 Source: Page 39-41 — Study Leave" in user_prompt


def test_detect_query_language_chhattisgarhi():
    # User question from screenshot
    assert detect_query_language("छात्रावास के फीस कतेक हे अउ मेस चार्ज कितना लगही?") == "chhattisgarhi"
    # Other Chhattisgarhi dialect queries
    assert detect_query_language("एमटेक बर हॉस्टल के फीस कतेक हावे?") == "chhattisgarhi"
    assert detect_query_language("जय जोहार, सीएसई म एडमिशन कइसे मिलही?") == "chhattisgarhi"
    assert detect_query_language("हॉस्टल के सुविधा बारे म बताव संगी") == "chhattisgarhi"
    assert detect_query_language("hostel ke fees katek he aau mess charge kitna laghi") == "chhattisgarhi"
    # Audio hint override
    assert detect_query_language("What is the fee?", hint_lang="hne") == "chhattisgarhi"


def test_detect_query_language_hindi():
    assert detect_query_language("IIIT Naya Raipur में B.Tech CSE की फीस कितनी है?") == "hindi"
    assert detect_query_language("छात्रावास का शुल्क और मेस चार्ज कितना लगेगा?") == "hindi"
    assert detect_query_language("hostel ki fees kitni hai aur mess charge kitna lagega") == "hindi"


def test_detect_query_language_english():
    assert detect_query_language("What is the hostel fee and mess charges at IIIT Naya Raipur?") == "english"
    assert detect_query_language("What is the procedure for study leave?") == "english"
    assert detect_query_language("What is the fee for the hostel?") == "english"
    assert detect_query_language("What are the eligibility criteria for the program?") == "english"
    assert detect_query_language("Tell me about the campus and the hostels") == "english"
    assert detect_query_language("Can you give me the details of the M.Tech admissions?") == "english"
    assert detect_query_language("Where is the institute located and what is the address?") == "english"
    assert detect_query_language("Hello there, how are you doing?", hint_lang="en") == "english"


def test_build_prompt_chhattisgarhi_system_prompt():
    context = RetrievalCandidate(
        chunk_index=1,
        text="Hostel fee is Rs 15000 per semester and mess fee is Rs 25000.",
        page_numbers=[6],
        section_title="Hostel and Mess Charges",
        heading_path=["FEES", "Hostel"],
    )

    system_prompt, user_prompt = build_prompt(
        "छात्रावास के फीस कतेक हे अउ मेस चार्ज कितना लगही?",
        [context],
    )

    assert "Chhattisgarhi (छत्तीसगढ़ी)" in system_prompt
    assert "जय जोहार" in system_prompt
    assert "रुपिया" in system_prompt
    assert "📄 Source: Page 6 — Hostel and Mess Charges" in user_prompt


def test_build_prompt_language_override():
    context = RetrievalCandidate(
        chunk_index=1,
        text="Sample text",
        page_numbers=[1],
        section_title="General",
        heading_path=[],
    )

    # Force English
    system_prompt, _ = build_prompt("कतेक हे", [context], language="english")
    assert "English" in system_prompt

    # Force Hindi
    system_prompt, _ = build_prompt("How much?", [context], language="hindi")
    assert "Hindi (written in Devanagari script)" in system_prompt


def test_build_no_context_response_multilingual():
    cg_response = build_no_context_response("chhattisgarhi")
    assert "मोला माफ करहू" in cg_response
    assert "नइ मिले हे" in cg_response

    hi_response = build_no_context_response("hindi")
    assert "मुझे क्षमा करें" in hi_response
    assert "नहीं मिली" in hi_response

    en_response = build_no_context_response("english")
    assert "I apologize" in en_response
    assert "not found" in en_response


def test_session_stickiness_and_reinforcement():
    # Hotkey dialect detection
    lang1 = detect_query_language("छात्रावास के फीस कतेक हे अउ मेस चार्ज कितना लगही?")
    assert lang1 == "chhattisgarhi"

    # Session stickiness preserves dialect even when STT produces standard Hindi spelling
    lang2 = detect_query_language("मेस का नियम क्या है?", session_lang=lang1)
    assert lang2 == "chhattisgarhi"

    # Clean switch to English
    lang3 = detect_query_language("What is the eligibility for B.Tech?", session_lang=lang2)
    assert lang3 == "english"

    # Reinforcement in user prompt
    candidate = RetrievalCandidate(
        chunk_index=1,
        text="Hostel fees is Rs 15000",
        page_numbers=[6],
        section_title="Fees",
        heading_path=[],
    )
    _, user_p = build_prompt("छात्रावास के फीस कतेक हे?", [candidate], language="chhattisgarhi")
    assert "CRITICAL LANGUAGE REQUIREMENT: Answer strictly and authentically in Chhattisgarhi" in user_p
    assert "जय जोहार!" in user_p


