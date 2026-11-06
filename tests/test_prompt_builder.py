"""test_prompt_builder.py — Tests for multi-language prompt construction and detection."""

import pytest
from app.prompt_builder import (
    build_no_context_response,
    build_prompt,
    detect_query_language,
    SYSTEM_PROMPT_CHHATTISGARHI,
    SYSTEM_PROMPT_ENGLISH,
    SYSTEM_PROMPT_HINDI,
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


@pytest.mark.parametrize("query,expected_lang", [
    ("What is the opening rank of DSAI in 2022?", "en"),
    ("Where is IIIT Naya Raipur campus located?", "en"),
    ("Can I take study leave for two semesters?", "en"),
    ("IIIT Naya Raipur कहाँ स्थित है?", "hi"),
    ("CSE ki opening rank kya hai", "hi"),
    ("Branch sliding ka rule kya hai", "hi"),
    ("Hostel facilities kaisa hai", "hi"),
    ("होस्टल के फीस कतेक लागही?", "hne"),
    ("जय जोहार, सीएसई म एडमिशन कइसे मिलही?", "hne"),
    ("Okhar fees katek hawe", "hne"),
    ("नवा रायपुर मा सीएसई के फीस कतेक हावे", "hne"),
    ("ओखर नियम काबर हावे", "hne"),
])
def test_detect_query_language(query, expected_lang):
    assert detect_query_language(query) == expected_lang


def test_build_prompt_selects_chhattisgarhi():
    context = RetrievalCandidate(
        chunk_index=1,
        text="Hostel fee is 25000 per semester.",
        page_numbers=[10],
        section_title="Fee Structure",
        heading_path=["ADMISSIONS", "Fee Structure"],
    )
    sys_prompt, _ = build_prompt("होस्टल के फीस कतेक लागही?", [context])
    assert "जय जोहार!" in sys_prompt
    assert "CHHATTISGARHI" in sys_prompt


def test_build_prompt_selects_hindi():
    context = RetrievalCandidate(
        chunk_index=1,
        text="Hostel fee is 25000 per semester.",
        page_numbers=[10],
        section_title="Fee Structure",
        heading_path=["ADMISSIONS", "Fee Structure"],
    )
    sys_prompt, _ = build_prompt("होस्टल की फीस कितनी है?", [context])
    assert "नमस्ते!" in sys_prompt
    assert "HINDI" in sys_prompt


def test_build_prompt_selects_english():
    context = RetrievalCandidate(
        chunk_index=1,
        text="Hostel fee is 25000 per semester.",
        page_numbers=[10],
        section_title="Fee Structure",
        heading_path=["ADMISSIONS", "Fee Structure"],
    )
    sys_prompt, _ = build_prompt("What is the hostel fee?", [context])
    assert "ENGLISH" in sys_prompt


def test_build_no_context_response_languages():
    hne_resp = build_no_context_response(language="hne")
    assert "मोला क्षमा करहू" in hne_resp
    assert "नई मिले" in hne_resp

    hi_resp = build_no_context_response(language="hi")
    assert "मुझे क्षमा करें" in hi_resp
    assert "नहीं मिली" in hi_resp

    en_resp = build_no_context_response(language="en")
    assert "I apologize" in en_resp
    assert "not found" in en_resp
