"""Answer prompt must provide exact citation format that the model can copy."""

from app.prompt_builder import build_prompt
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
