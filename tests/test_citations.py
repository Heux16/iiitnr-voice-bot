"""Test citation parser and public source formatting."""

from app.citations import (
    citation_pages_are_in_context,
    extract_cited_pages,
    strip_inline_source_lines,
)
from app.retrieval import RetrievalCandidate, format_sources


class TestCitationParsing:
    def test_single_page(self):
        assert extract_cited_pages("Source: Page 39") == [39]
        assert extract_cited_pages("Sumber: Halaman 39") == [39]

    def test_page_range(self):
        assert extract_cited_pages("Source: Page 39-40") == [39, 40]
        assert extract_cited_pages("Sumber: Halaman 39-40") == [39, 40]

    def test_page_list(self):
        assert extract_cited_pages("Page 39, 40 and 41") == [39, 40, 41]
        assert extract_cited_pages("Halaman 39, 40 dan 41") == [39, 40, 41]

    def test_dash_heading_not_parsed_as_page(self):
        answer = "Page 43 — 5. Study load and credit semester"
        assert extract_cited_pages(answer) == [43]
        answer_id = "Halaman 43 — 5. Beban studi dan beban kredit semester"
        assert extract_cited_pages(answer_id) == [43]

    def test_citation_valid_with_context(self):
        contexts = [RetrievalCandidate(chunk_index=0, text="x", page_numbers=[39, 40])]
        assert citation_pages_are_in_context("Page 39-40", contexts) is True
        assert citation_pages_are_in_context("Halaman 39-40", contexts) is True

    def test_citation_outside_context_rejected(self):
        contexts = [RetrievalCandidate(chunk_index=0, text="x", page_numbers=[39])]
        assert citation_pages_are_in_context("Page 40", contexts) is False
        assert citation_pages_are_in_context("Halaman 40", contexts) is False


class TestStripInlineSources:
    def test_strip_source_line(self):
        answer = "Main answer.\n\n📄 Source: Page 39 — Study Leave"
        assert strip_inline_source_lines(answer) == "Main answer."
        answer_id = "Jawaban utama.\n\n📄 Sumber: Halaman 39 — H. Cuti Studi"
        assert strip_inline_source_lines(answer_id) == "Jawaban utama."

    def test_preserve_answer_body(self):
        answer = "Point one.\nPoint two.\n📄 Source: Page 39"
        assert strip_inline_source_lines(answer) == "Point one.\nPoint two."
        answer_id = "Poin pertama.\nPoin kedua.\n📄 Sumber: Halaman 39"
        assert strip_inline_source_lines(answer_id) == "Poin pertama.\nPoin kedua."


class TestCleanSourceCard:
    def _candidate(self, index, pages, score, section):
        return RetrievalCandidate(
            chunk_index=index,
            text="x",
            page_numbers=pages,
            dense_score=score,
            section_title=section,
            heading_path=["PROGRAM", section],
        )

    def test_only_cited_pages_displayed(self):
        results = [
            self._candidate(0, [39], 0.82, "Study Leave"),
            self._candidate(1, [40], 0.81, "Study Leave"),
            self._candidate(2, [107, 108], 0.67, "LIBRARY"),
            self._candidate(3, [42], 0.76, "Lectures"),
        ]
        result = format_sources(
            results,
            cited_pages={39, 40},
            include_scores=False,
        )
        assert "Page 39–40" in result
        assert "LIBRARY" not in result
        assert "Lectures" not in result
        assert "relevance" not in result

    def test_overlap_merged_into_one_line(self):
        results = [
            self._candidate(0, [39], 0.82, "Study Leave"),
            self._candidate(1, [39, 40], 0.90, "Study Leave"),
            self._candidate(2, [40], 0.81, "Study Leave"),
        ]
        result = format_sources(results, cited_pages={39, 40}, include_scores=False)
        assert result.count("Page") == 1
        assert "Page 39–40" in result

    def test_operator_may_view_scores(self):
        result = format_sources(
            [self._candidate(0, [39], 0.82, "Study Leave")],
            cited_pages={39},
            include_scores=True,
        )
        assert "relevance" in result
        assert "82%" in result

    def test_different_documents_remain_separate(self):
        a = self._candidate(0, [39], 0.82, "Study Leave")
        b = self._candidate(1, [39], 0.81, "Decree")
        a.source_document = "guideline.md"
        b.source_document = "decree.md"
        result = format_sources([a, b], cited_pages={39}, include_scores=False)
        assert "guideline" in result
        assert "decree" in result
        assert len(result.splitlines()) == 2
