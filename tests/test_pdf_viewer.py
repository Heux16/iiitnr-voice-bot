"""Test path safety and source references for the embedded document/PDF viewer."""

from pathlib import Path
import pymupdf
import pytest

from app import pdf_viewer


@pytest.fixture(autouse=True)
def setup_test_pdf(tmp_path, monkeypatch):
    """Set up a test PDF file in a mock DATA_DIR so tests don't require external files."""
    test_pdf_name = "test_document.pdf"
    pdf_path = tmp_path / test_pdf_name
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test Document Content")
    doc.save(str(pdf_path))
    doc.close()

    monkeypatch.setattr(pdf_viewer, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pdf_viewer, "DEFAULT_PDF_NAME", test_pdf_name)
    return pdf_path


class TestSafePdfPath:
    def test_default_document_ditemukan(self):
        path = pdf_viewer._safe_pdf_path("")
        assert path is not None
        assert path.suffix.lower() == ".pdf"
        assert path.parent == pdf_viewer.DATA_DIR.resolve()

    def test_dokumen_tercatat_ditemukan(self):
        name = pdf_viewer.DEFAULT_PDF_NAME
        path = pdf_viewer._safe_pdf_path(name)
        assert path is not None
        assert path.name == name

    def test_path_traversal_ditolak(self):
        assert pdf_viewer._safe_pdf_path("..\\.streamlit\\secrets.toml") is None
        assert pdf_viewer._safe_pdf_path("../../.streamlit/secrets.toml") is None

    def test_file_bukan_pdf_ditolak(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pdf_viewer, "DATA_DIR", tmp_path)
        (tmp_path / "rahasia.txt").write_text("secret", encoding="utf-8")
        assert pdf_viewer._safe_pdf_path("rahasia.txt") is None

    def test_halaman_dirender_menjadi_png(self):
        data = pdf_viewer._pdf_page_image(
            str(pdf_viewer.DATA_DIR / pdf_viewer.DEFAULT_PDF_NAME),
            1,
        )
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(data) > 1_000

    def test_key_referensi_stabil(self):
        first = {
            "source_document": "pedoman.pdf",
            "pages": [39, 40],
        }
        second = {
            "source_document": "pedoman.pdf",
            "pages": [39, 40],
        }
        assert pdf_viewer._viewer_reference_key(first) == pdf_viewer._viewer_reference_key(second)
