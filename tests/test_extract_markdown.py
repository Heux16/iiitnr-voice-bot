"""test_extract_markdown.py — Pengujian ekstraksi dokumen Markdown."""

from __future__ import annotations

from pathlib import Path
import pytest

from ingestion.extract_markdown import extract_markdown_with_report
from ingestion.build_index import discover_documents


class TestExtractMarkdown:
    def test_ekstraksi_heading_dan_hierarki(self, tmp_path):
        md_content = """# Dokumen Panduan

## Bagian 1: Registrasi
Mahasiswa wajib melakukan registrasi pada setiap awal semester.
Syarat registrasi meliputi pembayaran SPP dan pengisian KRS.

### Sub-bagian 1.1: Batas Waktu
Batas waktu registrasi adalah dua minggu sebelum perkuliahan dimulai.

## Bagian 2: Cuti Akademik
Mahasiswa yang berhalangan dapat mengajukan cuti studi paling lama 2 semester.
"""
        doc = tmp_path / "panduan.md"
        doc.write_text(md_content, encoding="utf-8")

        chunks, report = extract_markdown_with_report(doc)

        assert len(chunks) >= 2
        assert report.lines_raw > 0
        assert report.lines_kept > 0
        assert report.pages_with_text == len(chunks)

        # Cek chunk pertama
        first = chunks[0]
        assert "Dokumen Panduan" in first.heading_path
        assert "Bagian 1: Registrasi" in first.heading_path
        assert "registrasi" in first.text.lower()
        assert first.source_document == "panduan.md"
        assert first.page_numbers == [1]

        # Cek chunk sub-bagian
        sub = next((c for c in chunks if "Batas Waktu" in c.section_title), None)
        assert sub is not None
        assert "Dokumen Panduan" in sub.heading_path
        assert "Bagian 1: Registrasi" in sub.heading_path
        assert "Sub-bagian 1.1: Batas Waktu" in sub.heading_path

    def test_dokumen_pendek_tetap_dihasilkan(self, tmp_path):
        md_content = "# Pengumuman Singkat\nKuliah diliburkan hari ini."
        doc = tmp_path / "singkat.md"
        doc.write_text(md_content, encoding="utf-8")

        chunks, report = extract_markdown_with_report(doc)
        assert len(chunks) == 1
        assert "Kuliah diliburkan" in chunks[0].text
        assert chunks[0].page_numbers == [1]


class TestDiscoverDocumentsMarkdown:
    def test_menemukan_pdf_dan_markdown(self, tmp_path):
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.md").touch()
        (tmp_path / "c.markdown").touch()
        (tmp_path / "d.txt").touch()
        (tmp_path / "README.md").touch()

        found = discover_documents(None, data_dir=tmp_path)
        names = [f.name for f in found]

        assert "a.pdf" in names
        assert "b.md" in names
        assert "c.markdown" in names
        assert "d.txt" not in names
        # README.md folder diabaikan agar tidak terindeks sebagai dokumen akademik
        assert "README.md" not in names

