"""
extract_markdown.py — Ekstrak teks dan hierarki bagian dari dokumen Markdown (.md).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from ingestion.chunking import Chunk
from ingestion.extract_text import ExtractionReport

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_long_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pecah teks yang lebih panjang dari max_chars menjadi potongan-potongan dengan overlap."""
    if len(text) <= max_chars:
        return [text]

    lines = text.split("\n")
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) + 1 > max_chars and current:
            pieces.append("\n".join(current))
            overlap_lines: list[str] = []
            ov_len = 0
            for prev in reversed(current):
                if ov_len + len(prev) + 1 <= overlap:
                    overlap_lines.insert(0, prev)
                    ov_len += len(prev) + 1
                else:
                    break
            current = overlap_lines + [line]
            current_len = sum(len(l) + 1 for l in current)
        else:
            current.append(line)
            current_len += len(line) + 1

    if current:
        pieces.append("\n".join(current))

    final_pieces: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            final_pieces.append(piece)
        else:
            words = piece.split(" ")
            w_curr: list[str] = []
            w_len = 0
            for w in words:
                if w_len + len(w) + 1 > max_chars and w_curr:
                    final_pieces.append(" ".join(w_curr))
                    w_curr = w_curr[-15:] + [w]
                    w_len = sum(len(x) + 1 for x in w_curr)
                else:
                    w_curr.append(w)
                    w_len += len(w) + 1
            if w_curr:
                final_pieces.append(" ".join(w_curr))

    merged: list[str] = []
    for p in final_pieces:
        p = p.strip()
        if not p:
            continue
        if merged and len(p) < 40:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def extract_markdown_with_report(
    md_path: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = 20,
) -> tuple[list[Chunk], ExtractionReport]:
    """
    Ekstrak dokumen Markdown menjadi chunk berjenjang dengan heading hierarchy.

    Args:
        md_path: Path berkas Markdown (.md / .markdown).
        chunk_size: Batas maksimum karakter per chunk.
        chunk_overlap: Jumlah karakter overlap saat memotong teks panjang.
        min_chunk_size: Ukuran minimum karakter untuk mempertahankan chunk.

    Returns:
        Tuple (daftar chunk, laporan ekstraksi).
    """
    raw_text = md_path.read_text(encoding="utf-8", errors="replace")
    raw_lines = raw_text.splitlines()

    report = ExtractionReport(
        lines_raw=len(raw_lines),
    )

    # Tangani YAML frontmatter jika ada di awal berkas
    frontmatter_info: list[str] = []
    start_idx = 0
    if raw_lines and raw_lines[0].strip() == "---":
        end_fm = -1
        for i in range(1, len(raw_lines)):
            if raw_lines[i].strip() == "---":
                end_fm = i
                break
        if end_fm != -1:
            for l in raw_lines[1:end_fm]:
                s = l.strip()
                if s and not s.startswith("source_sha256"):
                    frontmatter_info.append(s)
            start_idx = end_fm + 1

    sections: list[dict] = []
    heading_stack: list[tuple[int, str]] = []
    current_paragraphs: list[list[str]] = []
    current_p: list[str] = []

    def flush_paragraph() -> None:
        nonlocal current_p
        if current_p:
            current_paragraphs.append(current_p)
            current_p = []

    def flush_section() -> None:
        nonlocal current_paragraphs
        flush_paragraph()
        if current_paragraphs:
            sections.append({
                "heading_path": [title for _, title in heading_stack],
                "section_title": heading_stack[-1][1] if heading_stack else "",
                "paragraphs": current_paragraphs,
            })
            current_paragraphs = []

    kept_lines_count = 0

    for line in raw_lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        match = _HEADING_RE.match(stripped)
        if match:
            flush_section()
            level = len(match.group(1))
            title = match.group(2).strip()
            title = re.sub(r"[*_`]", "", title).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            kept_lines_count += 1
        else:
            current_p.append(stripped)
            kept_lines_count += 1

    flush_section()
    report.lines_kept = kept_lines_count

    # Sisipkan ringkasan frontmatter ke bagian pertama jika ada
    if frontmatter_info and sections:
        fm_text = "Metadata:\n" + "\n".join(f"- {item}" for item in frontmatter_info)
        sections[0]["paragraphs"].insert(0, [fm_text])

    chunks: list[Chunk] = []
    page_counter = 1

    for sec in sections:
        heading_path = sec["heading_path"]
        section_title = sec["section_title"]
        paragraphs = sec["paragraphs"]

        p_texts = [" ".join(p) for p in paragraphs if p]
        if not p_texts:
            continue

        # Kumpulkan paragraf-paragraf ke dalam blok
        accumulated = ""
        for p_text in p_texts:
            # Jika satu paragraf saja sudah melebihi chunk_size, pecah
            if len(p_text) > chunk_size:
                if accumulated.strip():
                    if len(accumulated.strip()) >= min_chunk_size:
                        chunks.append(Chunk(
                            text=accumulated.strip(),
                            page_numbers=[page_counter],
                            section_title=section_title,
                            heading_path=heading_path,
                            source_document=md_path.name,
                        ))
                        page_counter += 1
                    accumulated = ""

                for sub_piece in _split_long_text(p_text, max_chars=chunk_size, overlap=chunk_overlap):
                    chunks.append(Chunk(
                        text=sub_piece,
                        page_numbers=[page_counter],
                        section_title=section_title,
                        heading_path=heading_path,
                        source_document=md_path.name,
                    ))
                    page_counter += 1
                continue

            if not accumulated:
                accumulated = p_text
            elif len(accumulated) + len(p_text) + 2 <= chunk_size:
                accumulated += "\n\n" + p_text
            else:
                if len(accumulated.strip()) >= min_chunk_size:
                    chunk = Chunk(
                        text=accumulated.strip(),
                        page_numbers=[page_counter],
                        section_title=section_title,
                        heading_path=heading_path,
                        source_document=md_path.name,
                    )
                    chunks.append(chunk)
                    page_counter += 1

                words = accumulated.split()
                overlap_text = ""
                if len(words) > 10:
                    tail = " ".join(words[-20:])
                    if len(tail) <= chunk_overlap:
                        overlap_text = tail

                accumulated = f"{overlap_text}\n\n{p_text}".strip() if overlap_text else p_text

        if accumulated.strip() and len(accumulated.strip()) >= min_chunk_size:
            chunk = Chunk(
                text=accumulated.strip(),
                page_numbers=[page_counter],
                section_title=section_title,
                heading_path=heading_path,
                source_document=md_path.name,
            )
            chunks.append(chunk)
            page_counter += 1

    # Dokumen sangat pendek tanpa heading
    if not chunks and any(sec["paragraphs"] for sec in sections):
        all_text = "\n\n".join(
            " ".join(p) for sec in sections for p in sec["paragraphs"] if p
        ).strip()
        if all_text:
            first_sec = sections[0] if sections else {}
            chunks.append(Chunk(
                text=all_text,
                page_numbers=[1],
                section_title=first_sec.get("section_title", md_path.stem),
                heading_path=first_sec.get("heading_path", [md_path.stem]),
                source_document=md_path.name,
            ))

    report.pages_with_text = len(chunks)
    return chunks, report
