"""Interactive source component and document renderer (PDF & Markdown)."""

from __future__ import annotations

import base64
from pathlib import Path

import pymupdf
import streamlit as st

from app.config import DATA_DIR

DEFAULT_PDF_NAME = "Buku-Pedoman-Akademik-Fakultas-Teknologi-Industri-2025-2026.pdf"


@st.cache_data(show_spinner=False)
def _pdf_page_pdf_bytes(path_string: str, page_number: int) -> bytes:
    """Generate single-page PDF to open in native browser viewer."""
    document = pymupdf.open(path_string)
    try:
        if page_number < 1 or page_number > len(document):
            raise ValueError(f"Page {page_number} is outside the document.")
        single_page = pymupdf.open()
        try:
            single_page.insert_pdf(
                document,
                from_page=page_number - 1,
                to_page=page_number - 1,
            )
            return single_page.tobytes()
        finally:
            single_page.close()
    finally:
        document.close()


@st.cache_data(show_spinner=False)
def _pdf_page_image(path_string: str, page_number: int, zoom: float = 1.5) -> bytes:
    """Render a single PDF page as PNG."""
    document = pymupdf.open(path_string)
    try:
        if page_number < 1 or page_number > len(document):
            raise ValueError(
                f"Page {page_number} is outside the document ({len(document)} pages)."
            )
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom),
            alpha=False,
        )
        return pixmap.tobytes("png")
    finally:
        document.close()


def _safe_pdf_path(source_document: str) -> Path | None:
    """Resolve documents only within the data directory, preventing path traversal."""
    data_root = DATA_DIR.resolve()
    filename = Path(source_document).name if source_document else DEFAULT_PDF_NAME

    # First check direct child of DATA_DIR
    candidate = (data_root / filename).resolve()
    if candidate.exists() and candidate.is_file() and (candidate.parent == data_root or data_root in candidate.parents):
        if candidate.suffix.lower() in (".pdf", ".md", ".markdown"):
            return candidate

    # Search subdirectories under data_root
    for match in data_root.rglob(filename):
        if match.is_file() and data_root in match.resolve().parents:
            if match.suffix.lower() in (".pdf", ".md", ".markdown"):
                return match.resolve()

    return None


_safe_document_path = _safe_pdf_path


def _viewer_reference_key(reference: dict) -> str:
    document = reference.get("source_document") or DEFAULT_PDF_NAME
    pages = ",".join(str(page) for page in reference.get("pages", []))
    return f"{document}:{pages}"


def render_source_references(
    references: list[dict],
    key_prefix: str,
    public: bool = True,
) -> None:
    """Render sources as interactive buttons opening document viewer."""
    if not references:
        return

    st.markdown(
        '<div class="source-box"><strong>📚 Document Sources:</strong>'
        '<div style="margin-top:4px;color:var(--muted);font-size:0.82rem;">'
        'Click a source to display the cited document content.</div></div>',
        unsafe_allow_html=True,
    )

    for index, reference in enumerate(references):
        label = f"📄 Page {reference.get('page_label', '-')}"
        section = reference.get("section_title")
        if section:
            label += f" · {section}"
        if not public and reference.get("score") is not None:
            label += f" · relevance {reference['score']:.0%}"

        if st.button(
            label,
            key=f"{key_prefix}_source_{index}_{_viewer_reference_key(reference)}",
            type="secondary",
            width="stretch",
            help="Show cited document",
        ):
            st.session_state["pdf_viewer_reference"] = reference
            st.session_state["pdf_viewer_open"] = True
            st.rerun()


def render_pdf_viewer() -> None:
    """Display selected document content (PDF or Markdown) as a preview."""
    if not st.session_state.get("pdf_viewer_open"):
        return

    reference = st.session_state.get("pdf_viewer_reference") or {}
    path = _safe_pdf_path(reference.get("source_document", ""))
    if path is None:
        st.error("Source document is not available on the server.")
        return

    # Markdown file handling
    if path.suffix.lower() in (".md", ".markdown"):
        st.divider()
        with st.container(border=True):
            title = reference.get("section_title") or "Markdown Document Source"
            st.subheader(f"📖 View Markdown Document — {title}")
            st.caption(
                f"{path.name} · section/page: {reference.get('page_label', '-')}"
            )
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                st.markdown(content)
            except Exception as e:
                st.error(f"Failed to read Markdown file: {e}")

            col_close, col_download = st.columns([1, 1])
            with col_close:
                if st.button("Close viewer", key="close_md_viewer", width="stretch"):
                    st.session_state.pop("pdf_viewer_reference", None)
                    st.session_state["pdf_viewer_open"] = False
                    st.rerun()
            with col_download:
                st.download_button(
                    "Download Markdown",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/markdown",
                    key="download_source_md",
                    width="stretch",
                )
        return

    # PDF file handling
    pages = [int(page) for page in reference.get("pages", []) if str(page).isdigit()]
    if not pages:
        pages = [1]

    st.divider()
    with st.container(border=True):
        title = reference.get("section_title") or "Document Source"
        st.subheader(f"📖 View Document — {title}")
        st.caption(
            f"{path.name} · original page: {reference.get('page_label', '-')}. "
            "Selected page preview:"
        )

        selected_page = st.selectbox(
            "PDF Page",
            options=pages,
            index=0,
            key="pdf_viewer_selected_page",
            format_func=lambda page: f"Page {page}",
        )

        try:
            page_image = _pdf_page_image(str(path), selected_page)
            encoded_image = base64.b64encode(page_image).decode("ascii")
            st.markdown(
                f"""
                <div style="background:#ffffff;border:1px solid #d8e7ef;border-radius:8px;padding:18px;text-align:center;">
                  <img src="data:image/png;base64,{encoded_image}"
                       alt="Page {selected_page} of {path.name}"
                       style="display:block;width:100%;height:auto;margin:0 auto;" />
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Page {selected_page} · {path.name}")

            native_pdf = _pdf_page_pdf_bytes(str(path), selected_page)
            native_encoded = base64.b64encode(native_pdf).decode("ascii")
            st.markdown(
                f'<a href="data:application/pdf;base64,{native_encoded}#page=1" '
                'target="_blank" rel="noopener" '
                'style="display:block;text-align:center;margin:10px 0 2px;color:#14809F;font-weight:700;">'
                '↗ Open PDF in browser viewer</a>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.error("PDF page could not be rendered.")

        col_close, col_download = st.columns([1, 1])
        with col_close:
            if st.button("Close viewer", key="close_pdf_viewer", width="stretch"):
                st.session_state.pop("pdf_viewer_reference", None)
                st.session_state["pdf_viewer_open"] = False
                st.rerun()
        with col_download:
            st.download_button(
                "Download PDF",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/pdf",
                key="download_source_pdf",
                width="stretch",
            )
