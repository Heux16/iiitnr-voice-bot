"""
main_streamlit.py — Academic Document RAG Chatbot Interface.

Design follows the campus academic portal aesthetic: light blue background,
clean white bordered cards, cyan group headers, and crisp responsive components.
Style definitions are maintained in `app/theme.py`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answer_guard import diagnose_answer
from app.citations import (
    extract_cited_pages,
    strip_inline_source_lines,
)
from app.config import (
    DEFAULT_RETRIEVAL_CONFIG,
    EMBEDDING_MODEL,
    LLM_MODEL,
    UTILITY_MODEL,
    RetrievalConfig,
)
from app.llm_client import LLMConfigError, call_llm, call_llm_stream, effective_model
from app.pdf_viewer import render_pdf_viewer, render_source_references
from app.prompt_builder import build_no_context_response, build_prompt
from app.public_config import (
    app_mode,
    is_operator_mode,
    public_rate_limit,
    public_rate_window_seconds,
)
from app.rate_limit import SessionRateLimiter
from app.retrieval import (
    DocumentRetriever,
    RetrievalOutcome,
    build_source_references,
    format_sources,
)
from app.theme import (
    inject_theme,
    nav_label,
    render_brand,
    render_footer,
    render_lead,
    render_status_badge,
    render_welcome,
)

LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Academic RAG Chatbot | Documents",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()


# ──────────────────────────────────────────────
# Cache & Load Resources
# ──────────────────────────────────────────────

@st.cache_resource
def load_retriever() -> tuple[DocumentRetriever | None, str | None]:
    try:
        retriever = DocumentRetriever()
        return retriever, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error loading index: {e}"


@st.cache_data
def load_metadata_df() -> pd.DataFrame:
    metadata_path = Path(__file__).resolve().parent.parent / "index" / "metadata.json"
    if not metadata_path.exists():
        return pd.DataFrame()
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


@st.cache_data
def load_test_questions() -> list[dict]:
    eval_path = Path(__file__).resolve().parent.parent / "eval" / "test_questions.json"
    if not eval_path.exists():
        return []
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Retrieval Debug Panel
# ──────────────────────────────────────────────

def outcome_to_debug(outcome: RetrievalOutcome) -> dict:
    """Summarize retrieval outcome into session-safe dictionary."""
    used = {candidate.chunk_index for candidate in outcome.contexts}

    many_documents = len({
        c.source_document for c in outcome.candidates if c.source_document
    }) > 1

    rows = []
    for candidate in outcome.candidates:
        rows.append({
            "Used": "✅" if candidate.chunk_index in used else "",
            "Chunk": candidate.chunk_index,
            **({"Document": candidate.document_label[:24]} if many_documents else {}),
            "Page": candidate.page_label,
            "Section": candidate.heading_path_label or "-",
            "Retrieved by": candidate.retrieved_by,
            "Dense (cosine)": round(candidate.dense_score, 3),
            "BM25": round(candidate.lexical_score, 2),
            "RRF": round(candidate.fusion_score, 5),
            "Rerank (0-10)": (
                round(candidate.rerank_score, 1)
                if candidate.rerank_score is not None else None
            ),
            "Term coverage": f"{candidate.lexical_coverage:.0%}",
            "Matched terms": ", ".join(candidate.matched_terms) or "-",
        })

    return {
        "rows": rows,
        "gate": dict(outcome.gate),
        "timings_ms": dict(outcome.timings_ms),
        "effective_query": outcome.effective_query,
        "was_rewritten": outcome.was_rewritten,
        "refused": outcome.refused,
        "refusal_reason": outcome.refusal_reason,
        "rerank_failed": outcome.rerank_failed,
    }


def render_retrieval_debug(debug: dict) -> None:
    """Render retrieval metrics and score breakdown across pipeline stages."""
    if not debug:
        return

    rows = debug.get("rows") or []
    label = f"🔬 Retrieval Details ({len(rows)} candidates)"

    with st.expander(label, expanded=False):
        timings = debug.get("timings_ms") or {}
        if timings:
            stage_names = {
                "rewrite_ms": "Query rewriting",
                "dense_ms": "Dense (embed + FAISS)",
                "lexical_ms": "BM25",
                "fusion_ms": "RRF Fusion",
                "rerank_ms": "LLM Reranker",
            }
            cols = st.columns(len(timings) + 1)
            for col, (key, value) in zip(cols, timings.items()):
                col.metric(stage_names.get(key, key), f"{value:.0f} ms")
            cols[-1].metric("Total", f"{sum(timings.values()):.0f} ms")

        gate = debug.get("gate") or {}
        if gate:
            st.caption(
                "**Relevance Gate** — "
                f"max similarity {gate.get('max_dense_score', 0):.3f} "
                f"(threshold {gate.get('dense_threshold', 0):.2f}) · "
                f"max term coverage {gate.get('max_lexical_coverage', 0):.0%} "
                f"(threshold {gate.get('lexical_coverage_threshold', 0):.0%}) · "
                f"dense branch {'passed' if gate.get('dense_pass') else 'failed'}, "
                f"lexical branch {'passed' if gate.get('lexical_pass') else 'failed'}"
                + (
                    f", reranker {'passed' if gate.get('rerank_pass') else 'rejected'}"
                    if "rerank_pass" in gate else ""
                )
            )

        if debug.get("rerank_failed"):
            st.error(
                "Reranker failed to execute; fusion rank order was used and "
                "precision gate was skipped for this question. "
                "This usually happens when the API is busy — try again.",
                icon="🔥",
            )

        if debug.get("refused"):
            st.warning(f"Refused: {debug.get('refusal_reason', '-')}")

        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No candidates found in either search branch.")


def render_index_freshness(retriever: DocumentRetriever | None) -> None:
    """Warn when the index is out of sync with source documents."""
    if not retriever:
        return

    freshness = retriever.freshness()
    if not freshness.has_info or not freshness.is_stale:
        return

    pesan = "\n\n".join(f"- {m}" for m in freshness.messages())
    st.warning(
        f"**Index may be outdated.**\n\n{pesan}\n\n"
        "Run `python ingestion/build_index.py --yes` to rebuild.",
        icon="🕒",
    )


def render_hero() -> None:
    """Hero welcome card matching campus academic portal styling."""
    render_welcome(
        "Welcome to the IIIT-NR Academic Chatbot",
        "Ask questions about <strong>IIIT-NR Academic Guidelines, Faculty Profiles, Admissions, Curriculum, and Campus Information</strong>. "
        "Every answer is constructed strictly from official documents and includes "
        "page numbers and section titles so you can verify each detail directly.",
    )


# ──────────────────────────────────────────────
# View 1: Q&A Chat
# ──────────────────────────────────────────────

def render_chat_view(
    retriever: DocumentRetriever | None,
    config: RetrievalConfig,
    operator_mode: bool = False,
    rate_limiter: SessionRateLimiter | None = None,
) -> None:
    st.header("Academic Q&A")
    render_lead(
        "Ask questions in English about academic policies, credit requirements, "
        "thesis guidelines, leave of absence, or graduation."
    )

    if not retriever:
        if operator_mode:
            st.error(
                "Document index not found. Run "
                "`python ingestion/build_index.py` first."
            )
        else:
            st.error(
                "Chatbot service is being prepared. Please try again later "
                "or contact the administrator."
            )
        return

    # Preset Questions Picker
    preset_col, action_col = st.columns([3, 1])
    sample_queries = [
        "-- Select a sample question --",
        "Who is Dr. Abhishek Sharma and what is his research area?",
        "What are the admission requirements for B.Tech at IIIT-NR?",
        "What are the placement statistics and top recruiters?",
        "What is the curriculum and credit structure for B.Tech CSE?",
        "What campus facilities and research laboratories are available?",
    ]
    with preset_col:
        selected_sample = st.selectbox(
            "Sample questions",
            options=sample_queries,
            help="Select a common question to test the chatbot workflow directly.",
        )
    with action_col:
        st.write("")
        st.write("")
        if st.button("Send question", type="primary", disabled=(selected_sample == sample_queries[0])):
            st.session_state["queued_query"] = selected_sample
            st.rerun()

    # Session messages initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("source_refs"):
                    render_source_references(
                        msg["source_refs"],
                        key_prefix=f"history_{msg.get('message_id', id(msg))}",
                        public=not operator_mode,
                    )
                elif msg.get("sources"):
                    st.markdown(
                        f'<div class="source-box"><strong>📚 Document Sources:</strong><br>{msg["sources"]}</div>',
                        unsafe_allow_html=True,
                    )
                if operator_mode and msg.get("debug"):
                    render_retrieval_debug(msg["debug"])

    # Process queued query from preset if any
    input_query = None
    if "queued_query" in st.session_state:
        input_query = st.session_state.pop("queued_query")

    chat_input = st.chat_input("Type your question here...")
    query = input_query or chat_input

    if query:
        _handle_query(
            query,
            retriever,
            config,
            operator_mode=operator_mode,
            rate_limiter=rate_limiter,
        )

    render_pdf_viewer()


def _handle_query(
    query: str,
    retriever: DocumentRetriever,
    config: RetrievalConfig,
    operator_mode: bool = False,
    rate_limiter: SessionRateLimiter | None = None,
) -> None:
    if rate_limiter is not None:
        limit_result = rate_limiter.check_and_consume(st.session_state)
        if not limit_result.allowed:
            minutes = max(1, (limit_result.retry_after_seconds + 59) // 60)
            st.warning(
                f"Question rate limit reached "
                f"({limit_result.limit} questions per hour). "
                f"Please try again in about {minutes} minute(s).",
                icon="⏳",
            )
            return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(query)

    # Chat history excluding latest question
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar="🎓"):
        spinner_text = (
            "Rewriting question and searching documents..."
            if config.use_query_rewrite and chat_history
            else "Searching relevant document excerpts..."
        )

        try:
            with st.spinner(spinner_text):
                outcome = retriever.retrieve(
                    query,
                    config=config,
                    chat_history=chat_history,
                )
        except LLMConfigError as error:
            if operator_mode:
                st.error(f"⚠️ {error}")
            else:
                st.error(
                    "Search service is not ready. Please contact administrator.",
                    icon="⚠️",
                )
            st.session_state.messages.pop()
            return
        except Exception as error:
            if operator_mode:
                st.error(f"⚠️ Search failed: {error}")
            else:
                st.error(
                    "Search service encountered an error. "
                    "Please try again later.",
                    icon="⚠️",
                )
            st.session_state.messages.pop()
            return

        debug = outcome_to_debug(outcome)

        if outcome.was_rewritten and operator_mode:
            st.caption(
                f"🔁 Question rewritten for search: "
                f"*{outcome.effective_query}*"
            )

        if outcome.refused:
            response = build_no_context_response()
            st.markdown(response)
            if operator_mode:
                render_retrieval_debug(debug)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "debug": debug,
            })
            return

        system_prompt, user_prompt = build_prompt(
            outcome.original_query, outcome.contexts, chat_history
        )
        message_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in call_llm_stream(user_prompt, system_instruction=system_prompt):
                full_response += chunk
                live_text = strip_inline_source_lines(full_response)
                message_placeholder.markdown(live_text + " ▌")
        except Exception as error:
            LOGGER.warning("Streaming error, fallback: %s", error)
            if not full_response:
                full_response = call_llm(user_prompt, system_instruction=system_prompt)

        response = full_response

        context_pages = {
            page for context in outcome.contexts for page in context.page_numbers
        }
        diagnostic = diagnose_answer(response, context_pages)
        if diagnostic is not None:
            diagnostic_code, diagnostic_detail = diagnostic
            LOGGER.warning(
                "answer_guard_rejected code=%s detail=%s",
                diagnostic_code,
                diagnostic_detail,
            )
            if operator_mode:
                st.error(
                    "Chat provider response failed validation (page citation missing "
                    "or outside retrieval context). Review provider output."
                )
                st.caption(f"Diagnosis ({diagnostic_code}): {diagnostic_detail}")
            else:
                st.warning(
                    "The AI service could not return a verified response. "
                    "Please try again later or contact the administrator.",
                    icon="⚠️",
                )
            response = (
                "I apologize, but the AI service could not produce a verified answer "
                "at this time. Please try again later."
            )

        cited_pages = set(extract_cited_pages(response))
        source_refs = build_source_references(
            outcome.contexts,
            cited_pages=cited_pages,
        )
        display_response = strip_inline_source_lines(response)
        message_placeholder.markdown(display_response)

        sources_text = format_sources(
            outcome.contexts,
            cited_pages=cited_pages,
            include_scores=operator_mode,
        )
        if source_refs:
            render_source_references(
                source_refs,
                key_prefix=f"current_{len(st.session_state.messages)}",
                public=not operator_mode,
            )

        if operator_mode:
            render_retrieval_debug(debug)

        st.session_state.messages.append({
            "role": "assistant",
            "content": display_response,
            "sources": sources_text,
            "source_refs": source_refs,
            "message_id": len(st.session_state.messages),
            "debug": debug,
        })


# ──────────────────────────────────────────────
# View 2: Document Explorer
# ──────────────────────────────────────────────

def render_document_explorer(retriever: DocumentRetriever | None) -> None:
    st.header("Document & Index Explorer")
    render_lead(
        "Inspect text structure, tokenized chunks, and page distributions "
        "across indexed source documents."
    )

    if not retriever:
        st.warning("Index is not available yet.")
        return

    info = retriever.document_info
    df = load_metadata_df()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks", f"{info.get('total_chunks', 0):,}")
    col2.metric("Total Pages", f"{info.get('total_pages', 0)}")
    col3.metric("Page Range", f"p. {info.get('page_range', '-')}")
    col4.metric(
        "Pages per Chunk",
        f"{info.get('avg_pages_per_chunk', 0)} avg",
        help="Values closer to 1 ensure higher citation precision. "
             f"Maximum page span: {info.get('max_pages_per_chunk', 0)} pages.",
    )

    documents = info.get("documents") or []
    if len(documents) > 1:
        st.subheader("Documents in Index")
        st.dataframe(
            pd.DataFrame([
                {
                    "Document": doc["name"],
                    "Chunks": doc["chunks"],
                    "Pages": doc["pages"],
                }
                for doc in documents
            ]),
            hide_index=True,
            width="stretch",
        )

    lexical = info.get("lexical") or {}
    if lexical:
        st.caption(
            f"**BM25 Lexical Index** — {lexical.get('documents', 0)} documents · "
            f"{lexical.get('vocabulary', 0):,} unique terms · "
            f"average {lexical.get('avg_tokens_per_doc', 0)} tokens per chunk · "
            f"k1={lexical.get('k1')}, b={lexical.get('b')}. "
            "Constructed in-memory at application startup."
        )

    st.subheader("Search Document Content")
    search_term = st.text_input(
        "Filter chunk text",
        placeholder="Example: thesis, leave, curriculum, honors...",
    )

    display_df = df.copy()
    if search_term and not display_df.empty:
        haystack = display_df["text"].fillna("")
        if "section_title" in display_df.columns:
            haystack = haystack + " " + display_df["section_title"].fillna("")
        display_df = display_df[haystack.str.contains(search_term, case=False, na=False)]

    if not display_df.empty:
        st.caption(f"Displaying {min(len(display_df), 50)} of {len(display_df)} matching chunks")
        table_data = []
        for _, row in display_df.head(50).iterrows():
            pages = row.get("page_numbers") or []
            page_label = "-"
            if len(pages):
                first, last = min(pages), max(pages)
                page_label = str(first) if first == last else f"{first}-{last}"

            heading_path = row.get("heading_path")
            if isinstance(heading_path, list) and heading_path:
                heading = " › ".join(heading_path)
            else:
                heading = row.get("section_title") or "-"

            entri = {
                "Index": row.get("chunk_index"),
                "Page": page_label,
                "Section / Title": heading,
                "Characters": row.get("char_count"),
                "Text Excerpt": (row.get("text") or "")[:140] + "...",
            }
            if len(documents) > 1:
                name = row.get("source_document") or "-"
                entri = {"Document": name.rsplit(".", 1)[0][:28], **entri}
            table_data.append(entri)
        st.dataframe(pd.DataFrame(table_data), hide_index=True, width="stretch")
    else:
        st.info("No chunks match the search filter.")


# ──────────────────────────────────────────────
# View 3: Evaluation
# ──────────────────────────────────────────────

def render_evaluation(retriever: DocumentRetriever | None) -> None:
    st.header("RAG Quality Evaluation")
    render_lead(
        "Evaluate retrieval precision and out-of-scope question refusal."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Target Recall@4", "≥ 80.0%")
    col2.metric("Refusal Accuracy", "100.0%")
    col3.metric("Target Latency", "< 5.0 s")

    questions = load_test_questions()
    if not questions:
        st.info("File `eval/test_questions.json` not found.")
        return

    in_scope = [q for q in questions if q.get("category") == "in_scope"]
    out_scope = [q for q in questions if q.get("category") == "out_of_scope"]

    tab1, tab2 = st.tabs([f"In-Scope ({len(in_scope)} Questions)", f"Out-of-Scope ({len(out_scope)} Questions)"])

    with tab1:
        st.caption("Questions whose answers exist within the academic guidelines:")
        in_data = [
            {
                "ID": q["id"],
                "Question": q["question"],
                "Expected Keywords": ", ".join(q.get("expected_answer_contains", [])),
                "Category": "In-Scope",
            }
            for q in in_scope
        ]
        st.dataframe(pd.DataFrame(in_data), hide_index=True)

    with tab2:
        st.caption("Out-of-scope questions that must be politely declined (anti-hallucination):")
        out_data = [
            {
                "ID": q["id"],
                "Question": q["question"],
                "Expected Outcome": "Refused / Information not found",
                "Category": "Out-of-Scope",
            }
            for q in out_scope
        ]
        st.dataframe(pd.DataFrame(out_data), hide_index=True)


# ──────────────────────────────────────────────
# View 4: About
# ──────────────────────────────────────────────

def render_about(retriever: DocumentRetriever | None) -> None:
    st.header("About the Project")
    st.markdown(
        """
        This project implements a **Retrieval-Augmented Generation (RAG)** architecture to answer questions regarding campus academic policies and guidelines accurately and without hallucination.

        **Terms of Use & Disclaimer**
        This application serves as an educational reference assistant. Information provided is for academic guidance. For official administrative decisions, students must consult official faculty announcements and academic administration.
        """
    )

    st.subheader("Retrieval Pipeline")
    st.markdown(
        """
        Every user question passes through the following pipeline:

        1. **Query rewriting** — follow-up questions like *"what are the requirements?"* are converted
           into standalone queries using conversation history, **before** search runs.
        2. **Dual-branch retrieval** — *dense search* (Gemini embeddings + FAISS cosine similarity) captures
           semantic meaning, while *BM25 Okapi* captures exact keyword matches such as "GPA 3.50", "144 credits", or "Article 12".
        3. **Reciprocal Rank Fusion (RRF)** — ranks are merged by rank position rather than raw scores,
           seamlessly joining bounded cosine (0–1) and unbounded BM25 scores.
        4. **Reranking** — the cross-encoder LLM evaluates query and document excerpts together,
           scoring *answerability* rather than mere lexical similarity.
        5. **Multi-layer relevance gating** — similarity threshold, lexical term coverage, reranker score,
           and strict system instructions ensure out-of-scope questions are politely refused.
        """
    )

    st.subheader("Configuration & Technical Stack")
    info = retriever.document_info if retriever else {}
    index_info = info.get("index_info") or {}
    embedding_info = index_info.get("embedding") or {}
    lexical = info.get("lexical") or {}

    from app.index_status import indexed_documents

    sources = indexed_documents(index_info)
    if len(sources) > 1:
        label_documents = f"{len(sources)} documents: " + ", ".join(
            d.get("name", "?").rsplit(".", 1)[0][:32] for d in sources
        )
    elif sources:
        label_documents = sources[0].get("name", "-")
    else:
        label_documents = "Academic Guidelines 2025-2026"

    dimension = embedding_info.get("dimension")
    embedding_label = f"Google {EMBEDDING_MODEL}"
    if dimension:
        embedding_label += f" (dimension {dimension})"

    rows = [
        ("Institution", "IIIT Naya Raipur (Dr. SPM IIIT-NR)"),
        ("Source Documents", label_documents),
        ("LLM Generation Model", f"Google Gemini · {effective_model(LLM_MODEL)} (via API)"),
        ("Rerank & Rewrite Model", f"Google Gemini · {effective_model(UTILITY_MODEL)} (via API)"),
        ("Embedding Model", embedding_label),
        ("Embedding Task Type", embedding_info.get("task_type") or "generic"),
        ("Vector Store", "FAISS IndexFlatIP (cosine similarity)"),
        ("Lexical Index", f"BM25 Okapi in-memory, {lexical.get('vocabulary', 0):,} unique terms"),
        ("Total Indexed Chunks", f"{info.get('total_chunks', 0)} chunks"),
        ("Total Source Pages", f"{info.get('total_pages', 0)} pages"),
        (
            "Page Citation Precision",
            f"average {info.get('avg_pages_per_chunk', 0)} pages per chunk "
            f"(maximum {info.get('max_pages_per_chunk', 0)})",
        ),
        ("UI Framework", "Streamlit"),
        (
            "Hallucination Guardrails",
            "Similarity threshold + lexical coverage + reranker gate + strict system prompt",
        ),
    ]

    if index_info.get("built_at"):
        rows.append(("Index Built At", index_info["built_at"]))

    st.dataframe(
        pd.DataFrame(rows, columns=["Component", "Specification / Value"]),
        hide_index=True,
        width="stretch",
    )


# ──────────────────────────────────────────────
# Main Entry Point & Sidebar
# ──────────────────────────────────────────────

def _operator_main_legacy() -> None:
    render_hero()

    retriever, error = load_retriever()
    render_index_freshness(retriever)

    with st.sidebar:
        render_brand("IIIT-NR Chatbot", "Academic & Campus Assistant")

        nav_label("Navigation")
        page = st.radio(
            "Navigation",
            ["Q&A Chat", "Document Explorer", "Evaluation", "About"],
            label_visibility="collapsed",
        )

        st.divider()

        nav_label("System Status")
        if retriever:
            render_status_badge(
                "Vector Store",
                f"● Active Index · {retriever.total_chunks} Chunks",
                online=True,
            )
        else:
            render_status_badge("Vector Store", "● Index Not Ready", online=False)

        # Settings
        selected_documents: tuple[str, ...] = ()
        if retriever:
            available = retriever.source_documents
            if len(available) > 1:
                nav_label("Document Scope")
                chosen = st.multiselect(
                    "Search only in selected documents",
                    options=available,
                    default=available,
                    format_func=lambda name: name.rsplit(".", 1)[0][:42],
                    help="Leave empty to search all documents.",
                )
                if chosen and len(chosen) < len(available):
                    selected_documents = tuple(chosen)

        nav_label("Retrieval Pipeline")
        use_hybrid = st.toggle(
            "Hybrid search (BM25 + dense)",
            value=DEFAULT_RETRIEVAL_CONFIG.use_hybrid,
            help="Combines semantic embedding search with exact keyword matching (BM25) using Reciprocal Rank Fusion.",
        )
        use_rerank = st.toggle(
            "LLM Reranker",
            value=DEFAULT_RETRIEVAL_CONFIG.use_rerank,
            help="Evaluates candidate excerpts against the question simultaneously using cross-encoder scoring.",
        )
        if not use_rerank:
            st.warning(
                "Without the reranker, out-of-scope question refusal drops at retrieval level. "
                "Answers are still protected by system prompt, but irrelevant context may be included.",
                icon="⚠️",
            )
        use_rewrite = st.toggle(
            "Query rewriting",
            value=DEFAULT_RETRIEVAL_CONFIG.use_query_rewrite,
            help="Rewrites follow-up questions into standalone queries before retrieval.",
        )

        nav_label("Retrieval Parameters")
        top_k = st.slider(
            "Top-K Context Chunks", min_value=1, max_value=8,
            value=DEFAULT_RETRIEVAL_CONFIG.top_k,
            help="Number of document chunks sent to LLM as context",
        )
        threshold = st.slider(
            "Relevance Threshold", min_value=0.1, max_value=0.8,
            value=DEFAULT_RETRIEVAL_CONFIG.dense_threshold, step=0.05,
            help="Minimum cosine similarity for dense retrieval branch",
        )

        config = DEFAULT_RETRIEVAL_CONFIG.with_overrides(
            top_k=top_k,
            dense_threshold=threshold,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            use_query_rewrite=use_rewrite,
            document_filter=selected_documents,
        )

        st.divider()
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            f"Generation: {effective_model(LLM_MODEL)} · "
            f"Rerank/rewrite: {effective_model(UTILITY_MODEL)}"
        )

    if page == "Q&A Chat":
        render_chat_view(retriever, config, operator_mode=True)
    elif page == "Document Explorer":
        render_document_explorer(retriever)
    elif page == "Evaluation":
        render_evaluation(retriever)
    else:
        render_about(retriever)

    st.divider()
    render_footer(
        "IIIT-NR Academic & Campus RAG Chatbot"
    )


def main() -> None:
    """Run application in public or operator mode."""
    operator_mode = is_operator_mode()

    if operator_mode:
        _operator_main_legacy()
        return

    render_hero()
    retriever, error = load_retriever()

    limiter = SessionRateLimiter(
        limit=public_rate_limit(),
        window_seconds=public_rate_window_seconds(),
    )
    config = DEFAULT_RETRIEVAL_CONFIG

    with st.sidebar:
        render_brand("IIIT-NR Chatbot", "Academic & Campus Services")
        nav_label("Student Services")
        st.caption(
            "Ask academic, faculty, curriculum, and campus questions based on official IIIT-NR documents."
        )
        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("Use this information as guidance. For official administrative decisions, please consult academic administration.")

    render_chat_view(
        retriever,
        config,
        operator_mode=False,
        rate_limiter=limiter,
    )

    st.divider()
    render_footer(
        "IIIT-NR Academic & Campus RAG Chatbot"
    )


if __name__ == "__main__":
    main()
