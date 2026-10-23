"""Streamlit entry point for the RetrievalLab workbench."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import streamlit as st

from src.config import AppConfig, load_config
from src.document_parser import DocumentValidationError
from src.embedder import ArcticEmbedder, detect_preferred_device
from src.search_engine import DenseSearchEngine
from src.services import CorpusService, SearchService
from src.ui_components import render_about_method, render_results
from src.vector_index import available_backend_name


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINES = {
    "Dense": "Semantic retrieval over normalized Arctic-S embeddings.",
    "BM25": "Lexical retrieval using raw Okapi BM25 scores.",
    "Hybrid": "Dense and BM25 rankings combined with Reciprocal Rank Fusion.",
}


@st.cache_resource(show_spinner=False)
def get_embedder(model_id: str, dimension: int) -> ArcticEmbedder:
    """Load Arctic-S once per Streamlit server process."""

    return ArcticEmbedder(model_id, dimension)


def _sidebar(config: AppConfig, preferred_device: str) -> None:
    engine: DenseSearchEngine | None = st.session_state.get("engine")
    active_device = engine.embedder.device if engine is not None else preferred_device
    backend = engine.index_backend if engine is not None else available_backend_name()
    with st.sidebar:
        st.markdown(
            """
            <div class="app-brand">
                <svg class="app-brand__mark" viewBox="0 0 40 40" role="img" aria-label="RetrievalLab">
                    <path d="M20 3C21.7 13.5 26.5 18.3 37 20C26.5 21.7 21.7 26.5 20 37C18.3 26.5 13.5 21.7 3 20C13.5 18.3 18.3 13.5 20 3Z" fill="url(#brand-spectrum)"/>
                    <defs>
                        <linearGradient id="brand-spectrum" x1="7" y1="7" x2="34" y2="34" gradientUnits="userSpaceOnUse">
                            <stop stop-color="#8AB4F8"/>
                            <stop offset="0.45" stop-color="#A78BFA"/>
                            <stop offset="0.72" stop-color="#67E8F9"/>
                            <stop offset="1" stop-color="#F6C977"/>
                        </linearGradient>
                    </defs>
                </svg>
                <div>
                    <strong>RetrievalLab</strong>
                    <span>Local retrieval workbench</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="sidebar-label">Session status</p>', unsafe_allow_html=True)
        if engine is None:
            st.info("No corpus is active")
        else:
            st.success(f"Corpus ready · {len(engine.passages):,} passages")

        with st.expander("Advanced details"):
            st.markdown(f"**Profile**  \n{config.profile_name}")
            st.markdown(f"**Dense encoder**  \n{config.model_id}")
            st.markdown(f"**Embedding dimension**  \n{config.embedding_dimension}")
            st.markdown(f"**Candidate depth**  \n{config.candidate_depth}")
            st.markdown(f"**Final results**  \n{config.top_results}")
            st.markdown(
                f"**Chunking**  \n{config.chunk_size} tokens · {config.chunk_overlap} overlap"
            )
            st.markdown(f"**Compute device**  \n{active_device.upper()}")
            st.markdown(f"**Dense index backend**  \n{backend}")

        st.markdown(
            """
            <div class="privacy-note">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 5 6v5c0 4.6 2.8 8.1 7 10 4.2-1.9 7-5.4 7-10V6l-7-3Zm0 2.2 5 2.1V11c0 3.5-1.9 6.2-5 7.8C8.9 17.2 7 14.5 7 11V7.3l5-2.1Z"/></svg>
                <p><strong>Private by design</strong><span>Uploads, passages, and embeddings stay in this session. Results are retrieved evidence, never generated answers.</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_corpus_manifest(engine: DenseSearchEngine) -> None:
    """Show a bounded, literal manifest for the active corpus."""

    entries: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for passage in engine.passages:
        key = (passage.source_id, passage.document_id)
        if key in seen:
            continue
        seen.add(key)
        identity = passage.file_name
        if passage.document_id is not None:
            identity += f" · document {passage.document_id}"
        if passage.title:
            identity += f" · {passage.title}"
        entries.append(identity)

    with st.expander("Corpus manifest"):
        for entry in entries[:20]:
            st.text(entry)
        if len(entries) > 20:
            st.caption(f"Showing 20 of {len(entries):,} indexed documents.")


def _render_corpus_section(config: AppConfig) -> None:
    st.markdown('<p class="sidebar-label corpus-label">Knowledge base</p>', unsafe_allow_html=True)
    st.caption("Prepare one session-local corpus for every retrieval pipeline.")
    nonce = st.session_state.setdefault("uploader_nonce", 0)
    uploads = st.file_uploader(
        "Add scientific or technical documents",
        type=["pdf", "docx", "txt", "jsonl"],
        accept_multiple_files=True,
        key=f"corpus_upload_{nonce}",
        help=(
            "PDF, DOCX, TXT, or BEIR corpus JSONL. JSONL records require _id and text; "
            "title is optional. Up to 20 files and 50 MB total. OCR is not included."
        ),
    )
    total_size = sum(int(upload.size) for upload in uploads) if uploads else 0
    st.caption(
        f"Selected files: {len(uploads) if uploads else 0}/{config.max_files} · "
        f"Total size: {total_size / (1024 * 1024):.2f}/{config.max_total_bytes / (1024 * 1024):.0f} MB"
    )
    single_jsonl = bool(
        uploads
        and len(uploads) == 1
        and Path(uploads[0].name).suffix.lower() == ".jsonl"
    )
    bounded_jsonl = False
    jsonl_start_document = 1
    if single_jsonl:
        with st.expander("Large JSONL corpus options"):
            bounded_jsonl = st.checkbox(
                f"Index an ordered slice capped at {config.max_passages:,} passages",
                value=False,
                key=f"bounded_jsonl_{nonce}",
                help=(
                    "Enable this only when the complete JSONL corpus exceeds the deployment limit. "
                    "The app stops before splitting a document and clearly reports omitted records."
                ),
            )
            if bounded_jsonl:
                jsonl_start_document = int(
                    st.number_input(
                        "Start at indexable JSONL document",
                        min_value=1,
                        value=1,
                        step=1,
                        key=f"jsonl_start_document_{nonce}",
                        help=(
                            "One-based position among non-empty JSONL documents, in file order. "
                            "Use a later position to search another batch."
                        ),
                    )
                )
                st.warning(
                    "Bounded mode indexes only one slice. Retrieval results do not cover documents "
                    "outside the active slice."
                )
    process_column, clear_column = st.columns([2, 1])
    process = process_column.button(
        "Index corpus",
        type="primary",
        disabled=not uploads,
        width="stretch",
    )
    has_corpus = isinstance(st.session_state.get("engine"), DenseSearchEngine)
    clear = clear_column.button("Clear corpus", disabled=not has_corpus, width="stretch")
    if clear:
        st.session_state.pop("engine", None)
        st.session_state.pop("bm25_engine", None)
        st.session_state.pop("hybrid_engine", None)
        st.session_state.pop("corpus_summary", None)
        st.session_state.pop("search_response", None)
        st.session_state["uploader_nonce"] = nonce + 1
        st.rerun()

    if process:
        started = perf_counter()
        try:
            with st.status("Building a replacement corpus index…", expanded=True) as status:
                payloads = [(upload.name, upload.getvalue()) for upload in uploads]
                prepared = CorpusService(
                    config,
                    lambda: get_embedder(config.model_id, config.embedding_dimension),
                ).prepare(
                    payloads,
                    bounded_jsonl=bounded_jsonl,
                    jsonl_start_document=jsonl_start_document,
                    progress=status.write,
                )
                elapsed = perf_counter() - started
                st.session_state["engine"] = prepared.engine
                st.session_state["bm25_engine"] = prepared.bm25_engine
                st.session_state["hybrid_engine"] = prepared.hybrid_engine
                st.session_state["search_response"] = None
                st.session_state["corpus_summary"] = {
                    "files": prepared.files,
                    "documents": prepared.documents,
                    "passages": prepared.passages,
                    "embeddings": prepared.embeddings,
                    "seconds": elapsed,
                    "warnings": prepared.warnings,
                }
                status.update(label="Corpus indexed successfully.", state="complete", expanded=False)
        except (DocumentValidationError, ValueError) as exc:
            LOGGER.exception("Corpus indexing failed")
            st.error("The new corpus was not installed. The previous corpus is still active.")
            st.text(str(exc))
        except (RuntimeError, OSError):
            LOGGER.exception("Corpus indexing failed")
            st.error(
                "The replacement corpus could not be prepared. The previous corpus is still active. "
                "Check the local model setup and try again."
            )
        except Exception:
            LOGGER.exception("Unexpected corpus indexing failure")
            st.error(
                "The replacement corpus could not be prepared. The previous corpus is still active."
            )

    summary = st.session_state.get("corpus_summary")
    if summary:
        st.success("Knowledge base ready")
        metrics = st.columns(4)
        metrics[0].metric("Documents", summary.get("documents", summary["files"]))
        metrics[1].metric("Passages", summary["passages"])
        metrics[2].metric("Indexed", summary["embeddings"])
        metrics[3].metric("Preparation", f"{summary['seconds']:.1f} s")
        active_engine = st.session_state.get("engine")
        if isinstance(active_engine, DenseSearchEngine):
            _render_corpus_manifest(active_engine)
        if summary["warnings"]:
            with st.expander(f"Corpus notes ({len(summary['warnings'])})"):
                for warning in summary["warnings"]:
                    st.warning(warning)
    else:
        st.caption("Index documents to unlock the search canvas.")


def _render_search_section() -> None:
    pipeline = st.radio("Retrieval pipeline", list(PIPELINES), horizontal=True)
    engines = {
        "Dense": st.session_state.get("engine"),
        "BM25": st.session_state.get("bm25_engine"),
        "Hybrid": st.session_state.get("hybrid_engine"),
    }
    engine = engines[pipeline]
    ready = engine is not None and engine.is_indexed
    st.markdown(
        f'<p class="pipeline-description">{PIPELINES[pipeline]}</p>',
        unsafe_allow_html=True,
    )
    query = st.chat_input(
        "Ask the active corpus a precise question",
        disabled=not ready,
        key="retrieval_query",
    )
    if not ready:
        st.caption("Open Corpus controls and index documents to begin.")
    else:
        st.caption("Press Enter or select the send control to retrieve passages.")

    if query and query.strip():
        try:
            assert engine is not None
            st.session_state["search_response"] = SearchService(engine).search(query)
        except ValueError as exc:
            st.error("Retrieval could not be started.")
            st.text(str(exc))
        except RuntimeError:
            LOGGER.exception("Retrieval failed")
            st.error("Retrieval could not be completed with the selected pipeline.")


def main() -> None:
    """Render the complete local retrieval application."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    st.set_page_config(
        page_title="RetrievalLab · Local retrieval workbench",
        page_icon=":material/search_insights:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    stylesheet = (PROJECT_ROOT / "assets" / "app.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)
    config = load_config()
    preferred_device = detect_preferred_device()
    _sidebar(config, preferred_device)
    with st.sidebar:
        _render_corpus_section(config)
    st.markdown(
        """
        <section class="search-hero">
            <div class="hero-orbit" aria-hidden="true"></div>
            <svg class="hero-mark" viewBox="0 0 48 48" aria-hidden="true">
                <path d="M24 3C25.9 16.1 31.9 22.1 45 24C31.9 25.9 25.9 31.9 24 45C22.1 31.9 16.1 25.9 3 24C16.1 22.1 22.1 16.1 24 3Z" fill="url(#hero-spectrum)"/>
                <defs><linearGradient id="hero-spectrum" x1="9" y1="7" x2="39" y2="41"><stop stop-color="#8AB4F8"/><stop offset=".42" stop-color="#A78BFA"/><stop offset=".7" stop-color="#67E8F9"/><stop offset="1" stop-color="#F6C977"/></linearGradient></defs>
            </svg>
            <h1>What should we discover?</h1>
            <p>A modular workbench for comparing dense, lexical, and hybrid retrieval pipelines.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    _render_search_section()
    response = st.session_state.get("search_response")
    if response is not None:
        render_results(response, config=config)
    render_about_method()


if __name__ == "__main__":
    main()
