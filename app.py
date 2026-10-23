"""Streamlit entry point for the RetrievalLab workbench."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import streamlit as st

from src.config import AppConfig, load_config
from src.document_parser import DocumentValidationError
from src.embedder import ArcticEmbedder, detect_preferred_device
from src.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankerUnavailableError,
)
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
RERANKERS = ("Off", "Cross-Encoder")


@st.cache_resource(show_spinner=False)
def get_embedder(model_id: str, dimension: int) -> ArcticEmbedder:
    """Load Arctic-S once per Streamlit server process."""

    return ArcticEmbedder(model_id, dimension)


@st.cache_resource(show_spinner=False)
def get_cross_encoder_reranker(model_id: str, device: str) -> CrossEncoderReranker:
    """Create one lazy reranker resource per model and preferred device."""

    return CrossEncoderReranker(model_id, device=device)


def _sidebar(config: AppConfig, preferred_device: str) -> None:
    engine: DenseSearchEngine | None = st.session_state.get("engine")
    active_device = engine.embedder.device if engine is not None else preferred_device
    backend = engine.index_backend if engine is not None else available_backend_name()
    with st.sidebar:
        st.markdown(
            """
            <div class="app-brand">
                <svg class="app-brand__mark" viewBox="0 0 40 40" aria-hidden="true">
                    <rect x="3.5" y="3.5" width="33" height="33" rx="10"/>
                    <circle cx="12" cy="13" r="2"/>
                    <circle cx="12" cy="20" r="2"/>
                    <circle cx="12" cy="27" r="2"/>
                    <path d="M17 13h12M17 20h9M17 27h6"/>
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
            st.markdown(f"**Public reranker**  \n{DEFAULT_RERANKER_MODEL}")
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
    st.markdown(
        """
        <div class="corpus-heading">
            <div>
                <h2>Build your retrieval corpus</h2>
                <p>Prepare one session-local knowledge base for every retrieval pipeline.</p>
            </div>
            <span>PDF · DOCX · TXT · JSONL</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    st.markdown(
        """
        <div class="search-section-heading">
            <h2>Search the corpus</h2>
            <p>Choose how the evidence should be ranked.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="selector-label">Base retrieval</p>', unsafe_allow_html=True)
    pipeline = st.radio(
        "Base retrieval",
        list(PIPELINES),
        horizontal=True,
        key="retrieval_pipeline",
    )
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
    st.markdown('<p class="selector-label">Reranker</p>', unsafe_allow_html=True)
    reranker_choice = st.radio(
        "Reranker",
        RERANKERS,
        horizontal=True,
        index=0,
        key="reranker",
    )
    if reranker_choice == "Cross-Encoder":
        st.markdown(
            '<p class="reranker-description">Reranks the retrieved candidate set using '
            f'{DEFAULT_RERANKER_MODEL}.</p>',
            unsafe_allow_html=True,
        )
    query = st.chat_input(
        "Ask the active corpus a precise question",
        disabled=not ready,
        key="retrieval_query",
    )
    if not ready:
        st.caption("Index documents above to begin.")
    else:
        st.caption("Press Enter or select the send control to retrieve passages.")

    if query and query.strip():
        try:
            assert engine is not None
            reranker = None
            if reranker_choice == "Cross-Encoder":
                reranker = get_cross_encoder_reranker(
                    DEFAULT_RERANKER_MODEL,
                    detect_preferred_device(),
                )
            st.session_state["search_response"] = SearchService(engine, reranker).search(query)
        except RerankerUnavailableError as exc:
            LOGGER.exception("Cross-encoder reranking failed")
            st.error(
                "Cross-Encoder reranking is unavailable. The base retrieval system is still "
                "ready; select Off to search without reranking."
            )
            st.caption(str(exc))
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
    st.markdown(
        """
        <section class="search-hero">
            <h1>What should we discover?</h1>
            <p>A modular workbench for comparing dense, lexical, and hybrid retrieval pipelines.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True, key="corpus_workspace"):
        _render_corpus_section(config)
    _render_search_section()
    response = st.session_state.get("search_response")
    if response is not None:
        render_results(response, config=config)
    render_about_method()


if __name__ == "__main__":
    main()
