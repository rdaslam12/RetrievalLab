"""Streamlit entry point for the RetrievalLab workbench."""

from __future__ import annotations

import hashlib
from html import escape
import logging
from pathlib import Path
from time import perf_counter

import streamlit as st

from src.benchmark_service import (
    ALL_BENCHMARK_CONFIGURATIONS,
    DEFAULT_BENCHMARK_CONFIGURATIONS,
    BenchmarkService,
    PipelineConfiguration,
    StoredBenchmarkRun,
    benchmark_to_csv,
    benchmark_to_json,
)
from src.config import AppConfig, load_config
from src.document_parser import DocumentValidationError
from src.bm25_retriever import BM25SearchEngine
from src.embedder import (
    EmbeddingModelUnavailableError,
    SentenceTransformerEmbedder,
    detect_preferred_device,
)
from src.evaluation import (
    EvaluationDataError,
    corpus_document_ids,
    parse_qrels_tsv,
    parse_query_jsonl,
)
from src.evaluation_service import EvaluationService
from src.model_ids import ModelValidationError
from src.model_selection import invalidate_dense_indexes_for_model, resolve_model_choice
from src.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankerUnavailableError,
)
from src.search_engine import DenseSearchEngine
from src.remote_adapter import RemoteAdapterClient, RemoteDenseAdapter, RemoteRerankerAdapter
from src.remote_transport import AdapterError
from src.services import CorpusService, SearchService, build_dense_engines
from src.ui_components import (
    render_about_method,
    render_benchmark_report,
    render_evaluation_report,
    render_results,
)
from src.vector_index import available_backend_name


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINES = {
    "Dense": "Semantic retrieval using the selected dense encoder and local vector index.",
    "BM25": "Lexical retrieval using raw Okapi BM25 scores.",
    "Hybrid": "Dense and BM25 rankings combined with Reciprocal Rank Fusion.",
}
RERANKERS = ("Off", "Cross-Encoder")


@st.cache_resource(show_spinner=False)
def get_embedder(model_id: str) -> SentenceTransformerEmbedder:
    """Load one safe SentenceTransformer resource per model ID."""

    return SentenceTransformerEmbedder(model_id)


@st.cache_resource(show_spinner=False)
def get_cross_encoder_reranker(model_id: str, device: str) -> CrossEncoderReranker:
    """Create one lazy reranker resource per model and preferred device."""

    return CrossEncoderReranker(model_id, device=device)


def _remote_token(stage: str) -> str:
    return st.session_state.get(f"remote_{stage}_token", "")


def _reset_remote_connection(stage: str) -> None:
    st.session_state.pop(f"remote_{stage}_client", None)
    st.session_state.pop(f"remote_{stage}_error", None)
    for key in ("search_response", "evaluation_result", "benchmark_result"):
        st.session_state.pop(key, None)


def _model_source_changed(stage: str) -> None:
    st.session_state.pop(f"remote_{stage}_client", None)
    st.session_state.pop(f"remote_{stage}_error", None)
    for key in ("search_response", "evaluation_result", "benchmark_result"):
        st.session_state.pop(key, None)
    if stage == "reranker":
        mode = st.session_state.get("reranker_model_mode")
        st.session_state["reranker"] = (
            "Off" if mode == "Off" else "Remote Adapter" if mode == "Remote Adapter" else "Cross-Encoder"
        )


def _active_embedder(model_id: str):
    if st.session_state.get("embedding_model_mode") == "Remote Adapter":
        client = st.session_state.get("remote_embedding_client")
        if client is None or client.model_id != model_id:
            raise AdapterError("Test the selected dense adapter connection before indexing.")
        return RemoteDenseAdapter(client)
    return get_embedder(model_id)


def _active_reranker(model_id: str, device: str):
    if st.session_state.get("reranker_model_mode") == "Remote Adapter":
        client = st.session_state.get("remote_reranker_client")
        if client is None or client.model_id != model_id:
            raise AdapterError("Test the selected reranker adapter connection before reranking.")
        return RemoteRerankerAdapter(client)
    return get_cross_encoder_reranker(model_id, device)


def _test_remote_connection(stage: str) -> None:
    """Validate before the full rerun so later widgets keep their session state."""
    _reset_remote_connection(stage)
    try:
        client = RemoteAdapterClient(
            st.session_state.get(f"remote_{stage}_endpoint", ""),
            token_provider=lambda: _remote_token(stage),
        )
        if stage == "embedding":
            RemoteDenseAdapter(client)
        else:
            RemoteRerankerAdapter(client)
        st.session_state[f"remote_{stage}_client"] = client
    except AdapterError as exc:
        st.session_state[f"remote_{stage}_error"] = str(exc)


def _render_remote_settings(stage: str) -> None:
    st.text_input("Adapter endpoint URL", key=f"remote_{stage}_endpoint",
                  placeholder="https://adapter.example.org", on_change=_reset_remote_connection,
                  args=(stage,))
    st.text_input("API token (optional)", type="password", key=f"remote_{stage}_token",
                  on_change=_reset_remote_connection, args=(stage,))
    st.button("Test connection", key=f"test_{stage}_adapter",
              on_click=_test_remote_connection, args=(stage,))
    connection_error = st.session_state.get(f"remote_{stage}_error")
    if connection_error:
        st.error(connection_error)
    client = st.session_state.get(f"remote_{stage}_client")
    if client is not None:
        metadata = client.metadata
        details = [metadata.name, metadata.adapter_type]
        if metadata.dimension is not None:
            details.extend([f"{metadata.dimension} dimensions", metadata.similarity])
        details.append(metadata.protocol)
        st.text(" · ".join(details))


def _resolve_model_choices(
    config: AppConfig,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve current controls without loading either model."""

    embedding_error: str | None = None
    reranker_error: str | None = None
    try:
        embedding_model_id = resolve_model_choice(
            ("Default" if st.session_state.get("embedding_model_mode") == "Remote Adapter"
             else st.session_state.get("embedding_model_mode", "Default")),
            st.session_state.get("custom_embedding_model_id", ""),
            default_model_id=config.model_id,
            kind="embedding",
        )
        if st.session_state.get("embedding_model_mode") == "Remote Adapter":
            client = st.session_state.get("remote_embedding_client")
            if client is None:
                raise AdapterError("Test the dense adapter connection to validate its metadata.")
            embedding_model_id = client.model_id
    except (ModelValidationError, ValueError) as exc:
        embedding_model_id = None
        embedding_error = str(exc)
    try:
        reranker_model_id = resolve_model_choice(
            ("Default" if st.session_state.get("reranker_model_mode") in ("Off", "Remote Adapter")
             else st.session_state.get("reranker_model_mode", "Default")),
            st.session_state.get("custom_reranker_model_id", ""),
            default_model_id=DEFAULT_RERANKER_MODEL,
            kind="reranker",
        )
        if st.session_state.get("reranker_model_mode") == "Remote Adapter":
            client = st.session_state.get("remote_reranker_client")
            if client is None:
                raise AdapterError("Test the reranker adapter connection to validate its metadata.")
            reranker_model_id = client.model_id
        elif st.session_state.get("reranker_model_mode") == "Off":
            reranker_model_id = None
    except (ModelValidationError, ValueError) as exc:
        reranker_model_id = None
        reranker_error = str(exc)
    return embedding_model_id, reranker_model_id, embedding_error, reranker_error


def _render_model_settings(
    config: AppConfig,
    *,
    embedding_error: str | None,
    reranker_error: str | None,
) -> None:
    """Render compact, conditional selectors for the two model stages."""

    with st.expander("Model settings"):
        embedding_column, reranker_column = st.columns(2)
        with embedding_column:
            st.markdown("**Embedding model**")
            embedding_mode = st.radio(
                "Embedding model",
                ("Default", "Custom", "Remote Adapter"),
                format_func=lambda value: "Hugging Face" if value == "Custom" else value,
                horizontal=True,
                key="embedding_model_mode",
                on_change=_model_source_changed, args=("embedding",),
            )
            if embedding_mode == "Custom":
                st.text_input(
                    "Hugging Face embedding model ID",
                    key="custom_embedding_model_id",
                    placeholder="organization/sentence-transformer",
                )
            elif embedding_mode == "Remote Adapter":
                _render_remote_settings("embedding")
            else:
                st.caption(f"Default · `{config.model_id}`")
            st.caption("Changing this model requires rebuilding Dense and Hybrid indexes.")
            if embedding_error:
                st.error(embedding_error)

        with reranker_column:
            st.markdown("**Reranker model**")
            reranker_mode = st.radio(
                "Reranker model",
                ("Off", "Default", "Custom", "Remote Adapter"),
                index=1,
                format_func=lambda value: "Hugging Face" if value == "Custom" else value,
                horizontal=True,
                key="reranker_model_mode",
                on_change=_model_source_changed, args=("reranker",),
            )
            if reranker_mode == "Custom":
                st.text_input(
                    "Hugging Face CrossEncoder model ID",
                    key="custom_reranker_model_id",
                    placeholder="organization/cross-encoder",
                )
            elif reranker_mode == "Remote Adapter":
                _render_remote_settings("reranker")
            elif reranker_mode == "Default":
                st.caption(f"Default · `{DEFAULT_RERANKER_MODEL}`")
            st.caption("Changing this model affects reranking only; corpus embeddings are reused.")
            if reranker_error:
                st.error(reranker_error)

        st.caption(
            "Remote adapters let custom research models participate in RetrievalLab without "
            "uploading executable model code."
        )


def _sidebar(
    config: AppConfig,
    preferred_device: str,
    embedding_model_id: str | None,
    reranker_model_id: str | None,
) -> None:
    engine: DenseSearchEngine | None = st.session_state.get("engine")
    active_device = engine.embedder.device if engine is not None else preferred_device
    backend = engine.index_backend if engine is not None else available_backend_name()
    corpus_engine = engine or st.session_state.get("bm25_engine")
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
        if corpus_engine is None:
            st.info("No corpus is active")
        else:
            st.success(f"Corpus ready · {len(corpus_engine.passages):,} passages")

        with st.expander("Advanced details"):
            st.markdown(f"**Profile**  \n{config.profile_name}")
            st.markdown(f"**Dense encoder**  \n{embedding_model_id or 'Invalid selection'}")
            st.markdown(f"**Reranker model**  \n{reranker_model_id or 'Invalid selection'}")
            dimension = engine.embedder.dimension if engine is not None else "Detected at load"
            st.markdown(f"**Embedding dimension**  \n{dimension}")
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
                <p><strong>Private by design</strong><span>Indexes stay in this session. Remote adapters receive passage and query text for inference. Results are retrieved evidence, never generated answers.</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_corpus_manifest(engine: DenseSearchEngine | BM25SearchEngine) -> None:
    """Show a bounded, literal manifest for the active corpus."""

    entries: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for passage in engine.passages:
        key = (passage.source_id, passage.document_id)
        if key in seen:
            continue
        seen.add(key)
        evaluation_id = passage.document_id or passage.source_id
        identity = f"{passage.file_name} · evaluation ID {evaluation_id}"
        if passage.document_id is not None:
            identity += " · JSONL document_id"
        if passage.title:
            identity += f" · {passage.title}"
        entries.append(identity)

    with st.expander("Corpus manifest"):
        for entry in entries[:20]:
            st.text(entry)
        if len(entries) > 20:
            st.caption(f"Showing 20 of {len(entries):,} indexed documents.")


def _render_corpus_section(
    config: AppConfig,
    *,
    embedding_model_id: str | None,
    embedding_error: str | None,
    reranker_error: str | None,
) -> None:
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
    _render_model_settings(
        config,
        embedding_error=embedding_error,
        reranker_error=reranker_error,
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
    selected_pipeline = st.session_state.get("retrieval_pipeline", "Dense")
    dense_index_requested = selected_pipeline != "BM25"
    process = process_column.button(
        "Index corpus",
        type="primary",
        disabled=not uploads or (dense_index_requested and embedding_model_id is None),
        width="stretch",
    )
    has_corpus = isinstance(st.session_state.get("bm25_engine"), BM25SearchEngine)
    clear = clear_column.button("Clear corpus", disabled=not has_corpus, width="stretch")
    if clear:
        st.session_state.pop("engine", None)
        st.session_state.pop("bm25_engine", None)
        st.session_state.pop("hybrid_engine", None)
        st.session_state.pop("corpus_summary", None)
        st.session_state.pop("corpus_passages", None)
        st.session_state.pop("search_response", None)
        st.session_state.pop("dense_reindex_required", None)
        st.session_state.pop("evaluation_result", None)
        st.session_state.pop("benchmark_result", None)
        st.session_state["uploader_nonce"] = nonce + 1
        st.rerun()

    if process:
        started = perf_counter()
        try:
            with st.status("Building a replacement corpus index…", expanded=True) as status:
                payloads = [(upload.name, upload.getvalue()) for upload in uploads]
                embedder_loader = None
                if dense_index_requested:
                    assert embedding_model_id is not None
                    embedder_loader = lambda: _active_embedder(embedding_model_id)
                prepared = CorpusService(
                    config,
                    embedder_loader,
                ).prepare(
                    payloads,
                    bounded_jsonl=bounded_jsonl,
                    jsonl_start_document=jsonl_start_document,
                    progress=status.write,
                    build_dense=dense_index_requested,
                )
                elapsed = perf_counter() - started
                if prepared.engine is None:
                    st.session_state.pop("engine", None)
                    st.session_state.pop("hybrid_engine", None)
                    st.session_state["dense_reindex_required"] = True
                else:
                    st.session_state["engine"] = prepared.engine
                    st.session_state["hybrid_engine"] = prepared.hybrid_engine
                    st.session_state["dense_reindex_required"] = False
                st.session_state["bm25_engine"] = prepared.bm25_engine
                st.session_state["corpus_passages"] = prepared.passage_records
                st.session_state["search_response"] = None
                st.session_state.pop("evaluation_result", None)
                st.session_state.pop("benchmark_result", None)
                st.session_state["corpus_summary"] = {
                    "files": prepared.files,
                    "documents": prepared.documents,
                    "passages": prepared.passages,
                    "embeddings": prepared.embeddings,
                    "seconds": elapsed,
                    "warnings": prepared.warnings,
                    "embedding_model_id": (
                        prepared.engine.embedder.model_id if prepared.engine is not None else None
                    ),
                    "embedding_dimension": (
                        prepared.engine.embedder.dimension if prepared.engine is not None else None
                    ),
                }
                status.update(label="Corpus indexed successfully.", state="complete", expanded=False)
        except (DocumentValidationError, EmbeddingModelUnavailableError, ValueError) as exc:
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

    bm25_engine = st.session_state.get("bm25_engine")
    dense_engine = st.session_state.get("engine")
    if isinstance(bm25_engine, BM25SearchEngine) and not isinstance(
        dense_engine, DenseSearchEngine
    ):
        st.info(
            "BM25 is ready. Build the Dense and Hybrid indexes to use the selected embedding model."
        )
        rebuild = st.button(
            "Build Dense + Hybrid indexes",
            disabled=embedding_model_id is None,
            width="stretch",
            key="rebuild_dense_index",
        )
        if rebuild:
            assert embedding_model_id is not None
            started = perf_counter()
            try:
                with st.status("Building Dense and Hybrid indexes…", expanded=True) as status:
                    status.write(f"Loading {embedding_model_id}.")
                    dense_engine, hybrid_engine = build_dense_engines(
                        config,
                        bm25_engine.passages,
                        lambda: _active_embedder(embedding_model_id),
                        bm25_engine=bm25_engine,
                    )
                    st.session_state["engine"] = dense_engine
                    st.session_state["hybrid_engine"] = hybrid_engine
                    st.session_state["search_response"] = None
                    st.session_state.pop("evaluation_result", None)
                    st.session_state.pop("benchmark_result", None)
                    st.session_state["dense_reindex_required"] = False
                    summary = dict(st.session_state.get("corpus_summary", {}))
                    summary.update(
                        {
                            "embeddings": len(dense_engine.passages),
                            "seconds": perf_counter() - started,
                            "embedding_model_id": dense_engine.embedder.model_id,
                            "embedding_dimension": dense_engine.embedder.dimension,
                        }
                    )
                    st.session_state["corpus_summary"] = summary
                    status.update(
                        label="Dense and Hybrid indexes are ready.",
                        state="complete",
                        expanded=False,
                    )
                    st.rerun()
            except (EmbeddingModelUnavailableError, ModelValidationError, ValueError) as exc:
                LOGGER.exception("Dense index rebuild failed")
                st.error("Dense and Hybrid indexes could not be built with the selected model.")
                st.text(str(exc))
            except (RuntimeError, OSError):
                LOGGER.exception("Dense index rebuild failed")
                st.error("Dense and Hybrid indexes could not be built with the selected model.")

    summary = st.session_state.get("corpus_summary")
    if summary:
        st.success("Knowledge base ready")
        metrics = st.columns(4)
        metrics[0].metric("Documents", summary.get("documents", summary["files"]))
        metrics[1].metric("Passages", summary["passages"])
        metrics[2].metric("Indexed", summary["embeddings"])
        metrics[3].metric("Preparation", f"{summary['seconds']:.1f} s")
        active_engine = st.session_state.get("engine") or st.session_state.get("bm25_engine")
        if isinstance(active_engine, (DenseSearchEngine, BM25SearchEngine)):
            _render_corpus_manifest(active_engine)
        if summary["warnings"]:
            with st.expander(f"Corpus notes ({len(summary['warnings'])})"):
                for warning in summary["warnings"]:
                    st.warning(warning)
    else:
        st.caption("Index documents to unlock the search canvas.")


def _render_search_section(
    *,
    embedding_model_id: str | None,
    embedding_error: str | None,
    reranker_model_id: str | None,
    reranker_error: str | None,
) -> None:
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
    model_selection_ready = pipeline == "BM25" or embedding_model_id is not None
    ready = engine is not None and engine.is_indexed and model_selection_ready
    st.markdown(
        f'<p class="pipeline-description">{PIPELINES[pipeline]}</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="selector-label">Reranker</p>', unsafe_allow_html=True)
    reranker_choice = st.radio(
        "Reranker",
        (("Off", "Remote Adapter") if st.session_state.get("reranker_model_mode") == "Remote Adapter"
         else ("Off",) if st.session_state.get("reranker_model_mode") == "Off" else RERANKERS),
        horizontal=True,
        index=0,
        key="reranker",
    )
    if reranker_choice != "Off":
        selected_client = st.session_state.get("remote_reranker_client")
        selected_reranker = (selected_client.metadata.name if selected_client is not None
                            and reranker_choice == "Remote Adapter" else reranker_model_id or "unvalidated model")
        st.markdown(
            '<p class="reranker-description">Reranks the retrieved candidate set using '
            f'{escape(selected_reranker)}.</p>',
            unsafe_allow_html=True,
        )
        if reranker_error:
            st.error(reranker_error)
    query_ready = ready and not (
        reranker_choice != "Off" and reranker_model_id is None
    )
    query = st.chat_input(
        "Ask the active corpus a precise question",
        disabled=not query_ready,
        key="retrieval_query",
    )
    if not ready:
        if pipeline in {"Dense", "Hybrid"} and isinstance(
            st.session_state.get("bm25_engine"), BM25SearchEngine
        ):
            st.caption("Build the Dense and Hybrid indexes above to use this pipeline.")
        elif embedding_error and pipeline in {"Dense", "Hybrid"}:
            st.caption("Choose a valid embedding model to use this pipeline.")
        else:
            st.caption("Index documents above to begin.")
    elif not query_ready:
        st.caption("Choose a valid reranker model, or select Off.")
    else:
        st.caption("Press Enter or select the send control to retrieve passages.")

    if query and query.strip():
        try:
            assert engine is not None
            reranker = None
            if reranker_choice != "Off":
                assert reranker_model_id is not None
                reranker = _active_reranker(
                    reranker_model_id,
                    detect_preferred_device(),
                )
            st.session_state["search_response"] = SearchService(engine, reranker).search(query)
        except AdapterError as exc:
            st.error(str(exc))
        except RerankerUnavailableError as exc:
            LOGGER.exception("Cross-encoder reranking failed")
            st.error(
                "Cross-Encoder reranking is unavailable. The base retrieval system is still "
                "ready; select Off to search without reranking."
            )
            st.caption(str(exc))
        except ModelValidationError as exc:
            st.error("The selected model ID is invalid.")
            st.caption(str(exc))
        except ValueError as exc:
            st.error("Retrieval could not be started.")
            st.text(str(exc))
        except RuntimeError:
            LOGGER.exception("Retrieval failed")
            st.error("Retrieval could not be completed with the selected pipeline.")


def _evaluation_signature(
    engine: object,
    pipeline: str,
    reranker_choice: str,
    reranker_model_id: str | None,
    query_bytes: bytes,
    qrels_bytes: bytes,
) -> tuple[object, ...]:
    return (
        id(engine),
        pipeline,
        reranker_choice,
        reranker_model_id if reranker_choice != "Off" else None,
        hashlib.sha256(query_bytes).hexdigest(),
        hashlib.sha256(qrels_bytes).hexdigest(),
    )


def _benchmark_signature(
    engines: dict[str, object],
    configurations: tuple[PipelineConfiguration, ...],
    embedding_model_id: str | None,
    reranker_model_id: str | None,
    query_bytes: bytes,
    qrels_bytes: bytes,
) -> tuple[object, ...]:
    return (
        tuple(
            (configuration.name, id(engines.get(configuration.base_pipeline)))
            for configuration in configurations
        ),
        embedding_model_id,
        reranker_model_id if any(item.use_reranker for item in configurations) else None,
        hashlib.sha256(query_bytes).hexdigest(),
        hashlib.sha256(qrels_bytes).hexdigest(),
    )


def _render_evaluation_section(
    *,
    embedding_model_id: str | None,
    reranker_model_id: str | None,
    reranker_error: str | None,
) -> None:
    """Render single-pipeline evaluation and multi-pipeline benchmarking."""

    pipeline = st.session_state.get("retrieval_pipeline", "Dense")
    reranker_choice = st.session_state.get("reranker", "Off")
    engines = {
        "Dense": st.session_state.get("engine"),
        "BM25": st.session_state.get("bm25_engine"),
        "Hybrid": st.session_state.get("hybrid_engine"),
    }
    engine = engines[pipeline]

    with st.container(border=True, key="evaluation_workspace"):
        st.markdown(
            """
            <div class="evaluation-heading">
                <div>
                    <h2>Evaluate retrieval</h2>
                    <p>Measure one pipeline or compare several against the same labeled queries.</p>
                </div>
                <span>DOCUMENT LEVEL</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "Evaluation mode",
            ("Evaluate one pipeline", "Compare pipelines"),
            horizontal=True,
            key="evaluation_mode",
        )

        selected_configurations: tuple[PipelineConfiguration, ...] = ()
        if mode == "Evaluate one pipeline":
            reranker_label = reranker_choice
            if reranker_choice != "Off" and reranker_model_id:
                remote_client = st.session_state.get("remote_reranker_client")
                model_label = (remote_client.metadata.name if reranker_choice == "Remote Adapter"
                               and remote_client is not None else reranker_model_id)
                reranker_label = f"{reranker_choice} · {model_label}"
            st.caption(f"Active configuration: {pipeline} · Reranker {reranker_label}")
        else:
            configurations = tuple(
                PipelineConfiguration(item.base_pipeline, item.use_reranker,
                    "Remote Reranker" if st.session_state.get("reranker_model_mode") == "Remote Adapter"
                    else "Cross-Encoder")
                for item in ALL_BENCHMARK_CONFIGURATIONS
            )
            configuration_by_name = {item.name: item for item in configurations}
            stored_names = st.session_state.get("benchmark_configurations", [])
            if any(name not in configuration_by_name for name in stored_names):
                st.session_state["benchmark_configurations"] = [item.name for item in DEFAULT_BENCHMARK_CONFIGURATIONS]
            selected_names = st.multiselect(
                "Pipeline configurations",
                tuple(configuration_by_name),
                default=tuple(item.name for item in DEFAULT_BENCHMARK_CONFIGURATIONS),
                key="benchmark_configurations",
                help=(
                    "Each row composes an existing base retriever with the optional shared "
                    "selected reranking adapter."
                ),
            )
            selected_configurations = tuple(
                configuration_by_name[name] for name in selected_names
            )
            st.caption(
                "Every selected configuration uses the same active corpus, query IDs, qrels, "
                "candidate depth, and metric implementation."
            )

        query_column, qrels_column = st.columns(2)
        query_upload = query_column.file_uploader(
            "Query JSONL",
            type=["jsonl"],
            accept_multiple_files=False,
            key="evaluation_queries",
            help="One UTF-8 JSON object per line with _id and text fields.",
        )
        qrels_upload = qrels_column.file_uploader(
            "Qrels TSV",
            type=["tsv", "txt"],
            accept_multiple_files=False,
            key="evaluation_qrels",
            help="Three tab-separated fields: query-id, corpus-id, and non-negative score.",
        )

        with st.expander("File formats and active document IDs"):
            st.markdown("**Queries · JSONL**")
            st.code('{"_id": "q1", "text": "example query"}', language="json")
            st.markdown("**Relevance judgments · TSV**")
            st.code("query-id\tcorpus-id\tscore\nq1\tdoc-17\t1", language="text")
            st.caption(
                "The header is optional. Scores greater than 0 are relevant; graded scores are "
                "used by NDCG. JSONL corpora use document_id. PDF, DOCX, and TXT corpora use "
                "the stable source_id shown below."
            )
            corpus_engine = engines.get("BM25") or engines.get("Dense")
            if corpus_engine is not None and corpus_engine.is_indexed:
                identities = sorted(corpus_document_ids(corpus_engine.passages))
                st.code("\n".join(identities[:50]), language="text")
                if len(identities) > 50:
                    st.caption(f"Showing 50 of {len(identities):,} active evaluation IDs.")
            else:
                st.info("Index a corpus to see its evaluation IDs.")

        query_bytes = b""
        qrels_bytes = b""
        if query_upload is not None and qrels_upload is not None:
            query_bytes = query_upload.getvalue()
            qrels_bytes = qrels_upload.getvalue()

        if mode == "Evaluate one pipeline":
            engine_ready = engine is not None and engine.is_indexed
            reranker_ready = reranker_choice == "Off" or reranker_model_id is not None
            can_run = bool(
                engine_ready
                and reranker_ready
                and query_upload is not None
                and qrels_upload is not None
            )
            run_evaluation = st.button(
                "Run evaluation",
                type="primary",
                disabled=not can_run,
                width="stretch",
                key="run_evaluation",
            )
            if not engine_ready:
                st.info(f"Index the required corpus data before evaluating the {pipeline} pipeline.")
            elif not reranker_ready:
                st.error(reranker_error or "Choose a valid reranker model, or select Off.")
            elif query_upload is None or qrels_upload is None:
                st.caption("Add both evaluation files to enable the run.")

            signature: tuple[object, ...] | None = None
            if query_upload is not None and qrels_upload is not None and engine is not None:
                signature = _evaluation_signature(
                    engine,
                    pipeline,
                    reranker_choice,
                    reranker_model_id,
                    query_bytes,
                    qrels_bytes,
                )

            if run_evaluation:
                assert engine is not None and query_upload is not None and qrels_upload is not None
                try:
                    queries = parse_query_jsonl(query_bytes, file_name=query_upload.name)
                    qrels = parse_qrels_tsv(qrels_bytes, file_name=qrels_upload.name)
                    reranker = None
                    if reranker_choice != "Off":
                        assert reranker_model_id is not None
                        reranker = _active_reranker(
                            reranker_model_id,
                            detect_preferred_device(),
                        )
                    with st.status(
                        "Evaluating the active retrieval pipeline…", expanded=True
                    ) as status:
                        def update_progress(index: int, total: int, _query: object) -> None:
                            status.update(label=f"Evaluating query {index:,} of {total:,}…")

                        report = EvaluationService(engine, reranker).evaluate(
                            queries,
                            qrels,
                            progress=update_progress,
                        )
                        status.update(
                            label=f"Evaluation complete · {report.evaluated_queries:,} queries",
                            state="complete",
                            expanded=False,
                        )
                    st.session_state["evaluation_result"] = {
                        "signature": signature,
                        "report": report,
                    }
                except EvaluationDataError as exc:
                    st.session_state.pop("evaluation_result", None)
                    st.error("Evaluation files could not be validated.")
                    st.text(str(exc))
                except AdapterError as exc:
                    st.session_state.pop("evaluation_result", None)
                    st.error(str(exc))
                except RerankerUnavailableError as exc:
                    st.session_state.pop("evaluation_result", None)
                    st.error("Cross-Encoder reranking is unavailable for this evaluation.")
                    st.caption(str(exc))
                except (ValueError, RuntimeError):
                    st.session_state.pop("evaluation_result", None)
                    LOGGER.exception("Evaluation failed")
                    st.error("Evaluation could not be completed with the active configuration.")

            stored_result = st.session_state.get("evaluation_result")
            if stored_result and stored_result.get("signature") == signature:
                render_evaluation_report(stored_result["report"])
        else:
            dense_required = any(
                item.base_pipeline in {"Dense", "Hybrid"}
                for item in selected_configurations
            )
            bm25_required = any(
                item.base_pipeline in {"BM25", "Hybrid"}
                for item in selected_configurations
            )
            reranker_required = any(item.use_reranker for item in selected_configurations)
            dense_engine = engines.get("Dense")
            hybrid_engine = engines.get("Hybrid")
            bm25_engine = engines.get("BM25")
            dense_ready = bool(
                embedding_model_id is not None
                and dense_engine is not None
                and dense_engine.is_indexed
                and getattr(dense_engine.embedder, "model_id", None) == embedding_model_id
                and hybrid_engine is not None
                and hybrid_engine.is_indexed
            )
            bm25_ready = bool(bm25_engine is not None and bm25_engine.is_indexed)
            reranker_ready = not reranker_required or reranker_model_id is not None
            selection_ready = bool(selected_configurations)
            can_run = bool(
                selection_ready
                and (not dense_required or dense_ready)
                and (not bm25_required or bm25_ready)
                and reranker_ready
                and query_upload is not None
                and qrels_upload is not None
            )
            run_benchmark = st.button(
                "Run benchmark",
                type="primary",
                disabled=not can_run,
                width="stretch",
                key="run_benchmark",
            )
            if not selection_ready:
                st.error("Select at least one pipeline configuration.")
            elif dense_required and not dense_ready:
                st.error(
                    "Dense or Hybrid is selected, but the active Dense index is unavailable or "
                    "stale for the selected embedding model. Build Dense + Hybrid indexes first."
                )
            elif bm25_required and not bm25_ready:
                st.error("BM25 or Hybrid is selected, but no active BM25 index is available.")
            elif not reranker_ready:
                st.error(reranker_error or "Choose a validated reranker model for reranked rows.")
            elif query_upload is None or qrels_upload is None:
                st.caption("Add both evaluation files to enable the benchmark.")

            benchmark_signature: tuple[object, ...] | None = None
            if query_upload is not None and qrels_upload is not None:
                benchmark_signature = _benchmark_signature(
                    engines,
                    selected_configurations,
                    embedding_model_id,
                    reranker_model_id,
                    query_bytes,
                    qrels_bytes,
                )

            if run_benchmark:
                assert query_upload is not None and qrels_upload is not None
                try:
                    queries = parse_query_jsonl(query_bytes, file_name=query_upload.name)
                    qrels = parse_qrels_tsv(qrels_bytes, file_name=qrels_upload.name)
                    reranker_loader = None
                    if reranker_required:
                        assert reranker_model_id is not None
                        reranker_loader = lambda: _active_reranker(
                            reranker_model_id,
                            detect_preferred_device(),
                        )
                    with st.status(
                        "Running the shared-dataset benchmark…", expanded=True
                    ) as status:
                        def update_benchmark_progress(
                            configuration_name: str,
                            index: int,
                            total: int,
                            _query: object,
                        ) -> None:
                            status.update(
                                label=(
                                    f"{configuration_name} · query {index:,} of {total:,}"
                                )
                            )

                        benchmark = BenchmarkService(
                            engines,
                            embedding_model_id=embedding_model_id,
                            reranker_model_id=reranker_model_id,
                            reranker_loader=reranker_loader,
                        ).evaluate(
                            selected_configurations,
                            queries,
                            qrels,
                            progress=update_benchmark_progress,
                        )
                        status.update(
                            label=(
                                f"Benchmark complete · {len(benchmark.successful_results):,} "
                                f"of {len(benchmark.pipeline_results):,} configurations"
                            ),
                            state="complete",
                            expanded=False,
                        )
                    assert benchmark_signature is not None
                    st.session_state["benchmark_result"] = StoredBenchmarkRun(
                        signature=benchmark_signature,
                        result=benchmark,
                    )
                except EvaluationDataError as exc:
                    st.session_state.pop("benchmark_result", None)
                    st.error("Evaluation files could not be validated.")
                    st.text(str(exc))
                except (ValueError, RuntimeError) as exc:
                    st.session_state.pop("benchmark_result", None)
                    LOGGER.exception("Benchmark failed before a valid comparison was possible")
                    st.error("The benchmark could not be completed as a valid comparison.")
                    st.caption(str(exc))

            stored_benchmark = st.session_state.get("benchmark_result")
            if (
                isinstance(stored_benchmark, StoredBenchmarkRun)
                and stored_benchmark.signature == benchmark_signature
            ):
                render_benchmark_report(stored_benchmark.result)
                export_columns = st.columns(2)
                export_columns[0].download_button(
                    "Download aggregate CSV",
                    data=benchmark_to_csv(stored_benchmark.result),
                    file_name="retrievallab-benchmark.csv",
                    mime="text/csv",
                    width="stretch",
                    key="download_benchmark_csv",
                )
                export_columns[1].download_button(
                    "Download benchmark JSON",
                    data=benchmark_to_json(stored_benchmark.result),
                    file_name="retrievallab-benchmark.json",
                    mime="application/json",
                    width="stretch",
                    key="download_benchmark_json",
                )


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
    (
        embedding_model_id,
        reranker_model_id,
        embedding_error,
        reranker_error,
    ) = _resolve_model_choices(config)
    invalidate_dense_indexes_for_model(st.session_state, embedding_model_id or "unvalidated-selection")
    _sidebar(config, preferred_device, embedding_model_id, reranker_model_id)
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
        _render_corpus_section(
            config,
            embedding_model_id=embedding_model_id,
            embedding_error=embedding_error,
            reranker_error=reranker_error,
        )
    _render_search_section(
        embedding_model_id=embedding_model_id,
        embedding_error=embedding_error,
        reranker_model_id=reranker_model_id,
        reranker_error=reranker_error,
    )
    response = st.session_state.get("search_response")
    if response is not None:
        render_results(response, config=config)
    _render_evaluation_section(
        embedding_model_id=embedding_model_id,
        reranker_model_id=reranker_model_id,
        reranker_error=reranker_error,
    )
    render_about_method()


if __name__ == "__main__":
    main()
