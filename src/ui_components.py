"""Focused Streamlit rendering helpers; no model or index state lives here."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .config import AppConfig
from .evaluation_service import EvaluationReport
from .hybrid_retriever import DEFAULT_RRF_K
from .types import RankedPassage, SearchResponse


def _literal_block(text: str, *, class_name: str = "literal-text") -> None:
    """Render user-controlled text literally inside a styled block."""

    st.markdown(
        f'<div class="{class_name}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _metadata_html(result: RankedPassage) -> str:
    passage = result.passage
    location = f"Page {passage.page_number}" if passage.page_number is not None else "Document passage"
    metadata = []
    if passage.document_id is not None:
        metadata.append(f"Document {passage.document_id}")
    if passage.title:
        metadata.append(passage.title)
    metadata.extend([location, f"Passage {passage.passage_index}"])
    escaped = " · ".join(html.escape(item) for item in metadata)
    return (
        '<div class="result-heading">'
        f'<span class="rank-badge">#{result.rank}</span>'
        '<div class="result-identity">'
        f'<strong>{html.escape(passage.file_name)}</strong>'
        f'<span>{escaped}</span>'
        "</div></div>"
    )


def _stats_html(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        '<div class="result-stat">'
        f'<span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong>'
        "</div>"
        for label, value in items
    )
    return f'<div class="result-stats">{cells}</div>'


def render_result_card(result: RankedPassage, *, method: str) -> None:
    """Render one safe, readable retrieval result with method-specific metadata."""

    with st.container(border=True):
        st.markdown(_metadata_html(result), unsafe_allow_html=True)
        preview = result.passage.text
        if len(preview) > 700:
            preview = preview[:700].rstrip() + "…"
        _literal_block(preview, class_name="passage-text")
        if len(result.passage.text) > len(preview):
            with st.expander("Read full passage"):
                _literal_block(result.passage.text, class_name="passage-text passage-text-full")

        if result.reranker_score is not None:
            retrieval_rank = result.retrieval_rank or result.rank
            retrieval_score = (
                result.retrieval_score if result.retrieval_score is not None else result.score
            )
            change = result.rank_change or 0
            if change > 0:
                movement = f"Up {change}"
            elif change < 0:
                movement = f"Down {abs(change)}"
            else:
                movement = "No change"
            score_label = "Original retrieval score"
            if method == "Hybrid":
                score_label = "Original RRF score"
            elif method == "BM25":
                score_label = "Original BM25 score"
            elif method == "Dense":
                score_label = "Original dense score"
            stats = [
                ("Reranker score", f"{result.reranker_score:.5f}"),
                ("Final rank", f"#{result.rank}"),
                ("Original retrieval rank", f"#{retrieval_rank}"),
                ("Movement", movement),
                (score_label, f"{retrieval_score:.6f}"),
            ]
        elif method == "BM25":
            stats = [
                ("Raw BM25 score", f"{result.score:.5f}"),
                ("BM25 rank", f"#{result.rank}"),
            ]
        elif method == "Hybrid":
            rrf_score = result.rrf_score if result.rrf_score is not None else result.score
            stats = [("RRF score", f"{rrf_score:.6f}"), ("Hybrid rank", f"#{result.rank}")]
        else:
            stats = [
                ("Dense score", f"{result.score:.5f}"),
                ("Dense rank", f"#{result.rank}"),
            ]
        st.markdown(_stats_html(stats), unsafe_allow_html=True)

        if method == "Hybrid":
            dense = "Not in candidate set"
            if result.dense_rank is not None and result.dense_score is not None:
                dense = f"rank #{result.dense_rank} · raw score {result.dense_score:.5f}"
            bm25 = "Not in candidate set"
            if result.bm25_rank is not None and result.bm25_score is not None:
                bm25 = f"rank #{result.bm25_rank} · raw score {result.bm25_score:.5f}"
            st.markdown(
                '<div class="provenance-row">'
                f'<span><strong>Dense</strong> · {html.escape(dense)}</span>'
                f'<span><strong>BM25</strong> · {html.escape(bm25)}</span>'
                "</div>",
                unsafe_allow_html=True,
            )


def _render_pipeline_details(response: SearchResponse, config: AppConfig, *, method: str) -> None:
    details: list[tuple[str, str]] = [
        ("Pipeline", method),
        ("Candidate depth used", str(response.candidate_depth_used)),
        ("Indexed passages", f"{response.indexed_passages:,}"),
        ("Chunking", f"{config.chunk_size} tokens · {config.chunk_overlap} overlap"),
        ("Backend", response.index_backend),
    ]
    if method in {"Dense", "Hybrid"}:
        details.extend(
            [
                ("Dense model", response.dense_model_id or config.model_id),
                (
                    "Embedding dimension",
                    str(response.embedding_dimension or config.embedding_dimension),
                ),
                ("Device", response.device.upper()),
            ]
        )
    if method == "Hybrid":
        details.append(("RRF k", str(DEFAULT_RRF_K)))
    if method == "BM25":
        details.append(("Retriever", "Okapi BM25 lexical retrieval"))
    details.extend(
        [
            ("Reranking", "On" if response.reranking_used else "Off"),
            ("Retrieval latency", f"{response.retrieval_latency_ms:.1f} ms"),
        ]
    )
    if response.reranking_used:
        details.extend(
            [
                ("Reranker", response.reranker_name or "Cross-Encoder"),
                ("Reranking latency", f"{response.reranking_latency_ms:.1f} ms"),
            ]
        )
    details.append(("Total query latency", f"{response.total_query_latency_ms:.1f} ms"))
    st.dataframe(
        pd.DataFrame(details, columns=["Setting", "Value"]),
        hide_index=True,
        width="stretch",
    )


def _render_result_summary(response: SearchResponse) -> None:
    if response.reranking_used:
        summary = st.columns(4)
        summary[0].metric("Base retrieval", response.retrieval_method)
        summary[1].metric("Reranker", "Cross-Encoder")
        summary[2].metric("Results returned", len(response.results))
        summary[3].metric("Total query", f"{response.total_query_latency_ms:.1f} ms")
        st.caption(
            f"Retrieval: {response.retrieval_latency_ms:.1f} ms · "
            f"Reranking: {response.reranking_latency_ms:.1f} ms"
        )
    else:
        summary = st.columns(3)
        summary[0].metric("Pipeline", response.retrieval_method)
        summary[1].metric("Results returned", len(response.results))
        summary[2].metric("Total query", f"{response.total_query_latency_ms:.1f} ms")
    if not response.reranking_used and response.retrieval_method == "Hybrid":
        st.caption(
            "Aggregate Dense + BM25 retrieval and fusion: "
            f"{response.retrieval_latency_ms:.1f} ms"
        )
    elif not response.reranking_used:
        st.caption(f"Retrieval latency: {response.retrieval_latency_ms:.1f} ms")


def render_results(response: SearchResponse, *, config: AppConfig) -> None:
    """Render ranked evidence and pipeline details without changing response data."""

    method = response.retrieval_method
    st.header("Retrieved evidence")
    _literal_block(response.query, class_name="query-block")
    _render_result_summary(response)
    for result in response.results:
        render_result_card(result, method=method)
    with st.expander("Pipeline details"):
        _render_pipeline_details(response, config, method=method)

    if response.reranking_used:
        st.caption(
            "Reranker scores and original retrieval scores are separate raw quantities; "
            "neither is a probability or confidence value."
        )
    elif method == "BM25":
        st.caption("BM25 values are raw lexical retrieval scores, not probabilities or confidence.")
    elif method == "Hybrid":
        st.caption("RRF combines branch ranks; its scores are not probabilities or normalized raw scores.")
    else:
        st.caption("Dense values are retrieval scores, not probabilities or confidence.")


def render_evaluation_report(report: EvaluationReport) -> None:
    """Render aggregate metrics, latency summaries, and per-query diagnostics."""

    st.markdown("### Evaluation results")
    metric_rows = [
        {
            "Metric": metric_name,
            **{f"@{k}": report.metrics[metric_name][k] for k in report.k_values},
        }
        for metric_name in ("Recall", "Hit Rate", "MRR", "NDCG")
    ]
    metric_columns = {
        f"@{k}": st.column_config.NumberColumn(format="%.4f") for k in report.k_values
    }
    st.dataframe(
        pd.DataFrame(metric_rows),
        hide_index=True,
        width="stretch",
        column_config=metric_columns,
    )
    st.caption(
        "Each value is the arithmetic mean of per-query document-level metrics over every "
        "loaded query. Queries without positive judgments contribute 0."
    )

    dataset_columns = st.columns(2)
    dataset_columns[0].metric("Evaluated queries", report.evaluated_queries)
    dataset_columns[1].metric("Queries with judgments", report.queries_with_judgments)
    latency_count = 3 if report.mean_reranking_latency_ms is not None else 2
    latency_columns = st.columns(latency_count)
    latency_columns[0].metric(
        "Mean retrieval latency", f"{report.mean_retrieval_latency_ms:.1f} ms"
    )
    next_column = 1
    if report.mean_reranking_latency_ms is not None:
        latency_columns[1].metric(
            "Mean reranking latency", f"{report.mean_reranking_latency_ms:.1f} ms"
        )
        next_column = 2
    latency_columns[next_column].metric(
        "Mean total latency", f"{report.mean_total_latency_ms:.1f} ms"
    )
    for warning in report.warnings:
        st.warning(warning)

    st.markdown("### Per-query diagnostics")
    diagnostic_rows = [
        {
            "Query ID": result.query_id,
            "Query text": result.query_text,
            "Relevant documents": result.relevant_document_count,
            "First relevant rank": result.first_relevant_rank,
            "Hit@10": result.hit_at_10,
            "Retrieved document IDs (top 10)": ", ".join(result.retrieved_document_ids),
        }
        for result in report.query_results
    ]
    st.dataframe(
        pd.DataFrame(diagnostic_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Query text": st.column_config.TextColumn(width="large"),
            "Retrieved document IDs (top 10)": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "Passages are collapsed to their document_id, or stable source_id for PDF, DOCX, and "
        "TXT uploads. Only the highest-ranked passage occurrence determines document rank."
    )


def render_about_method() -> None:
    with st.expander("About the retrieval pipelines"):
        st.markdown(
            """
            **Dense** uses normalized SentenceTransformer-compatible query and passage embeddings with exact inner-product retrieval.

            **BM25** retrieves by lexical term evidence and preserves its raw Okapi BM25 score.

            **Hybrid** combines the independent Dense and BM25 candidate rankings with Reciprocal Rank Fusion.

            **Cross-Encoder** optionally reranks only the fixed candidate set returned by the selected base pipeline.
            """
        )
        st.info(
            "RetrievalLab retrieves passages from the active corpus. It does not generate answers."
        )
