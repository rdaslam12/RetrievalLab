"""Orchestration for repeated retrieval evaluation over active indexes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING

from .evaluation import (
    EVALUATION_K_VALUES,
    EvaluationQuery,
    deduplicate_ranked_documents,
    metrics_at_k,
    validate_evaluation_data,
)
from .services import SearchService
from .types import SearchResponse

if TYPE_CHECKING:
    from .interfaces import Reranker, RerankerAdapter, SearchEngine


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: str
    query_text: str
    relevant_document_count: int
    first_relevant_rank: int | None
    hit_at_10: bool
    retrieved_document_ids: tuple[str, ...]
    relevant_document_ids: tuple[str, ...]
    metrics: Mapping[int, Mapping[str, float]]
    search_response: SearchResponse


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    k_values: tuple[int, ...]
    metrics: Mapping[str, Mapping[int, float]]
    mean_retrieval_latency_ms: float
    mean_reranking_latency_ms: float | None
    mean_total_latency_ms: float
    evaluated_queries: int
    queries_with_judgments: int
    warnings: tuple[str, ...]
    query_results: tuple[QueryEvaluation, ...]
    dense_model_id: str | None = None
    reranker_model_id: str | None = None


class EvaluationService:
    """Evaluate queries through the public search service without rebuilding indexes.

    Aggregate metrics are arithmetic means of per-query values over every loaded
    query, including queries without qrels or positive relevance judgments. Those
    queries contribute zero, making the denominator explicit and stable.
    """

    def __init__(self, engine: SearchEngine, reranker: RerankerAdapter | Reranker | None = None) -> None:
        self.engine = engine
        self.reranker = reranker

    def evaluate(
        self,
        queries: Sequence[EvaluationQuery],
        qrels: Mapping[str, Mapping[str, float]],
        *,
        k_values: Sequence[int] = EVALUATION_K_VALUES,
        progress: Callable[[int, int, EvaluationQuery], None] | None = None,
    ) -> EvaluationReport:
        fixed_queries = tuple(queries)
        fixed_k = tuple(k_values)
        if not fixed_queries:
            raise ValueError("Evaluation requires at least one query.")
        if not fixed_k or any(k <= 0 for k in fixed_k) or len(set(fixed_k)) != len(fixed_k):
            raise ValueError("Evaluation K values must be unique positive integers.")
        warnings = validate_evaluation_data(fixed_queries, qrels, self.engine.passages)
        search_service = SearchService(self.engine, self.reranker)
        per_query: list[QueryEvaluation] = []
        total_queries = len(fixed_queries)
        for index, query in enumerate(fixed_queries, start=1):
            if progress is not None:
                progress(index, total_queries, query)
            # Use the complete fixed candidate set. This both supplies at least
            # max(K) documents when available and keeps reranking identical to search.
            response = search_service.search(query.text, result_limit=self.engine.candidate_depth)
            document_ranking = deduplicate_ranked_documents(response.results)
            relevance = qrels.get(query.query_id, {})
            query_metrics = {
                k: metrics_at_k(document_ranking, relevance, k) for k in fixed_k
            }
            relevant_ids = {
                document_id for document_id, score in relevance.items() if score > 0.0
            }
            first_relevant_rank = next(
                (
                    rank
                    for rank, document_id in enumerate(document_ranking, start=1)
                    if document_id in relevant_ids
                ),
                None,
            )
            per_query.append(
                QueryEvaluation(
                    query_id=query.query_id,
                    query_text=query.text,
                    relevant_document_count=len(relevant_ids),
                    first_relevant_rank=first_relevant_rank,
                    hit_at_10=bool(metrics_at_k(document_ranking, relevance, 10)["Hit Rate"]),
                    retrieved_document_ids=document_ranking[: max(fixed_k)],
                    relevant_document_ids=tuple(
                        document_id
                        for document_id in document_ranking[: max(fixed_k)]
                        if document_id in relevant_ids
                    ),
                    metrics=query_metrics,
                    search_response=response,
                )
            )

        aggregate = {
            metric_name: {
                k: fmean(result.metrics[k][metric_name] for result in per_query)
                for k in fixed_k
            }
            for metric_name in ("Recall", "Hit Rate", "MRR", "NDCG")
        }
        reranking_enabled = any(result.search_response.reranking_used for result in per_query)
        return EvaluationReport(
            dense_model_id=per_query[0].search_response.dense_model_id,
            reranker_model_id=per_query[0].search_response.reranker_name,
            k_values=fixed_k,
            metrics=aggregate,
            mean_retrieval_latency_ms=fmean(
                result.search_response.retrieval_latency_ms for result in per_query
            ),
            mean_reranking_latency_ms=(
                fmean(result.search_response.reranking_latency_ms for result in per_query)
                if reranking_enabled
                else None
            ),
            mean_total_latency_ms=fmean(
                result.search_response.total_query_latency_ms for result in per_query
            ),
            evaluated_queries=len(per_query),
            queries_with_judgments=sum(bool(qrels.get(query.query_id)) for query in fixed_queries),
            warnings=warnings,
            query_results=tuple(per_query),
        )
