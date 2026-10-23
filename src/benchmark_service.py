"""Typed orchestration and exports for multi-pipeline retrieval benchmarks."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from time import perf_counter
from typing import TYPE_CHECKING

from .evaluation import EVALUATION_K_VALUES, EvaluationQuery, validate_evaluation_data
from .evaluation_service import EvaluationReport, EvaluationService, QueryEvaluation

if TYPE_CHECKING:
    from .interfaces import Reranker, RerankerAdapter, SearchEngine


BASE_PIPELINES = ("Dense", "BM25", "Hybrid")


@dataclass(frozen=True, slots=True)
class PipelineConfiguration:
    """One composable base retriever plus an optional reranking layer."""

    base_pipeline: str
    use_reranker: bool = False
    reranker_label: str = "Cross-Encoder"

    def __post_init__(self) -> None:
        if self.base_pipeline not in BASE_PIPELINES:
            raise ValueError(f"Unsupported benchmark pipeline: {self.base_pipeline}")

    @property
    def name(self) -> str:
        suffix = f" + {self.reranker_label}" if self.use_reranker else ""
        return f"{self.base_pipeline}{suffix}"


DEFAULT_BENCHMARK_CONFIGURATIONS = tuple(
    PipelineConfiguration(name) for name in BASE_PIPELINES
)
ALL_BENCHMARK_CONFIGURATIONS = DEFAULT_BENCHMARK_CONFIGURATIONS + tuple(
    PipelineConfiguration(name, use_reranker=True) for name in BASE_PIPELINES
)


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkResult:
    """A successful or failed pipeline run with stable model identity."""

    configuration: PipelineConfiguration
    reranker_name: str
    embedding_model_id: str | None
    reranker_model_id: str | None
    report: EvaluationReport | None
    error: str | None = None

    @property
    def pipeline_name(self) -> str:
        if self.configuration.use_reranker and self.reranker_name.startswith("remote:"):
            return f"{self.configuration.base_pipeline} + Remote Reranker"
        return self.configuration.name

    @property
    def succeeded(self) -> bool:
        return self.report is not None and self.error is None

    @property
    def metrics(self) -> Mapping[str, Mapping[int, float]]:
        return {} if self.report is None else self.report.metrics

    @property
    def query_results(self) -> tuple[QueryEvaluation, ...]:
        return () if self.report is None else self.report.query_results

    @property
    def mean_retrieval_latency_ms(self) -> float | None:
        return None if self.report is None else self.report.mean_retrieval_latency_ms

    @property
    def mean_reranking_latency_ms(self) -> float | None:
        return None if self.report is None else self.report.mean_reranking_latency_ms

    @property
    def mean_total_latency_ms(self) -> float | None:
        return None if self.report is None else self.report.mean_total_latency_ms

    @property
    def evaluated_queries(self) -> int:
        return 0 if self.report is None else self.report.evaluated_queries


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Complete benchmark outcome, including failures and per-query evidence."""

    k_values: tuple[int, ...]
    query_ids: tuple[str, ...]
    candidate_depth: int
    total_duration_ms: float
    pipeline_results: tuple[PipelineBenchmarkResult, ...]
    warnings: tuple[str, ...]

    @property
    def successful_results(self) -> tuple[PipelineBenchmarkResult, ...]:
        return tuple(result for result in self.pipeline_results if result.succeeded)

    @property
    def failed_results(self) -> tuple[PipelineBenchmarkResult, ...]:
        return tuple(result for result in self.pipeline_results if not result.succeeded)


@dataclass(frozen=True, slots=True)
class StoredBenchmarkRun:
    """Session-safe benchmark result tied to the exact inputs that produced it."""

    signature: tuple[object, ...]
    result: BenchmarkResult


class BenchmarkService:
    """Evaluate selected compositions over shared, already-built retrieval engines."""

    def __init__(
        self,
        engines: Mapping[str, SearchEngine | None],
        *,
        embedding_model_id: str | None = None,
        reranker_model_id: str | None = None,
        reranker_loader: Callable[[], RerankerAdapter | Reranker] | None = None,
    ) -> None:
        self.engines = dict(engines)
        dense_engine = self.engines.get("Dense") or self.engines.get("Hybrid")
        dense_engine = getattr(dense_engine, "dense_engine", dense_engine)
        encoder = getattr(dense_engine, "embedder", None)
        actual_id = getattr(encoder, "model_id", None)
        self.embedding_model_id = (actual_id if getattr(encoder, "identity", {}).get("source") == "remote"
                                   else embedding_model_id or actual_id)
        self.reranker_model_id = reranker_model_id
        self.reranker_loader = reranker_loader

    def evaluate(
        self,
        configurations: Sequence[PipelineConfiguration],
        queries: Sequence[EvaluationQuery],
        qrels: Mapping[str, Mapping[str, float]],
        *,
        k_values: Sequence[int] = EVALUATION_K_VALUES,
        progress: Callable[[str, int, int, EvaluationQuery], None] | None = None,
    ) -> BenchmarkResult:
        fixed_configurations = tuple(configurations)
        fixed_queries = tuple(queries)
        fixed_k = tuple(k_values)
        if not fixed_configurations:
            raise ValueError("Select at least one benchmark configuration.")
        names = [configuration.name for configuration in fixed_configurations]
        if len(set(names)) != len(names):
            raise ValueError("Benchmark configurations must be unique.")
        if not fixed_queries:
            raise ValueError("Benchmarking requires at least one query.")

        available = {
            configuration.base_pipeline: self.engines.get(configuration.base_pipeline)
            for configuration in fixed_configurations
            if self.engines.get(configuration.base_pipeline) is not None
        }
        if not available:
            raise ValueError("None of the selected benchmark pipelines has an active index.")
        reference_engine = next(iter(available.values()))
        assert reference_engine is not None
        reference_passages = tuple(reference_engine.passages)
        candidate_depth = reference_engine.candidate_depth
        for name, engine in available.items():
            assert engine is not None
            if tuple(engine.passages) != reference_passages:
                raise ValueError(
                    f"{name} does not use the same active corpus as the other benchmark pipelines."
                )
            if engine.candidate_depth != candidate_depth:
                raise ValueError(
                    f"{name} uses a different candidate depth; the comparison would be invalid."
                )

        warnings = validate_evaluation_data(fixed_queries, qrels, reference_engine.passages)
        started = perf_counter()
        results: list[PipelineBenchmarkResult] = []
        shared_reranker: RerankerAdapter | Reranker | None = None
        reranker_load_error: Exception | None = None

        for configuration in fixed_configurations:
            engine = self.engines.get(configuration.base_pipeline)
            reranker_name = "Off"
            try:
                if engine is None or not engine.is_indexed:
                    raise RuntimeError(
                        f"The {configuration.base_pipeline} index is unavailable or stale."
                    )
                reranker = None
                if configuration.use_reranker:
                    reranker_name = self.reranker_model_id or "Cross-Encoder"
                    if reranker_load_error is not None:
                        raise reranker_load_error
                    if shared_reranker is None:
                        if self.reranker_loader is None:
                            raise RuntimeError("No compatible reranker adapter is selected.")
                        try:
                            shared_reranker = self.reranker_loader()
                        except Exception as exc:
                            reranker_load_error = exc
                            raise
                    reranker = shared_reranker
                    reranker_name = self.reranker_model_id or getattr(reranker, "model_id", reranker.name)
                    if getattr(reranker, "identity", {}).get("source") == "remote":
                        reranker_name = reranker.model_id

                def update_progress(
                    index: int,
                    total: int,
                    query: EvaluationQuery,
                ) -> None:
                    if progress is not None:
                        progress(configuration.name, index, total, query)

                report = EvaluationService(engine, reranker).evaluate(
                    fixed_queries,
                    qrels,
                    k_values=fixed_k,
                    progress=update_progress,
                )
                results.append(
                    PipelineBenchmarkResult(
                        configuration=configuration,
                        reranker_name=reranker_name,
                        embedding_model_id=(
                            self.embedding_model_id
                            if configuration.base_pipeline in {"Dense", "Hybrid"}
                            else None
                        ),
                        reranker_model_id=(
                            reranker_name if configuration.use_reranker else None
                        ),
                        report=report,
                    )
                )
            except Exception as exc:
                results.append(
                    PipelineBenchmarkResult(
                        configuration=configuration,
                        reranker_name=reranker_name,
                        embedding_model_id=(
                            self.embedding_model_id
                            if configuration.base_pipeline in {"Dense", "Hybrid"}
                            else None
                        ),
                        reranker_model_id=(
                            reranker_name if configuration.use_reranker else None
                        ),
                        report=None,
                        error=str(exc) or exc.__class__.__name__,
                    )
                )

        return BenchmarkResult(
            k_values=fixed_k,
            query_ids=tuple(query.query_id for query in fixed_queries),
            candidate_depth=candidate_depth,
            total_duration_ms=(perf_counter() - started) * 1000.0,
            pipeline_results=tuple(results),
            warnings=warnings,
        )


def benchmark_aggregate_rows(result: BenchmarkResult) -> list[dict[str, object]]:
    """Return stable, flat rows for display and CSV serialization."""

    rows: list[dict[str, object]] = []
    for pipeline in result.pipeline_results:
        row: dict[str, object] = {
            "Pipeline": pipeline.pipeline_name,
            "Reranker": pipeline.reranker_name,
            "Embedding model": pipeline.embedding_model_id or "Not applicable",
            "Reranker model": pipeline.reranker_model_id or "Off",
            "Status": "Succeeded" if pipeline.succeeded else "Failed",
        }
        for metric_name in ("Recall", "Hit Rate", "MRR", "NDCG"):
            for k in result.k_values:
                row[f"{metric_name}@{k}"] = (
                    pipeline.metrics[metric_name][k] if pipeline.succeeded else None
                )
        row.update(
            {
                "Mean retrieval latency (ms)": pipeline.mean_retrieval_latency_ms,
                "Mean reranking latency (ms)": pipeline.mean_reranking_latency_ms,
                "Mean total latency (ms)": pipeline.mean_total_latency_ms,
                "Evaluated queries": pipeline.evaluated_queries,
                "Error": pipeline.error or "",
            }
        )
        rows.append(row)
    return rows


def benchmark_to_csv(result: BenchmarkResult) -> str:
    rows = benchmark_aggregate_rows(result)
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _query_payload(query: QueryEvaluation) -> dict[str, object]:
    return {
        "query_id": query.query_id,
        "query_text": query.query_text,
        "relevant_document_count": query.relevant_document_count,
        "first_relevant_rank": query.first_relevant_rank,
        "hit_at_10": query.hit_at_10,
        "retrieved_document_ids": list(query.retrieved_document_ids),
        "relevant_document_ids_retrieved": list(query.relevant_document_ids),
        "metrics": {
            str(k): dict(metric_values) for k, metric_values in query.metrics.items()
        },
        "latency": {
            "retrieval_ms": query.search_response.retrieval_latency_ms,
            "reranking_ms": (
                query.search_response.reranking_latency_ms
                if query.search_response.reranking_used
                else None
            ),
            "total_ms": query.search_response.total_query_latency_ms,
        },
    }


def benchmark_to_json(result: BenchmarkResult) -> str:
    payload = {
        "configuration": {
            "pipelines": [item.pipeline_name for item in result.pipeline_results],
            "candidate_depth": result.candidate_depth,
            "k_values": list(result.k_values),
            "query_ids": list(result.query_ids),
        },
        "total_benchmark_duration_ms": result.total_duration_ms,
        "warnings": list(result.warnings),
        "results": [
            {
                "pipeline": pipeline.pipeline_name,
                "base_pipeline": pipeline.configuration.base_pipeline,
                "reranker": pipeline.reranker_name,
                "embedding_model_id": pipeline.embedding_model_id,
                "reranker_model_id": pipeline.reranker_model_id,
                "status": "succeeded" if pipeline.succeeded else "failed",
                "error": pipeline.error,
                "aggregate_metrics": {
                    metric_name: {
                        str(k): value for k, value in values.items()
                    }
                    for metric_name, values in pipeline.metrics.items()
                },
                "latency": {
                    "mean_retrieval_ms": pipeline.mean_retrieval_latency_ms,
                    "mean_reranking_ms": pipeline.mean_reranking_latency_ms,
                    "mean_total_ms": pipeline.mean_total_latency_ms,
                },
                "evaluated_queries": pipeline.evaluated_queries,
                "per_query": [_query_payload(query) for query in pipeline.query_results],
            }
            for pipeline in result.pipeline_results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
