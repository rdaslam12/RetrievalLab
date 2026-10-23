import csv
import json
from io import StringIO

import pytest

from src.benchmark_service import (
    BenchmarkService,
    PipelineConfiguration,
    benchmark_to_csv,
    benchmark_to_json,
)
from src.evaluation import EvaluationQuery
from src.evaluation_service import EvaluationService
from src.services import CorpusService

from .helpers import FakeEmbedder, FakeReranker, axis_vectors, make_test_config


def _prepared_corpus():
    payload = (
        b'{"_id":"doc-a","text":"alpha relevant evidence"}\n'
        b'{"_id":"doc-b","text":"beta lexical evidence"}\n'
        b'{"_id":"doc-c","text":"gamma other material"}\n'
    )
    embedder = FakeEmbedder(axis_vectors(3), axis_vectors(3)[0])
    prepared = CorpusService(make_test_config(), lambda: embedder).prepare(
        [("corpus.jsonl", payload)]
    )
    return prepared, embedder


def _dataset():
    queries = (
        EvaluationQuery("q1", "alpha"),
        EvaluationQuery("q2", "beta"),
    )
    qrels = {"q1": {"doc-a": 1.0}, "q2": {"doc-b": 1.0}}
    return queries, qrels


def _engines(prepared):
    return {
        "Dense": prepared.engine,
        "BM25": prepared.bm25_engine,
        "Hybrid": prepared.hybrid_engine,
    }


def test_dense_bm25_and_hybrid_benchmark_together_with_same_queries() -> None:
    prepared, _ = _prepared_corpus()
    queries, qrels = _dataset()
    configurations = tuple(
        PipelineConfiguration(name) for name in ("Dense", "BM25", "Hybrid")
    )

    result = BenchmarkService(
        _engines(prepared), embedding_model_id="fake/embedding-model"
    ).evaluate(configurations, queries, qrels)

    assert not result.failed_results
    assert [item.pipeline_name for item in result.pipeline_results] == [
        "Dense",
        "BM25",
        "Hybrid",
    ]
    assert result.query_ids == ("q1", "q2")
    assert all(
        tuple(query.query_id for query in pipeline.query_results) == result.query_ids
        for pipeline in result.successful_results
    )


def test_benchmark_metrics_match_single_pipeline_evaluation() -> None:
    prepared, _ = _prepared_corpus()
    queries, qrels = _dataset()

    benchmark = BenchmarkService(_engines(prepared)).evaluate(
        [PipelineConfiguration("BM25")], queries, qrels
    )
    single = EvaluationService(prepared.bm25_engine).evaluate(queries, qrels)

    assert benchmark.successful_results[0].metrics == single.metrics
    assert [item.metrics for item in benchmark.successful_results[0].query_results] == [
        item.metrics for item in single.query_results
    ]
    assert [
        item.retrieved_document_ids
        for item in benchmark.successful_results[0].query_results
    ] == [item.retrieved_document_ids for item in single.query_results]


def test_reranked_and_non_reranked_variants_coexist_and_share_reranker() -> None:
    prepared, _ = _prepared_corpus()
    queries, qrels = _dataset()
    reranker = FakeReranker()
    loader_calls = 0

    def load_reranker():
        nonlocal loader_calls
        loader_calls += 1
        return reranker

    result = BenchmarkService(
        _engines(prepared),
        embedding_model_id="fake/embedding-model",
        reranker_model_id=reranker.name,
        reranker_loader=load_reranker,
    ).evaluate(
        (
            PipelineConfiguration("Dense"),
            PipelineConfiguration("Dense", use_reranker=True),
            PipelineConfiguration("BM25", use_reranker=True),
        ),
        queries,
        qrels,
    )

    assert not result.failed_results
    assert loader_calls == 1
    assert reranker.calls == len(queries) * 2
    assert result.pipeline_results[0].reranker_name == "Off"
    assert result.pipeline_results[1].reranker_model_id == reranker.name


def test_dense_corpus_embeddings_are_not_rebuilt_for_dense_and_hybrid() -> None:
    prepared, embedder = _prepared_corpus()
    queries, qrels = _dataset()

    BenchmarkService(_engines(prepared)).evaluate(
        (PipelineConfiguration("Dense"), PipelineConfiguration("Hybrid")),
        queries,
        qrels,
    )

    assert embedder.passage_calls == 1
    assert prepared.hybrid_engine.dense_engine is prepared.engine
    assert prepared.hybrid_engine.dense_engine.embeddings is prepared.engine.embeddings


class _FailingEngine:
    def __init__(self, source_engine) -> None:
        self.passages = source_engine.passages
        self.candidate_depth = source_engine.candidate_depth
        self.top_results = source_engine.top_results
        self.is_indexed = True

    def search(self, query: str):
        raise RuntimeError("synthetic retrieval failure")

    def search_with_limit(self, query: str, result_limit: int):
        raise RuntimeError("synthetic retrieval failure")

    def search_candidates(self, query: str):
        raise RuntimeError("synthetic retrieval failure")


def test_one_pipeline_failure_is_named_and_successes_are_preserved() -> None:
    prepared, _ = _prepared_corpus()
    queries, qrels = _dataset()
    engines = _engines(prepared)
    engines["BM25"] = _FailingEngine(prepared.bm25_engine)

    result = BenchmarkService(engines).evaluate(
        (PipelineConfiguration("Dense"), PipelineConfiguration("BM25")),
        queries,
        qrels,
    )

    assert [item.pipeline_name for item in result.successful_results] == ["Dense"]
    assert result.failed_results[0].pipeline_name == "BM25"
    assert result.failed_results[0].error == "synthetic retrieval failure"


def test_results_retain_configuration_and_model_identity() -> None:
    prepared, _ = _prepared_corpus()
    queries, qrels = _dataset()
    reranker = FakeReranker()

    result = BenchmarkService(
        _engines(prepared),
        embedding_model_id="custom/embedding-model",
        reranker_model_id="custom/reranker-model",
        reranker_loader=lambda: reranker,
    ).evaluate(
        [PipelineConfiguration("Hybrid", use_reranker=True)],
        queries,
        qrels,
    )
    pipeline = result.successful_results[0]

    assert pipeline.pipeline_name == "Hybrid + Cross-Encoder"
    assert pipeline.embedding_model_id == "custom/embedding-model"
    assert pipeline.reranker_model_id == "custom/reranker-model"
    assert pipeline.reranker_name == "custom/reranker-model"


def test_per_query_results_align_by_query_id_across_configurations() -> None:
    prepared, _ = _prepared_corpus()
    queries, qrels = _dataset()

    result = BenchmarkService(_engines(prepared)).evaluate(
        (PipelineConfiguration("Dense"), PipelineConfiguration("BM25")),
        queries,
        qrels,
    )

    aligned = {
        pipeline.pipeline_name: {
            query.query_id: query for query in pipeline.query_results
        }
        for pipeline in result.successful_results
    }
    assert set(aligned["Dense"]) == set(aligned["BM25"]) == {"q1", "q2"}
    assert aligned["Dense"]["q1"].relevant_document_ids == ("doc-a",)
    assert aligned["BM25"]["q2"].relevant_document_ids == ("doc-b",)


def test_csv_and_json_exports_are_valid_and_serializable() -> None:
    prepared, _ = _prepared_corpus()
    queries, qrels = _dataset()
    result = BenchmarkService(
        _engines(prepared), embedding_model_id="fake/embedding-model"
    ).evaluate(
        (PipelineConfiguration("Dense"), PipelineConfiguration("BM25")),
        queries,
        qrels,
    )

    csv_rows = list(csv.DictReader(StringIO(benchmark_to_csv(result))))
    payload = json.loads(benchmark_to_json(result))

    assert [row["Pipeline"] for row in csv_rows] == ["Dense", "BM25"]
    assert "Recall@10" in csv_rows[0]
    assert payload["configuration"]["query_ids"] == ["q1", "q2"]
    assert payload["results"][0]["per_query"][0]["query_id"] == "q1"
    assert payload["results"][0]["per_query"][0]["retrieved_document_ids"]


def test_benchmark_does_not_mutate_search_or_evaluation_dataset_state() -> None:
    prepared, _ = _prepared_corpus()
    queries, qrels = _dataset()
    query_upload_state = b'{"_id":"q1","text":"alpha"}\n'
    qrels_upload_state = b"q1\tdoc-a\t1\n"
    interactive_response = object()
    state = {
        "search_response": interactive_response,
        "evaluation_queries_bytes": query_upload_state,
        "evaluation_qrels_bytes": qrels_upload_state,
    }
    qrels_before = {query_id: dict(judgments) for query_id, judgments in qrels.items()}

    BenchmarkService(_engines(prepared)).evaluate(
        (PipelineConfiguration("Dense"), PipelineConfiguration("BM25")),
        queries,
        qrels,
    )

    assert state["search_response"] is interactive_response
    assert state["evaluation_queries_bytes"] is query_upload_state
    assert state["evaluation_qrels_bytes"] is qrels_upload_state
    assert queries == _dataset()[0]
    assert qrels == qrels_before


def test_benchmark_rejects_mismatched_candidate_depth_before_running() -> None:
    prepared, _ = _prepared_corpus()
    queries, qrels = _dataset()
    other_prepared = CorpusService(
        make_test_config(candidate_depth=2),
        lambda: FakeEmbedder(axis_vectors(3), axis_vectors(3)[0]),
    ).prepare(
        [
            (
                "corpus.jsonl",
                b'{"_id":"doc-a","text":"alpha relevant evidence"}\n'
                b'{"_id":"doc-b","text":"beta lexical evidence"}\n'
                b'{"_id":"doc-c","text":"gamma other material"}\n',
            )
        ]
    )

    with pytest.raises(ValueError, match="different candidate depth"):
        BenchmarkService(
            {"Dense": prepared.engine, "BM25": other_prepared.bm25_engine}
        ).evaluate(
            (PipelineConfiguration("Dense"), PipelineConfiguration("BM25")),
            queries,
            qrels,
        )
