import math

import pytest

from src.evaluation import (
    EvaluationDataError,
    EvaluationQuery,
    deduplicate_ranked_documents,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    parse_qrels_tsv,
    parse_query_jsonl,
    recall_at_k,
    validate_evaluation_data,
)
from src.evaluation_service import EvaluationService
from src.services import CorpusService
from src.types import Passage, RankedPassage

from .helpers import FakeEmbedder, FakeReranker, axis_vectors, make_test_config


def _prepared_corpus():
    payload = (
        b'{"_id":"doc-a","text":"alpha relevant evidence"}\n'
        b'{"_id":"doc-b","text":"beta lexical evidence"}\n'
        b'{"_id":"doc-c","text":"gamma other material"}\n'
    )
    vectors = axis_vectors(3)
    embedder = FakeEmbedder(vectors, vectors[0])
    prepared = CorpusService(make_test_config(), lambda: embedder).prepare(
        [("corpus.jsonl", payload)]
    )
    return prepared, embedder


def test_recall_hit_rate_and_mrr_at_k_with_multiple_relevant_documents() -> None:
    ranking = ["doc-a", "doc-x", "doc-b"]
    relevance = {"doc-a": 1.0, "doc-b": 1.0}

    assert recall_at_k(ranking, relevance, 1) == 0.5
    assert recall_at_k(ranking, relevance, 3) == 1.0
    assert hit_rate_at_k(ranking, relevance, 1) == 1.0
    assert mrr_at_k(ranking, relevance, 3) == 1.0


def test_metrics_are_zero_when_no_relevant_result_is_retrieved() -> None:
    ranking = ["doc-x", "doc-y"]
    relevance = {"doc-a": 1.0}

    assert recall_at_k(ranking, relevance, 10) == 0.0
    assert hit_rate_at_k(ranking, relevance, 10) == 0.0
    assert mrr_at_k(ranking, relevance, 10) == 0.0
    assert ndcg_at_k(ranking, relevance, 10) == 0.0


def test_ndcg_at_k_with_binary_relevance() -> None:
    actual = ndcg_at_k(["doc-x", "doc-a"], {"doc-a": 1.0}, 2)

    assert actual == pytest.approx(1.0 / math.log2(3))


def test_ndcg_at_k_uses_graded_relevance() -> None:
    actual = ndcg_at_k(["low", "high"], {"high": 3.0, "low": 1.0}, 2)
    expected = (1.0 + 7.0 / math.log2(3)) / (7.0 + 1.0 / math.log2(3))

    assert actual == pytest.approx(expected)


def test_duplicate_passages_are_deduplicated_before_cutoff() -> None:
    ranking = ["doc-a", "doc-a", "doc-b"]
    relevance = {"doc-a": 1.0, "doc-b": 1.0}

    assert recall_at_k(ranking, relevance, 2) == 1.0
    assert mrr_at_k(["doc-x", "doc-x", "doc-a"], {"doc-a": 1.0}, 2) == 0.5


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"not json\n", "not valid JSON"),
        (b'{"_id":"q1"}\n', "text"),
        (b'{"_id":"q1","text":"ok"}\n{"_id":"q1","text":"again"}\n', "duplicate"),
        (b"\n", "no query records"),
    ],
)
def test_malformed_query_files_are_rejected(payload: bytes, message: str) -> None:
    with pytest.raises(EvaluationDataError, match=message):
        parse_query_jsonl(payload)


def test_query_parser_accepts_string_and_integer_ids() -> None:
    queries = parse_query_jsonl(
        b'{"_id":"q1","text":"first"}\n{"_id":2,"text":"second"}\n'
    )

    assert [(query.query_id, query.text) for query in queries] == [
        ("q1", "first"),
        ("2", "second"),
    ]


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"q1 doc-a 1\n", "tab-separated"),
        (b"q1\tdoc-a\tnot-a-score\n", "numeric"),
        (b"q1\tdoc-a\t-1\n", "non-negative"),
        (b"query-id\tcorpus-id\tscore\n", "no relevance judgments"),
        (b"q1\tdoc-a\t1\nq1\tdoc-a\t2\n", "duplicate judgment"),
    ],
)
def test_malformed_qrels_are_rejected(payload: bytes, message: str) -> None:
    with pytest.raises(EvaluationDataError, match=message):
        parse_qrels_tsv(payload)


def test_qrels_parser_supports_an_optional_header() -> None:
    with_header = parse_qrels_tsv(b"query-id\tcorpus-id\tscore\nq1\tdoc-a\t2\n")
    without_header = parse_qrels_tsv(b"q1\tdoc-a\t2\n")

    assert with_header == without_header == {"q1": {"doc-a": 2.0}}


def test_unknown_query_ids_are_rejected_before_evaluation() -> None:
    prepared, _ = _prepared_corpus()

    with pytest.raises(EvaluationDataError, match="unknown query IDs.*missing"):
        validate_evaluation_data(
            [EvaluationQuery("q1", "alpha")],
            {"missing": {"doc-a": 1.0}},
            prepared.passage_records,
        )


def test_unknown_corpus_document_ids_are_rejected_before_evaluation() -> None:
    prepared, _ = _prepared_corpus()

    with pytest.raises(EvaluationDataError, match="unknown corpus/document IDs.*doc-missing"):
        validate_evaluation_data(
            [EvaluationQuery("q1", "alpha")],
            {"q1": {"doc-missing": 1.0}},
            prepared.passage_records,
        )


def test_unresolved_ids_stop_before_any_retrieval_query_runs() -> None:
    prepared, embedder = _prepared_corpus()

    with pytest.raises(EvaluationDataError):
        EvaluationService(prepared.engine).evaluate(
            [EvaluationQuery("q1", "alpha")], {"q1": {"missing": 1.0}}
        )

    assert embedder.query_calls == 0


@pytest.mark.parametrize("engine_name", ["engine", "bm25_engine", "hybrid_engine"])
def test_evaluation_runs_with_dense_bm25_and_hybrid(engine_name: str) -> None:
    prepared, _ = _prepared_corpus()
    engine = getattr(prepared, engine_name)

    report = EvaluationService(engine).evaluate(
        [EvaluationQuery("q1", "alpha")], {"q1": {"doc-a": 1.0}}
    )

    assert report.evaluated_queries == 1
    assert report.queries_with_judgments == 1
    assert report.metrics["Hit Rate"][10] == 1.0
    assert report.query_results[0].retrieved_document_ids[0] == "doc-a"


def test_evaluation_with_reranking_scores_full_fixed_candidate_set() -> None:
    prepared, _ = _prepared_corpus()
    reranker = FakeReranker(scores=[0.0, 1.0, 2.0])

    report = EvaluationService(prepared.engine, reranker).evaluate(
        [EvaluationQuery("q1", "alpha")], {"q1": {"doc-c": 1.0}}
    )

    assert reranker.calls == 1
    assert len(reranker.candidate_ids[0]) == 3
    assert report.query_results[0].search_response.reranking_used
    assert report.query_results[0].retrieved_document_ids[0] == "doc-c"
    assert report.mean_reranking_latency_ms is not None


def test_evaluation_preserves_hybrid_provenance() -> None:
    prepared, _ = _prepared_corpus()

    report = EvaluationService(prepared.hybrid_engine).evaluate(
        [EvaluationQuery("q1", "alpha")], {"q1": {"doc-a": 1.0}}
    )

    response = report.query_results[0].search_response
    assert response.retrieval_method == "Hybrid"
    assert all(result.rrf_score is not None for result in response.results)
    assert any(result.dense_rank is not None for result in response.results)
    assert any(result.bm25_rank is not None for result in response.results)


def test_repeated_evaluation_queries_reuse_corpus_embeddings() -> None:
    prepared, embedder = _prepared_corpus()

    EvaluationService(prepared.engine).evaluate(
        [EvaluationQuery("q1", "alpha"), EvaluationQuery("q2", "beta")],
        {"q1": {"doc-a": 1.0}, "q2": {"doc-b": 1.0}},
    )

    assert embedder.passage_calls == 1
    assert embedder.query_calls == 2


def test_evaluation_does_not_mutate_interactive_search_state_or_index() -> None:
    prepared, _ = _prepared_corpus()
    interactive_response = object()
    session_state = {"search_response": interactive_response, "engine": prepared.engine}
    passages_before = prepared.engine.passages
    embeddings_before = prepared.engine.embeddings

    EvaluationService(prepared.engine).evaluate(
        [EvaluationQuery("q1", "alpha")], {"q1": {"doc-a": 1.0}}
    )

    assert session_state == {"search_response": interactive_response, "engine": prepared.engine}
    assert prepared.engine.passages is passages_before
    assert prepared.engine.embeddings is embeddings_before


def test_unjudged_queries_remain_in_metric_aggregation() -> None:
    prepared, _ = _prepared_corpus()

    report = EvaluationService(prepared.engine).evaluate(
        [EvaluationQuery("q1", "alpha"), EvaluationQuery("q2", "alpha")],
        {"q1": {"doc-a": 1.0}},
    )

    assert report.metrics["Hit Rate"][1] == 0.5
    assert report.queries_with_judgments == 1
    assert report.warnings


def test_ranked_passage_deduplication_uses_document_id_then_source_id() -> None:
    passages = [
        Passage(
            chunk_id=f"chunk-{index}",
            source_id="source-a" if index < 2 else "source-b",
            file_name="a.txt" if index < 2 else "b.txt",
            file_type="txt",
            page_number=None,
            passage_index=index + 1,
            text="text",
            token_count=1,
        )
        for index in range(3)
    ]
    results = [
        RankedPassage(passage=passage, rank=index + 1, score=1.0)
        for index, passage in enumerate(passages)
    ]

    assert deduplicate_ranked_documents(results) == ("source-a", "source-b")
