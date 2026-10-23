import pytest

from src.services import CorpusService, SearchService

from .helpers import FakeEmbedder, axis_vectors, make_test_config


def _corpus_service(passage_count: int) -> CorpusService:
    vectors = axis_vectors(passage_count)
    embedder = FakeEmbedder(vectors, vectors[0])
    return CorpusService(make_test_config(), lambda: embedder)


def test_corpus_service_builds_all_public_engines_from_document_bytes() -> None:
    prepared = _corpus_service(1).prepare([("paper.txt", b"scientific evidence")])

    assert prepared.engine.is_indexed
    assert prepared.bm25_engine.is_indexed
    assert prepared.hybrid_engine.is_indexed
    assert prepared.bm25_engine.passages is prepared.engine.passages
    assert prepared.hybrid_engine.passages is prepared.engine.passages
    assert prepared.files == 1
    assert prepared.documents == 1
    assert prepared.passages == 1
    assert prepared.embeddings == 1
    assert prepared.engine.passages[0].text == "scientific evidence"


def test_failed_replacement_leaves_existing_engine_available_to_the_caller() -> None:
    current = _corpus_service(1).prepare([("current.txt", b"working corpus")]).engine
    session_state = {"engine": current}

    with pytest.raises(ValueError, match="Unexpected passage count"):
        _corpus_service(1).prepare(
            [("replacement-a.txt", b"first"), ("replacement-b.txt", b"second")]
        )

    assert session_state["engine"] is current
    assert SearchService(current).search("query").indexed_passages == 1


def test_search_service_runs_dense_search() -> None:
    engine = _corpus_service(2).prepare(
        [("first.txt", b"first"), ("second.txt", b"second")]
    ).engine

    response = SearchService(engine).search("query")

    assert response.candidate_depth_used == 2
    assert [result.rank for result in response.results] == [1, 2]


def test_corpus_service_prepares_bm25_from_the_same_passages() -> None:
    prepared = _corpus_service(2).prepare(
        [("first.txt", b"chlorophyll evidence"), ("second.txt", b"unrelated material")]
    )

    response = SearchService(prepared.bm25_engine).search("chlorophyll")

    assert response.retrieval_method == "BM25"
    assert response.results[0].passage is prepared.engine.passages[0]


def test_corpus_service_prepares_hybrid_without_reprocessing_passages() -> None:
    vectors = axis_vectors(2)
    embedder = FakeEmbedder(vectors, vectors[0])
    prepared = CorpusService(make_test_config(), lambda: embedder).prepare(
        [("first.txt", b"chlorophyll evidence"), ("second.txt", b"unrelated material")]
    )

    response = SearchService(prepared.hybrid_engine).search("chlorophyll")

    assert response.retrieval_method == "Hybrid"
    assert prepared.hybrid_engine.dense_engine is prepared.engine
    assert prepared.hybrid_engine.bm25_engine is prepared.bm25_engine
    assert embedder.passage_calls == 1
    assert all(result.passage in prepared.engine.passages for result in response.results)


def test_search_service_preserves_direct_engine_results() -> None:
    engine = _corpus_service(3).prepare(
        [("first.txt", b"first"), ("second.txt", b"second"), ("third.txt", b"third")]
    ).engine

    direct = engine.search("same query")
    through_service = SearchService(engine).search("same query")

    assert through_service.query == direct.query
    assert through_service.candidate_depth_used == direct.candidate_depth_used
    assert through_service.indexed_passages == direct.indexed_passages
    assert through_service.results == direct.results
