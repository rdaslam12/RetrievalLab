import numpy as np

from src.search_engine import DenseSearchEngine
from src.types import Passage
from src.vector_index import NumpyVectorIndex

from .helpers import FakeEmbedder, axis_vectors, make_test_config


def _passages(count: int) -> list[Passage]:
    return [
        Passage(
            chunk_id=f"chunk-{index}",
            source_id=f"source-{index // 2}",
            file_name=f"document-{index // 2}.pdf",
            file_type="pdf",
            page_number=index + 1,
            passage_index=index + 1,
            text=f"Technical passage {index}",
            token_count=3,
        )
        for index in range(count)
    ]


def _engine(count: int) -> tuple[DenseSearchEngine, FakeEmbedder]:
    vectors = axis_vectors(count)
    embedder = FakeEmbedder(vectors, vectors[0])
    engine = DenseSearchEngine(
        make_test_config(),
        embedder,
        index_factory=lambda matrix: NumpyVectorIndex(matrix),
    )
    engine.index_passages(_passages(count))
    return engine, embedder


def test_dense_engine_preserves_candidate_and_top_counts() -> None:
    engine, _ = _engine(4)
    response = engine.search("query")
    assert response.candidate_depth_used == 4
    assert len(response.results) == 4


def test_dense_retrieval_scores_and_order_remain_unchanged() -> None:
    engine, embedder = _engine(3)

    response = engine.search("query")

    assert [result.passage.chunk_id for result in response.results] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    assert [result.score for result in response.results] == [1.0, 0.0, 0.0]
    assert response.retrieval_method == "Dense"
    assert embedder.query_calls == 1


def test_search_results_preserve_source_and_page_metadata() -> None:
    engine, _ = _engine(3)
    result = engine.search("query").results[0]
    assert result.passage.file_name == "document-0.pdf"
    assert result.passage.page_number == 1
    assert result.passage.source_id == "source-0"


def test_empty_queries_are_rejected() -> None:
    engine, _ = _engine(1)
    for query in ("", "   \n"):
        try:
            engine.search(query)
        except ValueError as exc:
            assert "non-empty" in str(exc)
        else:
            raise AssertionError("Expected empty query rejection")


def test_corpus_embeddings_are_reused_for_repeated_queries() -> None:
    engine, embedder = _engine(2)
    engine.search("first")
    engine.search("second")
    assert embedder.passage_calls == 1
    assert embedder.query_calls == 2


def test_passage_and_query_embeddings_are_l2_normalized() -> None:
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[0, 0] = 3.0
    vectors[1, 1] = 4.0
    query = np.zeros(384, dtype=np.float32)
    query[0] = 2.0
    embedder = FakeEmbedder(vectors, query)
    engine = DenseSearchEngine(
        make_test_config(),
        embedder,
        index_factory=lambda matrix: NumpyVectorIndex(matrix),
    )
    engine.index_passages(_passages(2))

    response = engine.search("query")

    assert np.allclose(np.linalg.norm(engine.embeddings, axis=1), 1.0)
    assert response.results[0].score == 1.0


def test_one_passage_corpus_searches_successfully() -> None:
    engine, _ = _engine(1)
    response = engine.search("query")
    assert response.candidate_depth_used == 1
    assert len(response.results) == 1
