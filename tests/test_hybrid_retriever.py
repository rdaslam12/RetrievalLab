import pytest

from src.bm25_retriever import BM25SearchEngine
from src.hybrid_retriever import DEFAULT_RRF_K, HybridSearchEngine, reciprocal_rank_fusion
from src.search_engine import DenseSearchEngine
from src.services import SearchService
from src.types import Passage, VectorHit
from src.vector_index import NumpyVectorIndex

from .helpers import FakeEmbedder, axis_vectors, make_test_config


def _passages(count: int) -> tuple[Passage, ...]:
    return tuple(
        Passage(
            chunk_id=f"chunk-{index}",
            source_id=f"source-{index}",
            file_name=f"document-{index}.txt",
            file_type="txt",
            page_number=None,
            passage_index=1,
            text=f"passage {index}",
            token_count=2,
            document_id=f"document-{index}",
        )
        for index in range(count)
    )


def test_rrf_math_rewards_overlap_and_preserves_branch_only_candidates() -> None:
    passages = _passages(3)
    dense_hits = [VectorHit(0, 0.9, 1), VectorHit(1, 0.8, 2)]
    bm25_hits = [VectorHit(1, 4.0, 1), VectorHit(2, 3.0, 2)]

    fused = reciprocal_rank_fusion(passages, dense_hits, bm25_hits)

    assert [result.passage.chunk_id for result in fused] == ["chunk-1", "chunk-0", "chunk-2"]
    assert fused[0].rrf_score == pytest.approx(
        1 / (DEFAULT_RRF_K + 2) + 1 / (DEFAULT_RRF_K + 1)
    )
    assert fused[1].rrf_score == pytest.approx(1 / (DEFAULT_RRF_K + 1))
    assert fused[2].rrf_score == pytest.approx(1 / (DEFAULT_RRF_K + 2))
    assert fused[1].bm25_rank is None
    assert fused[2].dense_rank is None


def test_rrf_preserves_raw_scores_metadata_and_identity() -> None:
    passages = _passages(2)
    dense_hits = [VectorHit(1, 0.75, 1)]
    bm25_hits = [VectorHit(1, 6.5, 2)]

    result = reciprocal_rank_fusion(passages, dense_hits, bm25_hits)[0]

    assert result.passage is passages[1]
    assert result.passage.document_id == "document-1"
    assert result.dense_rank == 1
    assert result.dense_score == 0.75
    assert result.bm25_rank == 2
    assert result.bm25_score == 6.5


def test_rrf_ties_use_original_corpus_order() -> None:
    passages = _passages(2)

    fused = reciprocal_rank_fusion(
        passages,
        [VectorHit(1, 0.9, 1)],
        [VectorHit(0, 5.0, 1)],
    )

    assert [result.passage.chunk_id for result in fused] == ["chunk-0", "chunk-1"]


def _hybrid_engine() -> tuple[HybridSearchEngine, FakeEmbedder]:
    passages = _passages(3)
    vectors = axis_vectors(3)
    embedder = FakeEmbedder(vectors, vectors[0])
    dense = DenseSearchEngine(
        make_test_config(),
        embedder,
        index_factory=lambda matrix: NumpyVectorIndex(matrix),
    )
    dense.index_passages(passages)
    bm25 = BM25SearchEngine(make_test_config(), dense.passages)
    return HybridSearchEngine(dense, bm25), embedder


def test_hybrid_runs_both_public_branches_and_preserves_provenance() -> None:
    engine, embedder = _hybrid_engine()

    response = SearchService(engine).search("passage 1")

    assert response.retrieval_method == "Hybrid"
    assert embedder.query_calls == 1
    assert all(result.rrf_score is not None for result in response.results)
    assert any(result.dense_rank is not None for result in response.results)
    assert any(result.bm25_rank is not None for result in response.results)
