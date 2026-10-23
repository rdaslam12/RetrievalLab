from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from src.bm25_retriever import BM25SearchEngine
from src.hybrid_retriever import HybridSearchEngine
from src.reranker import CrossEncoderReranker, rerank_candidates
from src.search_engine import DenseSearchEngine
from src.services import SearchService
from src.types import Passage, RankedPassage
from src.vector_index import NumpyVectorIndex

from .helpers import FakeEmbedder, FakeReranker, axis_vectors, make_test_config


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


def _engines(count: int = 6) -> tuple[
    DenseSearchEngine,
    BM25SearchEngine,
    HybridSearchEngine,
    FakeEmbedder,
]:
    passages = _passages(count)
    vectors = axis_vectors(count)
    embedder = FakeEmbedder(vectors, vectors[0])
    dense = DenseSearchEngine(
        make_test_config(),
        embedder,
        index_factory=lambda matrix: NumpyVectorIndex(matrix),
    )
    dense.index_passages(passages)
    bm25 = BM25SearchEngine(make_test_config(), dense.passages)
    return dense, bm25, HybridSearchEngine(dense, bm25), embedder


def test_reranking_changes_order_without_changing_the_candidate_set() -> None:
    passages = _passages(3)
    candidates = tuple(
        RankedPassage(
            passage=passage,
            rank=index,
            score=1.0 / index,
            retrieval_rank=index,
            retrieval_score=1.0 / index,
        )
        for index, passage in enumerate(passages, start=1)
    )

    reranked = rerank_candidates("query", candidates, FakeReranker([0.1, 0.9, 0.2]))

    assert [item.passage.chunk_id for item in reranked] == ["chunk-1", "chunk-2", "chunk-0"]
    assert {item.passage.chunk_id for item in reranked} == {
        item.passage.chunk_id for item in candidates
    }
    assert reranked[0].rank == 1
    assert reranked[0].retrieval_rank == 2
    assert reranked[0].retrieval_score == 0.5
    assert reranked[0].score == 0.5
    assert reranked[0].reranker_score == 0.9
    assert reranked[0].rank_change == 1
    assert reranked[-1].rank_change == -2


def test_reranker_ties_preserve_original_retrieval_order() -> None:
    candidates = tuple(
        RankedPassage(passage=passage, rank=index, score=0.0)
        for index, passage in enumerate(_passages(3), start=1)
    )

    reranked = rerank_candidates("query", candidates, FakeReranker([1.0, 1.0, 1.0]))

    assert [item.passage.chunk_id for item in reranked] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]


@pytest.mark.parametrize("engine_index", [0, 1, 2], ids=["dense", "bm25", "hybrid"])
def test_all_public_base_retrievers_support_reranking(engine_index: int) -> None:
    engines = _engines()
    engine = engines[engine_index]
    reranker = FakeReranker()
    expected_candidates = engine.search_candidates("passage").results

    response = SearchService(engine, reranker).search("passage")

    assert reranker.candidate_ids[-1] == tuple(
        item.passage.chunk_id for item in expected_candidates
    )
    assert len(response.results) == len(expected_candidates)
    assert response.results[0].passage.chunk_id == expected_candidates[-1].passage.chunk_id
    assert response.reranking_used is True
    assert response.reranker_name == reranker.name
    assert response.reranking_latency_ms >= 0.0
    assert response.total_query_latency_ms >= response.reranking_latency_ms


@pytest.mark.parametrize("engine_index", [0, 1, 2], ids=["dense", "bm25", "hybrid"])
def test_full_candidate_depth_is_reranked_before_final_truncation(engine_index: int) -> None:
    engines = _engines(105)
    engine = engines[engine_index]
    reranker = FakeReranker()

    response = SearchService(engine, reranker).search("passage")

    assert response.candidate_depth_used == 100
    assert len(reranker.candidate_ids[0]) == 100
    assert len(response.results) == 10
    assert response.results[0].passage.chunk_id == "chunk-99"


def test_hybrid_branch_provenance_survives_reranking() -> None:
    _, _, hybrid, _ = _engines()

    response = SearchService(hybrid, FakeReranker()).search("passage 2")

    assert all(item.rrf_score is not None for item in response.results)
    assert any(item.dense_rank is not None and item.dense_score is not None for item in response.results)
    assert any(item.bm25_rank is not None and item.bm25_score is not None for item in response.results)
    assert all(item.retrieval_score == item.rrf_score for item in response.results)


def test_reranker_off_preserves_existing_retrieval_behavior() -> None:
    dense, _, _, _ = _engines()

    direct = dense.search("passage")
    through_service = SearchService(dense).search("passage")

    assert through_service.results == direct.results
    assert through_service.candidate_depth_used == direct.candidate_depth_used
    assert through_service.retrieval_method == direct.retrieval_method
    assert through_service.reranking_used is False
    assert through_service.reranker_name is None
    assert through_service.reranking_latency_ms == 0.0


def test_reranked_repeated_searches_do_not_rebuild_corpus_embeddings() -> None:
    dense, _, _, embedder = _engines()
    service = SearchService(dense, FakeReranker())

    service.search("first")
    service.search("second")

    assert embedder.passage_calls == 1
    assert embedder.query_calls == 2


class _FakeCrossEncoder:
    def __init__(self) -> None:
        self.predict_calls = 0

    def eval(self) -> object:
        return self

    def to(self, device: str) -> object:
        return self

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.predict_calls += 1
        return np.arange(len(inputs), dtype=np.float32)


def test_cross_encoder_model_loading_is_lazy_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[tuple[str, str, str]] = []
    model = _FakeCrossEncoder()

    def fake_loader(model_id: str, device: str, cache_folder: str) -> tuple[_FakeCrossEncoder, str]:
        loads.append((model_id, device, cache_folder))
        return model, device

    monkeypatch.setattr("src.reranker._load_cached_cross_encoder", fake_loader)
    reranker = CrossEncoderReranker(device="cpu")
    candidates: Sequence[RankedPassage] = (
        RankedPassage(passage=_passages(1)[0], rank=1, score=1.0),
    )

    assert loads == []
    reranker.score("first", candidates)
    reranker.score("second", candidates)

    assert len(loads) == 1
    assert model.predict_calls == 2
