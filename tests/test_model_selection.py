from __future__ import annotations

from collections.abc import Sequence
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.bm25_retriever import BM25SearchEngine
from src.config import load_config
from src.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingModelUnavailableError,
    SentenceTransformerEmbedder,
    _load_cached_sentence_transformer,
)
from src.hybrid_retriever import HybridSearchEngine
from src.model_ids import ModelValidationError, validate_huggingface_model_id
from src.model_selection import invalidate_dense_indexes_for_model, resolve_model_choice
from src.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankerUnavailableError,
    _load_cached_cross_encoder,
)
from src.search_engine import DenseSearchEngine
from src.services import CorpusService, SearchService, build_dense_engines
from src.types import Passage, RankedPassage
from src.vector_index import NumpyVectorIndex

from .helpers import FakeEmbedder, WhitespaceTokenizer, axis_vectors, make_test_config


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
        )
        for index in range(count)
    )


def _dense_engine(model_id: str, *, dimension: int = 8) -> tuple[DenseSearchEngine, FakeEmbedder]:
    passages = _passages(3)
    vectors = axis_vectors(3, dimension)
    embedder = FakeEmbedder(vectors, vectors[0], model_id=model_id)
    engine = DenseSearchEngine(
        make_test_config(),
        embedder,
        index_factory=lambda matrix: NumpyVectorIndex(matrix),
    )
    engine.index_passages(passages)
    return engine, embedder


def test_default_model_choices_remain_unchanged() -> None:
    config = load_config()

    assert config.model_id == DEFAULT_EMBEDDING_MODEL == "Snowflake/snowflake-arctic-embed-s"
    assert DEFAULT_RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert resolve_model_choice(
        "Default", "ignored/custom", default_model_id=config.model_id, kind="embedding"
    ) == config.model_id
    assert resolve_model_choice(
        "Default", "ignored/custom", default_model_id=DEFAULT_RERANKER_MODEL, kind="reranker"
    ) == DEFAULT_RERANKER_MODEL


def test_custom_embedding_id_is_passed_to_the_loader_and_dimension_is_inferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[tuple[str, str, str]] = []
    fake_model = SimpleNamespace(tokenizer=WhitespaceTokenizer())

    def fake_loader(model_id: str, device: str, cache_folder: str) -> tuple[object, str, int]:
        loaded.append((model_id, device, cache_folder))
        return fake_model, device, 7

    monkeypatch.setattr("src.embedder._load_cached_sentence_transformer", fake_loader)

    embedder = SentenceTransformerEmbedder("research/custom-embedder", device="cpu")

    assert loaded[0][0] == "research/custom-embedder"
    assert embedder.model_id == "research/custom-embedder"
    assert embedder.dimension == 7


def test_dense_index_uses_inferred_model_dimension_instead_of_config_dimension() -> None:
    engine, _ = _dense_engine("research/seven-dimensional", dimension=7)

    response = engine.search("query")

    assert engine.embeddings is not None
    assert engine.embeddings.shape == (3, 7)
    assert response.embedding_dimension == 7
    assert response.dense_model_id == "research/seven-dimensional"
    assert engine.config.embedding_dimension == 384


def test_embedding_model_change_invalidates_then_rebuilds_only_dense_indexes() -> None:
    original, _ = _dense_engine("research/model-a")
    bm25 = BM25SearchEngine(make_test_config(), original.passages)
    state: dict[str, object] = {
        "engine": original,
        "bm25_engine": bm25,
        "hybrid_engine": HybridSearchEngine(original, bm25),
        "search_response": original.search("query"),
        "corpus_summary": {"embeddings": 3},
    }

    changed = invalidate_dense_indexes_for_model(state, "research/model-b")

    assert changed is True
    assert "engine" not in state
    assert "hybrid_engine" not in state
    assert "search_response" not in state
    assert state["bm25_engine"] is bm25
    assert state["corpus_summary"] == {
        "embeddings": 0,
        "embedding_model_id": None,
        "embedding_dimension": None,
    }

    vectors = axis_vectors(3, 5)
    replacement = FakeEmbedder(vectors, vectors[0], model_id="research/model-b")
    dense, hybrid = build_dense_engines(
        make_test_config(),
        bm25.passages,
        lambda: replacement,
        bm25_engine=bm25,
    )

    assert dense.embedder is replacement
    assert dense.embeddings is not None and dense.embeddings.shape == (3, 5)
    assert hybrid.bm25_engine is bm25
    assert replacement.passage_calls == 1


def test_changing_only_reranker_does_not_rebuild_corpus_embeddings() -> None:
    dense, embedder = _dense_engine("research/stable-embedder")
    original_embeddings = dense.embeddings

    class NamedReranker:
        def __init__(self, name: str) -> None:
            self.name = name

        def score(
            self, query: str, candidates: Sequence[RankedPassage]
        ) -> Sequence[float]:
            return tuple(float(index) for index in range(len(candidates)))

    SearchService(dense, NamedReranker("research/reranker-a")).search("first")
    SearchService(dense, NamedReranker("research/reranker-b")).search("second")

    assert dense.embeddings is original_embeddings
    assert embedder.passage_calls == 1
    assert embedder.query_calls == 2


def test_successful_embedding_and_reranker_models_are_cached_by_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    embedding_loads: list[str] = []
    reranker_loads: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            embedding_loads.append(model_id)
            self.tokenizer = WhitespaceTokenizer()

        def eval(self) -> object:
            return self

        def get_sentence_embedding_dimension(self) -> int:
            return 13

    class FakeCrossEncoder:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            reranker_loads.append(model_id)

        def eval(self) -> object:
            return self

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer,
            CrossEncoder=FakeCrossEncoder,
        ),
    )
    cache_folder = str(tmp_path)
    _load_cached_sentence_transformer.cache_clear()
    _load_cached_cross_encoder.cache_clear()
    try:
        first_embedding = _load_cached_sentence_transformer(
            "research/cached-embedder", "cpu", cache_folder
        )[0]
        second_embedding = _load_cached_sentence_transformer(
            "research/cached-embedder", "cpu", cache_folder
        )[0]
        _load_cached_sentence_transformer("research/other-embedder", "cpu", cache_folder)
        first_reranker = _load_cached_cross_encoder(
            "research/cached-reranker", "cpu", cache_folder
        )[0]
        second_reranker = _load_cached_cross_encoder(
            "research/cached-reranker", "cpu", cache_folder
        )[0]
        _load_cached_cross_encoder("research/other-reranker", "cpu", cache_folder)
    finally:
        _load_cached_sentence_transformer.cache_clear()
        _load_cached_cross_encoder.cache_clear()

    assert first_embedding is second_embedding
    assert first_reranker is second_reranker
    assert embedding_loads == ["research/cached-embedder", "research/other-embedder"]
    assert reranker_loads == ["research/cached-reranker", "research/other-reranker"]


@pytest.mark.parametrize(
    "model_id",
    ["../local-model", "/tmp/model", "https://huggingface.co/org/model", "weights.pt", "a/b/c"],
)
def test_unsafe_or_non_hub_model_references_surface_clear_errors(model_id: str) -> None:
    with pytest.raises(ModelValidationError, match="Hugging Face"):
        validate_huggingface_model_id(model_id, kind="embedding")


def test_incompatible_models_surface_stage_specific_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    class BrokenModel:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            raise TypeError("incompatible architecture")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=BrokenModel, CrossEncoder=BrokenModel),
    )
    cache_folder = str(tmp_path)
    _load_cached_sentence_transformer.cache_clear()
    _load_cached_cross_encoder.cache_clear()
    try:
        with pytest.raises(
            EmbeddingModelUnavailableError,
            match="not a compatible SentenceTransformer",
        ):
            _load_cached_sentence_transformer("research/not-an-embedder", "cpu", cache_folder)
        with pytest.raises(RerankerUnavailableError, match="not a compatible Hugging Face"):
            _load_cached_cross_encoder("research/not-a-reranker", "cpu", cache_folder)
    finally:
        _load_cached_sentence_transformer.cache_clear()
        _load_cached_cross_encoder.cache_clear()


def test_bm25_preparation_does_not_load_an_embedding_model() -> None:
    calls = 0

    def forbidden_loader() -> FakeEmbedder:
        nonlocal calls
        calls += 1
        raise AssertionError("embedding loader must not be called")

    prepared = CorpusService(make_test_config(), forbidden_loader).prepare(
        [("paper.txt", b"lexical evidence remains searchable")],
        build_dense=False,
    )

    assert calls == 0
    assert prepared.engine is None
    assert prepared.hybrid_engine is None
    assert prepared.embeddings == 0
    assert prepared.bm25_engine.search("lexical").results[0].passage.text.startswith("lexical")


def test_custom_cross_encoder_reranks_the_fixed_retrieved_candidate_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense, _ = _dense_engine("research/stable-embedder")
    expected_candidates = dense.search_candidates("query").results

    class RecordingCrossEncoder:
        def __init__(self) -> None:
            self.pairs: list[tuple[str, str]] = []

        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
            self.pairs = list(pairs)
            return np.arange(len(pairs), dtype=np.float32)

    model = RecordingCrossEncoder()
    monkeypatch.setattr(
        "src.reranker._load_cached_cross_encoder",
        lambda model_id, device, cache_folder: (model, device),
    )
    reranker = CrossEncoderReranker("research/custom-cross-encoder", device="cpu")

    response = SearchService(dense, reranker).search("query")

    expected_texts = [candidate.passage.text for candidate in expected_candidates]
    assert [text for _, text in model.pairs] == expected_texts
    assert {result.passage.chunk_id for result in response.results} == {
        candidate.passage.chunk_id for candidate in expected_candidates
    }
    assert response.reranker_name == "research/custom-cross-encoder"
