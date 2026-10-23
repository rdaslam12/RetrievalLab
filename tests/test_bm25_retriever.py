import math

import pytest

from src.bm25_retriever import BM25Retriever, BM25SearchEngine
from src.services import SearchService
from src.types import Passage

from .helpers import make_test_config


def _passage(index: int, text: str, *, document_id: str | None = None) -> Passage:
    return Passage(
        chunk_id=f"chunk-{index}",
        source_id=f"source-{index}",
        file_name=f"document-{index}.jsonl",
        file_type="jsonl",
        page_number=None,
        passage_index=index + 1,
        text=text,
        token_count=len(text.split()),
        document_id=document_id,
        title=f"Title {index}",
    )


def test_bm25_indexes_and_ranks_exact_lexical_matches() -> None:
    passages = [
        _passage(0, "photosynthesis chlorophyll sunlight"),
        _passage(1, "galaxy orbit telescope"),
        _passage(2, "protein folding assay"),
    ]

    hits = BM25Retriever(passages).search("chlorophyll", k=3)

    assert hits[0].passage_index == 0
    assert hits[0].rank == 1
    assert hits[0].score > hits[1].score
    assert hits[1].score == 0.0


def test_bm25_preserves_raw_score_and_passage_metadata() -> None:
    passages = [
        _passage(0, "marker marker marker", document_id="relevant-doc"),
        _passage(1, "unrelated terms", document_id="other-doc"),
    ]
    engine = BM25SearchEngine(make_test_config(), passages)

    response = SearchService(engine).search("marker")

    result = response.results[0]
    idf = math.log(1.0 + (2 - 1 + 0.5) / (1 + 0.5))
    average_length = 2.5
    denominator = 3 + 1.5 * (1 - 0.75 + 0.75 * (3 / average_length))
    expected_score = idf * (3 * (1.5 + 1)) / denominator
    assert result.score == pytest.approx(expected_score)
    assert result.passage.document_id == "relevant-doc"
    assert result.passage.title == "Title 0"
    assert result.passage.chunk_id == "chunk-0"
    assert response.retrieval_method == "BM25"


def test_bm25_operates_without_an_embedder() -> None:
    engine = BM25SearchEngine(
        make_test_config(),
        [_passage(0, "lexical retrieval"), _passage(1, "dense embeddings")],
    )

    response = engine.search("lexical")

    assert response.results[0].passage.chunk_id == "chunk-0"
    assert response.device == "cpu"
