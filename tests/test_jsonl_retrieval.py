import json

import numpy as np

from src.chunker import chunk_sections
from src.document_parser import parse_corpus
from src.search_engine import DenseSearchEngine
from src.vector_index import NumpyVectorIndex

from .helpers import FakeEmbedder, WhitespaceTokenizer, make_test_config


def test_beir_jsonl_corpus_can_be_indexed_and_retrieved() -> None:
    payload = (
        b'{"_id":"dense","title":"Dense retrieval","text":"Embeddings find passages."}\n'
        b'{"_id":"lexical","title":"Lexical retrieval","text":"Terms find passages."}\n'
    )
    parsed = parse_corpus([("corpus.jsonl", payload)])
    passages = chunk_sections(parsed.sections, WhitespaceTokenizer())
    passage_vectors = np.zeros((2, 384), dtype=np.float32)
    passage_vectors[0, 0] = 1.0
    passage_vectors[1, 1] = 1.0
    query_vector = passage_vectors[0].copy()
    engine = DenseSearchEngine(
        make_test_config(),
        FakeEmbedder(passage_vectors, query_vector),
        index_factory=lambda matrix: NumpyVectorIndex(matrix),
    )

    engine.index_passages(passages)
    response = engine.search("How are embeddings used?")

    assert response.results[0].passage.document_id == "dense"
    assert response.results[0].passage.title == "Dense retrieval"
    assert response.results[0].passage.file_type == "jsonl"


def test_large_jsonl_bounded_slice_indexes_and_searches_configured_limit() -> None:
    test_limit = 50
    payload = "\n".join(
        json.dumps({"_id": f"doc-{index}", "text": f"scientific document {index}"})
        for index in range(test_limit + 1)
    ).encode("utf-8")
    parsed = parse_corpus([("corpus-slice.jsonl", payload)])
    passages = chunk_sections(
        parsed.sections,
        WhitespaceTokenizer(),
        max_passages=test_limit,
        truncate_at_limit=True,
    )
    passage_vectors = np.zeros((test_limit, 384), dtype=np.float32)
    passage_vectors[:, 0] = 1.0
    query_vector = passage_vectors[0].copy()
    engine = DenseSearchEngine(
        make_test_config(),
        FakeEmbedder(passage_vectors, query_vector),
        index_factory=lambda matrix: NumpyVectorIndex(matrix),
    )

    engine.index_passages(passages)
    response = engine.search("scientific evidence")

    assert make_test_config().max_passages == 35_000
    assert len(passages) == test_limit
    assert passages[-1].document_id == "doc-49"
    assert response.indexed_passages == test_limit
    assert response.candidate_depth_used == test_limit
    assert len(response.results) == 10
