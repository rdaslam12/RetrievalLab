import numpy as np

from src.vector_index import NumpyVectorIndex


def test_vector_retrieval_ranks_normalized_inner_products() -> None:
    root_two = np.sqrt(2.0)
    embeddings = np.asarray(
        [[1.0, 0.0], [1.0 / root_two, 1.0 / root_two], [0.0, 1.0]], dtype=np.float32
    )
    hits = NumpyVectorIndex(embeddings).search(np.asarray([1.0, 0.0], dtype=np.float32), 3)
    assert [hit.passage_index for hit in hits] == [0, 1, 2]
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert np.allclose([hit.score for hit in hits], [1.0, 1.0 / root_two, 0.0])


def test_exact_scoring_is_deterministic() -> None:
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(12, 8)).astype(np.float32)
    query = rng.normal(size=8).astype(np.float32)
    index = NumpyVectorIndex(embeddings)
    first = index.search(query, 7)
    second = index.search(query, 7)
    assert first == second

