"""Deterministic test doubles shared across unit tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from src.config import AppConfig, load_config
from src.types import RankedPassage


class WhitespaceTokenizer:
    """Small reversible tokenizer for offline chunking tests."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        result = []
        for token in text.split():
            if token not in self.token_to_id:
                token_id = len(self.token_to_id) + 1
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            result.append(self.token_to_id[token])
        return result

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return " ".join(self.id_to_token[int(token_id)] for token_id in token_ids)


class FakeEmbedder:
    def __init__(self, passage_vectors: np.ndarray, query_vector: np.ndarray) -> None:
        self.passage_vectors = np.asarray(passage_vectors, dtype=np.float32)
        self.query_vector = np.asarray(query_vector, dtype=np.float32)
        self.tokenizer = WhitespaceTokenizer()
        self.device = "cpu"
        self.passage_calls = 0
        self.query_calls = 0

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        self.passage_calls += 1
        if len(texts) != len(self.passage_vectors):
            raise ValueError("Unexpected passage count")
        return self.passage_vectors.copy()

    def encode_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        return self.query_vector.copy()


class FakeReranker:
    """Deterministic score-only reranker that records its complete input set."""

    name = "fake/deterministic-reranker"

    def __init__(self, scores: Sequence[float] | None = None) -> None:
        self.scores = tuple(scores) if scores is not None else None
        self.calls = 0
        self.queries: list[str] = []
        self.candidate_ids: list[tuple[str, ...]] = []

    def score(self, query: str, candidates: Sequence[RankedPassage]) -> Sequence[float]:
        self.calls += 1
        self.queries.append(query)
        self.candidate_ids.append(tuple(item.passage.chunk_id for item in candidates))
        if self.scores is not None:
            if len(self.scores) != len(candidates):
                raise ValueError("Fake score count does not match candidates")
            return self.scores
        return tuple(float(index) for index in range(len(candidates)))


def make_test_config(**changes: object) -> AppConfig:
    return replace(load_config(), **changes)


def axis_vectors(count: int, dimension: int = 384) -> np.ndarray:
    vectors = np.zeros((count, dimension), dtype=np.float32)
    for index in range(count):
        vectors[index, index % dimension] = 1.0
    return vectors
