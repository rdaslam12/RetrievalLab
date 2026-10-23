"""Lightweight retrieval component contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypeVar

import numpy as np

from .types import Passage, RankedPassage, SearchResponse, VectorHit


QueryT = TypeVar("QueryT", contravariant=True)


class Retriever(Protocol[QueryT]):
    """Retrieve ranked candidates from a strategy-appropriate query."""

    @property
    def backend(self) -> str: ...

    def search(self, query: QueryT, k: int) -> list[VectorHit]: ...


class SearchEngine(Protocol):
    """Run one configured retrieval strategy."""

    @property
    def is_indexed(self) -> bool: ...

    @property
    def top_results(self) -> int: ...

    @property
    def candidate_depth(self) -> int: ...

    @property
    def passages(self) -> tuple[Passage, ...]: ...

    def search(self, query: str) -> SearchResponse: ...

    def search_with_limit(self, query: str, result_limit: int) -> SearchResponse:
        """Return a caller-selected result count within the fixed candidate set."""

        ...

    def search_candidates(self, query: str) -> SearchResponse:
        """Return the fixed first-stage candidate set without final truncation."""

        ...


class Reranker(Protocol):
    """Score an already-retrieved, fixed candidate set for one query."""

    @property
    def name(self) -> str: ...

    def score(self, query: str, candidates: Sequence[RankedPassage]) -> Sequence[float]: ...


class DenseEncoderAdapter(Protocol):
    """Architecture-independent, role-aware dense inference."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def similarity(self) -> str: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...


class RerankerAdapter(Protocol):
    """Score texts in input order; scores must be comparable across batches.

    input_kind distinguishes this contract from the legacy RankedPassage API.
    """

    input_kind: Literal["text"]

    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


def adapter_model_id(adapter: object) -> str:
    """Use explicit non-secret identity when available, otherwise the model name."""
    return getattr(adapter, "model_id", getattr(adapter, "name", "Unknown adapter"))
