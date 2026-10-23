"""Lightweight retrieval component contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

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
