"""Lightweight retrieval component contracts."""

from __future__ import annotations

from typing import Protocol, TypeVar

from .types import SearchResponse, VectorHit


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

    def search(self, query: str) -> SearchResponse: ...
