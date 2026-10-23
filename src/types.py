"""Typed objects shared by parsing, indexing, search, and UI layers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SourceSection:
    """An ordered extractable unit; a PDF unit corresponds to one page."""

    source_id: str
    file_name: str
    file_type: str
    page_number: int | None
    text: str
    document_id: str | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class Passage:
    """One independently searchable, token-bounded passage."""

    chunk_id: str
    source_id: str
    file_name: str
    file_type: str
    page_number: int | None
    passage_index: int
    text: str
    token_count: int
    document_id: str | None = None
    title: str | None = None


@dataclass(slots=True)
class ParsedCorpus:
    """Parsed ordered sections and user-visible extraction warnings."""

    sections: list[SourceSection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_processed: int = 0
    total_bytes: int = 0


@dataclass(frozen=True, slots=True)
class VectorHit:
    passage_index: int
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class RankedPassage:
    """A ranked passage with retrieval and optional reranking provenance.

    ``rank`` is the displayed/final rank and ``score`` remains the first-stage
    retrieval score for backwards compatibility. Reranked results additionally
    carry the unambiguous first-stage fields below.
    """

    passage: Passage
    rank: int
    score: float
    rrf_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    reranker_score: float | None = None
    rank_change: int | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    candidate_depth_used: int
    indexed_passages: int
    results: tuple[RankedPassage, ...]
    retrieval_latency_ms: float
    total_query_latency_ms: float
    index_backend: str
    device: str
    retrieval_method: str
    reranking_used: bool = False
    reranker_name: str | None = None
    reranking_latency_ms: float = 0.0
    dense_model_id: str | None = None
    embedding_dimension: int | None = None
