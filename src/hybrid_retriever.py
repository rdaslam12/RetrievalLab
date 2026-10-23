"""Reciprocal Rank Fusion over the existing dense and BM25 rankings."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .bm25_retriever import BM25SearchEngine
from .search_engine import DenseSearchEngine
from .types import Passage, RankedPassage, SearchResponse, VectorHit


DEFAULT_RRF_K = 60


@dataclass(slots=True)
class _FusionEntry:
    passage: Passage
    corpus_order: int
    rrf_score: float = 0.0
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


def reciprocal_rank_fusion(
    passages: tuple[Passage, ...],
    dense_hits: list[VectorHit],
    bm25_hits: list[VectorHit],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> tuple[RankedPassage, ...]:
    """Fuse rankings by stable chunk identity and preserve branch provenance."""

    if rrf_k < 0:
        raise ValueError("RRF k must be non-negative")
    entries: dict[str, _FusionEntry] = {}
    for branch, hits in (("dense", dense_hits), ("bm25", bm25_hits)):
        for hit in hits:
            passage = passages[hit.passage_index]
            entry = entries.setdefault(
                passage.chunk_id,
                _FusionEntry(passage=passage, corpus_order=hit.passage_index),
            )
            entry.rrf_score += 1.0 / (rrf_k + hit.rank)
            if branch == "dense":
                entry.dense_rank = hit.rank
                entry.dense_score = hit.score
            else:
                entry.bm25_rank = hit.rank
                entry.bm25_score = hit.score

    ordered = sorted(
        entries.values(),
        key=lambda entry: (-entry.rrf_score, entry.corpus_order, entry.passage.chunk_id),
    )
    return tuple(
        RankedPassage(
            passage=entry.passage,
            rank=rank,
            score=entry.rrf_score,
            rrf_score=entry.rrf_score,
            dense_rank=entry.dense_rank,
            dense_score=entry.dense_score,
            bm25_rank=entry.bm25_rank,
            bm25_score=entry.bm25_score,
        )
        for rank, entry in enumerate(ordered, start=1)
    )


class HybridSearchEngine:
    """Run dense and BM25 independently, then fuse their fixed-depth rankings."""

    def __init__(
        self,
        dense_engine: DenseSearchEngine,
        bm25_engine: BM25SearchEngine,
        *,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        dense_ids = tuple(passage.chunk_id for passage in dense_engine.passages)
        bm25_ids = tuple(passage.chunk_id for passage in bm25_engine.passages)
        if dense_ids != bm25_ids:
            raise ValueError("Hybrid retrieval requires identical ordered passage corpora")
        if rrf_k < 0:
            raise ValueError("RRF k must be non-negative")
        self.dense_engine = dense_engine
        self.bm25_engine = bm25_engine
        self.config = dense_engine.config
        self.passages = dense_engine.passages
        self.rrf_k = rrf_k

    @property
    def is_indexed(self) -> bool:
        return self.dense_engine.is_indexed and self.bm25_engine.is_indexed

    @property
    def index_backend(self) -> str:
        return f"RRF(k={self.rrf_k}) over dense + BM25"

    @property
    def top_results(self) -> int:
        return self.config.top_results

    def _search(self, query: str, *, result_limit: int) -> SearchResponse:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Enter a non-empty search query.")
        total_started = perf_counter()
        retrieval_started = perf_counter()
        _, _, dense_hits = self.dense_engine.retrieve_candidates(cleaned_query)
        _, bm25_hits = self.bm25_engine.retrieve_candidates(cleaned_query)
        fused = reciprocal_rank_fusion(
            self.passages,
            dense_hits,
            bm25_hits,
            rrf_k=self.rrf_k,
        )
        retrieval_latency = (perf_counter() - retrieval_started) * 1000.0
        candidate_depth = min(self.config.candidate_depth, len(self.passages))
        candidates = fused[:candidate_depth]
        top_count = min(result_limit, len(candidates))
        return SearchResponse(
            query=cleaned_query,
            candidate_depth_used=candidate_depth,
            indexed_passages=len(self.passages),
            results=tuple(
                RankedPassage(
                    passage=result.passage,
                    rank=result.rank,
                    score=result.score,
                    rrf_score=result.rrf_score,
                    dense_rank=result.dense_rank,
                    dense_score=result.dense_score,
                    bm25_rank=result.bm25_rank,
                    bm25_score=result.bm25_score,
                    retrieval_rank=result.rank,
                    retrieval_score=result.score,
                )
                for result in candidates[:top_count]
            ),
            retrieval_latency_ms=retrieval_latency,
            total_query_latency_ms=(perf_counter() - total_started) * 1000.0,
            index_backend=self.index_backend,
            device=str(self.dense_engine.embedder.device),
            retrieval_method="Hybrid",
        )

    def search(self, query: str) -> SearchResponse:
        return self._search(query, result_limit=self.config.top_results)

    def search_candidates(self, query: str) -> SearchResponse:
        """Return the fixed-depth fused candidate set for optional reranking."""

        return self._search(query, result_limit=self.config.candidate_depth)
