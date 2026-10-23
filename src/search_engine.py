"""Exact dense retrieval over one reusable, in-memory corpus index."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Protocol, Sequence

import numpy as np

from .config import AppConfig
from .interfaces import Retriever
from .types import Passage, RankedPassage, SearchResponse, VectorHit
from .vector_index import create_vector_index


class EmbedderLike(Protocol):
    device: str
    dimension: int
    model_id: str

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, query: str) -> np.ndarray: ...


IndexFactory = Callable[[np.ndarray], Retriever[np.ndarray]]


def _unit_normalize(matrix: np.ndarray, dimension: int) -> np.ndarray:
    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[1] != dimension:
        raise ValueError(f"Expected [N, {dimension}] embeddings, got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Embeddings must have non-zero norms")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class DenseSearchEngine:
    """Encode a corpus once and reuse its normalized embeddings across queries."""

    def __init__(
        self,
        config: AppConfig,
        embedder: EmbedderLike,
        *,
        index_factory: IndexFactory = create_vector_index,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self._index_factory = index_factory
        self.passages: tuple[Passage, ...] = ()
        self.embeddings: np.ndarray | None = None
        self.vector_index: Retriever[np.ndarray] | None = None

    @property
    def is_indexed(self) -> bool:
        return bool(self.passages) and self.vector_index is not None and self.embeddings is not None

    @property
    def corpus_statistics(self) -> dict[str, int]:
        return {
            "files": len({passage.source_id for passage in self.passages}),
            "passages": len(self.passages),
            "embeddings": 0 if self.embeddings is None else int(self.embeddings.shape[0]),
        }

    @property
    def index_backend(self) -> str:
        return self.vector_index.backend if self.vector_index is not None else "Not indexed"

    @property
    def top_results(self) -> int:
        return self.config.top_results

    def index_passages(self, passages: Sequence[Passage]) -> None:
        """Build replacement state locally and install it only after complete success."""

        candidate_passages = tuple(passages)
        if not candidate_passages:
            raise ValueError("Cannot index an empty passage collection")
        if len(candidate_passages) > self.config.max_passages:
            raise ValueError(f"At most {self.config.max_passages} passages may be indexed")
        identifiers = [passage.chunk_id for passage in candidate_passages]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Passage chunk IDs must be unique")
        candidate_embeddings = _unit_normalize(
            self.embedder.encode_passages([passage.text for passage in candidate_passages]),
            self.embedder.dimension,
        )
        if candidate_embeddings.shape[0] != len(candidate_passages):
            raise ValueError("Encoder returned a different number of embeddings than passages")
        candidate_index = self._index_factory(candidate_embeddings)
        self.passages = candidate_passages
        self.embeddings = candidate_embeddings
        self.vector_index = candidate_index

    def retrieve_candidates(self, query: str) -> tuple[str, np.ndarray, list[VectorHit]]:
        """Return the exact dense candidate ranking at the configured depth."""

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Enter a non-empty search query.")
        if not self.is_indexed or self.vector_index is None or self.embeddings is None:
            raise RuntimeError("Index a corpus before searching.")
        query_vector = _unit_normalize(
            self.embedder.encode_query(cleaned_query), self.embedder.dimension
        )[0]
        candidate_depth = min(self.config.candidate_depth, len(self.passages))
        return cleaned_query, query_vector, self.vector_index.search(query_vector, candidate_depth)

    def _search(self, query: str, *, result_limit: int) -> SearchResponse:
        total_started = perf_counter()
        retrieval_started = perf_counter()
        cleaned_query, _, hits = self.retrieve_candidates(query)
        retrieval_latency = (perf_counter() - retrieval_started) * 1000.0
        top_count = min(result_limit, len(hits))
        results = tuple(
            RankedPassage(
                passage=self.passages[hit.passage_index],
                rank=hit.rank,
                score=hit.score,
                retrieval_rank=hit.rank,
                retrieval_score=hit.score,
            )
            for hit in hits[:top_count]
        )
        return SearchResponse(
            query=cleaned_query,
            candidate_depth_used=min(self.config.candidate_depth, len(self.passages)),
            indexed_passages=len(self.passages),
            results=results,
            retrieval_latency_ms=retrieval_latency,
            total_query_latency_ms=(perf_counter() - total_started) * 1000.0,
            index_backend=self.vector_index.backend,
            device=str(self.embedder.device),
            retrieval_method="Dense",
            dense_model_id=self.embedder.model_id,
            embedding_dimension=self.embedder.dimension,
        )

    def search(self, query: str) -> SearchResponse:
        """Retrieve and return the configured number of dense results."""

        return self._search(query, result_limit=self.config.top_results)

    def search_candidates(self, query: str) -> SearchResponse:
        """Return the fixed-depth dense candidate set for optional reranking."""

        return self._search(query, result_limit=self.config.candidate_depth)
