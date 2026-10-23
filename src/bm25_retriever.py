"""Small in-memory Okapi BM25 retriever over indexed passages."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import math
import re
from time import perf_counter

from .config import AppConfig
from .types import Passage, RankedPassage, SearchResponse, VectorHit


_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """Standard Okapi BM25 with deterministic original-order tie breaking."""

    backend = "BM25 lexical (Okapi)"

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not passages:
            raise ValueError("Cannot build BM25 over an empty passage collection")
        if k1 <= 0.0:
            raise ValueError("BM25 k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("BM25 b must be in [0, 1]")
        self.passages = tuple(passages)
        self.k1 = float(k1)
        self.b = float(b)
        postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequencies: Counter[str] = Counter()
        lengths: list[int] = []
        for passage_index, passage in enumerate(self.passages):
            frequencies = Counter(_tokenize(passage.text))
            lengths.append(sum(frequencies.values()))
            for term, frequency in frequencies.items():
                postings[term].append((passage_index, frequency))
                document_frequencies[term] += 1
        self._postings = dict(postings)
        self._lengths = tuple(lengths)
        self._average_length = sum(lengths) / len(lengths)
        document_count = len(self.passages)
        self._idf = {
            term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    @property
    def size(self) -> int:
        return len(self.passages)

    def search(self, query: str, k: int) -> list[VectorHit]:
        if k <= 0:
            return []
        scores = [0.0] * self.size
        for term in _tokenize(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for passage_index, frequency in self._postings[term]:
                length = self._lengths[passage_index]
                length_ratio = length / self._average_length if self._average_length else 0.0
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                scores[passage_index] += idf * (frequency * (self.k1 + 1.0)) / denominator
        order = sorted(range(self.size), key=lambda index: (-scores[index], index))[
            : min(k, self.size)
        ]
        return [
            VectorHit(passage_index=index, score=scores[index], rank=rank)
            for rank, index in enumerate(order, start=1)
        ]


class BM25SearchEngine:
    """Adapt BM25 hits to the application's existing search response."""

    def __init__(self, config: AppConfig, passages: Sequence[Passage]) -> None:
        candidate_passages = tuple(passages)
        if len(candidate_passages) > config.max_passages:
            raise ValueError(f"At most {config.max_passages} passages may be indexed")
        identifiers = [passage.chunk_id for passage in candidate_passages]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Passage chunk IDs must be unique")
        self.config = config
        self.passages = candidate_passages
        self.retriever = BM25Retriever(candidate_passages)

    @property
    def is_indexed(self) -> bool:
        return bool(self.passages)

    @property
    def index_backend(self) -> str:
        return self.retriever.backend

    @property
    def top_results(self) -> int:
        return self.config.top_results

    def retrieve_candidates(self, query: str) -> tuple[str, list[VectorHit]]:
        """Return the unchanged BM25 first-stage candidate ranking."""

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Enter a non-empty search query.")
        candidate_depth = min(self.config.candidate_depth, len(self.passages))
        return cleaned_query, self.retriever.search(cleaned_query, candidate_depth)

    def _search(self, query: str, *, result_limit: int) -> SearchResponse:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Enter a non-empty search query.")
        total_started = perf_counter()
        retrieval_started = perf_counter()
        _, hits = self.retrieve_candidates(cleaned_query)
        retrieval_latency = (perf_counter() - retrieval_started) * 1000.0
        candidate_depth = min(self.config.candidate_depth, len(self.passages))
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
            candidate_depth_used=candidate_depth,
            indexed_passages=len(self.passages),
            results=results,
            retrieval_latency_ms=retrieval_latency,
            total_query_latency_ms=(perf_counter() - total_started) * 1000.0,
            index_backend=self.retriever.backend,
            device="cpu",
            retrieval_method="BM25",
        )

    def search(self, query: str) -> SearchResponse:
        return self._search(query, result_limit=self.config.top_results)

    def search_candidates(self, query: str) -> SearchResponse:
        """Return the fixed-depth BM25 candidate set for optional reranking."""

        return self._search(query, result_limit=self.config.candidate_depth)
