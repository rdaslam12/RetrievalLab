"""Application workflows for corpus preparation and retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from .bm25_retriever import BM25SearchEngine
from .chunker import TokenizerLike, WhitespaceTokenizer, chunk_sections
from .config import AppConfig
from .document_parser import DocumentValidationError, parse_corpus
from .hybrid_retriever import HybridSearchEngine
from .reranker import rerank_candidates
from .search_engine import DenseSearchEngine, EmbedderLike
from .types import Passage, SearchResponse

if TYPE_CHECKING:
    from .interfaces import Reranker, SearchEngine


class CorpusEmbedder(EmbedderLike, Protocol):
    tokenizer: TokenizerLike


@dataclass(frozen=True, slots=True)
class PreparedCorpus:
    """A complete replacement corpus that is safe for the caller to install."""

    engine: DenseSearchEngine | None
    bm25_engine: BM25SearchEngine
    hybrid_engine: HybridSearchEngine | None
    passage_records: tuple[Passage, ...]
    files: int
    documents: int
    passages: int
    embeddings: int
    warnings: tuple[str, ...]


class CorpusService:
    """Prepare a complete indexed corpus without mutating application state."""

    def __init__(
        self,
        config: AppConfig,
        embedder_loader: Callable[[], CorpusEmbedder] | None = None,
    ) -> None:
        self.config = config
        self._embedder_loader = embedder_loader

    def prepare(
        self,
        files: Sequence[tuple[str, bytes]],
        *,
        bounded_jsonl: bool = False,
        jsonl_start_document: int = 1,
        progress: Callable[[str], None] | None = None,
        build_dense: bool = True,
    ) -> PreparedCorpus:
        """Parse, chunk, encode, and index a replacement corpus locally."""

        report_progress = progress or (lambda _message: None)
        report_progress("Validating uploads and extracting text.")
        parsed = parse_corpus(
            files,
            max_files=self.config.max_files,
            max_total_bytes=self.config.max_total_bytes,
        )
        sections_to_chunk = parsed.sections
        if bounded_jsonl:
            if jsonl_start_document > len(parsed.sections):
                raise DocumentValidationError(
                    f"The selected start position is {jsonl_start_document:,}, but the "
                    f"JSONL corpus has only {len(parsed.sections):,} indexable documents."
                )
            sections_to_chunk = parsed.sections[jsonl_start_document - 1 :]

        embedder: CorpusEmbedder | None = None
        if build_dense:
            if self._embedder_loader is None:
                raise ValueError("Dense indexing requires an embedding model loader.")
            report_progress("Loading the selected embedding model and tokenizer.")
            embedder = self._embedder_loader()
            tokenizer = embedder.tokenizer
        else:
            report_progress("Preparing BM25 passages without loading an embedding model.")
            tokenizer = WhitespaceTokenizer()
        report_progress("Chunking with 350-token windows and 50-token overlap.")
        passages = chunk_sections(
            sections_to_chunk,
            tokenizer,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            max_passages=self.config.max_passages,
            truncate_at_limit=bounded_jsonl,
        )
        corpus_warnings = list(parsed.warnings)
        indexed_source_ids = {passage.source_id for passage in passages}
        if bounded_jsonl:
            total_documents = len(parsed.sections)
            indexed_documents = len(indexed_source_ids)
            last_document = jsonl_start_document + indexed_documents - 1
            omitted_documents = total_documents - indexed_documents
            if omitted_documents:
                corpus_warnings.append(
                    "Bounded JSONL slice: indexed "
                    f"{indexed_documents:,} of {total_documents:,} indexable documents "
                    f"(positions {jsonl_start_document:,}–{last_document:,} in file order) "
                    f"as {len(passages):,} passages. Retrieval excludes "
                    f"{omitted_documents:,} documents outside this slice."
                )

        passage_records = tuple(passages)
        bm25_engine = BM25SearchEngine(self.config, passage_records)
        engine: DenseSearchEngine | None = None
        hybrid_engine: HybridSearchEngine | None = None
        if embedder is not None:
            report_progress(f"Encoding {len(passages)} passages and creating the exact index.")
            engine = DenseSearchEngine(self.config, embedder)
            engine.index_passages(passage_records)
            hybrid_engine = HybridSearchEngine(engine, bm25_engine)
        return PreparedCorpus(
            engine=engine,
            bm25_engine=bm25_engine,
            hybrid_engine=hybrid_engine,
            passage_records=passage_records,
            files=parsed.files_processed,
            documents=len(indexed_source_ids),
            passages=len(passages),
            embeddings=len(passages) if engine is not None else 0,
            warnings=tuple(corpus_warnings),
        )


def build_dense_engines(
    config: AppConfig,
    passages: Sequence[Passage],
    embedder_loader: Callable[[], CorpusEmbedder],
    *,
    bm25_engine: BM25SearchEngine | None = None,
) -> tuple[DenseSearchEngine, HybridSearchEngine]:
    """Build Dense and Hybrid over an existing corpus without reparsing uploads."""

    passage_records = tuple(passages)
    embedder = embedder_loader()
    dense_engine = DenseSearchEngine(config, embedder)
    dense_engine.index_passages(passage_records)
    lexical_engine = bm25_engine or BM25SearchEngine(config, passage_records)
    return dense_engine, HybridSearchEngine(dense_engine, lexical_engine)


class SearchService:
    """Run base retrieval and an optional independent reranking stage."""

    def __init__(self, engine: SearchEngine, reranker: Reranker | None = None) -> None:
        self.engine = engine
        self.reranker = reranker

    def search(self, query: str) -> SearchResponse:
        if self.reranker is None:
            return self.engine.search(query)

        total_started = perf_counter()
        retrieval_response = self.engine.search_candidates(query)
        reranking_started = perf_counter()
        reranked = rerank_candidates(
            retrieval_response.query,
            retrieval_response.results,
            self.reranker,
        )
        reranking_latency = (perf_counter() - reranking_started) * 1000.0
        return replace(
            retrieval_response,
            results=reranked[: self.engine.top_results],
            reranking_used=True,
            reranker_name=self.reranker.name,
            reranking_latency_ms=reranking_latency,
            total_query_latency_ms=(perf_counter() - total_started) * 1000.0,
        )
