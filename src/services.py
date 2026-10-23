"""Application workflows for corpus preparation and retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .bm25_retriever import BM25SearchEngine
from .chunker import TokenizerLike, chunk_sections
from .config import AppConfig
from .document_parser import DocumentValidationError, parse_corpus
from .hybrid_retriever import HybridSearchEngine
from .interfaces import SearchEngine
from .search_engine import DenseSearchEngine, EmbedderLike
from .types import SearchResponse


class CorpusEmbedder(EmbedderLike, Protocol):
    tokenizer: TokenizerLike


@dataclass(frozen=True, slots=True)
class PreparedCorpus:
    """A complete replacement corpus that is safe for the caller to install."""

    engine: DenseSearchEngine
    bm25_engine: BM25SearchEngine
    hybrid_engine: HybridSearchEngine
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
        embedder_loader: Callable[[], CorpusEmbedder],
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

        report_progress("Loading the Arctic-S encoder and tokenizer.")
        embedder = self._embedder_loader()
        report_progress("Chunking with 350-token windows and 50-token overlap.")
        passages = chunk_sections(
            sections_to_chunk,
            embedder.tokenizer,
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

        report_progress(f"Encoding {len(passages)} passages and creating the exact index.")
        engine = DenseSearchEngine(self.config, embedder)
        engine.index_passages(passages)
        bm25_engine = BM25SearchEngine(self.config, engine.passages)
        hybrid_engine = HybridSearchEngine(engine, bm25_engine)
        return PreparedCorpus(
            engine=engine,
            bm25_engine=bm25_engine,
            hybrid_engine=hybrid_engine,
            files=parsed.files_processed,
            documents=len(indexed_source_ids),
            passages=len(passages),
            embeddings=len(passages),
            warnings=tuple(corpus_warnings),
        )


class SearchService:
    """Run retrieval without presentation-layer dependencies."""

    def __init__(self, engine: SearchEngine) -> None:
        self.engine = engine

    def search(self, query: str) -> SearchResponse:
        return self.engine.search(query)
