"""Arctic-S tokenizer-based overlapping passage chunking."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Protocol, Sequence

from .types import Passage, SourceSection


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...


class WhitespaceTokenizer:
    """Small reversible tokenizer for BM25-only corpus preparation."""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        token_ids: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            token_ids.append(self._token_to_id[token])
        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return " ".join(self._id_to_token[int(token_id)] for token_id in token_ids)


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace without touching punctuation or case."""

    return re.sub(r"\s+", " ", text).strip()


def chunk_sections(
    sections: Sequence[SourceSection],
    tokenizer: TokenizerLike,
    *,
    chunk_size: int = 350,
    overlap: int = 50,
    max_passages: int = 35_000,
    truncate_at_limit: bool = False,
) -> list[Passage]:
    """Chunk sections independently, optionally stopping at a whole-section boundary."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if max_passages <= 0:
        raise ValueError("max_passages must be positive")

    passages: list[Passage] = []
    per_source_index: defaultdict[str, int] = defaultdict(int)
    step = chunk_size - overlap
    for section in sections:
        normalized = normalize_whitespace(section.text)
        if not normalized:
            continue
        token_ids = list(tokenizer.encode(normalized, add_special_tokens=False))
        if not token_ids:
            continue
        section_passages: list[Passage] = []
        starting_passage_index = per_source_index[section.source_id]
        start = 0
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            window = token_ids[start:end]
            text = normalize_whitespace(
                tokenizer.decode(
                    window,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
            if text:
                passage_index = starting_passage_index + len(section_passages) + 1
                identity = (
                    f"{section.source_id}|{section.page_number}|{passage_index}|{start}|{end}"
                )
                chunk_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                section_passages.append(
                    Passage(
                        chunk_id=f"chunk-{chunk_hash}",
                        source_id=section.source_id,
                        file_name=section.file_name,
                        file_type=section.file_type,
                        page_number=section.page_number,
                        passage_index=passage_index,
                        text=text,
                        token_count=len(window),
                        document_id=section.document_id,
                        title=section.title,
                    )
                )
            if end >= len(token_ids):
                break
            next_start = start + step
            if next_start <= start:  # defensive guard against future configuration regressions
                raise RuntimeError("Chunking made no forward progress")
            start = next_start
        if len(passages) + len(section_passages) > max_passages:
            if truncate_at_limit and passages:
                break
            raise ValueError(
                f"Corpus generated more than {max_passages} passages. "
                "Reduce the corpus or use bounded JSONL slice mode."
            )
        passages.extend(section_passages)
        if section_passages:
            per_source_index[section.source_id] = section_passages[-1].passage_index
    if not passages:
        raise ValueError("No non-empty passages could be generated.")
    return passages
