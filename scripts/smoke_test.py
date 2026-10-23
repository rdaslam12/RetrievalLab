"""Offline synthetic smoke test; set RUN_MODEL_SMOKE=1 to use real Arctic-S."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services import CorpusService, SearchService  # noqa: E402
from src.config import load_config  # noqa: E402
from src.types import RankedPassage  # noqa: E402


class SmokeTokenizer:
    def __init__(self) -> None:
        self.forward: dict[str, int] = {}
        self.reverse: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        result = []
        for token in text.split():
            if token not in self.forward:
                token_id = len(self.forward) + 1
                self.forward[token] = token_id
                self.reverse[token_id] = token
            result.append(self.forward[token])
        return result

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return " ".join(self.reverse[int(token_id)] for token_id in token_ids)


class DeterministicSmokeEmbedder:
    device = "cpu"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.model_id = "smoke/deterministic-embedder"
        self.tokenizer = SmokeTokenizer()

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"(?u)\b\w+\b", text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dimension] += 1.0
        if not np.any(vector):
            vector[0] = 1.0
        return vector

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return self._vector(query)


class DeterministicSmokeReranker:
    name = "smoke/deterministic-reranker"

    def score(self, query: str, candidates: Sequence[RankedPassage]) -> list[float]:
        return [float(len(candidate.passage.text)) for candidate in candidates]


def main() -> None:
    config = load_config()
    use_real_model = os.getenv("RUN_MODEL_SMOKE") == "1"
    if use_real_model:
        from src.embedder import ArcticEmbedder

        embedder = ArcticEmbedder(config.model_id)
    else:
        embedder = DeterministicSmokeEmbedder(config.embedding_dimension)
    payloads = [
        (path.name, path.read_bytes()) for path in sorted((ROOT / "sample_data").glob("*.txt"))
    ]
    payloads.append(
        (
            "synthetic_corpus.jsonl",
            b'{"_id":"jsonl-dense","title":"Semantic retrieval",'
            b'"text":"Dense embeddings retrieve semantically related passages."}\n',
        )
    )
    prepared = CorpusService(config, lambda: embedder).prepare(payloads)
    dense = SearchService(prepared.engine).search("dense embeddings retrieve passages")
    lexical = SearchService(prepared.bm25_engine).search("dense embeddings")
    hybrid = SearchService(prepared.hybrid_engine).search("dense embeddings")
    reranker = DeterministicSmokeReranker()
    reranked = (
        SearchService(prepared.engine, reranker).search("dense embeddings"),
        SearchService(prepared.bm25_engine, reranker).search("dense embeddings"),
        SearchService(prepared.hybrid_engine, reranker).search("dense embeddings"),
    )
    if not any(passage.document_id == "jsonl-dense" for passage in prepared.engine.passages):
        raise RuntimeError("JSONL document identity was not preserved during chunking")
    for name, response in (("dense", dense), ("bm25", lexical), ("hybrid", hybrid)):
        if not response.results:
            raise RuntimeError(f"{name} retrieval returned no results")
    if not all(response.reranking_used and response.results for response in reranked):
        raise RuntimeError("One or more public pipelines failed deterministic reranking")
    if not any(result.passage.document_id == "jsonl-dense" for result in lexical.results):
        raise RuntimeError("JSONL document was not returned by lexical retrieval")
    if not any(result.passage.document_id == "jsonl-dense" for result in hybrid.results):
        raise RuntimeError("JSONL document was not returned by hybrid retrieval")
    print(
        "Smoke test passed:",
        f"files={prepared.files}",
        f"passages={prepared.passages}",
        f"dense_backend={dense.index_backend}",
        "pipelines=dense,bm25,hybrid",
        "reranking=passed",
    )


if __name__ == "__main__":
    main()
