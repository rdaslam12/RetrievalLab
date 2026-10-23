"""Frozen Snowflake Arctic-S sentence-transformer wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np


LOGGER = logging.getLogger(__name__)


def detect_preferred_device() -> str:
    """Select CUDA, then Apple MPS, then CPU."""

    try:
        import torch
    except ImportError:  # pragma: no cover - dependency validation occurs in setup
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _normalize_and_validate(array: np.ndarray, expected_dim: int) -> np.ndarray:
    vectors = np.asarray(array, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise ValueError(f"Expected embeddings shaped [N, {expected_dim}], got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Encoder returned NaN or infinite embedding values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Encoder returned a zero-length embedding")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class ArcticEmbedder:
    """Cached-at-the-UI frozen encoder with safe MPS fallback behavior."""

    def __init__(
        self,
        model_id: str = "Snowflake/snowflake-arctic-embed-s",
        expected_dimension: int = 384,
        *,
        device: str | None = None,
        batch_size: int = 32,
        cache_folder: Path | str | None = None,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.expected_dimension = expected_dimension
        self.device = device or detect_preferred_device()
        self.batch_size = batch_size
        self.cache_folder = Path(cache_folder) if cache_folder else (
            Path(__file__).resolve().parents[1] / "model_cache" / "huggingface" / "hub"
        )
        repo_cache = self.cache_folder / f"models--{model_id.replace('/', '--')}" / "snapshots"
        local_files_only = repo_cache.is_dir() and any(repo_cache.iterdir())
        LOGGER.info("Loading frozen encoder model=%s device=%s", model_id, self.device)
        try:
            self.model = SentenceTransformer(
                model_id,
                device=self.device,
                cache_folder=str(self.cache_folder),
                local_files_only=local_files_only,
            )
        except Exception:
            if self.device != "mps":
                raise
            LOGGER.warning("Arctic-S failed to load on MPS; retrying on CPU", exc_info=True)
            self.device = "cpu"
            self.model = SentenceTransformer(
                model_id,
                device="cpu",
                cache_folder=str(self.cache_folder),
                local_files_only=local_files_only,
            )
        self.model.eval()
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension != expected_dimension:
            raise ValueError(
                f"Encoder dimension is {dimension}; RetrievalLab requires {expected_dimension}."
            )
        self.tokenizer = self.model.tokenizer
        self._torch = torch

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        if not texts:
            return np.empty((0, self.expected_dimension), dtype=np.float32)
        kwargs = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if is_query:
            kwargs["prompt_name"] = "query"
        try:
            with self._torch.inference_mode():
                output = self.model.encode(list(texts), **kwargs)
        except Exception:
            if self.device != "mps":
                raise
            LOGGER.warning("Arctic-S execution failed on MPS; moving the encoder to CPU", exc_info=True)
            self.model.to("cpu")
            self.device = "cpu"
            with self._torch.inference_mode():
                output = self.model.encode(list(texts), **kwargs)
        return _normalize_and_validate(output, self.expected_dimension)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Encode passages without the query prompt."""

        return self._encode(texts, is_query=False)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query using Arctic-S's configured ``query`` prompt."""

        return self._encode([query], is_query=True)[0]
