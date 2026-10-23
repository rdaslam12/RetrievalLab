"""Safe, cached SentenceTransformer embedding models."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .model_ids import validate_huggingface_model_id


LOGGER = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-s"


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when a selected SentenceTransformer cannot be loaded or used."""


class SentenceTransformerLike(Protocol):
    tokenizer: object

    def eval(self) -> object: ...

    def to(self, device: str) -> object: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: list[str], **kwargs: object) -> object: ...


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
        raise EmbeddingModelUnavailableError(
            f"Embedding model returned shape {vectors.shape}; expected [N, {expected_dim}]."
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingModelUnavailableError(
            "Embedding model returned NaN or infinite embedding values."
        )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise EmbeddingModelUnavailableError("Embedding model returned a zero-length embedding.")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _has_cached_snapshot(cache_folder: str, model_id: str) -> bool:
    repo_cache = Path(cache_folder) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return repo_cache.is_dir() and any(repo_cache.iterdir())


@lru_cache(maxsize=4)
def _load_cached_sentence_transformer(
    model_id: str,
    device: str,
    cache_folder: str,
) -> tuple[SentenceTransformerLike, str, int]:
    """Load and validate one trusted-code-disabled model per configuration."""

    model_id = validate_huggingface_model_id(model_id, kind="embedding")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise EmbeddingModelUnavailableError(
            "sentence-transformers is unavailable; the embedding model could not be loaded."
        ) from exc

    local_files_only = _has_cached_snapshot(cache_folder, model_id)

    def build(target_device: str) -> tuple[SentenceTransformerLike, int]:
        try:
            model = SentenceTransformer(
                model_id,
                device=target_device,
                cache_folder=cache_folder,
                local_files_only=local_files_only,
                trust_remote_code=False,
            )
            model.eval()
            dimension = model.get_sentence_embedding_dimension()
            tokenizer = getattr(model, "tokenizer", None)
            if not isinstance(dimension, int) or dimension <= 0:
                raise TypeError("the model does not report a positive embedding dimension")
            if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
                raise TypeError("the model does not expose a compatible tokenizer")
            return model, dimension
        except EmbeddingModelUnavailableError:
            raise
        except Exception as exc:
            raise EmbeddingModelUnavailableError(
                f"Embedding model '{model_id}' is not a compatible SentenceTransformer model. "
                "Check the Hugging Face model ID and model files."
            ) from exc

    LOGGER.info("Loading embedding model=%s device=%s", model_id, device)
    try:
        model, dimension = build(device)
        return model, device, dimension
    except EmbeddingModelUnavailableError:
        if device != "mps":
            raise
        LOGGER.warning("Embedding model failed to load on MPS; retrying on CPU", exc_info=True)
        try:
            model, dimension = build("cpu")
            return model, "cpu", dimension
        except EmbeddingModelUnavailableError as cpu_exc:
            raise EmbeddingModelUnavailableError(
                f"Embedding model '{model_id}' could not be loaded on MPS or CPU."
            ) from cpu_exc


class SentenceTransformerEmbedder:
    """SentenceTransformer wrapper with inferred dimensions and safe MPS fallback."""

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        cache_folder: Path | str | None = None,
    ) -> None:
        import torch

        self.model_id = validate_huggingface_model_id(model_id, kind="embedding")
        self.device = device or detect_preferred_device()
        self.batch_size = batch_size
        self.cache_folder = Path(cache_folder) if cache_folder else (
            Path(__file__).resolve().parents[1] / "model_cache" / "huggingface" / "hub"
        )
        self.model, self.device, self.dimension = _load_cached_sentence_transformer(
            self.model_id,
            self.device,
            str(self.cache_folder),
        )
        self.tokenizer = self.model.tokenizer
        self._torch = torch

    def _model_encode(self, texts: list[str], *, is_query: bool) -> object:
        kwargs: dict[str, object] = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if is_query:
            if self.model_id == DEFAULT_EMBEDDING_MODEL:
                # Preserve the original Arctic-S query path byte-for-byte at the API boundary.
                kwargs["prompt_name"] = "query"
                return self.model.encode(texts, **kwargs)
            encode_query = getattr(self.model, "encode_query", None)
            if callable(encode_query):
                return encode_query(texts, **kwargs)
            prompts = getattr(self.model, "prompts", {})
            if isinstance(prompts, dict) and "query" in prompts:
                kwargs["prompt_name"] = "query"
        return self.model.encode(texts, **kwargs)

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            with self._torch.inference_mode():
                output = self._model_encode(list(texts), is_query=is_query)
        except Exception as exc:
            if self.device != "mps":
                raise EmbeddingModelUnavailableError(
                    f"Embedding model '{self.model_id}' could not encode the requested text."
                ) from exc
            LOGGER.warning("Embedding execution failed on MPS; moving model to CPU", exc_info=True)
            try:
                self.model.to("cpu")
                self.device = "cpu"
                with self._torch.inference_mode():
                    output = self._model_encode(list(texts), is_query=is_query)
            except Exception as cpu_exc:
                raise EmbeddingModelUnavailableError(
                    f"Embedding model '{self.model_id}' could not encode the requested text."
                ) from cpu_exc
        return _normalize_and_validate(np.asarray(output), self.dimension)

    @property
    def name(self) -> str:
        return self.model_id

    @property
    def similarity(self) -> str:
        return "cosine"

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=True)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Encode passages with the selected model's document/default behavior."""

        return self._encode(texts, is_query=False)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query using the model's query prompt when it defines one."""

        return self._encode([query], is_query=True)[0]


# Backwards-compatible public name for existing integrations and smoke-test opt-ins.
ArcticEmbedder = SentenceTransformerEmbedder
