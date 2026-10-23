"""Exact normalized inner-product vector indexes."""

from __future__ import annotations

from abc import ABC, abstractmethod
import sys

import numpy as np

from .types import VectorHit


MACOS_NUMPY_BACKEND = "NumPy exact inner product (macOS FAISS/OpenMP-safe fallback)"


def _faiss_runtime_supported() -> bool:
    """Avoid a known fatal duplicate-OpenMP runtime in macOS FAISS wheels.

    This is not an accuracy compromise: both available backends perform exhaustive
    inner-product scoring. FAISS remains the primary backend on supported platforms.
    """

    return sys.platform != "darwin"


def _validate_matrix(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Embeddings must be a non-empty 2D matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _validate_query(query: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    if vector.shape != (dimension,):
        raise ValueError(f"Query must have shape ({dimension},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("Query contains NaN or infinite values")
    return np.ascontiguousarray(vector, dtype=np.float32)


class VectorIndex(ABC):
    """Backend-neutral exact retrieval interface."""

    backend: str

    @property
    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[VectorHit]: ...


class NumpyVectorIndex(VectorIndex):
    """Deterministic brute-force fallback for exact inner-product search."""

    backend = "NumPy exact inner product"

    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = _validate_matrix(embeddings).copy()

    @property
    def size(self) -> int:
        return self._embeddings.shape[0]

    @property
    def dimension(self) -> int:
        return self._embeddings.shape[1]

    def search(self, query: np.ndarray, k: int) -> list[VectorHit]:
        if k <= 0:
            return []
        vector = _validate_query(query, self.dimension)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = self._embeddings @ vector
        if not np.isfinite(scores).all():
            raise ValueError("Dense similarity scores overflowed; use smaller dot-product vector magnitudes")
        indices = np.arange(self.size)
        order = np.lexsort((indices, -scores))[: min(k, self.size)]
        return [
            VectorHit(int(index), float(scores[index]), rank)
            for rank, index in enumerate(order, start=1)
        ]


class FaissVectorIndex(VectorIndex):
    """FAISS IndexFlatIP exact retrieval backend."""

    backend = "FAISS IndexFlatIP (exact)"

    def __init__(self, embeddings: np.ndarray) -> None:
        import faiss

        matrix = _validate_matrix(embeddings)
        self._dimension = matrix.shape[1]
        self._size = matrix.shape[0]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(matrix)

    @property
    def size(self) -> int:
        return self._size

    @property
    def dimension(self) -> int:
        return self._dimension

    def search(self, query: np.ndarray, k: int) -> list[VectorHit]:
        if k <= 0:
            return []
        vector = _validate_query(query, self.dimension)
        count = min(k, self.size)
        scores, indices = self._index.search(vector.reshape(1, -1), count)
        if not np.isfinite(scores).all():
            raise ValueError("Dense similarity scores overflowed; use smaller dot-product vector magnitudes")
        pairs = [(int(index), float(score)) for index, score in zip(indices[0], scores[0]) if index >= 0]
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return [
            VectorHit(index, score, rank)
            for rank, (index, score) in enumerate(pairs, start=1)
        ]


def create_vector_index(embeddings: np.ndarray, *, prefer_faiss: bool = True) -> VectorIndex:
    """Create the primary FAISS exact index or its tested exact NumPy fallback."""

    if prefer_faiss and _faiss_runtime_supported():
        try:
            return FaissVectorIndex(embeddings)
        except ImportError:
            pass
    index = NumpyVectorIndex(embeddings)
    if prefer_faiss and not _faiss_runtime_supported():
        index.backend = MACOS_NUMPY_BACKEND
    return index


def available_backend_name() -> str:
    """Report the backend a newly built index would use."""

    if not _faiss_runtime_supported():
        return MACOS_NUMPY_BACKEND
    try:
        import faiss  # noqa: F401
    except ImportError:
        return NumpyVectorIndex.backend
    return FaissVectorIndex.backend
