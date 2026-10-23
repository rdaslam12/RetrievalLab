"""Public, retrieval-independent cross-encoder reranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .embedder import detect_preferred_device
from .model_ids import validate_huggingface_model_id
from .types import RankedPassage

if TYPE_CHECKING:
    from .interfaces import Reranker


LOGGER = logging.getLogger(__name__)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class RerankerUnavailableError(RuntimeError):
    """Raised when a selected reranker cannot be loaded or executed."""


class CrossEncoderLike(Protocol):
    def eval(self) -> object: ...

    def to(self, device: str) -> object: ...

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object: ...


@lru_cache(maxsize=4)
def _load_cached_cross_encoder(
    model_id: str,
    device: str,
    cache_folder: str,
) -> tuple[CrossEncoderLike, str]:
    """Load one model per configuration, retrying MPS loads on CPU."""

    model_id = validate_huggingface_model_id(model_id, kind="reranker")
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RerankerUnavailableError(
            "sentence-transformers is unavailable; the cross-encoder could not be loaded."
        ) from exc

    repo_cache = Path(cache_folder) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    local_files_only = repo_cache.is_dir() and any(repo_cache.iterdir())

    def build(target_device: str) -> CrossEncoderLike:
        model = CrossEncoder(
            model_id,
            device=target_device,
            cache_folder=cache_folder,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        model.eval()
        return model

    LOGGER.info("Loading cross-encoder model=%s device=%s", model_id, device)
    try:
        return build(device), device
    except Exception as exc:
        if device != "mps":
            raise RerankerUnavailableError(
                f"Reranker '{model_id}' is not a compatible Hugging Face CrossEncoder model "
                f"or could not be loaded on {device}."
            ) from exc
        LOGGER.warning("Cross-encoder failed to load on MPS; retrying on CPU", exc_info=True)
        try:
            return build("cpu"), "cpu"
        except Exception as cpu_exc:
            raise RerankerUnavailableError(
                f"Reranker '{model_id}' is not a compatible Hugging Face CrossEncoder model "
                "or could not be loaded on MPS or CPU."
            ) from cpu_exc


def rerank_candidates(
    query: str,
    candidates: Sequence[RankedPassage],
    reranker: Reranker,
) -> tuple[RankedPassage, ...]:
    """Rerank exactly ``candidates`` while preserving first-stage provenance."""

    fixed_candidates = tuple(candidates)
    if not fixed_candidates:
        return ()
    scores = np.asarray(reranker.score(query, fixed_candidates), dtype=np.float64).reshape(-1)
    if scores.shape != (len(fixed_candidates),):
        raise ValueError(
            "Reranker returned a different number of scores than retrieved candidates."
        )
    if not np.isfinite(scores).all():
        raise ValueError("Reranker returned NaN or infinite scores.")

    # Python's stable sort keeps retrieval order for equal reranker scores.
    order = sorted(range(len(fixed_candidates)), key=lambda index: -float(scores[index]))
    reranked: list[RankedPassage] = []
    for final_rank, candidate_index in enumerate(order, start=1):
        candidate = fixed_candidates[candidate_index]
        retrieval_rank = (
            candidate.retrieval_rank if candidate.retrieval_rank is not None else candidate.rank
        )
        retrieval_score = (
            candidate.retrieval_score if candidate.retrieval_score is not None else candidate.score
        )
        reranked.append(
            replace(
                candidate,
                rank=final_rank,
                score=retrieval_score,
                retrieval_rank=retrieval_rank,
                retrieval_score=retrieval_score,
                reranker_score=float(scores[candidate_index]),
                rank_change=retrieval_rank - final_rank,
            )
        )
    return tuple(reranked)


class CrossEncoderReranker:
    """Lazy ``sentence_transformers.CrossEncoder`` passage reranker."""

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        cache_folder: Path | str | None = None,
    ) -> None:
        self.model_id = validate_huggingface_model_id(model_id, kind="reranker")
        self.device = device or detect_preferred_device()
        self.batch_size = batch_size
        self.cache_folder = Path(cache_folder) if cache_folder else (
            Path(__file__).resolve().parents[1] / "model_cache" / "huggingface" / "hub"
        )
        self._model: CrossEncoderLike | None = None

    @property
    def name(self) -> str:
        return self.model_id

    def _get_model(self) -> CrossEncoderLike:
        if self._model is None:
            self._model, self.device = _load_cached_cross_encoder(
                self.model_id,
                self.device,
                str(self.cache_folder),
            )
        return self._model

    def _predict(
        self,
        model: CrossEncoderLike,
        pairs: list[tuple[str, str]],
    ) -> object:
        return model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def score(self, query: str, candidates: Sequence[RankedPassage]) -> Sequence[float]:
        if not candidates:
            return ()
        pairs = [(query, candidate.passage.text) for candidate in candidates]
        model = self._get_model()
        try:
            predictions = self._predict(model, pairs)
        except Exception as exc:
            if self.device != "mps":
                raise RerankerUnavailableError(
                    f"Reranker {self.model_id} could not score the retrieved candidates."
                ) from exc
            LOGGER.warning("Cross-encoder execution failed on MPS; moving to CPU", exc_info=True)
            try:
                model.to("cpu")
                self.device = "cpu"
                predictions = self._predict(model, pairs)
            except Exception as cpu_exc:
                raise RerankerUnavailableError(
                    f"Reranker {self.model_id} could not score the retrieved candidates."
                ) from cpu_exc

        scores = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if scores.shape != (len(candidates),):
            raise RerankerUnavailableError(
                f"Reranker {self.model_id} returned an unexpected score shape."
            )
        if not np.isfinite(scores).all():
            raise RerankerUnavailableError(
                f"Reranker {self.model_id} returned invalid scores."
            )
        return scores.tolist()
