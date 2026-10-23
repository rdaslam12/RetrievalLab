"""Model-choice resolution and dense-index lifecycle helpers."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .model_ids import validate_huggingface_model_id
from .search_engine import DenseSearchEngine
from .interfaces import adapter_model_id


def resolve_model_choice(
    mode: str,
    custom_model_id: str,
    *,
    default_model_id: str,
    kind: str,
) -> str:
    """Resolve a Default/Custom control into one validated model ID."""

    if mode == "Default":
        return default_model_id
    if mode not in ("Custom", "Hugging Face"):
        raise ValueError(f"Unsupported {kind} model selection: {mode}")
    return validate_huggingface_model_id(custom_model_id, kind=kind)


def invalidate_dense_indexes_for_model(
    state: MutableMapping[str, Any],
    selected_model_id: str,
) -> bool:
    """Discard only model-dependent indexes when their embedding model changes."""

    engine = state.get("engine")
    if not isinstance(engine, DenseSearchEngine):
        return False
    active_model_id = adapter_model_id(engine.embedder)
    if active_model_id == selected_model_id:
        return False
    state.pop("engine", None)
    state.pop("hybrid_engine", None)
    state.pop("search_response", None)
    state["dense_reindex_required"] = True
    summary = state.get("corpus_summary")
    if isinstance(summary, dict):
        updated_summary = dict(summary)
        updated_summary["embeddings"] = 0
        updated_summary["embedding_model_id"] = None
        updated_summary["embedding_dimension"] = None
        state["corpus_summary"] = updated_summary
    return True
