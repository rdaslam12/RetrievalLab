"""Validation helpers for safe Hugging Face Hub model selection."""

from __future__ import annotations

import re


_MODEL_ID_PATTERN = re.compile(
    r"^(?=.{1,96}$)(?:[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/)?"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
_BLOCKED_MODEL_SUFFIXES = (".pt", ".pth", ".ckpt", ".py", ".exe", ".bin", ".sh")


class ModelValidationError(ValueError):
    """Raised when a user-provided model reference is not a safe Hub model ID."""


def validate_huggingface_model_id(model_id: str, *, kind: str) -> str:
    """Return a normalized Hub model ID or raise a user-facing validation error."""

    normalized = model_id.strip()
    label = kind.capitalize()
    if not normalized:
        raise ModelValidationError(f"Enter a Hugging Face model ID for the {kind} model.")
    if normalized.lower().endswith(_BLOCKED_MODEL_SUFFIXES):
        raise ModelValidationError(
            f"{label} model must be a Hugging Face model ID, not a model file."
        )
    if (
        normalized.startswith(("/", "./", "../", "~"))
        or "://" in normalized
        or "\\" in normalized
        or normalized.count("/") > 1
        or "--" in normalized
        or ".." in normalized
        or not _MODEL_ID_PATTERN.fullmatch(normalized)
    ):
        raise ModelValidationError(
            f"Invalid {kind} model ID '{normalized}'. Use a Hugging Face Hub ID such as "
            "organization/model-name. Local paths, URLs, uploaded checkpoints, and code are not supported."
        )
    return normalized
