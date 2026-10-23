"""Validated public runtime configuration for RetrievalLab."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "retrievallab.json"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime settings shared by the public retrieval pipelines."""

    profile_name: str
    model_id: str
    embedding_dimension: int
    candidate_depth: int
    top_results: int
    chunk_size: int = 350
    chunk_overlap: int = 50
    max_files: int = 20
    max_total_bytes: int = 50 * 1024 * 1024
    max_passages: int = 35_000

    def validate(self) -> None:
        if self.model_id != "Snowflake/snowflake-arctic-embed-s":
            raise ValueError("RetrievalLab supports only Snowflake Arctic-S")
        if self.embedding_dimension != 384:
            raise ValueError("Snowflake Arctic-S must use embedding_dimension=384")
        if self.candidate_depth != 100:
            raise ValueError("candidate_depth must be 100")
        if self.top_results <= 0 or self.top_results > self.candidate_depth:
            raise ValueError("top_results must be between 1 and candidate_depth")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and below chunk_size")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and strictly validate the public JSON runtime profile."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)
    required = {
        "profile_name",
        "model_id",
        "embedding_dimension",
        "candidate_depth",
        "top_results",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Configuration is missing fields: {', '.join(missing)}")
    unexpected = sorted(raw.keys() - required)
    if unexpected:
        raise ValueError(f"Configuration has unsupported fields: {', '.join(unexpected)}")
    config = AppConfig(
        profile_name=str(raw["profile_name"]),
        model_id=str(raw["model_id"]),
        embedding_dimension=int(raw["embedding_dimension"]),
        candidate_depth=int(raw["candidate_depth"]),
        top_results=int(raw["top_results"]),
    )
    config.validate()
    return config
