"""Central configuration for paths, model names, and pipeline parameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    if override := os.environ.get("NUTRIGUIDE_DATA_DIR"):
        return Path(override)
    # src/nutriguide/config.py -> project root / data
    return Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Config:
    data_dir: Path = field(default_factory=_default_data_dir)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    chunk_size: int = 800  # characters per chunk
    chunk_overlap: int = 150  # overlap so context isn't cut mid-thought

    top_k: int = 4
    relevance_threshold: float = 0.35  # cosine score below which we refuse to answer
    max_new_tokens: int = 400
    condense_max_new_tokens: int = 64
    history_turns: int = 6  # messages of chat history given to the model

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    @property
    def pdf_path(self) -> Path:
        return self.data_dir / "Dietary_Guidelines_for_Americans_2020-2025.pdf"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "metadata.json"
