"""Shared model-loading helpers (device selection, embedder)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    device = best_device()
    logger.info("Loading embedding model %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)
