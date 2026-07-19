"""FAISS-backed passage retrieval."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from nutriguide.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Passage:
    page_number: int
    text: str
    score: float
    source: str = "text"


class Retriever:
    """Embeds a query and returns the top-k passages by cosine similarity.

    ``embedder`` needs an ``encode`` method compatible with
    ``SentenceTransformer.encode``; it is injectable for testing.
    """

    def __init__(self, index, metadata: list[dict], embedder):
        self._index = index
        self._metadata = metadata
        self._embedder = embedder

    @classmethod
    def from_config(cls, config: Config) -> Retriever:
        import faiss

        from nutriguide.models import load_embedder

        if not config.index_path.exists():
            raise FileNotFoundError(
                f"{config.index_path} not found - run `pixi run build-index` first"
            )
        logger.info("Loading FAISS index from %s", config.index_path)
        index = faiss.read_index(str(config.index_path))
        with open(config.metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        if index.ntotal != len(metadata):
            raise ValueError(
                f"Index/metadata mismatch: {index.ntotal} vectors vs "
                f"{len(metadata)} chunks - rebuild with `pixi run build-index`"
            )
        return cls(index, metadata, load_embedder(config.embedding_model))

    def search(self, query: str, k: int) -> list[Passage]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        q_emb = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self._index.search(q_emb, k)
        return [
            Passage(
                page_number=self._metadata[idx]["page_number"],
                text=self._metadata[idx]["text"],
                score=float(score),
                source=self._metadata[idx].get("source", "text"),
            )
            for score, idx in zip(scores[0], indices[0], strict=True)
            if idx >= 0  # FAISS pads with -1 when there are fewer results than k
        ]
