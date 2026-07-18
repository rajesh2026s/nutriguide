import faiss
import numpy as np

from nutriguide.retriever import Retriever


class FakeEmbedder:
    """Deterministic 4-dim embedder: each known text maps to a fixed unit vector."""

    VECTORS = {
        "sodium": [1.0, 0.0, 0.0, 0.0],
        "vegetables": [0.0, 1.0, 0.0, 0.0],
        "protein": [0.0, 0.0, 1.0, 0.0],
    }

    def encode(self, texts, **kwargs):
        return np.array([self.VECTORS[t] for t in texts], dtype="float32")


def make_retriever():
    embedder = FakeEmbedder()
    vectors = embedder.encode(["sodium", "vegetables"])
    index = faiss.IndexFlatIP(4)
    index.add(vectors)
    metadata = [
        {"page_number": 31, "text": "sodium limits", "source": "table"},
        {"page_number": 45, "text": "eat vegetables"},  # legacy entry without "source"
    ]
    return Retriever(index, metadata, embedder)


def test_search_returns_best_match_first():
    results = make_retriever().search("sodium", k=2)
    assert results[0].page_number == 31
    assert results[0].score > results[1].score
    assert results[0].source == "table"
    assert results[1].source == "text"  # default for legacy metadata


def test_k_larger_than_index_is_clamped():
    results = make_retriever().search("vegetables", k=10)
    assert len(results) == 2


def test_empty_index():
    retriever = Retriever(faiss.IndexFlatIP(4), [], FakeEmbedder())
    assert retriever.search("protein", k=4) == []
