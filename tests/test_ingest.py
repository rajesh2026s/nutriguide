import pytest

from nutriguide.ingest import chunk_text, linearize_table


def test_short_text_single_chunk():
    assert chunk_text("hello world", chunk_size=800, overlap=150) == ["hello world"]


def test_chunks_overlap_and_respect_size():
    words = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(words, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share text (overlap)
    assert chunks[0][-20:] in words
    joined = " ".join(chunks)
    assert "word0" in joined and "word499" in joined


def test_chunks_break_on_whitespace():
    words = " ".join(f"word{i}" for i in range(500))
    for chunk in chunk_text(words, chunk_size=200, overlap=50)[:-1]:
        assert not chunk.endswith("wor")  # no mid-word cuts at boundaries


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=100, overlap=100)


def test_linearize_table_basic():
    table = [
        ["Nutrient", "Daily Limit"],
        ["Sodium", "2,300 mg"],
        ["Saturated fat", "10% of calories"],
    ]
    out = linearize_table(table)
    assert "Nutrient: Sodium; Daily Limit: 2,300 mg." in out
    assert "Nutrient: Saturated fat; Daily Limit: 10% of calories." in out


def test_linearize_table_skips_empty_cells():
    table = [["A", "B"], [None, "x"], ["y", None]]
    out = linearize_table(table)
    assert out == "B: x.\nA: y."


def test_linearize_table_empty():
    assert linearize_table([]) == ""
    assert linearize_table([["only", "header"]]) == ""
