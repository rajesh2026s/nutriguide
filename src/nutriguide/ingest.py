"""Build the FAISS index from the source PDF.

Extracts running text *and* tables from each page (tables are linearized into
"header: cell" rows so figures buried in tabular layouts remain retrievable),
splits everything into overlapping chunks, embeds them, and writes the index
plus chunk metadata.

Run whenever the source PDF changes:  pixi run build-index
"""

from __future__ import annotations

import json
import logging
from typing import Any

from nutriguide.config import Config

logger = logging.getLogger(__name__)


def linearize_table(table: list[list[str | None]]) -> str:
    """Turn an extracted table into retrieval-friendly text, one sentence per row.

    The first row is treated as the header; each data row becomes
    "header1: cell1; header2: cell2; ...". Cells the extractor could not read
    (None or empty) are skipped.
    """
    rows = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in table]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2:
        return ""

    header, *data = rows
    lines = []
    for row in data:
        pairs = [f"{h}: {c}" for h, c in zip(header, row, strict=False) if h and c]
        if pairs:
            lines.append("; ".join(pairs) + ".")
    return "\n".join(lines)


def extract_pages(pdf_path) -> list[dict[str, Any]]:
    """Extract one record per text unit: page body text and each table separately."""
    import pdfplumber

    units: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                units.append({"page_number": page_no, "text": text, "source": "text"})
            for table in page.extract_tables():
                table_text = linearize_table(table)
                if table_text:
                    units.append({"page_number": page_no, "text": table_text, "source": "table"})
    return units


def chunk_text(
    text: str, chunk_size: int, overlap: int
) -> list[str]:
    """Split text into overlapping chunks, breaking at whitespace where possible."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # back up to the last whitespace so we don't cut a word in half
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary != -1:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(units: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    all_chunks = []
    for unit in units:
        for piece in chunk_text(unit["text"], config.chunk_size, config.chunk_overlap):
            all_chunks.append(
                {"page_number": unit["page_number"], "text": piece, "source": unit["source"]}
            )
    return all_chunks


def build_index(config: Config | None = None) -> None:
    import faiss

    from nutriguide.models import load_embedder

    config = config or Config()
    logger.info("Reading %s", config.pdf_path)
    units = extract_pages(config.pdf_path)
    logger.info("Extracted %d text/table units", len(units))

    all_chunks = build_chunks(units, config)
    logger.info("Created %d chunks", len(all_chunks))

    embedder = load_embedder(config.embedding_model)
    embeddings = embedder.encode(
        [c["text"] for c in all_chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.index_path))
    with open(config.metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    logger.info("Saved %s and %s", config.index_path, config.metadata_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_index()


if __name__ == "__main__":
    main()
