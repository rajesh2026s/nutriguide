"""
Extracts text from the USDA PDF, splits it into overlapping chunks,
embeds each chunk, and builds a FAISS index.
Run this once locally whenever the source PDF changes:  python build_index.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
from sentence_transformers import SentenceTransformer
import faiss

PDF_PATH = "Dietary_Guidelines_for_Americans_2020-2025.pdf"
INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.json"

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap so context isn't cut mid-thought

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page_number": i + 1, "text": text})
    return pages


def chunk_text(text, page_number, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"page_number": page_number, "text": chunk})
        start += chunk_size - overlap
    return chunks


def main():
    print(f"Reading {PDF_PATH} ...")
    pages = extract_pages(PDF_PATH)
    print(f"Extracted text from {len(pages)} pages")

    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page["text"], page["page_number"]))
    print(f"Created {len(all_chunks)} chunks")

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # so inner product == cosine similarity

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved {INDEX_PATH} and {METADATA_PATH}")


if __name__ == "__main__":
    main()