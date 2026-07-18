import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch                      # import torch before faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss                      # faiss imported last

INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.json"
TOP_K = 4

print("Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Loading generation model (this can take a minute on first run)...")
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=200,     # trimmed slightly too
    device=0,               # 0 = first GPU (the T4)
)


def retrieve(query: str, k: int = TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    results = [
        {"score": float(scores[0][i]), **metadata[indices[0][i]]}
        for i in range(len(indices[0]))
    ]
    return results


def build_prompt(user_message: str, constraints: str, passages: list) -> str:
    context = "\n\n".join(
        f"[Page {p['page_number']}] {p['text']}" for p in passages
    )

    if constraints:
        constraint_block = f"""
STRICT DIETARY CONSTRAINT: {constraints}
You MUST only recommend foods that comply with this constraint. Do not mention
or recommend any food that violates it, even if it appears in the context below.
If the context only offers non-compliant examples, say so explicitly and suggest
the user consult a dietitian for compliant alternatives, rather than listing
non-compliant foods.
"""
    else:
        constraint_block = ""

    return f"""You are NutriGuide, a dietary assistant grounded in the USDA Dietary
Guidelines for Americans. Answer ONLY using the context below. Do not diagnose. Do not
recommend medication or dosages.
{constraint_block}
Context:
{context}

Question: {user_message}

Answer (remember to strictly respect the dietary constraint above, if any):"""


def answer_query(user_message: str, constraints: str = "") -> str:
    passages = retrieve(user_message)
    prompt = build_prompt(user_message, constraints, passages)

    output = generator(prompt, do_sample=False)[0]["generated_text"]
    response = output[len(prompt):].strip()

    pages = sorted(set(p["page_number"] for p in passages))
    page_list = ", ".join(str(p) for p in pages)
    return f"{response}\n\n*Source: USDA Dietary Guidelines for Americans, page(s) {page_list}*"