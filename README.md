# 🥗 NutriGuide

A retrieval-augmented (RAG) nutrition chat assistant grounded in the **USDA
Dietary Guidelines for Americans, 2020–2025**. Ask a nutrition question, get a
concise answer with page-level citations back to the source document.

> ⚠️ Educational prototype. NutriGuide provides general dietary information
> only — not personalized medical or nutritional advice.

## How it works

```
question -> guardrails -> query condensation -> FAISS retrieval -> generation -> constraint check
```

- **Ingestion** (`nutriguide.ingest`): extracts running text *and* tables from
  the source PDF with `pdfplumber`. Tables are linearized into `header: cell`
  sentences so tabular figures (sodium limits, calorie tables, …) stay
  retrievable. Text is split into overlapping chunks, embedded with
  `all-MiniLM-L6-v2`, and stored in a FAISS inner-product index.
- **Conversational memory** (`nutriguide.pipeline`): follow-up questions
  ("which of those are vegetarian-friendly?") are rewritten into standalone
  queries using the chat history before retrieval, and recent turns are passed
  to the generator.
- **Constraint enforcement** (`nutriguide.constraints`): dietary constraints
  are not just prompted — generated answers are checked against a lexicon of
  non-compliant foods (with negation handling, so "avoid bacon" is fine). A
  violation triggers one corrective regeneration; anything remaining is
  flagged visibly in the answer.
- **Guardrails** (`nutriguide.guardrails`): low-relevance retrievals get an
  out-of-scope response; medication/dosage questions are refused; questions
  mentioning personal health conditions get a personalized-advice disclaimer.

## Setup

Requires [pixi](https://pixi.sh). The project is defined in `pyproject.toml`
(pixi workspace tables included).

```sh
pixi install
```

## Usage

```sh
# rebuild the FAISS index from the PDF in data/ (only needed when the PDF changes)
pixi run build-index

# launch the chat UI
pixi run app
```

The first launch downloads the embedding and generation models from Hugging
Face. GPU (CUDA) is used automatically when available; otherwise the app runs
on CPU.

## Development

```sh
pixi run test   # pytest
pixi run lint   # ruff
```

## Project layout

```
data/                     source PDF + built FAISS index and chunk metadata
src/nutriguide/
  config.py               paths, model names, pipeline parameters
  ingest.py               PDF text/table extraction, chunking, index build
  retriever.py            FAISS search
  generator.py            chat-template generation (device/dtype auto)
  constraints.py          dietary-constraint lexicon + compliance checking
  guardrails.py           medical-scope classification, out-of-scope handling
  pipeline.py             RAG orchestration (memory, retrieval, verification)
  app.py                  Gradio UI
tests/                    unit tests (no model downloads required)
```

## Known limitations

- Constraint checking is lexicon-based; uncommon constraints (e.g. "keto")
  are handled by prompting only.
- PDF table extraction is best-effort; complex multi-level tables may still
  linearize imperfectly.
- The system is scoped to general dietary information. It does not account
  for individual health conditions, medications, or lab values, and clinical
  questions are refused by design.
