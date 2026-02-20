# RAG Knowledge Worker — Constitution of Estonia

A RAG (Retrieval-Augmented Generation) application over the Constitution of Estonia. It converts the constitution PDF to Markdown, builds a vector knowledge base in Chroma, and answers questions via Gemini 2.0 Flash (OpenRouter) with retrieval over OpenAI embeddings.

## Stack

- **Encoder:** OpenAI `text-embedding-3-small`
- **LLM (chunking, answers, evaluation judge):** Gemini 2.0 Flash via OpenRouter
- **Vector store:** Chroma
- **UI:** Gradio

## Setup

1. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync
   ```

2. **Environment variables** (create a `.env` in the project root):

   - `OPENAI_API_KEY` — used for embeddings only
   - `OPENROUTER_API_KEY` — used for Gemini 2.0 Flash (chunking, answers, eval judge)

## Usage

1. **Build the knowledge base** (convert PDF(s) in `assets/` to Markdown under `knowledge-base/constitution/`):

   ```bash
   uv run scripts/build_knowledge_base.py
   ```

2. **Ingest** (chunk with Gemini, embed with OpenAI, write to Chroma):

   ```bash
   uv run ingest.py
   ```

3. **Run the chat UI**:

   ```bash
   uv run app.py
   ```

4. **Run evaluation** (retrieval + LLM-as-judge on `evaluation/tests.jsonl`):

   ```bash
   uv run evaluation/eval.py 0       # single test by index
   uv run evaluation/eval.py --all   # all tests and aggregate metrics
   ```

5. **Run evaluation dashboard** (Gradio UI with retrieval and answer metrics, bar charts by category):

   ```bash
   uv run evaluator.py
   ```

## Project layout

- `assets/` — source PDF(s), e.g. `estonian_constitution.pdf`
- `scripts/build_knowledge_base.py` — PDF → Markdown
- `knowledge-base/` — generated Markdown (e.g. `knowledge-base/constitution/*.md`)
- `ingest.py` — chunking + embeddings → Chroma
- `vector_db/` — Chroma persistence (created by ingest)
- `answer.py` — RAG retrieval and answer generation
- `app.py` — Gradio chat UI
- `evaluator.py` — Gradio evaluation dashboard (retrieval + answer metrics, charts)
- `evaluation/` — test definitions (`tests.jsonl`), loader (`test.py`), runner (`eval.py`)

All user-facing text, docstrings, and comments are in English.
