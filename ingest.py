"""
Ingest Constitution of Estonia documentation from knowledge-base/ into Chroma:
LLM-based chunking with Gemini 2.0 Flash, OpenAI text-embedding-3-small
embeddings, PersistentClient. Load .env from project root.
"""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from chromadb import PersistentClient
import litellm
from litellm import completion
from openai import OpenAI

# Reduce repeated "Provider List" / "LiteLLM.Info" messages on API errors and retries.
litellm.suppress_debug_info = True
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
DB_NAME = str(PROJECT_ROOT / "vector_db")
COLLECTION_NAME = "docs"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge-base"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_MODEL = "openrouter/google/gemini-2.0-flash-001"
AVERAGE_CHUNK_SIZE = 1200
WORKERS = 5  # One document in KB; use 1 to avoid idle pool. Increase if you add many docs.
# Split a single large .md into sections so each LLM call gets a smaller input (avoids timeouts).
MAX_DOCUMENT_CHARS = 25_000
COMPLETION_TIMEOUT = 180  # seconds per LLM call
WAIT = wait_exponential(multiplier=1, min=4, max=120)

client = OpenAI()


class Result(BaseModel):
    """Single chunk with content and metadata for Chroma."""

    page_content: str
    metadata: dict


class Chunk(BaseModel):
    """One chunk produced by the LLM: headline, summary, original text."""

    headline: str = Field(
        description="A brief heading for this chunk, likely to be matched by a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content to answer common questions",
    )
    original_text: str = Field(
        description="The original text of this chunk from the document, unchanged",
    )

    def as_result(self, document: dict) -> Result:
        metadata = {"source": document["source"], "type": document["type"]}
        content = f"{self.headline}\n\n{self.summary}\n\n{self.original_text}"
        return Result(page_content=content, metadata=metadata)


class Chunks(BaseModel):
    """List of chunks from the LLM."""

    chunks: list[Chunk]


def _split_large_document(text: str, source: str, doc_type: str) -> list[dict]:
    """Split text by markdown ## headings so each piece is under MAX_DOCUMENT_CHARS for faster LLM calls."""
    if len(text) <= MAX_DOCUMENT_CHARS:
        return [{"type": doc_type, "source": source, "text": text}]
    parts = re.split(r"\n(?=## )", text)
    out = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i > 0:
            part = "## " + part
        out.append({"type": doc_type, "source": source, "text": part})
    return out


def fetch_documents() -> list[dict]:
    """Load all .md and .mdx files from knowledge-base/, grouped by top-level folder (type)."""
    documents = []
    if not KNOWLEDGE_BASE_PATH.is_dir():
        print(
            f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}. "
            "Run scripts/build_knowledge_base.py first."
        )
        return documents
    for folder in sorted(KNOWLEDGE_BASE_PATH.iterdir()):
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for ext in ("*.md", "*.mdx"):
            for file in folder.rglob(ext):
                try:
                    text = file.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    print(f"Skip {file}: {e}")
                    continue
                source = file.as_posix()
                for doc in _split_large_document(text, source, doc_type):
                    documents.append(doc)
    print(f"Loaded {len(documents)} documents from {KNOWLEDGE_BASE_PATH}")
    return documents


def make_prompt(document: dict) -> str:
    """Build the chunking prompt for the LLM."""
    n = max(1, len(document["text"]) // AVERAGE_CHUNK_SIZE)
    return f"""You split a document into overlapping chunks for a RAG knowledge base.

The document is the Constitution of Estonia. Section: {document["type"]}. Source: {document["source"]}.

A chatbot will use these chunks to answer questions about the Constitution of Estonia. Split the document so the entire content is covered across chunks with some overlap (~25% or ~50 words). Aim for about {n} or more chunks. For each chunk provide: headline (short), summary (a few sentences), and original_text (exact copy of the segment). Respond with JSON only: a single object with key "chunks" containing a list of objects with keys "headline", "summary", "original_text".

Document:

{document["text"]}
"""


def _extract_json_from_content(content: str) -> str:
    """Extract JSON string from LLM response (may be wrapped in markdown or text)."""
    content = (content or "").strip()
    if not content:
        return "{}"
    # Try to find JSON object
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        return match.group(0)
    # Try code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        return match.group(1).strip()
    return content


@retry(wait=WAIT)
def process_document(document: dict) -> list[Result]:
    """Chunk one document using the LLM and return list of Result for Chroma."""
    messages = [{"role": "user", "content": make_prompt(document)}]
    response = completion(
        model=CHUNK_MODEL,
        messages=messages,
        timeout=COMPLETION_TIMEOUT,
    )
    raw = response.choices[0].message.content or "{}"
    json_str = _extract_json_from_content(raw)
    parsed = Chunks.model_validate_json(json_str)
    return [c.as_result(document) for c in parsed.chunks]


def create_chunks(documents: list[dict]) -> list[Result]:
    """Chunk all documents (sequential when WORKERS <= 1)."""
    if WORKERS <= 1:
        chunks = []
        for doc in tqdm(documents, desc="Chunking"):
            chunks.extend(process_document(doc))
        return chunks
    from multiprocessing import Pool

    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(process_document, documents),
            total=len(documents),
            desc="Chunking",
        ):
            chunks.extend(result)
    return chunks


def create_embeddings(chunks: list[Result]) -> None:
    """Embed all chunk texts with OpenAI and upsert into Chroma."""
    if not chunks:
        print("No chunks to embed. Skipping.")
        return
    chroma = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)
    texts = [c.page_content for c in chunks]
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(e.embedding for e in resp.data)
    ids = [str(i) for i in range(len(chunks))]
    metas = [c.metadata for c in chunks]
    coll = chroma.get_or_create_collection(COLLECTION_NAME)
    coll.add(ids=ids, embeddings=all_embeddings, documents=texts, metadatas=metas)
    print(f"Vector store created with {coll.count()} chunks at {DB_NAME}")


def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "Error: OPENROUTER_API_KEY is not set. "
            "Add it to .env for chunking with Gemini 2.0 Flash.",
            file=sys.stderr,
        )
        return
    documents = fetch_documents()
    if not documents:
        return
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
