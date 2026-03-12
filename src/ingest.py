import os
import pdfplumber
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_PATH = Path("docs")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-small"

# ── Step 1: Extract text from a PDF ───────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page by page from a PDF. Returns list of {page, text} dicts."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page": page_num,
                    "text": text.strip()
                })
    return pages


# ── Step 2: Split pages into chunks ───────────────────────────────────────────
def chunk_pages(pages: list[dict], filename: str) -> list[dict]:
    """Split page text into overlapping chunks. Preserves metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "content": split,
                "metadata": {
                    "filename": filename,
                    "page": page["page"],
                    "chunk_index": i
                }
            })
    return chunks


# ── Step 3: Embed a batch of chunks ───────────────────────────────────────────
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Call OpenAI embeddings API and attach embedding to each chunk."""
    texts = [chunk["content"] for chunk in chunks]

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    for i, embedding_obj in enumerate(response.data):
        chunks[i]["embedding"] = embedding_obj.embedding

    return chunks


# ── Step 4: Store chunks in Supabase ──────────────────────────────────────────
def store_chunks(chunks: list[dict]):
    """Insert chunks into the documents table in Supabase."""
    rows = [
        {
            "content": chunk["content"],
            "metadata": chunk["metadata"],
            "embedding": chunk["embedding"]
        }
        for chunk in chunks
    ]
    supabase.table("documents").insert(rows).execute()
    print(f"  Stored {len(rows)} chunks")


# ── Main ingestion loop ────────────────────────────────────────────────────────
def ingest_all():
    pdf_files = list(DOCS_PATH.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in /docs folder.")
        return

    print(f"Found {len(pdf_files)} PDFs\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        # Extract
        pages = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(pages)} pages")

        # Chunk
        chunks = chunk_pages(pages, pdf_path.name)
        print(f"  Created {len(chunks)} chunks")

        # Embed
        chunks = embed_chunks(chunks)
        print(f"  Embedded {len(chunks)} chunks")

        # Store
        store_chunks(chunks)
        print(f"  Done: {pdf_path.name}\n")

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_all()