import os
import re
import pdfplumber
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as DocxDocument
import openpyxl

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from llm import chat

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
DOCS_PATH = Path(DOCS_DIR)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}

# ── Prompt injection patterns to block ────────────────────────────────────────
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "forget everything above",
    "new instructions:",
    "you are now",
    "act as if you",
    "system prompt:",
    "override your",
    "jailbreak",
]

def check_prompt_injection(pages: list[dict]) -> bool:
    """Return True if the extracted text contains prompt injection patterns."""
    all_text = " ".join(p["text"] for p in pages).lower()
    return any(pattern in all_text for pattern in INJECTION_PATTERNS)


def check_relevance(pages: list[dict], ext: str = "") -> bool:
    """Return True if the document is relevant to the construction knowledge base."""
    # Sample up to the first 1500 chars to keep the check fast and cheap.
    # .xlsx files front-load column headers before any data rows, so give
    # them a larger sample to reach content that actually shows relevance.
    sample_size = 3000 if ext == ".xlsx" else 1500
    sample = " ".join(p["text"] for p in pages)[:sample_size]

    answer = chat([
        {
            "role": "system",
            "content": (
                "You are a document relevance classifier for a construction industry knowledge base "
                "that covers commercial roofing, glazing, curtain walls, building envelope systems, "
                "building codes, site safety, and related technical specifications.\n\n"
                "Construction-related terms include things like: \"MPa\", \"f'c\", \"slump\", "
                "\"mix design\", \"W/CM\", \"concrete\", \"aggregate\", \"CSA\", \"ASTM\", "
                "\"compressive strength\", \"QC\", \"pour\", \"batch\".\n\n"
                "Respond with ONLY 'yes' if the document is relevant to any of those topics, "
                "or 'no' if it is not (e.g. invoices, personal documents, unrelated business docs)."
            ),
        },
        {"role": "user", "content": f"Document sample:\n{sample}"},
    ]).lower().strip(" .")
    return answer == "yes"

# ── Step 1: Extract text ──────────────────────────────────────────────────────
# Every extractor returns pages as {"page": n, "text": str, "lines": [(line, level)]}.
# `level` is a heading level (1 = top) when the line is a heading, else None.
# Spreadsheets instead return {"page": n, "text": str, "chunks": [str, ...]} —
# they are pre-chunked by rows so a table row is never cut in half.

_NUMBERED_HEADING = re.compile(r"^(?:(?:PART|SECTION|Section|Part)\s+)?(\d+(?:\.\d+)*)\.?\s*[—–:-]?\s+(.+)$")
_WORD = re.compile(r"[A-Za-z][A-Za-z'/&()-]*")


def _title_like(text: str) -> bool:
    words = [w for w in _WORD.findall(text) if len(w.strip("()")) > 3]
    if not words:
        return False
    capitalised = sum(1 for w in words if w.strip("(")[0].isupper())
    return capitalised / len(words) >= 0.6


def heading_level(line: str):
    """Return a heading level for a line that looks like a section heading, else None."""
    s = line.strip()
    if not (3 <= len(s) <= 90) or s.endswith((".", ",", ";")):
        return None
    m = _NUMBERED_HEADING.match(s)
    if m:
        rest = m.group(2).strip()
        if (len(rest.split()) <= 9 and not re.search(r"\d|%", rest) and _title_like(rest)):
            depth = m.group(1).count(".") + 1
            if re.match(r"^(PART|Part)\b", s):
                depth = 1
            return depth
        return None
    letters = [c for c in s if c.isalpha()]
    words = [w for w in _WORD.findall(s) if len(w) >= 3]
    if (len(letters) >= 6 and len(words) >= 2 and len(s.split()) <= 10
            and not re.search(r"\d", s)
            and sum(c.isupper() for c in letters) / len(letters) > 0.85):
        return 1
    return None


def _lines_to_pages(lines: list[tuple], chars_per_page: int = 3000) -> list[dict]:
    """Group (line, level) tuples into virtual pages at line boundaries."""
    pages, current, size = [], [], 0
    for line, level in lines:
        if current and size + len(line) > chars_per_page:
            pages.append(current)
            current, size = [], 0
        current.append((line, level))
        size += len(line) + 1
    if current:
        pages.append(current)
    return [{"page": i + 1, "text": "\n".join(l for l, _ in p), "lines": p} for i, p in enumerate(pages)]


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page by page from a PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                lines = [(l, heading_level(l)) for l in text.split("\n") if l.strip()]
                pages.append({"page": page_num, "text": text, "lines": lines})
    return pages


def _table_rows_as_text(rows: list[list[str]]) -> list[str]:
    """Render table rows so each row carries its column names, e.g.
    'Application: Wide flange beams | Standard: CSA G40.21 | Yield Strength: 350 MPa'."""
    rows = [[c.strip() for c in r] for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    header = rows[0]
    if len(rows) == 1 or sum(1 for c in header if c) < 2:
        return [" | ".join(c for c in r if c) for r in rows]
    out = [" | ".join(c for c in header if c)]
    for r in rows[1:]:
        cells = [f"{h}: {v}" if h and v and h != v else v for h, v in zip(header, r)]
        out.append(" | ".join(c for c in cells if c))
    return out


def extract_text_from_docx(docx_path: Path) -> list[dict]:
    """Extract a .docx in reading order — tables stay under the heading they belong to."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = DocxDocument(docx_path)
    lines: list[tuple] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            level = None
            if style.startswith("Heading"):
                digits = re.findall(r"\d+", style)
                level = int(digits[0]) if digits else 1
            elif style == "Title":
                level = 1
            else:
                level = heading_level(text)
            lines.append((text, level))
        elif tag == "tbl":
            table = Table(child, doc)
            rows = []
            for row in table.rows:
                cells, prev = [], None
                for cell in row.cells:      # merged cells repeat — keep one copy
                    if cell._tc is not prev:
                        cells.append(cell.text)
                    prev = cell._tc
                rows.append(cells)
            lines.extend((t, None) for t in _table_rows_as_text(rows))
    return _lines_to_pages(lines)


def clean_text(text: str) -> str:
    """Remove decorative separator lines and collapse runs of blank lines."""
    cleaned = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and set(stripped).issubset(set("-=_*# ")):
            continue
        cleaned.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))


def extract_text_from_txt(txt_path: Path) -> list[dict]:
    """Extract a plain text file into virtual pages, detecting headings."""
    text = clean_text(txt_path.read_text(encoding="utf-8", errors="ignore"))
    lines = [(l.strip(), heading_level(l.strip().rstrip(":"))) for l in text.split("\n") if l.strip()]
    return _lines_to_pages(lines)


XLSX_CHUNK_CHARS = 3500


def sheet_to_page(sheet_index: int, sheet_name: str, rows: list[list[str]]):
    """Render one sheet's rows as a pre-chunked page. Rows are labelled with their
    column names and grouped into whole-row chunks; title lines and notes are
    repeated in every chunk so each chunk can be understood on its own."""
    rows = [[("" if c is None else str(c)).strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return None

    def filled(r):
        return sum(1 for c in r if c)

    header_idx = next((i for i in range(len(rows) - 1)
                       if filled(rows[i]) >= 3 and filled(rows[i + 1]) >= 3), None)
    if header_idx is None:
        titles, body, notes = [], [" ".join(c for c in r if c) for r in rows], []
    else:
        titles = [" | ".join(c for c in r if c) for r in rows[:header_idx]]
        header = rows[header_idx]
        body, notes = [], []
        for r in rows[header_idx + 1:]:
            if filled(r) >= 2:
                body.append(" | ".join(f"{h}: {v}" if h else v for h, v in zip(header, r) if v))
            else:
                notes.append(" ".join(c for c in r if c))
        if body:
            body = [f"Columns: {' | '.join(c for c in header if c)}"] + body

    prefix = "\n".join([f"Sheet: {sheet_name}"] + titles)
    notes_text = "\n".join(notes)
    budget = max(800, XLSX_CHUNK_CHARS - len(prefix) - len(notes_text))
    groups, current, size = [], [], 0
    for line in body:
        if current and size + len(line) > budget:
            groups.append(current)
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current or not groups:
        groups.append(current)
    chunks, start = [], 0
    for g in groups:
        span = f"(rows {start + 1}-{start + len(g)} of {len(body)})" if len(groups) > 1 else ""
        parts = [prefix + (" " + span if span else "")] + g + ([notes_text] if notes_text else [])
        chunks.append("\n".join(p for p in parts if p))
        start += len(g)
    return {"page": sheet_index, "text": "\n".join([prefix] + body + notes),
            "chunks": chunks, "section": f"Sheet: {sheet_name}"}


def extract_text_from_xlsx(file_path: Path) -> list[dict]:
    """One pre-chunked page per sheet (see sheet_to_page)."""
    wb = openpyxl.load_workbook(file_path, data_only=True)
    pages = []
    for sheet_index, sheet_name in enumerate(wb.sheetnames, start=1):
        page = sheet_to_page(sheet_index, sheet_name, [list(r) for r in wb[sheet_name].iter_rows(values_only=True)])
        if page:
            pages.append(page)
    return pages


# ── Dispatch extractor by file type ───────────────────────────────────────────
def extract_text(file_path) -> list[dict]:
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".xlsx":
        return extract_text_from_xlsx(file_path)
    return []


# ── Step 2: Split pages into chunks ───────────────────────────────────────────
def document_title(filename: str, pages: list[dict]) -> str:
    """Readable title: the filename plus the document's own first line."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    first = ""
    for p in pages:
        for line in p.get("text", "").split("\n"):
            if line.strip():
                first = line.strip()
                break
        if first:
            break
    if first and not first.lower().startswith("sheet:") and first.lower() != stem.lower():
        return f"{stem} — {first[:80]}"
    return stem


def chunk_pages(pages: list[dict], filename: str) -> list[dict]:
    """Section-aware chunking. Text is grouped under the heading it belongs to,
    small sections are merged, large ones are split with overlap, and every
    chunk records its document title and section path in metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )
    title = document_title(filename, pages)
    chunks: list[dict] = []
    stack: list[tuple] = []   # (level, heading text) — carried across pages

    def add(content: str, page: int, section: str, index: int):
        section = " / ".join(p for p in (x.strip() for x in section.split(" / ")) if p)[:240]
        prev = chunks[-1] if chunks else None
        # Fold a tiny leftover into the previous chunk on the same page
        if (prev and len(content) < 150 and prev["metadata"]["page"] == page
                and len(prev["content"]) + len(content) < CHUNK_SIZE + 300):
            prev["content"] += "\n" + content
            return
        chunks.append({
            "content": content,
            "metadata": {"filename": filename, "page": page, "chunk_index": index,
                         "section": section, "doc_title": title},
        })

    for page in pages:
        if "chunks" in page:   # spreadsheets arrive pre-chunked by whole rows
            for i, text in enumerate(page["chunks"]):
                add(text, page["page"], page.get("section", ""), i)
            continue

        lines = page.get("lines") or [(l, None) for l in page["text"].split("\n") if l.strip()]
        # 1. Group lines into blocks, one per section
        blocks: list[dict] = []
        for line, level in lines:
            if level is not None:
                # An ALL-CAPS subheading under a numbered PART/SECTION heading
                # belongs beneath it, not beside it.
                is_caps = not re.search(r"\d", line) and line.strip().rstrip(":").upper() == line.strip().rstrip(":")
                if (level == 1 and is_caps and stack
                        and re.match(r"^(PART|SECTION|Part|Section)?\s*\d", stack[0][1])):
                    level = 2
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, line.strip()))
                blocks.append({"section": " > ".join(h for _, h in stack), "lines": [line]})
            else:
                if not blocks:
                    blocks.append({"section": " > ".join(h for _, h in stack), "lines": []})
                blocks[-1]["lines"].append(line)

        # 2. Merge small neighbouring blocks, split big ones
        index = 0
        pending_text, pending_sections = "", []
        def flush():
            nonlocal pending_text, pending_sections, index
            if pending_text.strip():
                add(pending_text.strip(), page["page"], " / ".join(dict.fromkeys(pending_sections)), index)
                index += 1
            pending_text, pending_sections = "", []

        for block in blocks:
            text = "\n".join(block["lines"]).strip()
            if not text:
                continue
            if len(text) > CHUNK_SIZE:
                flush()
                for piece in splitter.split_text(text):
                    add(piece, page["page"], block["section"], index)
                    index += 1
            elif len(pending_text) + len(text) + 1 > CHUNK_SIZE:
                flush()
                pending_text, pending_sections = text, [block["section"]]
            else:
                pending_text = f"{pending_text}\n{text}" if pending_text else text
                pending_sections.append(block["section"])
        flush()
    return chunks


def chunk_header(metadata: dict) -> str:
    """Context line that travels with a chunk into embeddings, reranking and the answer prompt."""
    title = metadata.get("doc_title") or metadata.get("filename", "")
    section = metadata.get("section") or ""
    return f"[{title}{' — ' + section if section else ''}]"


# ── Step 3: Embed a batch of chunks ───────────────────────────────────────────
EMBED_BATCH = 100


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed each chunk together with its document/section header, so a chunk
    like 'Overlap Width 3 inch' is still findable as 'TPO hot-air welding'."""
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start: start + EMBED_BATCH]
        texts = [f"{chunk_header(c['metadata'])}\n{c['content']}" for c in batch]
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        for c, emb in zip(batch, response.data):
            c["embedding"] = emb.embedding
    return chunks


# ── Step 4: Store chunks in Supabase ──────────────────────────────────────────
BATCH_SIZE = 50

def store_chunks(chunks: list[dict], tenant_id: str = "default",
                  version: int = 1, is_latest: bool = True, original_filename: str = ""):
    """Insert chunks into the documents table in Supabase in batches."""
    rows = [
        {
            "content": chunk["content"],
            "metadata": chunk["metadata"],
            "embedding": chunk["embedding"],
            "tenant_id": tenant_id,
            "version": version,
            "is_latest": is_latest,
            "original_filename": original_filename or chunk["metadata"].get("filename", ""),
        }
        for chunk in chunks
    ]
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        supabase.table("documents").insert(batch).execute()
    print(f"  Stored {len(rows)} chunks (version={version})")


# ── Step 5: Register document in registry ─────────────────────────────────────
def register_document(filename: str, chunk_count: int, tenant_id: str = "default",
                       version: int = 1, is_latest: bool = True, original_filename: str = ""):
    """Record the document in document_registry with upload timestamp."""
    supabase.table("document_registry").upsert({
        "filename": filename,
        "chunk_count": chunk_count,
        "tenant_id": tenant_id,
        "version": version,
        "is_latest": is_latest,
        "original_filename": original_filename or filename,
    }, on_conflict="filename,tenant_id").execute()


# ── Main ingestion loop ────────────────────────────────────────────────────────
def ingest_all():
    all_files = [
        f for ext in SUPPORTED_EXTENSIONS
        for f in DOCS_PATH.glob(f"*{ext}")
    ]

    if not all_files:
        print("No supported files found in /docs folder.")
        return

    print(f"Found {len(all_files)} files\n")

    for file_path in all_files:
        existing = supabase.table("documents") \
            .select("id") \
            .eq("metadata->>filename", file_path.name) \
            .limit(1) \
            .execute()
        if existing.data:
            print(f"Skipping (already ingested): {file_path.name}")
            continue

        print(f"Processing: {file_path.name}")
        pages = extract_text(file_path)
        print(f"  Extracted {len(pages)} pages")

        chunks = chunk_pages(pages, file_path.name)
        print(f"  Created {len(chunks)} chunks")

        chunks = embed_chunks(chunks)
        print(f"  Embedded {len(chunks)} chunks")

        store_chunks(chunks, tenant_id="default", version=1, is_latest=True, original_filename=file_path.name)
        register_document(file_path.name, len(chunks), tenant_id="default",
                          version=1, is_latest=True, original_filename=file_path.name)
        print(f"  Done: {file_path.name}\n")

    print("Ingestion complete.")


def get_existing_versions(filename: str, tenant_id: str = "default") -> list[dict]:
    """Return all existing version records for a filename from document_registry."""
    # Match on either column: registry rows created before the versioning
    # migration have filename set but original_filename empty.
    found = {}
    for column in ("original_filename", "filename"):
        rows = supabase.table("document_registry") \
            .select("filename, version, is_latest, uploaded_at, chunk_count") \
            .eq(column, filename) \
            .eq("tenant_id", tenant_id) \
            .execute().data
        for r in rows:
            found[(r["filename"], r["version"])] = r
    return sorted(found.values(), key=lambda r: r["version"] or 1)


def get_document_version_history(filename: str, tenant_id: str = "default") -> list[dict]:
    """Return all versions of a document with their version numbers and upload dates."""
    return get_existing_versions(filename, tenant_id)


def ingest_single_file(file_path: str, filename: str, tenant_id: str = "default",
                        force_new_version: bool = False):
    """Ingest a single file — used for runtime uploads from the UI.
    If force_new_version=True and a document with the same filename exists,
    mark old versions as not latest and ingest as a new version."""
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"

    print(f"Ingesting: {filename} (tenant: {tenant_id}, force_new_version={force_new_version})")

    existing_versions = get_existing_versions(filename, tenant_id)

    if existing_versions and not force_new_version:
        return False, f"{filename} has already been ingested. No duplicates added."

    path = Path(file_path)
    pages = extract_text(path)

    if not pages:
        return False, "Could not extract text from this file. It may be scanned, image-based, or empty."

    if check_prompt_injection(pages):
        os.remove(path)
        return False, "This document was rejected because it contains content that is not allowed."

    if not check_relevance(pages, ext=path.suffix.lower()):
        os.remove(path)
        return False, "This document does not appear to be relevant to the construction knowledge base. Only roofing, glazing, building codes, safety, and related technical documents are accepted."

    # Determine version number — previous versions are only demoted AFTER the
    # new file has passed every check, so a rejected upload never leaves the
    # document without a current version.
    if existing_versions:
        new_version = max(v["version"] for v in existing_versions) + 1
    else:
        new_version = 1

    chunks = chunk_pages(pages, filename)
    chunks = embed_chunks(chunks)

    if existing_versions:
        # Match on metadata->>filename: every chunk has it, while legacy rows
        # ingested before the versioning migration have no original_filename.
        supabase.table("documents") \
            .update({"is_latest": False}) \
            .eq("metadata->>filename", filename) \
            .eq("tenant_id", tenant_id) \
            .execute()
        supabase.table("document_registry") \
            .update({"is_latest": False}) \
            .eq("original_filename", filename) \
            .eq("tenant_id", tenant_id) \
            .execute()
        print(f"  Marked {len(existing_versions)} previous version(s) as not latest")

    store_chunks(chunks, tenant_id, version=new_version, is_latest=True, original_filename=filename)
    register_document(filename, len(chunks), tenant_id,
                      version=new_version, is_latest=True, original_filename=filename)

    print(f"Ingested {len(chunks)} chunks from {filename} (version {new_version})")
    return True, {"chunks": len(chunks), "version": new_version}


# Backward-compatible alias
ingest_single_pdf = ingest_single_file

if __name__ == "__main__":
    ingest_all()