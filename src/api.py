import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Annotated
import shutil
from dotenv import load_dotenv
from supabase import create_client

from generate import answer_query
from ingest import ingest_single_file, get_existing_versions, extract_text, SUPPORTED_EXTENSIONS
from security import scan_for_injection

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

app = FastAPI(title="Flynn FlinMind API")

# ── Guarantee CORS header on every response, including unhandled 500s ─────────
# CORSMiddleware alone misses exceptions caught by ServerErrorMiddleware (the
# outermost Starlette layer), so we add a thin wrapper that always sets the
# header after the rest of the stack has run.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

class ForceCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            response = StarletteResponse("Internal Server Error", status_code=500)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

app.add_middleware(ForceCORSMiddleware)

# ── CORS — allows React frontend to talk to this API ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    history: Optional[list[dict]] = []

class SourceModel(BaseModel):
    index: int
    filename: str
    page: int
    relevance: float
    content: str
    uploaded_at: Optional[str] = ""
    version: Optional[int] = 1
    is_latest: Optional[bool] = True

class QueryResponse(BaseModel):
    question: str
    query_type: str
    answer: str
    sources: list[SourceModel]


# ── Ensure CORS headers are present on all error responses ────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={"Access-Control-Allow-Origin": "*"},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ── Upload filename safety ─────────────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


def safe_upload_name(name: Optional[str]) -> Optional[str]:
    """Reduce a client-supplied filename to a bare base name. Backslashes are
    treated as path separators too (Windows clients), so "..\\x.pdf" and
    "../x.pdf" both become "x.pdf". Returns None for names that are empty or
    become empty ("", "..", "dir/", NUL bytes)."""
    if not name or "\x00" in name:
        return None
    base = os.path.basename(name.replace("\\", "/")).strip()
    if base in ("", ".", ".."):
        return None
    return base


def path_inside_docs(name: str) -> Optional[str]:
    """Absolute path for `name` inside DOCS_DIR, or None if it would resolve elsewhere."""
    root = os.path.realpath(DOCS_DIR)
    target = os.path.realpath(os.path.join(root, name))
    return target if os.path.commonpath([root, target]) == root else None


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok", "service": "Flynn FlinMind"}


class FormatSourceRequest(BaseModel):
    content: str


@app.post("/format-source")
async def format_source(request: FormatSourceRequest):
    """Format a raw document excerpt using OpenAI."""
    from llm import chat

    system_prompt = """You are a helpful assistant. Format the following raw construction document excerpt into clean, readable markdown. If it contains table data, format it as a proper markdown table with | separators. If it contains a list, format it as bullet points. Preserve all technical values exactly. Do not add or remove any information."""

    formatted = await run_in_threadpool(chat, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Format this excerpt:\n\n{request.content}"},
    ])
    return {"formatted": formatted}


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    x_tenant_id: Annotated[str, Header()] = "default",
):
    try:
        result = await run_in_threadpool(
            answer_query, request.question, history=request.history, tenant_id=x_tenant_id
        )

        # Enrich sources with upload dates from registry
        reg_result = supabase.table("document_registry") \
            .select("filename, uploaded_at") \
            .eq("tenant_id", x_tenant_id) \
            .execute()
        reg_lookup = {r["filename"]: r["uploaded_at"] for r in reg_result.data}

        sources = [
            SourceModel(
                index=i + 1,
                filename=s["filename"],
                page=s["page"],
                relevance=s["relevance"],
                content=s["content"],
                uploaded_at=reg_lookup.get(s["filename"], ""),
                version=s.get("version", 1),
                is_latest=s.get("is_latest", True),
            )
            for i, s in enumerate(result["sources"])
        ]

        return QueryResponse(
            question=result["query"],
            query_type=result["query_type"],
            answer=result["answer"],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents(x_tenant_id: Annotated[str, Header()] = "default"):
    result = supabase.table("document_registry") \
        .select("filename, uploaded_at, chunk_count, version, is_latest") \
        .eq("tenant_id", x_tenant_id) \
        .order("uploaded_at", desc=False) \
        .execute()
    return {"documents": result.data}


@app.delete("/documents/{filename}")
def delete_document(filename: str, x_tenant_id: Annotated[str, Header()] = "default"):
    safe_name = safe_upload_name(filename)
    local_path = path_inside_docs(safe_name) if safe_name else None
    if not local_path:
        return JSONResponse({"detail": "Invalid filename."}, status_code=400,
                            headers={"Access-Control-Allow-Origin": "*"})
    supabase.table("documents").delete() \
        .eq("metadata->>filename", safe_name) \
        .eq("tenant_id", x_tenant_id) \
        .execute()
    supabase.table("document_registry").delete() \
        .eq("filename", safe_name) \
        .eq("tenant_id", x_tenant_id) \
        .execute()
    if os.path.isfile(local_path):
        os.remove(local_path)
    return {"message": f"Deleted {safe_name}"}


MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# Magic bytes for each supported type
MAGIC_BYTES: dict[str, bytes] = {
    ".pdf":  b"%PDF-",
    ".docx": b"PK\x03\x04",  # DOCX is a ZIP archive
    ".xlsx": b"PK\x03\x04",  # XLSX is also a ZIP archive
    ".txt":  b"",             # No magic bytes for plain text
}

# Expected MIME types per extension
EXPECTED_MIME: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt":  "text/plain",
}

ACCEPTED_MIME: dict[str, set] = {
    ".pdf":  {"application/pdf", "application/x-pdf"},
    ".docx": {EXPECTED_MIME[".docx"], "application/msword", "application/zip", "application/x-zip-compressed"},
    ".xlsx": {EXPECTED_MIME[".xlsx"], "application/vnd.ms-excel", "application/zip", "application/x-zip-compressed"},
    ".txt":  {"text/plain", "text/markdown", "text/csv"},
}
GENERIC_MIME = {"application/octet-stream", "binary/octet-stream"}

# A .docx / .xlsx is a ZIP archive that must contain this part
OFFICE_PART = {".docx": "word/document.xml", ".xlsx": "xl/workbook.xml"}


def content_matches_extension(content: bytes, ext: str) -> bool:
    """Validate the actual bytes, not the client-declared type."""
    import io
    import zipfile
    if ext == ".pdf":
        return content.startswith(MAGIC_BYTES[".pdf"])
    if ext in OFFICE_PART:
        if not content.startswith(MAGIC_BYTES[ext]):
            return False
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                return OFFICE_PART[ext] in z.namelist()
        except zipfile.BadZipFile:
            return False
    if ext == ".txt":
        if b"\x00" in content[:8192]:
            return False
        try:
            content.decode("utf-8")
            return True
        except UnicodeDecodeError:
            try:
                content.decode("cp1252")
                return True
            except UnicodeDecodeError:
                return False
    return False


CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    force_new_version: bool = Form(False),
    x_tenant_id: Annotated[str, Header()] = "default",
):
    def err(status: int, detail: str):
        return JSONResponse({"detail": detail}, status_code=status, headers=CORS_HEADERS)

    # 0. Filename safety — never trust the client-supplied name as a path
    safe_name = safe_upload_name(file.filename)
    if not safe_name:
        return err(400, "Invalid filename.")

    ext = os.path.splitext(safe_name.lower())[1]

    # 1. Extension check
    if ext not in SUPPORTED_EXTENSIONS:
        return err(400, "Unsupported file type. Only PDF, Word (.docx), Excel (.xlsx), and plain text (.txt) files are allowed.")

    # 2. Read into memory so we can validate before saving
    content = await file.read()

    # 3. File size check
    if len(content) > MAX_FILE_SIZE_BYTES:
        return err(400, "File too large. Maximum allowed size is 20MB.")

    # 4. Declared MIME type — browsers disagree on Office types (e.g. Windows sends
    # application/vnd.ms-excel for .xlsx, curl sends octet-stream), so only reject
    # a declared type that is clearly something else (an image, video, executable…).
    declared_mime = (file.content_type or "").split(";")[0].strip().lower()
    if declared_mime and declared_mime not in ACCEPTED_MIME[ext] | GENERIC_MIME:
        return err(400, "File type mismatch. The file contents do not match the file extension.")

    # 5. Content check — the bytes must really be this kind of file
    if not content_matches_extension(content, ext):
        return err(400, "File type mismatch. The file contents do not match the file extension.")

    # 5. Duplicate check — return 409 if file exists and force_new_version is false
    if not force_new_version:
        existing = get_existing_versions(safe_name, x_tenant_id)
        if existing:
            return JSONResponse(
                {
                    "status": "duplicate",
                    "message": f"{safe_name} already exists",
                    "existing_versions": existing,
                },
                status_code=409,
                headers=CORS_HEADERS,
            )

    # 6. Save to disk and ingest
    save_path = path_inside_docs(safe_name)
    if not save_path:
        return err(400, "Invalid filename.")
    try:
        os.makedirs(DOCS_DIR, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(content)

        # Scan extracted text for prompt injection before ingesting
        pages = await run_in_threadpool(extract_text, save_path)
        all_text = " ".join(p["text"] for p in pages)
        scan = scan_for_injection(all_text)
        if not scan["safe"]:
            os.remove(save_path)
            return err(400, "Document rejected: potential prompt injection detected. Please upload construction-related documents only.")

        success, result = await run_in_threadpool(
            ingest_single_file, save_path, safe_name, x_tenant_id,
            force_new_version=force_new_version,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return err(500, str(e))

    if not success:
        return err(400, result)

    return JSONResponse(
        {"message": f"Successfully ingested {safe_name}", "chunks": result["chunks"], "version": result["version"]},
        headers=CORS_HEADERS,
    )
