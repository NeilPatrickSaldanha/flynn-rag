"""Rebuild every stored document's chunks with the current chunker.

Usage:
    python src/reindex.py            # dry run: shows what would change
    python src/reindex.py --apply    # back up, rebuild, swap

For each (tenant, filename, version) the source text is:
  - the file in docs/ when it is the latest version AND its text matches what is
    stored (so we never re-index a different file that happens to share a name);
  - otherwise the text reconstructed from the stored chunks (older versions,
    files that only exist on the deployed server).
Every existing row is written to backups/ first. New chunks are inserted before
old ones are deleted, so a failure part-way leaves the old chunks in place.
Legacy duplicate rows (ingested before versioning, no original_filename) are
folded into the version they duplicate.
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openpyxl
from ingest import (DOCS_PATH, extract_text, chunk_pages, embed_chunks, heading_level,
                    sheet_to_page, supabase)
from retrieve import _execute

ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = ROOT / "backups"


def fetch_all_rows() -> list[dict]:
    rows, start = [], 0
    while True:
        page = _execute(supabase.table("documents")
                        .select("id, content, metadata, embedding, tenant_id, version, is_latest, original_filename")
                        .order("id")
                        .range(start, start + 499)).data
        rows += page
        if len(page) < 500:
            return rows
        start += 500


def tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def merged_text(rows: list[dict]) -> str:
    by_page = defaultdict(list)
    for r in rows:
        by_page[r["metadata"].get("page", 1)].append(r)
    parts = []
    for page_no in sorted(by_page):
        ordered = sorted(by_page[page_no], key=lambda r: r["metadata"].get("chunk_index", 0))
        text = ordered[0]["content"]
        for r in ordered[1:]:
            text = _merge_overlap(text, r["content"])
        parts.append(text)
    return "\n".join(parts)


def file_matches_stored(path: Path, rows: list[dict]) -> bool:
    """Exact content check: the local file must contain the same words, the
    same number of times, as the stored version (tolerating only words the old
    splitter cut in two). A version that changed even one number fails."""
    stored = merged_text(rows)
    if path.suffix.lower() == ".xlsx":
        wb = openpyxl.load_workbook(path, data_only=True)
        file_cells = Counter(tokens(" ".join(" ".join("" if c is None else str(c) for c in r)
                                             for ws in wb for r in ws.iter_rows(values_only=True))))
        return Counter(tokens(stored.replace("Sheet:", " "))) - Counter(tokens(" ".join(wb.sheetnames))) == file_cells \
            or not (Counter(tokens(stored)) - file_cells - Counter(tokens("sheet " + " ".join(wb.sheetnames)))) \
            and not (file_cells - Counter(tokens(stored)))
    if path.suffix.lower() == ".docx":
        # compare against raw paragraph + cell text (new extraction adds column labels)
        from docx import Document
        d = Document(path)
        raw = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                prev = None
                for cell in row.cells:
                    if cell._tc is not prev:
                        raw.append(cell.text)
                    prev = cell._tc
        file_counts = Counter(tokens(" ".join(raw)))
    else:
        file_counts = Counter(tokens(" ".join(p["text"] for p in extract_text(path))))
    stored_counts = Counter(tokens(stored))
    stored_extra = stored_counts - file_counts
    file_extra = file_counts - stored_counts
    fragments = set(stored_extra)
    for word, n in file_extra.items():
        if not any(word.startswith(a) and word[len(a):] in fragments for a in fragments):
            return False
        for a in list(fragments):
            if word.startswith(a) and word[len(a):] in fragments:
                stored_extra[a] -= n
                stored_extra[word[len(a):]] -= n
                break
    leftover = +stored_extra
    # A few repeated words can survive an imperfect overlap merge; numbers never may.
    return sum(leftover.values()) <= 3 and not any(re.search(r"\d", w) for w in leftover)


def _merge_overlap(prev: str, nxt: str) -> str:
    """Join two consecutive chunks, removing the overlap the splitter added."""
    for k in range(min(len(prev), len(nxt), 400), 9, -1):
        if prev.endswith(nxt[:k]):
            return prev + nxt[k:]
    return prev + "\n" + nxt


def reconstruct_pages(rows: list[dict], filename: str) -> list[dict]:
    """Rebuild page text from stored chunks (ordered by page, chunk_index)."""
    by_page = defaultdict(list)
    for r in rows:
        by_page[r["metadata"].get("page", 1)].append(r)
    pages = []
    for page_no in sorted(by_page):
        ordered = sorted(by_page[page_no], key=lambda r: r["metadata"].get("chunk_index", 0))
        text = ordered[0]["content"]
        for r in ordered[1:]:
            text = _merge_overlap(text, r["content"])
        if filename.lower().endswith(".xlsx"):
            lines = text.split("\n")
            sheet = lines[0][len("Sheet: "):] if lines and lines[0].startswith("Sheet: ") else f"Sheet {page_no}"
            sheet_rows = [l.split("\t") for l in lines[1:] if l.strip()]
            built = sheet_to_page(page_no, sheet, sheet_rows)
            if built:
                pages.append(built)
        else:
            lines = [(l, heading_level(l.strip().rstrip(":"))) for l in text.split("\n") if l.strip()]
            pages.append({"page": page_no, "text": text, "lines": lines})
    return pages


def plan(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tenant_id"], r["metadata"].get("filename"))].append(r)

    jobs = []
    for (tenant, filename), file_rows in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        max_version = max(r["version"] or 1 for r in file_rows)
        by_version = defaultdict(list)
        for r in file_rows:
            by_version[r["version"] or 1].append(r)
        for version, vrows in sorted(by_version.items()):
            # Prefer rows written by the versioned upload path; legacy rows
            # (no original_filename) are only used when nothing else exists.
            proper = [r for r in vrows if r.get("original_filename")]
            legacy = [r for r in vrows if not r.get("original_filename")]
            source_rows = proper or legacy
            local = DOCS_PATH / filename
            source, pages = "chunks", None
            if version == max_version and local.exists():
                if file_matches_stored(local, source_rows):
                    source, pages = "file (exact match)", extract_text(local)
                else:
                    source = "chunks (local file differs from stored version)"
            if pages is None:
                pages = reconstruct_pages(source_rows, filename)
            jobs.append({
                "tenant": tenant, "filename": filename, "version": version,
                "is_latest": version == max_version, "source": source, "pages": pages,
                "old_ids": [r["id"] for r in vrows],
                "legacy_dupes": len(legacy) if proper else 0,
            })
    return jobs


def main(apply: bool):
    rows = fetch_all_rows()
    print(f"{len(rows)} stored chunks")
    jobs = plan(rows)
    for j in jobs:
        new = chunk_pages(j["pages"], j["filename"])
        j["chunks"] = new
        print(f"  {j['tenant']:8} {j['filename'][:42]:42} v{j['version']} latest={j['is_latest']!s:5} "
              f"old={len(j['old_ids']):3} new={len(new):3} legacy_dupes={j['legacy_dupes']:2} src={j['source']}")
    if not apply:
        print("\nDry run only. Re-run with --apply to back up and rebuild.")
        return

    BACKUP_DIR.mkdir(exist_ok=True)
    backup = BACKUP_DIR / f"documents_backup_{time.strftime('%Y%m%d_%H%M%S')}.json"
    backup.write_text(json.dumps(rows))
    print(f"\nBacked up {len(rows)} rows to {backup}")

    for j in jobs:
        chunks = embed_chunks(j["chunks"])
        payload = [{
            "content": c["content"], "metadata": c["metadata"], "embedding": c["embedding"],
            "tenant_id": j["tenant"], "version": j["version"], "is_latest": j["is_latest"],
            "original_filename": j["filename"],
        } for c in chunks]
        for i in range(0, len(payload), 50):
            _execute(supabase.table("documents").insert(payload[i:i + 50]))
        for i in range(0, len(j["old_ids"]), 100):
            _execute(supabase.table("documents").delete().in_("id", j["old_ids"][i:i + 100]))
        if j["is_latest"]:
            _execute(supabase.table("document_registry")
                     .update({"chunk_count": len(chunks), "version": j["version"], "is_latest": True})
                     .eq("filename", j["filename"]).eq("tenant_id", j["tenant"]))
        print(f"  rebuilt {j['filename']} v{j['version']}: {len(j['old_ids'])} -> {len(chunks)} chunks")
    print("Re-index complete.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
