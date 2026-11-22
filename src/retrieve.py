import os
import math
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

print("[STARTUP] retrieve.py v2 loaded — filename-first merge, alias map")

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


def _execute(query_builder, attempts: int = 3):
    """Run a Supabase request, retrying transient connection drops (these happen
    when several requests share the HTTP/2 client from the API's thread pool)."""
    for attempt in range(attempts):
        try:
            return query_builder.execute()
        except Exception as e:  # httpx.RemoteProtocolError / ConnectError etc.
            transient = any(t in type(e).__name__ for t in ("Protocol", "Connect", "Read", "Timeout")) \
                or "disconnected" in str(e).lower()
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(0.3 * (attempt + 1))
TOP_K = 5  # Number of chunks to retrieve

# ── Step 1: Embed the query ────────────────────────────────────────────────────
def embed_query(query: str) -> list[float]:
    """Convert the user's query into a vector using the same model as ingestion."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    return response.data[0].embedding


# ── Step 2: Search Supabase for similar chunks ─────────────────────────────────
def semantic_search(query_embedding: list[float], top_k: int = TOP_K, tenant_id: str = "default",
                     version_filter: str = "latest") -> list[dict]:
    """Find the most semantically similar chunks using cosine similarity."""
    # Request extra results so we have enough after version filtering
    fetch_count = top_k * 3
    print("  [DEBUG] calling match_documents RPC...")
    results = _execute(supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.10,
            "match_count": fetch_count,
            "tenant_id": tenant_id,
        }
    )).data
    print(f"  [DEBUG] match_documents returned {len(results)} results")

    # Default version fields — PostgREST schema cache may not expose new columns yet
    for r in results:
        r.setdefault("version", 1)
        r.setdefault("is_latest", True)

    # Post-filter by version since the RPC doesn't support version filtering
    if version_filter == "latest":
        results = [r for r in results if r.get("is_latest", True)]
    elif version_filter != "all":
        try:
            v = int(version_filter)
            results = [r for r in results if r.get("version", 1) == v]
        except (ValueError, TypeError):
            results = [r for r in results if r.get("is_latest", True)]

    print(f"  [DEBUG] after version filter: {len(results)} results")
    return results[:top_k]


# ── Step 3: Keyword search ─────────────────────────────────────────────────────
def _apply_version_filter(query_builder, version_filter):
    """Apply version filtering to a Supabase query builder."""
    if version_filter == "latest":
        query_builder = query_builder.eq("is_latest", True)
    elif version_filter != "all":
        # Specific version number
        try:
            query_builder = query_builder.eq("version", int(version_filter))
        except (ValueError, TypeError):
            query_builder = query_builder.eq("is_latest", True)
    return query_builder


KEYWORD_STOPWORDS = {"what", "which", "when", "where", "does", "that", "this", "with", "from",
                     "have", "must", "should", "would", "could", "your", "their", "there",
                     "about", "into", "they", "them", "than", "then", "also", "minimum",
                     "maximum", "required", "requirement", "requirements", "need", "needs"}


def _keywords(query: str) -> list[str]:
    """Distinct query terms worth matching. Identifiers with digits ('FP-006',
    'INS-009', 'C35-20-300', '4:12') are kept even when short."""
    out = []
    for raw in query.split():
        w = raw.strip("?.,!\"'()[]").lower()
        if not w or w in KEYWORD_STOPWORDS:
            continue
        if len(w) > 3 or (re.search(r"\d", w) and len(w) >= 2):
            out.append(w)
    return list(dict.fromkeys(out))


def keyword_search(query: str, top_k: int = TOP_K, tenant_id: str = "default",
                   version_filter: str = "latest") -> list[dict]:
    """Keyword match ranked by IDF-style weights: a term that appears in only a
    few chunks (an ID like 'FP-006', a value like '35 km/h') counts far more
    than a common word like 'inspection'."""
    words = _keywords(query)
    if not words:
        return []

    limit = top_k * 3
    chunk_scores: dict[str, dict] = {}

    for word in words:
        q = supabase.table("documents") \
            .select("id, content, metadata, version, is_latest") \
            .eq("tenant_id", tenant_id) \
            .ilike("content", f"%{word}%") \
            .limit(limit)
        q = _apply_version_filter(q, version_filter)
        rows = _execute(q).data
        if not rows:
            continue
        weight = math.log(1 + limit / len(rows)) * (1.5 if re.search(r"\d", word) else 1.0)
        for row in rows:
            rid = row["id"]
            if rid not in chunk_scores:
                row.setdefault("version", 1)
                row.setdefault("is_latest", True)
                chunk_scores[rid] = {"row": row, "score": 0.0}
            chunk_scores[rid]["score"] += weight

    ranked = sorted(chunk_scores.values(), key=lambda x: x["score"], reverse=True)
    print(f"  [KEYWORD] {len(chunk_scores)} unique chunks, top scores: {[round(r['score'], 2) for r in ranked[:5]]}")
    return [item["row"] for item in ranked[:top_k]]


# ── Step 4: Dynamic alias map ──────────────────────────────────────────────────
FILENAME_STOPWORDS = {"the", "and", "of", "a", "an", "in", "for", "to",
                      "with", "by", "is", "it", "be", "as", "at", "or", "on",
                      # Construction-generic terms that appear in many document names
                      "manual", "guide", "specification", "specifications", "spec",
                      "standard", "standards", "requirements", "requirement",
                      "installation", "document", "report", "sheet", "data",
                      "technical", "general", "product", "system", "systems"}

def build_alias_map(tenant_id: str) -> dict:
    """Fetch all registered filenames and build alias → [filename] lookup.
    Each filename part split by underscore becomes an alias, so:
      doc6_curtain_wall_installation_manual.pdf
    produces aliases: doc6, curtain, wall, installation, manual"""
    rows = _execute(supabase.table("document_registry")
                    .select("filename")
                    .eq("tenant_id", tenant_id))
    alias_map: dict = {}
    for row in rows.data:
        filename = row["filename"]
        stem = filename.lower()
        if "." in stem:
            stem = stem[:stem.rfind(".")]
        for part in stem.split("_"):
            if part and part not in FILENAME_STOPWORDS:
                alias_map.setdefault(part, [])
                if filename not in alias_map[part]:
                    alias_map[part].append(filename)
    return alias_map


def match_filenames_from_query(query: str, alias_map: dict) -> tuple:
    """Return (filenames explicitly referenced in the query, is_specific_match).
    Multi-token signals (bigrams, "document N") are more specific than a lone
    generic word (e.g. "test" matching several unrelated "qa_test_*" docs), so
    they take priority — a specific match like "test3" shouldn't be diluted by
    weaker single-word matches on documents that just share a common fragment.
    is_specific_match tells callers whether the match came from one of these
    high-confidence multi-token signals (True) or the generic single-word
    fallback (False) — needed to tell "this document" apart from "documents
    that happen to share a word with the query"."""
    raw_words = [w.strip("?.,!\"'").lower() for w in query.split()]
    matched = []
    seen = set()

    def _add(filenames):
        for fn in filenames:
            if fn not in seen:
                matched.append(fn)
                seen.add(fn)

    # bigrams: "doc 5" → "doc5", "curtain wall" → "curtainwall"
    for i in range(len(raw_words) - 1):
        bigram = raw_words[i] + raw_words[i + 1]
        if bigram in alias_map:
            _add(alias_map[bigram])

    # "document N" → "docN"
    for i, w in enumerate(raw_words):
        if w == "document" and i + 1 < len(raw_words) and raw_words[i + 1].isdigit():
            docn = "doc" + raw_words[i + 1]
            if docn in alias_map:
                _add(alias_map[docn])

    # A specific multi-token match is confident enough on its own — don't dilute
    # it with weaker single-word matches on unrelated documents.
    if matched:
        return matched, True

    # Single-word matches: prefer the INTERSECTION across every matched word
    # over the union — e.g. "TPO" alone matches two files, but "TPO" AND
    # "datasheet" together only agree on one, which is a much stronger signal
    # than either word alone. Only fall back to the looser union (matches any
    # one word) when no single document is consistent with every matched word.
    matched_words = list(dict.fromkeys(w for w in raw_words if w in alias_map))
    word_match_sets = [set(alias_map[w]) for w in matched_words]
    if word_match_sets:
        intersection = set.intersection(*word_match_sets)
        # One ordinary word ("inspection", "building") is not the user naming a
        # document. Count it as specific only with a distinctive token that has a
        # digit ("doc5", "test3") or two+ filename words that agree on one file.
        distinctive = any(re.search(r"\d", w) for w in matched_words)
        if len(intersection) == 1 and (distinctive or len(matched_words) >= 2):
            return list(intersection), True

    for w in raw_words:
        if w in alias_map:
            _add(alias_map[w])

    return matched, False


# ── Step 5: Filename search ────────────────────────────────────────────────────
def filename_search(query: str, top_k: int = TOP_K, tenant_id: str = "default",
                     version_filter: str = "latest") -> tuple:
    """Retrieve chunks from all documents explicitly referenced in the query.
    Works for any uploaded document dynamically — reads aliases from Supabase.
    Returns (chunks, is_specific_match)."""
    alias_map = build_alias_map(tenant_id)
    matched_filenames, is_specific_match = match_filenames_from_query(query, alias_map)
    print(f"  [FILENAME] matched: {matched_filenames} (specific={is_specific_match})")

    if not matched_filenames:
        return [], is_specific_match

    seen_ids: set = set()
    results: list[dict] = []

    # Fetch ALL chunks for each matched file — no limit, so summarization
    # queries get every page of the document
    for filename in matched_filenames:
        q = supabase.table("documents") \
            .select("id, content, metadata, version, is_latest") \
            .eq("tenant_id", tenant_id) \
            .eq("metadata->>filename", filename)
        q = _apply_version_filter(q, version_filter)
        rows = _execute(q)
        for row in rows.data:
            if row["id"] not in seen_ids:
                row.setdefault("version", 1)
                row.setdefault("is_latest", True)
                seen_ids.add(row["id"])
                results.append(row)

    return results, is_specific_match


# ── Step 6: Combine and deduplicate results ────────────────────────────────────
def hybrid_search(query: str, top_k: int = TOP_K, tenant_id: str = "default",
                  original_query: str = None, version_filter: str = "latest") -> dict:
    """Run semantic, keyword, and filename search — merge and deduplicate.
    Returns {"chunks": [...], "target_filename": str | None, "filename_match_specific": bool}."""
    # Always retrieve enough candidates for the reranker — top_k controls
    # final output, but initial retrieval needs a wider net
    retrieval_k = max(top_k * 2, 15)

    query_embedding = embed_query(query)

    semantic_results = semantic_search(query_embedding, retrieval_k, tenant_id, version_filter=version_filter)
    keyword_results = keyword_search(query, retrieval_k, tenant_id, version_filter=version_filter)

    # Always run filename search against the original user query — doc references
    # like "doc5" or "doc 5" must not be lost to LLM rewriting
    filename_results, filename_match_specific = filename_search(original_query or query, retrieval_k, tenant_id,
                                        version_filter=version_filter)

    # If filename search matched exactly one document, expose it as the target
    target_filename = None
    if filename_results:
        matched_files = {c["metadata"]["filename"] for c in filename_results}
        if len(matched_files) == 1:
            target_filename = next(iter(matched_files))

    # Merge order matters: downstream code caps the list BEFORE reranking, so
    # relevance order must survive the merge.
    #  - Specific filename match (user named the doc): its chunks go first.
    #  - Otherwise (a generic word like "TPO"/"curtain" merely overlapped a
    #    filename): those chunks are NOT evidence of relevance. Rank semantic +
    #    keyword hits by Reciprocal Rank Fusion and put the filename chunks last,
    #    so they can't push the genuinely matching chunk out of the candidate cap.
    def _rrf(results_lists):
        scores: dict = {}
        rows: dict = {}
        for lst in results_lists:
            for rank, chunk in enumerate(lst):
                scores[chunk["id"]] = scores.get(chunk["id"], 0.0) + 1.0 / (60 + rank)
                rows.setdefault(chunk["id"], chunk)
        return [rows[i] for i in sorted(scores, key=scores.get, reverse=True)]

    if filename_match_specific:
        ordered = [("filename", filename_results),
                   ("semantic", semantic_results),
                   ("keyword", keyword_results)]
    else:
        sem_ids = {c["id"] for c in semantic_results}
        for c in (semantic_results + keyword_results):
            c["_label"] = "semantic" if c["id"] in sem_ids else "keyword"
        fused = _rrf([semantic_results, keyword_results])
        # "filename_weak": the doc's name merely shares a word with the query —
        # these chunks get no relevance floor in the reranker.
        ordered = [(None, fused), ("filename_weak", filename_results)]

    seen_ids: set = set()
    combined: list[dict] = []
    for source_label, source_results in ordered:
        for chunk in source_results:
            if chunk["id"] not in seen_ids:
                chunk["source"] = source_label or chunk.pop("_label", "semantic")
                chunk.setdefault("version", 1)
                chunk.setdefault("is_latest", True)
                combined.append(chunk)
                seen_ids.add(chunk["id"])

    # When filename search matched specific docs, return all filename chunks
    # plus top_k from other sources — ensures full document coverage
    if filename_results:
        return {"chunks": combined, "target_filename": target_filename,
                "filename_match_specific": filename_match_specific}
    return {"chunks": combined[:retrieval_k + 5], "target_filename": None,
            "filename_match_specific": False}


# ── Test retrieval ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "What is the minimum TPO membrane thickness for commercial roofing?"

    print(f"Query: {test_query}\n")
    results = hybrid_search(test_query)

    for i, chunk in enumerate(results, 1):
        print(f"Result {i} [{chunk['source']}]")
        print(f"  File: {chunk['metadata']['filename']}")
        print(f"  Page: {chunk['metadata']['page']}")
        print(f"  Content: {chunk['content'][:150]}...")
        print()