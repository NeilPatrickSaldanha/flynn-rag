import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

print("[STARTUP] retrieve.py v2 loaded — filename-first merge, alias map")

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
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
    print("  [DEBUG] calling match_documents RPC...")
    results = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.2,
            "match_count": top_k,
            "tenant_id": tenant_id,
        }
    ).execute().data
    print(f"  [DEBUG] match_documents returned {len(results)} results")

    # Default version fields — PostgREST schema cache may not expose new columns yet
    for r in results:
        r.setdefault("version", 1)
        r.setdefault("is_latest", True)

    return results


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


def keyword_search(query: str, top_k: int = TOP_K, tenant_id: str = "default",
                   version_filter: str = "latest") -> list[dict]:
    """Find chunks whose content contains any significant word from the query."""
    # Use individual words (length > 3) so partial matches work
    words = [w.strip("?.,!\"'") for w in query.split() if len(w.strip("?.,!\"'")) > 3]
    if not words:
        return []

    seen_ids: set = set()
    results: list[dict] = []

    for word in words:
        rows = supabase.table("documents") \
            .select("id, content, metadata") \
            .eq("tenant_id", tenant_id) \
            .ilike("content", f"%{word}%") \
            .limit(top_k) \
            .execute()
        for row in rows.data:
            if row["id"] not in seen_ids:
                row.setdefault("version", 1)
                row.setdefault("is_latest", True)
                seen_ids.add(row["id"])
                results.append(row)
        if len(results) >= top_k:
            break

    return results[:top_k]


# ── Step 4: Dynamic alias map ──────────────────────────────────────────────────
FILENAME_STOPWORDS = {"the", "and", "of", "a", "an", "in", "for", "to",
                      "with", "by", "is", "it", "be", "as", "at", "or", "on"}

def build_alias_map(tenant_id: str) -> dict:
    """Fetch all registered filenames and build alias → [filename] lookup.
    Each filename part split by underscore becomes an alias, so:
      doc6_curtain_wall_installation_manual.pdf
    produces aliases: doc6, curtain, wall, installation, manual"""
    rows = supabase.table("document_registry") \
        .select("filename") \
        .eq("tenant_id", tenant_id) \
        .execute()
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


def match_filenames_from_query(query: str, alias_map: dict) -> list:
    """Return filenames explicitly referenced in the query via alias matching."""
    raw_words = [w.strip("?.,!\"'").lower() for w in query.split()]
    matched = []
    seen = set()

    def _add(filenames):
        for fn in filenames:
            if fn not in seen:
                matched.append(fn)
                seen.add(fn)

    for w in raw_words:
        if w in alias_map:
            _add(alias_map[w])

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

    return matched


# ── Step 5: Filename search ────────────────────────────────────────────────────
def filename_search(query: str, top_k: int = TOP_K, tenant_id: str = "default",
                     version_filter: str = "latest") -> list[dict]:
    """Retrieve chunks from all documents explicitly referenced in the query.
    Works for any uploaded document dynamically — reads aliases from Supabase."""
    alias_map = build_alias_map(tenant_id)
    matched_filenames = match_filenames_from_query(query, alias_map)
    print(f"  [FILENAME] matched: {matched_filenames}")

    if not matched_filenames:
        return []

    seen_ids: set = set()
    results: list[dict] = []

    # Fetch ALL chunks for each matched file — no limit, so summarization
    # queries get every page of the document
    for filename in matched_filenames:
        rows = supabase.table("documents") \
            .select("id, content, metadata") \
            .eq("tenant_id", tenant_id) \
            .eq("metadata->>filename", filename) \
            .execute()
        for row in rows.data:
            if row["id"] not in seen_ids:
                row.setdefault("version", 1)
                row.setdefault("is_latest", True)
                seen_ids.add(row["id"])
                results.append(row)

    return results


# ── Step 6: Combine and deduplicate results ────────────────────────────────────
def hybrid_search(query: str, top_k: int = TOP_K, tenant_id: str = "default",
                  original_query: str = None, version_filter: str = "latest") -> dict:
    """Run semantic, keyword, and filename search — merge and deduplicate.
    Returns {"chunks": [...], "target_filename": str | None}."""
    query_embedding = embed_query(query)

    semantic_results = semantic_search(query_embedding, top_k, tenant_id, version_filter=version_filter)
    keyword_results = keyword_search(query, top_k, tenant_id, version_filter=version_filter)

    # Always run filename search against the original user query — doc references
    # like "doc5" or "doc 5" must not be lost to LLM rewriting
    filename_results = filename_search(original_query or query, top_k, tenant_id,
                                        version_filter=version_filter)

    # If filename search matched exactly one document, expose it as the target
    target_filename = None
    if filename_results:
        matched_files = {c["metadata"]["filename"] for c in filename_results}
        if len(matched_files) == 1:
            target_filename = next(iter(matched_files))

    seen_ids: set = set()
    combined: list[dict] = []

    # Filename results go first — if the user asked for a specific doc, those
    # chunks must not be evicted by unrelated semantic/keyword hits
    for source_label, source_results in [
        ("filename", filename_results),
        ("semantic", semantic_results),
        ("keyword", keyword_results),
    ]:
        for chunk in source_results:
            if chunk["id"] not in seen_ids:
                chunk["source"] = source_label
                # Ensure version fields are present on every chunk
                chunk.setdefault("version", 1)
                chunk.setdefault("is_latest", True)
                combined.append(chunk)
                seen_ids.add(chunk["id"])

    # When filename search matched specific docs, return all filename chunks
    # plus top_k from other sources — ensures full document coverage
    if filename_results:
        return {"chunks": combined, "target_filename": target_filename}
    return {"chunks": combined[:top_k + 3], "target_filename": None}


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