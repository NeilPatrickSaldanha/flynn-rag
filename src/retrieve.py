import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

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
def semantic_search(query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    """Find the most semantically similar chunks using cosine similarity."""
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.3,
            "match_count": top_k
        }
    ).execute()

    return result.data


# ── Step 3: Keyword search ─────────────────────────────────────────────────────
def keyword_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Find chunks containing exact or close keyword matches."""
    result = supabase.table("documents") \
        .select("id, content, metadata") \
        .ilike("content", f"%{query}%") \
        .limit(top_k) \
        .execute()

    return result.data


# ── Step 4: Combine and deduplicate results ────────────────────────────────────
def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Run both semantic and keyword search, merge and deduplicate results."""
    query_embedding = embed_query(query)

    semantic_results = semantic_search(query_embedding, top_k)
    keyword_results = keyword_search(query, top_k)

    # Merge — deduplicate by id
    seen_ids = set()
    combined = []

    for chunk in semantic_results:
        if chunk["id"] not in seen_ids:
            chunk["source"] = "semantic"
            combined.append(chunk)
            seen_ids.add(chunk["id"])

    for chunk in keyword_results:
        if chunk["id"] not in seen_ids:
            chunk["source"] = "keyword"
            combined.append(chunk)
            seen_ids.add(chunk["id"])

    return combined[:top_k + 3]  # Return slightly more for reranking later


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