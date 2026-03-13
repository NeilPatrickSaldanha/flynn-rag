import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from retrieve import hybrid_search
from query_understanding import classify_query

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Step 1: Rewrite query using conversation history ──────────────────────────
def rewrite_query_with_history(query: str, history: list[dict]) -> str:
    """
    If the query is a follow-up, rewrite it as a standalone question.
    If it's already standalone, return it unchanged.
    """
    if not history:
        return query

    # Build a summary of recent history — last 3 exchanges is enough
    recent = history[-6:]  # 3 user + 3 assistant messages
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in recent
    ])

    system_prompt = """You are a query rewriting system.

Given a conversation history and a follow-up question, rewrite the follow-up 
into a complete standalone question that makes sense without the history.

If the question is already standalone and doesn't reference anything from history, 
return it exactly as-is.

Return ONLY the rewritten question. No explanation, no preamble."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation history:\n{history_text}\n\nFollow-up question: {query}"}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ── Step 2: Rerank chunks by relevance ────────────────────────────────────────
def rerank_chunks(chunks: list[dict], query: str) -> list[dict]:
    if not chunks:
        return []

    system_prompt = """You are a relevance scoring system.
Score each chunk from 0.0 to 1.0 based on how directly it answers the query.
0.0 = completely irrelevant
0.5 = tangentially related
1.0 = directly answers the query

Respond ONLY with a JSON array of scores in the same order as the chunks.
Example: [0.9, 0.3, 0.7, 0.1, 0.8]
No other text."""

    chunks_text = "\n\n".join([
        f"Chunk {i+1}:\n{chunk['content']}"
        for i, chunk in enumerate(chunks)
    ])

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nChunks:\n{chunks_text}"}
        ],
        temperature=0
    )

    scores = json.loads(response.choices[0].message.content)

    for i, chunk in enumerate(chunks):
        chunk["relevance_score"] = scores[i] if i < len(scores) else 0.0

    ranked = sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)
    filtered = [c for c in ranked if c["relevance_score"] >= 0.3]

    return filtered


# ── Step 3: Assemble context from chunks ──────────────────────────────────────
def assemble_context(chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        filename = chunk["metadata"]["filename"]
        page = chunk["metadata"]["page"]
        content = chunk["content"]
        context_parts.append(
            f"[Source {i}: {filename}, Page {page}]\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


# ── Step 4: Generate answer ────────────────────────────────────────────────────
def generate_answer(query: str, context: str, query_type: str, history: list[dict]) -> str:
    type_instructions = {
        "lookup": "Provide a direct, precise answer. Lead with the specific value or requirement.",
        "comparison": "Structure your answer as a clear comparison. Use the sources to contrast the options.",
        "summarization": "Provide a structured summary organized by topic. Be comprehensive but concise.",
        "reasoning": "Think through the problem using the provided sources. Show your reasoning and cite evidence."
    }

    instruction = type_instructions.get(query_type, type_instructions["lookup"])

    system_prompt = f"""You are a construction industry technical assistant specializing in 
commercial roofing, glazing, curtain walls, and building envelope systems.

Answer questions using ONLY the provided source documents. 
Always cite your sources using [Source N] notation.
If the answer is not in the provided sources, say so clearly — do not fabricate.

{instruction}"""

    # Build messages with history
    messages = [{"role": "system", "content": system_prompt}]

    # Add last 3 exchanges from history for context
    if history:
        for msg in history[-6:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Add current question with context
    messages.append({
        "role": "user",
        "content": f"Question: {query}\n\nSources:\n{context}"
    })

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1
    )

    return response.choices[0].message.content


# ── Main pipeline ──────────────────────────────────────────────────────────────
def answer_query(query: str, history: list[dict] = None) -> dict:
    if history is None:
        history = []

    print(f"\nProcessing: {query}")

    # Rewrite query if it's a follow-up
    standalone_query = rewrite_query_with_history(query, history)
    if standalone_query != query:
        print(f"  Rewritten: {standalone_query}")

    # Classify and rewrite for retrieval
    query_info = classify_query(standalone_query)
    query_type = query_info["type"]
    rewritten = query_info["rewritten_query"]
    top_k = query_info["top_k"]

    print(f"  Type: {query_type} | Top K: {top_k}")

    # Retrieve
    chunks = hybrid_search(rewritten, top_k=top_k)
    print(f"  Retrieved: {len(chunks)} chunks")

    # Rerank
    ranked_chunks = rerank_chunks(chunks, rewritten)
    print(f"  After reranking: {len(ranked_chunks)} chunks kept")

    # Assemble context
    context = assemble_context(ranked_chunks)

    # Generate with history
    answer = generate_answer(query, context, query_type, history)

    return {
        "query": query,
        "query_type": query_type,
        "answer": answer,
        "sources": [
    {
        "filename": c["metadata"]["filename"],
        "page": c["metadata"]["page"],
        "relevance": c["relevance_score"],
        "content": c["content"]
    }
    for c in ranked_chunks
]
    }