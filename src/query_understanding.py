import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Query classification ───────────────────────────────────────────────────────
def classify_query(query: str) -> dict:
    """
    Classify the query type and rewrite it for better retrieval.
    Returns: { type, rewritten_query, top_k, requires_multiple_docs }
    """

    system_prompt = """You are a query analysis system for a construction industry knowledge base.
    
Classify the user's query into one of these types:
- lookup: seeking a specific fact, value, or requirement (e.g. "what is the minimum thickness")
- comparison: comparing two or more things (e.g. "compare TPO vs EPDM")
- summarization: asking for an overview of a topic (e.g. "summarize fall protection requirements")
- reasoning: requires combining information from multiple sources (e.g. "what roofing system should I use for climate zone 7")

Also rewrite the query to be more precise for document retrieval — expand abbreviations, add domain context.

Respond ONLY in this exact JSON format, no other text:
{
  "type": "lookup|comparison|summarization|reasoning",
  "rewritten_query": "expanded and precise version of the query",
  "top_k": 3,
  "requires_multiple_docs": true
}

Rules for top_k:
- lookup: 3
- comparison: 8
- summarization: 6
- reasoning: 8

Rules for requires_multiple_docs:
- lookup: false
- comparison: true
- summarization: false
- reasoning: true"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0
    )

    import json
    result = json.loads(response.choices[0].message.content)
    return result


# ── Test query understanding ───────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is the minimum TPO thickness?",
        "Compare TPO and EPDM membranes",
        "Summarize the fall protection requirements",
        "What roofing system should I use for a building in Winnipeg?"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        result = classify_query(query)
        print(f"  Type: {result['type']}")
        print(f"  Rewritten: {result['rewritten_query']}")
        print(f"  Top K: {result['top_k']}")
        print(f"  Multi-doc: {result['requires_multiple_docs']}")
        print()