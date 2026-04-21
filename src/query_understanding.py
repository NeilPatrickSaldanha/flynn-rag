import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Query preprocessing ────────────────────────────────────────────────────────
def preprocess_query(raw_query: str) -> dict:
    """
    Lightweight mechanical cleanup before classification.
    Format instruction detection is handled by the classifier LLM.
    Returns: { clean_query }
    """
    text = raw_query

    # 1. Strip voice-to-text filler words
    filler_patterns = [
        r'\blike you know\b',
        r'\bso like\b',
        r'\bi mean\b',
        r'\bkind of\b',
        r'\bbasically\b',
        r'\bum\b',
        r'\buh\b',
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 2. Normalize spaced acronyms (e.g. "T P O" → "TPO")
    text = re.sub(r'\b([A-Z])((?:\s+[A-Z])+)\b', lambda m: (m.group(1) + m.group(2)).replace(' ', ''), text)

    # Clean up extra whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()

    print(f"[PREPROCESS] raw_query      : {raw_query}")
    print(f"[PREPROCESS] clean_query    : {text}")

    return {
        "clean_query": text,
    }


# ── Query classification ───────────────────────────────────────────────────────
def classify_query(query: str) -> dict:
    """
    Classify the query type, extract format instructions, and rewrite for retrieval.
    Uses the LLM to intelligently detect how the user wants the answer formatted.
    Returns: { type, rewritten_query, top_k, requires_multiple_docs,
               format_instruction, clean_query }
    """
    preprocessed = preprocess_query(query)
    cleaned = preprocessed["clean_query"]

    system_prompt = """You are a query analysis system for a construction industry knowledge base.

Your job is to do THREE things:

1. CLASSIFY the user's query into one of these types:
- lookup: seeking a specific fact, value, or requirement (e.g. "what is the minimum thickness")
- comparison: comparing two or more things (e.g. "compare TPO vs EPDM")
- summarization: asking for an overview of a topic (e.g. "summarize fall protection requirements")
- reasoning: requires combining information from multiple sources (e.g. "what roofing system should I use for climate zone 7")

2. EXTRACT any formatting or style instructions the user has given about HOW they want the answer delivered. These are phrases like "answer in one word", "in one sentence", "briefly", "no explanation", "with detailed explanation", "in bullet points", "step by step", "explain in detail", "I NEED MORE EXPLANATION", "keep it short", etc. Also detect the tone — if the user is frustrated or emphasizing (e.g. ALL CAPS), acknowledge that in the instruction. Return the extracted instruction as a clear directive. If no formatting instruction is found, return an empty string.

3. REWRITE the query for document retrieval — expand abbreviations, add domain context, and REMOVE any formatting instructions from the rewritten query (those go in format_instruction, not here).
IMPORTANT: If the user references a specific document by name or number (e.g. "doc5", "doc 5", "document 2", "the safety manual", "the TPO datasheet"), preserve that reference exactly as-is in the rewritten query.

Respond ONLY in this exact JSON format, no other text:
{
  "type": "lookup|comparison|summarization|reasoning",
  "rewritten_query": "clean expanded query for retrieval, WITHOUT formatting instructions",
  "format_instruction": "the user's formatting/style directive, or empty string if none",
  "top_k": 3,
  "requires_multiple_docs": true
}

Rules for top_k:
- lookup: 3
- comparison: 8
- summarization: 20
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
            {"role": "user", "content": cleaned}
        ],
        temperature=0
    )

    import json
    result = json.loads(response.choices[0].message.content)

    # Detect version intent from the cleaned query
    result["version_intent"] = detect_version_intent(cleaned)
    # clean_query is the user's actual words (fillers removed) — NOT the LLM rewrite.
    # rewritten_query is used for retrieval/embedding; clean_query is shown to the LLM
    # as the user's question and used for filename matching.
    result["clean_query"] = cleaned

    print(f"[CLASSIFY] type={result['type']} | format_instruction={result.get('format_instruction','')}")
    print(f"[CLASSIFY] rewritten_query={result['rewritten_query']}")

    return result


def detect_version_intent(query: str) -> str:
    """Detect if the user is asking about a specific document version.
    Returns: 'latest' (default), 'previous', or a specific version number string."""
    q = query.lower()

    # Specific version number: "v1", "v2", "version 1", "version 2"
    match = re.search(r'\bv(?:ersion)?\s*(\d+)\b', q)
    if match:
        return match.group(1)

    # Previous / old version phrases
    previous_phrases = [
        "old version", "older version", "previous version", "earlier version",
        "original document", "original version", "before the update",
        "prior version", "first version", "initial version",
    ]
    for phrase in previous_phrases:
        if phrase in q:
            return "previous"

    return "latest"


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