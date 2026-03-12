import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from generate import answer_query

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Ground truth test set ──────────────────────────────────────────────────────
# These are questions where we know the correct answer from our documents.
# In a real system you'd have domain experts write these.

TEST_SET = [
    {
        "question": "What is the minimum TPO membrane thickness for commercial roofing?",
        "expected_answer": "60 mils (1.52 mm)",
        "expected_source": "doc1_roofing_installation_manual.pdf"
    },
    {
        "question": "At what height is fall protection mandatory on construction sites in Ontario?",
        "expected_answer": "10 feet (3 metres)",
        "expected_source": "doc4_site_safety_manual.pdf"
    },
    {
        "question": "What is the minimum effective R-value for roofs in Climate Zone 7?",
        "expected_answer": "R-31.0",
        "expected_source": "doc2_building_code_reference.pdf"
    },
    {
        "question": "What is the minimum seam overlap width for TPO hot-air welding?",
        "expected_answer": "3 inches (76 mm)",
        "expected_source": "doc5_tpo_product_datasheet.pdf"
    },
    {
        "question": "What is the maximum air infiltration rate for curtain wall systems?",
        "expected_answer": "0.06 cfm/sq ft at 1.57 psf",
        "expected_source": "doc3_glazing_curtain_wall_spec.pdf"
    },
    {
        "question": "What is the fire watch duration after torch work ceases?",
        "expected_answer": "60 minutes",
        "expected_source": "doc4_site_safety_manual.pdf"
    },
    {
        "question": "What is the warranty period for the ArmorFlex 25-Year Premium Warranty?",
        "expected_answer": "25 years",
        "expected_source": "doc5_tpo_product_datasheet.pdf"
    },
    {
        "question": "What colour is the ArmorFlex TPO membrane?",
        "expected_answer": "NOT_IN_DOCUMENTS",
        "expected_source": None
    },
]

# ── Evaluator ──────────────────────────────────────────────────────────────────
def evaluate_answer(question: str, expected: str, actual_answer: str, sources_used: list) -> dict:
    """Use LLM to score the answer against the expected answer."""

    if expected == "NOT_IN_DOCUMENTS":
        # Special case — we expect the system to say it doesn't know
        system_prompt = """You are evaluating a RAG system's ability to correctly refuse to answer 
questions that are not covered in its source documents.

The correct behavior is to clearly state the answer is not available in the sources.
Score 1.0 if the system correctly refused or said it couldn't find the answer.
Score 0.0 if the system hallucinated an answer.

Respond ONLY with JSON: {"score": 0.0, "reasoning": "explanation"}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nSystem answer: {actual_answer}"}
            ],
            temperature=0
        )
    else:
        system_prompt = """You are evaluating a RAG system's answer against a known correct answer.

Score the answer from 0.0 to 1.0:
1.0 = answer contains the correct information
0.5 = answer is partially correct or vague
0.0 = answer is wrong or hallucinated

Also check if the answer is faithful to sources (not fabricated).

Respond ONLY with JSON: {"score": 0.0, "reasoning": "explanation", "faithful": true}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nExpected: {expected}\n\nActual: {actual_answer}"}
            ],
            temperature=0
        )

    result = json.loads(response.choices[0].message.content)

    # Check source coverage
    if expected == "NOT_IN_DOCUMENTS":
        source_hit = True  # Not applicable
    else:
        source_filenames = [s["filename"] for s in sources_used]
        expected_source = next((t["expected_source"] for t in TEST_SET if t["question"] == question), None)
        source_hit = any(expected_source in f for f in source_filenames) if expected_source else True

    return {
        "score": result.get("score", 0.0),
        "reasoning": result.get("reasoning", ""),
        "faithful": result.get("faithful", True),
        "source_hit": source_hit
    }


# ── Run evaluation ─────────────────────────────────────────────────────────────
def run_evaluation():
    print("Running evaluation suite...\n")
    print("=" * 60)

    results = []

    for test in TEST_SET:
        question = test["question"]
        expected = test["expected_answer"]

        print(f"Q: {question}")

        # Run pipeline
        result = answer_query(question)
        actual_answer = result["answer"]
        sources_used = result["sources"]

        # Evaluate
        eval_result = evaluate_answer(question, expected, actual_answer, sources_used)

        score = eval_result["score"]
        reasoning = eval_result["reasoning"]
        source_hit = eval_result["source_hit"]

        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
        source_status = "✅" if source_hit else "❌"

        print(f"  Score: {status} {score:.2f} | Source hit: {source_status}")
        print(f"  Reasoning: {reasoning}")
        print()

        results.append({
            "question": question,
            "score": score,
            "source_hit": source_hit,
            "faithful": eval_result.get("faithful", True)
        })

    # Summary
    avg_score = sum(r["score"] for r in results) / len(results)
    source_hit_rate = sum(1 for r in results if r["source_hit"]) / len(results)
    faithfulness_rate = sum(1 for r in results if r["faithful"]) / len(results)

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print(f"  Average answer score:  {avg_score:.2f} / 1.00")
    print(f"  Source hit rate:       {source_hit_rate:.0%}")
    print(f"  Faithfulness rate:     {faithfulness_rate:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
