"""Single place for OpenAI chat calls and model choice.

FAST_MODEL handles the small structured steps (classify, rerank, rewrite,
relevance check). ANSWER_MODEL writes the final answer. Both can be
overridden with environment variables without touching code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FAST_MODEL = os.getenv("FAST_MODEL", "gpt-5.4-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-5.5")
ANSWER_REASONING = os.getenv("ANSWER_REASONING", "low")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3, timeout=90)


def _reasoning_kwargs(model: str, effort: str) -> dict:
    # GPT-5.x models take reasoning_effort; temperature is only honoured at "none".
    if model.startswith("gpt-5"):
        kw = {"reasoning_effort": effort}
        if effort == "none":
            kw["temperature"] = 0
        return kw
    return {"temperature": 0}


def chat(messages: list[dict], *, answer: bool = False, json_mode: bool = False) -> str:
    """Run a chat completion and return the text.
    answer=True uses ANSWER_MODEL (final answer); otherwise FAST_MODEL."""
    model = ANSWER_MODEL if answer else FAST_MODEL
    effort = ANSWER_REASONING if answer else "none"
    kwargs = _reasoning_kwargs(model, effort)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return (response.choices[0].message.content or "").strip()
