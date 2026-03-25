"""Prompt injection scanner for uploaded documents."""

INJECTION_PATTERNS: dict[str, list[str]] = {
    "instruction_override": [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard",
        "forget your instructions",
        "new instructions",
        "you are now",
        "act as",
        "pretend you are",
        "your new role",
    ],
    "system_prompt_probing": [
        "reveal your prompt",
        "show your system prompt",
        "what are your instructions",
        "print your instructions",
        "output your prompt",
    ],
    "role_hijacking": [
        "you are an ai that",
        "from now on you",
        "override",
        "jailbreak",
        "developer mode",
        "dan mode",
    ],
}


def scan_for_injection(text: str) -> dict:
    """Scan extracted document text for prompt injection patterns.

    Returns:
        {
            "safe": bool,
            "reason": str,          # empty if safe
            "matched_patterns": list[str],
        }
    """
    lower = text.lower()
    matched: list[str] = []

    for category, phrases in INJECTION_PATTERNS.items():
        for phrase in phrases:
            if phrase in lower:
                matched.append(phrase)

    if matched:
        reason = f"Detected {len(matched)} suspicious pattern(s): {', '.join(repr(p) for p in matched)}"
        return {"safe": False, "reason": reason, "matched_patterns": matched}

    return {"safe": True, "reason": "", "matched_patterns": []}
