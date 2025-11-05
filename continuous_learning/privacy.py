import re
from typing import Dict, List


EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d \-]{7,}\d")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def sanitize_text(text: str) -> Dict[str, List[str]]:
    """Redact common PII patterns from text, returning cleaned text and flags.

    Args:
        text: input string

    Returns:
        dict: {"text": cleaned_text, "flags": [labels]}
    """
    flags: List[str] = []
    cleaned = text

    def _redact(regex: re.Pattern, label: str):
        nonlocal cleaned
        if regex.search(cleaned):
            flags.append(label)
            cleaned = regex.sub(f"[REDACTED_{label}]", cleaned)

    _redact(EMAIL_RE, "EMAIL")
    _redact(PHONE_RE, "PHONE")
    _redact(SSN_RE, "SSN")
    _redact(CREDIT_CARD_RE, "CARD")

    # Collapse excessive whitespace after redactions
    cleaned = " ".join(cleaned.split())
    return {"text": cleaned, "flags": flags}


def validate_input(text: str) -> bool:
    """Basic validation to ensure well-formed user input."""
    if not isinstance(text, str):
        return False
    if len(text.strip()) == 0:
        return False
    # Restrict extremely long inputs to prevent abuse
    if len(text) > 8000:
        return False
    # Simple control: disallow non-printable characters
    if any(ord(ch) < 9 for ch in text):
        return False
    return True