"""Deterministic weak parser for subject display names.

Parsed fields are weak stacker features only; they are never labels or
prediction substitutes.
"""

from __future__ import annotations

import re
from typing import Any


def parse_subject_display_name(name: str) -> dict[str, Any]:
    raw = str(name or "").strip()
    text = raw.lower()
    provider = "unknown"
    family = "unknown"
    size_bucket = "unknown"
    confidence = 0.0
    method = "fallback_unknown"
    for candidate, patterns in [
        ("openai", [r"\bgpt[-_ ]", r"\bo[134]\b", r"openai/"]),
        ("anthropic", [r"\bclaude\b", r"anthropic/"]),
        ("google", [r"\bgemini\b", r"google/"]),
        ("meta", [r"\bllama\b", r"meta-llama/"]),
        ("mistral", [r"\bmistral\b", r"\bmixtral\b", r"mistralai/"]),
        ("qwen", [r"\bqwen\b", r"qwen/"]),
        ("deepseek", [r"\bdeepseek\b", r"deepseek-ai/"]),
        ("cohere", [r"\bcommand[-_ ]?r\b", r"\bcohere\b", r"cohere/"]),
        ("microsoft", [r"\bphi[-_ ]", r"microsoft/"]),
        ("xai", [r"\bgrok\b", r"\bxai\b"]),
        ("databricks", [r"\bdbrx\b", r"databricks/"]),
        ("aws", [r"\btitan\b", r"\bamazon\b", r"\baws\b"]),
    ]:
        if any(re.search(pattern, text) for pattern in patterns):
            provider = candidate
            confidence = 0.75
            method = "regex_partial"
            break
    for pattern, value in [
        (r"gpt[-_ ]?4o", "gpt-4o"),
        (r"gpt[-_ ]?4", "gpt-4"),
        (r"claude[-_ ]?3\.?5[-_ ]?sonnet", "claude-3.5-sonnet"),
        (r"claude[-_ ]?3[-_ ]?opus", "claude-3-opus"),
        (r"claude[-_ ]?3[-_ ]?haiku", "claude-3-haiku"),
        (r"claude", "claude"),
        (r"gemini[-_ ]?2", "gemini-2"),
        (r"gemini[-_ ]?1\.?5", "gemini-1.5"),
        (r"llama[-_ ]?3\.?3", "llama-3.3"),
        (r"llama[-_ ]?3\.?1", "llama-3.1"),
        (r"llama[-_ ]?3", "llama-3"),
        (r"mistral[-_ ]?large", "mistral-large"),
        (r"mixtral", "mixtral"),
        (r"qwen[-_ ]?2\.?5", "qwen-2.5"),
        (r"deepseek[-_ ]?v3", "deepseek-v3"),
        (r"deepseek[-_ ]?r1", "deepseek-r1"),
        (r"command[-_ ]?r[-_ ]?plus", "command-r-plus"),
        (r"phi[-_ ]?4", "phi-4"),
        (r"grok[-_ ]?2", "grok-2"),
    ]:
        if re.search(pattern, text):
            family = value
            confidence = max(confidence, 0.90)
            method = "regex_high_confidence"
            break
    size_billions = None
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*b\b", text)
    if m:
        size_billions = float(m.group(1))
        size_bucket = (
            "tiny"
            if size_billions < 3
            else "small"
            if size_billions < 10
            else "medium"
            if size_billions < 35
            else "large"
            if size_billions < 100
            else "xlarge"
        )
        confidence = max(confidence, 0.85)
    elif any(tok in text for tok in ["mini", "haiku", "small", "nano"]):
        size_bucket = "small"
        confidence = max(confidence, 0.60)
    elif any(tok in text for tok in ["large", "opus", "pro", "ultra"]):
        size_bucket = "large"
        confidence = max(confidence, 0.60)
    if provider == "unknown" and family == "unknown":
        confidence = 0.0
        method = "fallback_unknown"
    return {
        "provider_guess": provider,
        "family_guess": family,
        "size_bucket_guess": size_bucket,
        "size_billions_guess": size_billions,
        "parser_confidence": float(confidence),
        "parse_method": method,
    }
