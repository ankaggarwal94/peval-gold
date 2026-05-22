"""TDD red→green tests for ``peval_gold.data.normalize``.

Batch 2 deliverable (S1B) for the gold-track workflow. These tests are
written BEFORE any implementation; the red phase should fail with
``ModuleNotFoundError`` and the green phase should pass without touching
HuggingFace at all.

Schema reminder
---------------
The canonical row that flows through the gold-track laboratory mirrors
what ``predict()`` sees at runtime, plus the registry-keyed identifiers
needed for held-out splitting:

- ``benchmark``: runtime identifier (e.g. ``mmlupro``, ``ai2d_test``).
- ``condition``: literal ``"none"`` when the HF parquet stores ``null``,
  matching the kit's ``starting_kit/README.md:268-271`` convention.
- ``subject_content``: starts with ``Name: <display_name>`` per
  ``starting_kit/README.md:102-115``.
- ``item_content``: free text item content.
- ``response``: float64 raw response (NOT pre-binarized; D-7 binarization
  is a separate ``peval_gold.data.filters`` concern).
- ``subject_id``: registry primary key; falls back to the parsed
  ``Name:`` line when missing (sufficient for synthetic tests).
- ``item_id``: registry primary key; falls back to a deterministic
  content-hash when missing.
- Optional metadata: ``category`` (post-2026-05-17 spec overhaul),
  ``domain``, ``modality``, ``family`` (registry-backed, often absent).

The HF dataset uses the parallel ``benchmark_id`` / ``test_condition``
key names in its raw response parquets (see ``notebooks/load_data.py``
``RESPONSE_FEATURES`` block); the normalizer accepts EITHER shape so
that callers can hand it raw HF rows OR pre-joined rows from
``notebooks/load_data.to_training_example``.
"""

from __future__ import annotations

import numpy as np
import pytest

from peval_gold.data.normalize import (
    coerce_response_to_float,
    normalize_row,
    parse_subject_name,
)


# ---------------------------------------------------------------------------
# 1. normalize_row — happy path on a joined row dict
# ---------------------------------------------------------------------------


def test_normalize_row_accepts_joined_predict_shape_row() -> None:
    """A ``to_training_example``-shaped row passes through with all canonical keys."""
    raw = {
        "benchmark": "mmlupro",
        "condition": "none",
        "subject_content": "Name: GPT-4o\nOrganization: OpenAI",
        "item_content": "Solve 2x + 3 = 7.",
        "response": 1.0,
        "subject_id": "subj-abc",
        "item_id": "item-xyz",
    }
    out = normalize_row(raw)

    assert out["benchmark"] == "mmlupro"
    assert out["condition"] == "none"
    assert out["subject_content"] == "Name: GPT-4o\nOrganization: OpenAI"
    assert out["item_content"] == "Solve 2x + 3 = 7."
    assert out["response"] == pytest.approx(1.0)
    assert isinstance(out["response"], float)
    assert out["subject_id"] == "subj-abc"
    assert out["item_id"] == "item-xyz"


def test_normalize_row_accepts_raw_hf_response_shape_row() -> None:
    """A raw HF response row (``benchmark_id`` + ``test_condition``) normalizes too."""
    raw = {
        "subject_id": "04eb984f",
        "item_id": "9e0fe5e5",
        "benchmark_id": "mmlupro",
        "trial": 1,
        "test_condition": None,
        "response": 0.0,
        "correct_answer": "C",
        "trace": None,
    }
    out = normalize_row(raw)

    assert out["benchmark"] == "mmlupro"
    assert out["condition"] == "none"
    assert out["subject_id"] == "04eb984f"
    assert out["item_id"] == "9e0fe5e5"
    assert out["response"] == pytest.approx(0.0)


def test_normalize_row_maps_null_test_condition_to_literal_none() -> None:
    """``None`` and the empty string both map to ``"none"`` (kit convention)."""
    a = normalize_row({"benchmark": "mmlupro", "condition": None, "response": 0.0})
    b = normalize_row({"benchmark": "mmlupro", "condition": "", "response": 0.0})
    c = normalize_row({"benchmark_id": "mmlupro", "test_condition": None, "response": 1.0})
    assert a["condition"] == "none"
    assert b["condition"] == "none"
    assert c["condition"] == "none"


def test_normalize_row_preserves_non_none_condition() -> None:
    out = normalize_row(
        {"benchmark": "mmlupro", "condition": "zero-shot", "response": 1.0}
    )
    assert out["condition"] == "zero-shot"


def test_normalize_row_optional_fields_default_cleanly() -> None:
    """Optional fields (``category``, ``domain``, ``modality``, ``family``) are absent by default."""
    out = normalize_row({"benchmark": "mmlupro", "response": 1.0})
    # The four optional metadata keys are NOT in the output unless provided.
    for k in ("category", "domain", "modality", "family"):
        assert k not in out or out[k] is None


def test_normalize_row_passes_through_optional_metadata_when_present() -> None:
    raw = {
        "benchmark": "mmlupro",
        "response": 1.0,
        "category": "sample",
        "domain": ["math"],
        "modality": ["text"],
        "family": "GPT",
    }
    out = normalize_row(raw)
    assert out["category"] == "sample"
    assert out["domain"] == ["math"]
    assert out["modality"] == ["text"]
    assert out["family"] == "GPT"


def test_normalize_row_subject_id_fallback_parses_name() -> None:
    """When ``subject_id`` is absent, fall back to parsing ``subject_content``'s ``Name:`` line."""
    raw = {
        "benchmark": "mmlupro",
        "subject_content": "Name: Claude-Opus-3.5\nOrganization: Anthropic",
        "response": 1.0,
    }
    out = normalize_row(raw)
    assert out["subject_id"] == "Claude-Opus-3.5"


def test_normalize_row_item_id_fallback_to_content_hash() -> None:
    """When ``item_id`` is absent, fall back to a deterministic content-hash."""
    raw = {
        "benchmark": "mmlupro",
        "item_content": "Solve 2x + 3 = 7.",
        "response": 1.0,
    }
    out = normalize_row(raw)
    assert isinstance(out["item_id"], str)
    assert out["item_id"]  # non-empty
    # Determinism: same content → same hash.
    out2 = normalize_row(dict(raw))
    assert out["item_id"] == out2["item_id"]


def test_normalize_row_defensive_defaults_for_missing_text() -> None:
    """Missing ``subject_content`` / ``item_content`` default to empty strings."""
    out = normalize_row({"benchmark": "mmlupro", "response": 1.0})
    assert out["subject_content"] == ""
    assert out["item_content"] == ""


def test_normalize_row_does_not_trust_correct_answer_field() -> None:
    """``correct_answer`` is a runtime-absent field; the normalizer must NOT
    surface it in the canonical row even when present in the raw HF row.
    """
    raw = {
        "benchmark_id": "mmlupro",
        "response": 1.0,
        "correct_answer": "C",
        "trace": "some trace text",
    }
    out = normalize_row(raw)
    assert "correct_answer" not in out
    assert "trace" not in out


# ---------------------------------------------------------------------------
# 2. parse_subject_name
# ---------------------------------------------------------------------------


def test_parse_subject_name_extracts_name_line() -> None:
    out = parse_subject_name("Name: GPT-4o\nOrganization: OpenAI")
    assert out == "GPT-4o"


def test_parse_subject_name_skips_leading_blank_lines() -> None:
    out = parse_subject_name("\n\nName: Llama-3-8B\nOrganization: Meta")
    assert out == "Llama-3-8B"


def test_parse_subject_name_fallback_to_first_80_chars_when_no_prefix() -> None:
    bare = "Some short text"
    assert parse_subject_name(bare) == bare

    long_text = "x" * 200
    assert parse_subject_name(long_text) == "x" * 80


def test_parse_subject_name_empty_input_returns_empty() -> None:
    assert parse_subject_name("") == ""
    assert parse_subject_name("\n\n\n") == ""


def test_parse_subject_name_strips_surrounding_whitespace() -> None:
    out = parse_subject_name("Name:   Mistral-7B   ")
    assert out == "Mistral-7B"


# ---------------------------------------------------------------------------
# 3. coerce_response_to_float
# ---------------------------------------------------------------------------


def test_coerce_response_accepts_native_types() -> None:
    assert coerce_response_to_float(0) == 0.0
    assert coerce_response_to_float(1) == 1.0
    assert coerce_response_to_float(0.5) == 0.5
    assert coerce_response_to_float(True) == 1.0
    assert coerce_response_to_float(False) == 0.0
    assert isinstance(coerce_response_to_float(0), float)
    assert isinstance(coerce_response_to_float(True), float)


def test_coerce_response_accepts_numpy_scalar() -> None:
    out = coerce_response_to_float(np.float64(0.7))
    assert isinstance(out, float)
    assert out == pytest.approx(0.7)


def test_coerce_response_rejects_none() -> None:
    with pytest.raises(ValueError):
        coerce_response_to_float(None)


def test_coerce_response_rejects_non_numeric_string() -> None:
    with pytest.raises(ValueError):
        coerce_response_to_float("abc")


def test_coerce_response_accepts_numeric_string() -> None:
    """``"0.7"`` is unambiguously numeric — accept it for HF parquet
    interop edge cases where Arrow surfaces a stringified float.
    """
    assert coerce_response_to_float("0.7") == pytest.approx(0.7)
