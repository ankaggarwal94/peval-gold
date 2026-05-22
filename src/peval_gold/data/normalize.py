"""Canonical row normalization for the gold-track laboratory.

The HF dataset ``aims-foundations/measurement-db`` ships per-benchmark
response parquets with this 8-column schema (see ``notebooks/load_data.py``
``RESPONSE_FEATURES``):

    subject_id, item_id, benchmark_id, trial, test_condition,
    response, correct_answer, trace

The hosted ``predict()`` runtime sees this dict shape per the post-2026-05-17
spec overhaul (``starting_kit/README.md:139-144``, ``263-265``,
``starting_kit/sample_code_the wrapped predict() module``):

    {"benchmark": ..., "condition": ..., "subject_content": ...,
     "item_content": ..., "category"?: ...}

Neither shape is the "right" one — they coexist. The team's
``notebooks/load_data.to_training_example`` produces the joined,
``predict()``-style dict; the raw HF parquet produces the 8-column shape.
``normalize_row`` accepts EITHER and emits the gold-track canonical
schema.

Canonical schema
----------------

Required keys (always present):

- ``benchmark``: str. Runtime identifier (e.g. ``mmlupro``). Resolves to
  ``raw["benchmark"]`` or ``raw["benchmark_id"]`` or empty string.
- ``condition``: str. Literal ``"none"`` when source value is ``None``
  or empty (kit ``starting_kit/README.md:268-271`` convention).
- ``subject_content``: str. Starts with ``Name:`` per the kit's
  ``render_subject_content`` reference; defaults to empty string.
- ``item_content``: str. Free-text item content; defaults to empty.
- ``response``: float64. Raw response value (NOT pre-binarized — that's
  ``peval_gold.data.filters``'s job per D-7).
- ``subject_id``: str. Registry primary key. Fallback chain when missing:
  parse from ``subject_content``'s ``Name:`` line via
  :func:`parse_subject_name`.
- ``item_id``: str. Registry primary key. Fallback when missing: a
  deterministic 16-char hex digest of ``item_content``.

Optional keys (passed through when present in the input):

- ``category``: str. Organizer-internal grouping (post-overhaul kit
  convention, ``starting_kit/sample_data/test/test_items.csv``).
- ``domain``: str | list[str]. Registry-backed; ``benchmarks.parquet``
  stores this as a list of strings.
- ``modality``: str | list[str]. Same shape as ``domain``.
- ``family``: str. Registry-backed model family
  (``subjects.parquet:family``; currently null at the pinned revision
  per workspace Bug 14, but the schema slot exists).

Fields deliberately NOT surfaced
--------------------------------

The raw HF row carries ``correct_answer`` and ``trace`` columns. These
are runtime-absent (the hosted ``predict()`` does NOT receive them). Per
the project's "don't trust kit docstrings as the contract" rule
(``CLAUDE.md`` "Kit Contract Caveats") the normalizer drops them so no
downstream gold-track code can accidentally peek at oracle data during
training and silently break at runtime.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

CANONICAL_REQUIRED_KEYS: tuple[str, ...] = (
    "benchmark",
    "condition",
    "subject_content",
    "item_content",
    "response",
    "subject_id",
    "item_id",
)

CANONICAL_OPTIONAL_KEYS: tuple[str, ...] = (
    "category",
    "domain",
    "modality",
    "family",
)

# Whitelist for what _passes through_. Any other key in the raw row is
# dropped from the canonical output — this includes oracle-only fields
# (``correct_answer``, ``trace``) AND any future HF column we haven't
# explicitly opted into. Keep it deliberate.
_PASSTHROUGH_KEYS: frozenset[str] = frozenset(CANONICAL_OPTIONAL_KEYS)


def normalize_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce a raw HF or joined row into the canonical gold-track schema.

    Accepts both shapes — see module docstring. Defaults are conservative:
    missing text fields become ``""``, missing identifiers fall back to
    parsed-from-content values (so synthetic test rows can omit
    ``subject_id``/``item_id`` and still produce useful canonical rows).

    Parameters
    ----------
    raw : Mapping[str, Any]
        A single row dict in either raw HF response shape or joined
        ``predict()``-input shape. Must contain a ``response`` value
        coercible via :func:`coerce_response_to_float`.

    Returns
    -------
    dict[str, Any]
        Canonical row containing every key in
        :data:`CANONICAL_REQUIRED_KEYS` plus any present optional keys
        from :data:`CANONICAL_OPTIONAL_KEYS`. The oracle-only
        ``correct_answer`` and ``trace`` columns are intentionally
        dropped.

    Raises
    ------
    ValueError
        If ``raw["response"]`` (or, equivalently, ``raw["label"]`` when
        only the joined shape is provided) cannot be coerced to a float.
    """
    benchmark = _first_str(raw, ("benchmark", "benchmark_id"), default="")
    raw_condition = _first(raw, ("condition", "test_condition"), default=None)
    condition = _normalize_condition(raw_condition)

    subject_content = _first_str(raw, ("subject_content",), default="")
    item_content = _first_str(raw, ("item_content", "content"), default="")

    response_val = _first(raw, ("response", "label"), default=None)
    response = coerce_response_to_float(response_val)

    subject_id_raw = _first(raw, ("subject_id",), default=None)
    if subject_id_raw is None or str(subject_id_raw).strip() == "":
        subject_id = parse_subject_name(subject_content)
    else:
        subject_id = str(subject_id_raw)

    item_id_raw = _first(raw, ("item_id",), default=None)
    if item_id_raw is None or str(item_id_raw).strip() == "":
        item_id = _content_hash(item_content)
    else:
        item_id = str(item_id_raw)

    out: dict[str, Any] = {
        "benchmark": str(benchmark),
        "condition": condition,
        "subject_content": subject_content,
        "item_content": item_content,
        "response": response,
        "subject_id": subject_id,
        "item_id": item_id,
    }

    for key in _PASSTHROUGH_KEYS:
        if key in raw and raw[key] is not None:
            out[key] = raw[key]

    return out


def parse_subject_name(subject_content: str) -> str:
    """Extract the display name from a kit-formatted ``subject_content`` string.

    The kit's ``render_subject_content`` reference produces a multi-line
    string whose first non-empty line is ``"Name: <display_name>"`` and
    whose subsequent lines are optional ``Organization:``,
    ``Parameters:``, ``Released:``, ``Family:`` metadata (see
    ``starting_kit/README.md:102-115``).

    Behavior:

    - Skips leading blank lines.
    - When the first non-empty line starts with ``"Name:"`` (case
      sensitive per kit), return the text after the colon, stripped.
    - When there is no ``Name:`` prefix, return the first 80 chars of
      the input as a degenerate fallback (so :func:`normalize_row` can
      still produce a non-empty ``subject_id`` for synthetic test rows).
    - On empty / whitespace-only input, return ``""``.
    """
    if not subject_content:
        return ""
    lines = subject_content.splitlines()
    first_nonempty: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped:
            first_nonempty = stripped
            break
    if first_nonempty is None:
        return ""
    if first_nonempty.startswith("Name:"):
        return first_nonempty[len("Name:") :].strip()
    return first_nonempty[:80]


def coerce_response_to_float(value: Any) -> float:
    """Coerce a response value to a native Python float.

    Accepts:

    - native ``int`` / ``float`` / ``bool`` (per Python's ``bool`` is an
      ``int`` rule, ``True`` / ``False`` map to ``1.0`` / ``0.0``);
    - numpy scalars (e.g. ``np.float64(0.7)``);
    - numeric strings (``"0.7"`` → ``0.7``) for HF parquet Arrow-edge
      cases.

    Rejects with :class:`ValueError`:

    - ``None``;
    - non-numeric strings (``"abc"``);
    - NaN is preserved (passed through) so callers can decide how to
      handle it — the binarization filter drops NaN-response rows.

    Returns
    -------
    float
        Always a native Python float. The input shape does not leak
        through to the output type.
    """
    if value is None:
        raise ValueError("response value is None; expected numeric scalar")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"response value {value!r} is not a numeric string") from exc
    raise ValueError(f"response value {value!r} of type {type(value).__name__} is not numeric")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _first(raw: Mapping[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    """Return the first present, non-None key value or the default."""
    for k in keys:
        if k in raw and raw[k] is not None:
            return raw[k]
    return default


def _first_str(raw: Mapping[str, Any], keys: tuple[str, ...], default: str) -> str:
    val = _first(raw, keys, default=default)
    if val is None:
        return default
    return str(val)


def _normalize_condition(value: Any) -> str:
    """Map ``None``/empty to the literal ``"none"`` per kit convention."""
    if value is None:
        return "none"
    text = str(value).strip()
    if not text:
        return "none"
    return text


def _content_hash(text: str) -> str:
    """Deterministic 16-char hex digest of an item-content string."""
    if not text:
        text = ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:16]


# Re-export math.isnan for callers that want to scrub NaN explicitly
# without an extra import; the binarization filters use it.
__all__ = [
    "CANONICAL_OPTIONAL_KEYS",
    "CANONICAL_REQUIRED_KEYS",
    "coerce_response_to_float",
    "normalize_row",
    "parse_subject_name",
]


# Silence flake8 unused-import linting if math.isnan is referenced
# elsewhere in this module without being directly used here.
_ = math  # noqa: F841
