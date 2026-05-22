"""Item-text formatters for the gold-track template-correct NCF (H1).

The current shipped the hosted runtime artifact (``the weights file``) was
trained by encoding the raw ``item_content`` string and concatenating
the resulting 768-d embedding with the subject embedding. The hosted
``predict()`` runtime passes ``benchmark`` and ``condition`` to every
call, but the encoder never sees them — Batch-0 CoVe Claim 1 in
``docs/walkthroughs/012_gold_track_current_state_cove.md`` is the
direct evidence.

This module exposes three pure, deterministic formatters that decide
WHAT text the encoder embeds for the item side:

- :func:`item_only` — current behavior; the encoder embeds just the
  ``item_content`` string. Kept so the gold-track laboratory can
  reproduce the legacy baseline number byte-for-byte.
- :func:`canonical_item` — H1 deliverable. Prepends ``Benchmark:`` and
  ``Condition:`` lines so the encoder sees the two signals the current
  shipped artifact demonstrably ignores. Format is the spec-fixed
  ``"Benchmark: ...\\nCondition: ...\\nItem:\\n..."``.
- :func:`rich_item` — optional variant that also surfaces ``domain``
  and ``modality`` when they are present in the row dict. Used for a
  potential follow-up if H1 promotes.

Discipline
----------

All three are:

- pure (zero side effects, no global state, no I/O);
- deterministic (same input → identical output, byte-for-byte);
- defensive against missing optional keys (``rich_item`` returns a
  shorter but still valid string when ``domain`` / ``modality`` are
  absent or empty).

The module is stdlib-only: no torch, no numpy, no sentence-transformers.
If H1 promotes to a the hosted runtime submission, this same module can be
re-used inside ``the wrapped submission's `` without violating the no-network /
no-heavy-import discipline (``tests/test_runtime_no_network_imports.py``
already enforces that primitive scan).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping


def item_only(row: Mapping[str, Any]) -> str:
    """Return the row's raw ``item_content`` string.

    This is the legacy formatter — same text the current shipped
    ``the wrapped predict() module:_ncf_logit`` encodes. Kept so the gold-track
    laboratory can reproduce -class numbers without subtle
    formatting drift.

    Parameters
    ----------
    row : Mapping[str, Any]
        Runtime-shaped dict; only ``item_content`` is consulted.

    Returns
    -------
    str
        ``row["item_content"]`` coerced to ``str``. An empty string when
        the key is absent (defensive; production callers should never
        pass content-less rows).
    """
    value = row.get("item_content", "")
    return value if isinstance(value, str) else str(value)


def canonical_item(row: Mapping[str, Any]) -> str:
    """Return ``"Benchmark: ...\\nCondition: ...\\nItem:\\n<item_content>"``.

    H1 deliverable. The hosted ``predict()`` runtime passes ``benchmark``
    (e.g. ``"mmlupro"``, ``"ai2d_test"``) and ``condition`` (e.g.
    ``"cot"``, ``"direct"``, ``"none"``) on every call but the current
    encoder ignores both. This formatter prepends them so the embedding
    captures the signal.

    Format (spec-fixed, do not reorder):

    .. code-block:: text

        Benchmark: <benchmark>
        Condition: <condition>
        Item:
        <item_content>

    Parameters
    ----------
    row : Mapping[str, Any]
        Runtime-shaped dict. The three keys ``benchmark``, ``condition``,
        and ``item_content`` are read. Missing keys coerce to the empty
        string — the output still has the right shape so the encoder
        produces a deterministic embedding for content-less rows.

    Returns
    -------
    str
        Multi-line templated string.
    """
    benchmark = _as_str(row.get("benchmark", ""))
    condition = _as_str(row.get("condition", ""))
    item_content = _as_str(row.get("item_content", ""))
    return (
        f"Benchmark: {benchmark}\n"
        f"Condition: {condition}\n"
        f"Item:\n"
        f"{item_content}"
    )


def rich_item(row: Mapping[str, Any]) -> str:
    """Optional richer variant that includes ``domain`` / ``modality`` if present.

    Use only when the row carries the ``domain`` and/or ``modality``
    optional keys (lifted from ``benchmarks.parquet`` by
    :mod:`peval_gold.data.hf_loader`). Empty-string and ``None`` values
    are treated as missing (same convention as
    :func:`peval_gold.data.normalize`'s ``_stringify_subject_value``).

    Format (lines emitted only when the source key has a non-empty
    value):

    .. code-block:: text

        Benchmark: <benchmark>
        Condition: <condition>
        Domain: <domain>      # only when row["domain"] is non-empty
        Modality: <modality>  # only when row["modality"] is non-empty
        Item:
        <item_content>

    Registry-typed ``domain`` / ``modality`` values are lists of strings
    (``benchmarks.parquet`` schema per :mod:`peval_gold.data.registry`);
    they are rendered as comma-joined strings so the encoder sees a
    flat, deterministic format regardless of source list ordering.

    Parameters
    ----------
    row : Mapping[str, Any]
        Runtime-shaped dict with optional ``domain`` / ``modality``
        slots. Behavior degrades gracefully when both are missing — the
        output reduces to :func:`canonical_item`'s format.

    Returns
    -------
    str
        Multi-line templated string.
    """
    benchmark = _as_str(row.get("benchmark", ""))
    condition = _as_str(row.get("condition", ""))
    item_content = _as_str(row.get("item_content", ""))

    parts: list[str] = [
        f"Benchmark: {benchmark}",
        f"Condition: {condition}",
    ]

    domain = _flatten_optional(row.get("domain"))
    if domain:
        parts.append(f"Domain: {domain}")

    modality = _flatten_optional(row.get("modality"))
    if modality:
        parts.append(f"Modality: {modality}")

    parts.append("Item:")
    parts.append(item_content)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public registry of named templates (used by TemplateNCF runtime predictor
# and notebooks/train_ncf_template.py to resolve --item-template flag values)
# ---------------------------------------------------------------------------


TEMPLATES: dict[str, Any] = {
    "item_only": item_only,
    "canonical_item": canonical_item,
    "rich_item": rich_item,
}


def get_template(name: str):
    """Return the template function registered under ``name``.

    Raises
    ------
    ValueError
        When ``name`` is not in :data:`TEMPLATES`. The error message
        lists the accepted names so callers can correct the typo.
    """
    if name not in TEMPLATES:
        raise ValueError(
            f"unknown item template {name!r}; "
            f"accepted names: {sorted(TEMPLATES)}"
        )
    return TEMPLATES[name]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _as_str(value: Any) -> str:
    """Coerce a value to ``str``; ``None`` becomes empty string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _flatten_optional(value: Any) -> str:
    """Render a domain/modality value as a comma-joined deterministic string.

    Accepts:

    - ``None`` or empty string → ``""`` (caller should skip the line);
    - ``str`` → returned as-is (after strip);
    - iterable of strings → comma-joined, stripped, empty entries dropped.

    Lists are NOT sorted — the registry's ordering is the gold-track
    convention (``benchmarks.parquet:domain`` is operator-curated and we
    want the encoder to see consistent ordering across calls for the
    same benchmark).
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Iterable):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return ", ".join(parts)
    return str(value).strip()


__all__ = [
    "TEMPLATES",
    "canonical_item",
    "get_template",
    "item_only",
    "rich_item",
]
