"""HuggingFace response-table loader for the gold-track laboratory.

Thin wrapper over the ``aims-foundations/measurement-db`` per-benchmark
response parquets that returns canonical rows produced by
:func:`peval_gold.data.normalize.normalize_row`.

Why a separate module from ``notebooks/load_data.py``:

- ``notebooks/load_data.py`` ships ``to_training_example`` which returns
  a four-field ``predict()``-style dict (``benchmark``, ``condition``,
  ``subject_content``, ``item_content``, ``label``) and is locked in by
  ``tests/test_load_data.py``. We don't want to perturb it.
- The gold-track substrate wants a 7-required-key canonical schema (see
  ``peval_gold.data.normalize``) that includes ``subject_id`` and
  ``item_id`` for held-out splitting, and that drops oracle-only
  ``correct_answer`` / ``trace`` columns.
- This loader produces the gold-track canonical schema directly by
  joining responses against the three registry tables, then handing
  each joined row through :func:`normalize_row` for the final coercion.

Cache behavior
--------------

Both ``load_responses`` and ``load_all_responses`` use the default
HuggingFace cache (``~/.cache/huggingface/datasets/``). If the pinned
revision ``589ccfdb…`` is already present in the cache, no network
fetch happens (`local_files_only` is NOT set explicitly so this still
works in Posture A; HF's loader falls back to the cache when offline).

Errors
------

When the HF cache is cold AND no auth is configured (Posture A with no
prior fetch), ``datasets.load_dataset`` will raise a clear
``FileNotFoundError`` or HTTP error. Callers (notably
``scripts/gold_audit_measurement_db.py``) translate that into a clean
exit-2 with a usable message.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from peval_gold.data.normalize import normalize_row
from peval_gold.data.registry import (
    DEFAULT_REVISION,
    REPO_ID,
    load_benchmarks,
    load_items,
    load_subjects,
)

# The 16 per-benchmark response parquets at the pinned revision per
# ``the wrapped submission's ncf_head.meta.json:7-23``.
ALL_BENCHMARKS: tuple[str, ...] = (
    "afrimedqa",
    "agentdojo",
    "ai2d_test",
    "androidworld",
    "bfcl",
    "cybench",
    "hle",
    "livecodebench",
    "matharena",
    "mathvista_mini",
    "mmbench_v11",
    "mmlupro",
    "mtbench",
    "rewardbench",
    "swebench",
    "ultrafeedback",
)

# Metadata fields we lift from ``benchmarks.parquet`` into each
# canonical row's optional ``domain`` and ``modality`` keys.
_BENCHMARK_OPTIONAL_FIELDS: tuple[str, ...] = ("domain", "modality", "category")


def load_responses(
    benchmark: str,
    revision: str = DEFAULT_REVISION,
    repo_id: str = REPO_ID,
    subjects_by_id: dict[str, dict[str, Any]] | None = None,
    items_by_id: dict[str, dict[str, Any]] | None = None,
    benchmarks_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load one benchmark's response parquet, joined to registries, normalized.

    Parameters
    ----------
    benchmark : str
        Benchmark identifier (e.g. ``"mmlupro"``). The corresponding
        parquet is ``{benchmark}.parquet`` at the dataset root.
    revision : str
        Pinned HF git SHA (default: ``589ccfdb…``).
    repo_id : str
        Dataset repo id (default: ``aims-foundations/measurement-db``).
    subjects_by_id, items_by_id, benchmarks_by_id : dict | None
        Pre-loaded registry lookups. When ``None`` each is fetched
        lazily on first call. Passing pre-loaded lookups is much faster
        when calling this function across many benchmarks (see
        :func:`load_all_responses`).

    Returns
    -------
    list[dict[str, Any]]
        Canonical rows per :mod:`peval_gold.data.normalize`.
    """
    from datasets import load_dataset

    parquet = f"{benchmark}.parquet"
    ds = load_dataset(repo_id, data_files=parquet, revision=revision, split="train")

    subjects_by_id = subjects_by_id or load_subjects(revision=revision, repo_id=repo_id)
    items_by_id = items_by_id or load_items(revision=revision, repo_id=repo_id)
    benchmarks_by_id = benchmarks_by_id or load_benchmarks(revision=revision, repo_id=repo_id)

    out: list[dict[str, Any]] = []
    for raw in ds:
        joined = _join_one(raw, subjects_by_id, items_by_id, benchmarks_by_id)
        out.append(normalize_row(joined))
    return out


def load_all_responses(
    revision: str = DEFAULT_REVISION,
    repo_id: str = REPO_ID,
    benchmarks: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Load and concatenate responses across many benchmarks.

    Loads each registry once and reuses the lookups across the inner
    :func:`load_responses` calls, which is ~3x faster than calling
    :func:`load_responses` per benchmark naively.

    Parameters
    ----------
    benchmarks : Iterable[str] | None
        Subset of benchmark identifiers to load. When ``None`` defaults
        to :data:`ALL_BENCHMARKS` (all 16 at the pinned revision).
    """
    chosen = list(benchmarks) if benchmarks is not None else list(ALL_BENCHMARKS)

    subjects_by_id = load_subjects(revision=revision, repo_id=repo_id)
    items_by_id = load_items(revision=revision, repo_id=repo_id)
    benchmarks_by_id = load_benchmarks(revision=revision, repo_id=repo_id)

    out: list[dict[str, Any]] = []
    for bench in chosen:
        out.extend(
            load_responses(
                bench,
                revision=revision,
                repo_id=repo_id,
                subjects_by_id=subjects_by_id,
                items_by_id=items_by_id,
                benchmarks_by_id=benchmarks_by_id,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _join_one(
    raw: dict[str, Any],
    subjects_by_id: dict[str, dict[str, Any]],
    items_by_id: dict[str, dict[str, Any]],
    benchmarks_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Join a raw response row to its registry rows.

    Produces a dict that :func:`normalize_row` understands directly.
    Drops the oracle-only ``correct_answer`` and ``trace`` columns at
    this layer so :func:`normalize_row` does not have to re-defend.
    """
    subject = subjects_by_id.get(raw["subject_id"], {})
    item = items_by_id.get(raw["item_id"], {})
    benchmark = benchmarks_by_id.get(raw["benchmark_id"], {})

    joined: dict[str, Any] = {
        "benchmark": benchmark.get("benchmark_id") or raw["benchmark_id"],
        "condition": raw.get("test_condition"),
        "subject_content": _format_subject_content(subject, fallback_id=raw["subject_id"]),
        "item_content": item.get("content") or "",
        "response": raw.get("response"),
        "subject_id": raw["subject_id"],
        "item_id": raw["item_id"],
    }

    for field in _BENCHMARK_OPTIONAL_FIELDS:
        value = benchmark.get(field)
        if value is not None:
            joined[field] = value

    family = subject.get("family")
    if family is not None:
        joined["family"] = family

    return joined


def _format_subject_content(subject: dict[str, Any], fallback_id: str) -> str:
    """Mirror ``notebooks/load_data.format_subject_content`` byte-for-byte.

    We keep the implementation here (rather than importing from
    ``notebooks/load_data``) so the gold-track substrate does not depend
    on ``sys.path`` munging into ``notebooks/``. Equivalence with the
    notebooks version is locked in by ``tests/test_load_data.py``'s
    ``test_format_subject_content_matches_kit_render_reference`` — any
    drift between the two is caught by the existing test rather than
    silently diverging.
    """
    display_name = _stringify(subject.get("display_name")) or fallback_id
    lines = [f"Name: {display_name}"]
    for key, label in (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    ):
        value = _stringify(subject.get(key))
        if value is not None:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return " ".join(text.split())


__all__ = [
    "ALL_BENCHMARKS",
    "load_all_responses",
    "load_responses",
]
