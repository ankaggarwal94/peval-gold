"""Registry-table helpers for the gold-track laboratory.

The HF dataset ``aims-foundations/measurement-db`` ships three registry
parquets that downstream code joins against the per-benchmark response
parquets:

- ``subjects.parquet`` keyed by ``subject_id`` (currently 909 rows at
  revision ``589ccfdb…``). Columns: ``subject_id``, ``display_name``,
  ``provider``, ``hub_repo``, ``revision``, ``params``,
  ``release_date``, ``raw_labels_seen``, ``notes``. Note that
  ``provider`` / ``params`` / ``release_date`` are all-null at the
  pinned revision (workspace Bug 14); the schema slots remain so future
  organizer backfills flow through automatically.
- ``items.parquet`` keyed by ``item_id`` (~104K rows). Columns:
  ``item_id``, ``benchmark_id``, ``raw_item_id``, ``content``,
  ``correct_answer``, ``content_hash``.
- ``benchmarks.parquet`` keyed by ``benchmark_id`` (16 rows). Columns:
  ``benchmark_id``, ``name``, ``version``, ``license``, ``source_url``,
  ``description``, ``modality`` (list of strings), ``domain`` (list of
  strings), ``response_type``, ``response_scale``, ``categorical``,
  ``paper_url``, ``release_date``.

The team's ``notebooks/load_data.py:build_lookups`` produces the same
shape but uses HF Dataset row-iteration which is slow at registry scale.
This module re-exports the same convention but is the gold-track-owned
copy so that future evolution doesn't perturb the production trainer.
"""

from __future__ import annotations

from typing import Any

REPO_ID = "aims-foundations/measurement-db"
DEFAULT_REVISION = "589ccfdb8e82e6e0b5e35e9d23cd83a6df85018f"


def load_subjects(
    revision: str = DEFAULT_REVISION,
    repo_id: str = REPO_ID,
) -> dict[str, dict[str, Any]]:
    """Load ``subjects.parquet`` and return a dict keyed by ``subject_id``.

    Lazy-imports ``datasets`` so importing the rest of ``peval_gold.data``
    on a machine without HF installed does not break unrelated tests.
    """
    from datasets import load_dataset

    ds = load_dataset(repo_id, data_files="subjects.parquet", revision=revision, split="train")
    return {row["subject_id"]: row for row in ds}


def load_items(
    revision: str = DEFAULT_REVISION,
    repo_id: str = REPO_ID,
) -> dict[str, dict[str, Any]]:
    """Load ``items.parquet`` and return a dict keyed by ``item_id``."""
    from datasets import load_dataset

    ds = load_dataset(repo_id, data_files="items.parquet", revision=revision, split="train")
    return {row["item_id"]: row for row in ds}


def load_benchmarks(
    revision: str = DEFAULT_REVISION,
    repo_id: str = REPO_ID,
) -> dict[str, dict[str, Any]]:
    """Load ``benchmarks.parquet`` and return a dict keyed by ``benchmark_id``."""
    from datasets import load_dataset

    ds = load_dataset(
        repo_id,
        data_files="benchmarks.parquet",
        revision=revision,
        split="train",
    )
    return {row["benchmark_id"]: row for row in ds}


__all__ = [
    "DEFAULT_REVISION",
    "REPO_ID",
    "load_benchmarks",
    "load_items",
    "load_subjects",
]
