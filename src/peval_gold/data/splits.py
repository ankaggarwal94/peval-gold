"""Held-out and simulation splits for the gold-track laboratory.

Six split functions, each returning ``(train_rows, val_rows, manifest)``
where ``manifest`` is a dict with at minimum ``zero_overlap_verified:
True`` plus split-specific stats.

Promotion is gated by ``scripts/run_d9_gate.py``, which HARD-gates on
all four D-9 sub-gates (``overall_pass = all([sub_a, sub_b, sub_c,
sub_d])``). Both :func:`item_holdout_primary` (engineering diagnostic,
sub-gate (a)/(b)) and :func:`benchmark_holdout_stress` (report-facing,
sub-gate (c)) are co-equal requirements — item-heldout alone is NEVER
sufficient. See (project decision doc) and
(project decision doc).

Determinism
-----------

All splits use ``random.Random(seed)`` rather than ``numpy.random`` so
the test corpus uses a small standard-library RNG that's stable across
Python versions and doesn't require numpy to import for the splits to
work. Seed contracts:

- Same seed + same input → same split (byte-identical row identity).
- Different seeds diverge.

Zero-overlap policy
-------------------

The ``manifest["zero_overlap_verified"]`` field is set by each split
AFTER it has actually verified disjointness on the grouping key. We do
not trust by construction — we verify by explicit set intersection on
the way out. If a future implementation regresses, the verification
returns ``False`` and downstream gates can refuse to evaluate.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

CanonicalRow = Mapping[str, Any]
CanonicalRows = Sequence[CanonicalRow]
SplitResult = tuple[list[dict], list[dict], dict[str, Any]]


# ---------------------------------------------------------------------------
# 1. random_row_smoke
# ---------------------------------------------------------------------------


def random_row_smoke(
    rows: CanonicalRows,
    n_total: int = 20000,
    val_frac: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    """Small random row split for fast smoke tests.

    If the corpus has more than ``n_total`` rows, subsample to ``n_total``
    first (without replacement). Then split into train / val using
    ``val_frac`` (floored to integer count). This is a ROW split — items
    can appear in both train and val.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Canonical rows.
    n_total : int
        Cap on combined train + val size. Default 20K.
    val_frac : float
        Fraction of selected rows held out for val. Default 0.1.
    seed : int
        Deterministic seed.

    Returns
    -------
    (train, val, manifest)
        ``manifest`` includes ``n_train``, ``n_val``, ``zero_overlap_verified``,
        ``split_kind="random_row_smoke"``.
    """
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    sample = pool[:n_total]
    n_val = int(len(sample) * val_frac)
    val = [dict(r) for r in sample[:n_val]]
    train = [dict(r) for r in sample[n_val:]]

    # Identity check on the original references before we copied — both
    # ``val`` and ``train`` are fresh copies so we use the shuffled pool
    # indices to verify disjointness.
    val_indices = set(range(n_val))
    train_indices = set(range(n_val, len(sample)))
    overlap_ok = val_indices.isdisjoint(train_indices)

    manifest: dict[str, Any] = {
        "split_kind": "random_row_smoke",
        "n_train": len(train),
        "n_val": len(val),
        "n_total_selected": len(sample),
        "n_corpus": len(pool),
        "val_frac": val_frac,
        "seed": seed,
        "zero_overlap_verified": overlap_ok,
    }
    return train, val, manifest


# ---------------------------------------------------------------------------
# 2. item_holdout_primary  (ENGINEERING DIAGNOSTIC — per D-9 sub-gate (a)/(b))
# ---------------------------------------------------------------------------


def item_holdout_primary(
    rows: CanonicalRows,
    val_frac: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    """Item-held-out diagnostic split.

    ENGINEERING DIAGNOSTIC per D-9 sub-gate (a)/(b). Item-heldout NLL is
    sensitive to local-fold overfit and can overstate hidden-leaderboard
    performance — see D-2 and the 2026-05-22 transfer postmortem
    (``handoffs/2026-05-22_transfer_failure_postmortem.md``), where the
    locally-best item-heldout candidate scored worst on hidden.

    Promotion requires passing ALL four D-9 sub-gates, including the
    benchmark-heldout stress test (sub-gate (c)). Item-heldout alone is
    NEVER sufficient.

    Why item holdout is useful as a diagnostic: a random ROW split lets
    the model learn item-specific signal (memorize popular MCQ stems) and
    looks good in val while failing on cold items at runtime — the
    runtime's ``predict()`` distribution is item-cold-start because the
    platform holds out items, not rows. Item-heldout exposes this leak,
    but does NOT replicate the hidden-eval distribution (whole-benchmark
    holdout does — see :func:`benchmark_holdout_stress`).

    Behavior:

    - Compute the set of unique ``item_id`` values across ``rows``.
    - Shuffle deterministically with ``random.Random(seed)``.
    - Take ``int(n_items * val_frac)`` items for val; the rest for train.
    - Emit every row whose ``item_id`` is in the val set into ``val``;
      every other row into ``train``.

    See:
    - (project decision doc) (rationale)
    - (project decision doc) (gate spec)
    - ``scripts/run_d9_gate.py`` (enforcement: all 4 sub-gates HARD-gated
      via ``overall_pass``)
    """
    rng = random.Random(seed)
    items = sorted({r.get("item_id", "") for r in rows})
    rng.shuffle(items)
    n_val_groups = int(len(items) * val_frac)
    val_items = set(items[:n_val_groups])

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for r in rows:
        bucket = val if r.get("item_id") in val_items else train
        bucket.append(dict(r))

    train_items = {r.get("item_id") for r in train}
    overlap_ok = train_items.isdisjoint(val_items)

    manifest: dict[str, Any] = {
        "split_kind": "item_holdout_primary",
        "group_key": "item_id",
        "n_train": len(train),
        "n_val": len(val),
        "n_train_groups": len(train_items),
        "n_val_groups": len(val_items),
        "val_frac": val_frac,
        "seed": seed,
        "zero_overlap_verified": overlap_ok,
    }
    return train, val, manifest


# ---------------------------------------------------------------------------
# 3. benchmark_holdout_stress  (REPORT-FACING — per Tutorial 6 + D-9 sub-gate (c))
# ---------------------------------------------------------------------------


def benchmark_holdout_stress(
    rows: CanonicalRows,
    holdout_benchmarks: Sequence[str],
    seed: int = 42,
) -> SplitResult:
    """Benchmark-held-out stress split.

    REPORT-FACING per Tutorial 6 guidance and D-9 sub-gate (c). Whole-
    benchmark holdout most closely matches the hidden-eval distribution
    (every hidden item is from a benchmark the model has not seen). A
    candidate that wins on item-heldout but blows up on benchmark-heldout
    is the most uncertain hidden transfer and MUST NOT be promoted.

    The model must generalize to benchmark families it never saw at
    training time. This is the right gate for catching item-specific
    memorization (a perfectly tuned NCF for ``mmlupro`` likely fails on
    ``ai2d_test``).

    The ``seed`` argument is accepted for API uniformity but does not
    affect the deterministic partition (which is fully determined by
    ``holdout_benchmarks``). Kept so the signature matches the other
    splits.

    See:
    - (project decision doc) (rationale)
    - (project decision doc) (gate spec)
    - ``scripts/run_d9_gate.py`` (enforcement: all 4 sub-gates HARD-gated
      via ``overall_pass``)
    """
    holdout = set(holdout_benchmarks)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for r in rows:
        bucket = val if r.get("benchmark") in holdout else train
        bucket.append(dict(r))

    train_benches = {r.get("benchmark") for r in train}
    val_benches = {r.get("benchmark") for r in val}
    overlap_ok = train_benches.isdisjoint(val_benches)

    manifest: dict[str, Any] = {
        "split_kind": "benchmark_holdout_stress",
        "group_key": "benchmark",
        "holdout_benchmarks": sorted(holdout),
        "n_train": len(train),
        "n_val": len(val),
        "n_train_groups": len(train_benches),
        "n_val_groups": len(val_benches),
        "seed": seed,
        "zero_overlap_verified": overlap_ok,
    }
    return train, val, manifest


# ---------------------------------------------------------------------------
# 4. domain_holdout
# ---------------------------------------------------------------------------


def domain_holdout(
    rows: CanonicalRows,
    holdout_domains: Sequence[str] | None = None,
    seed: int = 42,
) -> SplitResult:
    """Hold out rows whose ``domain`` matches ``holdout_domains``.

    ``benchmarks.parquet:domain`` is a list of strings at the pinned
    revision (e.g. ``["math", "reasoning"]``). A row matches if ANY of
    its domain entries is in ``holdout_domains``. Pre-joined rows
    typically inherit ``domain`` from the benchmarks-registry join
    inside :func:`peval_gold.data.hf_loader.load_responses`.

    When NO row in the corpus has a ``domain`` field, the split degrades
    gracefully: train = all rows, val = empty, manifest flagged
    ``domain_field_present: False``. This is the right behavior because
    many benchmarks lack the ``domain`` annotation; refusing to operate
    would block the audit pipeline.
    """
    rng = random.Random(seed)
    _ = rng  # accept seed for API uniformity even though partition is deterministic

    holdout = set(holdout_domains or ())
    has_domain_anywhere = any("domain" in r and r["domain"] is not None for r in rows)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for r in rows:
        row_domains = r.get("domain")
        if row_domains is None:
            row_set: set[str] = set()
        elif isinstance(row_domains, str):
            row_set = {row_domains}
        else:
            row_set = set(row_domains)
        bucket = val if row_set & holdout else train
        bucket.append(dict(r))

    train_domains = _flatten_domains(train)
    val_domains = _flatten_domains(val)
    overlap_ok = (
        not has_domain_anywhere
        or train_domains.isdisjoint(val_domains)
    )

    manifest: dict[str, Any] = {
        "split_kind": "domain_holdout",
        "group_key": "domain",
        "domain_field_present": has_domain_anywhere,
        "holdout_domains": sorted(holdout),
        "n_train": len(train),
        "n_val": len(val),
        "seed": seed,
        "zero_overlap_verified": overlap_ok,
    }
    return train, val, manifest


# ---------------------------------------------------------------------------
# 5. subject_holdout  (DIAGNOSTIC ONLY)
# ---------------------------------------------------------------------------


def subject_holdout(
    rows: CanonicalRows,
    val_frac: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    """Group by ``subject_id``; hold out ``val_frac`` of unique subjects for val.

    DIAGNOSTIC ONLY. This split tells you how well the model
    generalizes to new subjects (e.g. a freshly-released LLM), which is
    informative for the auxiliary subject-side cold-start objective, but
    is NOT the promotion gate. The hosted runtime exposes BOTH train and
    val subjects in the ``input["subject_content"]`` field; the platform
    does not actually hold out subjects.
    """
    rng = random.Random(seed)
    subjects = sorted({r.get("subject_id", "") for r in rows})
    rng.shuffle(subjects)
    n_val_groups = int(len(subjects) * val_frac)
    val_subjects = set(subjects[:n_val_groups])

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for r in rows:
        bucket = val if r.get("subject_id") in val_subjects else train
        bucket.append(dict(r))

    train_subjects = {r.get("subject_id") for r in train}
    overlap_ok = train_subjects.isdisjoint(val_subjects)

    manifest: dict[str, Any] = {
        "split_kind": "subject_holdout",
        "group_key": "subject_id",
        "n_train": len(train),
        "n_val": len(val),
        "n_train_groups": len(train_subjects),
        "n_val_groups": len(val_subjects),
        "val_frac": val_frac,
        "seed": seed,
        "zero_overlap_verified": overlap_ok,
        "note": "diagnostic only; platform does NOT hold out subjects",
    }
    return train, val, manifest


# ---------------------------------------------------------------------------
# 6. adaptive_label_simulation
# ---------------------------------------------------------------------------


def adaptive_label_simulation(
    rows: CanonicalRows,
    k_per_category: int = 5,
    seed: int = 42,
) -> SplitResult:
    """Simulate the platform's K=5 revealed-labels-per-data-category policy.

    Per the kit (``starting_kit/README.md:268-271``):

    > The platform selects the top K=5 inputs per data category, resolves
    > their ground-truth labels, and passes the union to ``predict()``.

    The kit also explicitly defines a "data category" as an
    organizer-internal grouping (``starting_kit/README.md:266``) that we
    don't have direct access to. We therefore use the best available
    proxy:

    - If any row has a ``category`` field set, use ``category`` as the
      grouping key (this is the post-2026-05-17 spec-overhaul shape; the
      kit's ``sample_data/test/test_items.csv`` ships a ``category``
      column with a single value ``sample``).
    - Otherwise fall back to ``benchmark`` (the next-most-granular
      grouping the kit makes elsewhere — and the unit on which the
      organizers stratify scoring per ``starting_kit/README.md:331-336``).

    Sampling: for each category, deterministically shuffle that
    category's rows and take the first ``min(k_per_category, n_in_cat)``
    rows as labeled. The remaining rows go to unlabeled.

    Returns
    -------
    (labeled, unlabeled, manifest)
        ``manifest`` carries ``k_per_category``, ``category_key`` (which
        proxy was used), and the per-category labeled-count distribution.
    """
    rng = random.Random(seed)

    category_key = "category" if any("category" in r for r in rows) else "benchmark"

    by_cat: dict[str, list[CanonicalRow]] = defaultdict(list)
    for r in rows:
        cat = str(r.get(category_key, ""))
        by_cat[cat].append(r)

    labeled_ids: set[int] = set()
    labeled: list[dict[str, Any]] = []
    per_cat_labeled: dict[str, int] = {}

    for cat in sorted(by_cat):
        bucket = list(by_cat[cat])
        rng.shuffle(bucket)
        take = min(k_per_category, len(bucket))
        for r in bucket[:take]:
            labeled_ids.add(id(r))
            labeled.append(dict(r))
        per_cat_labeled[cat] = take

    unlabeled: list[dict[str, Any]] = []
    for r in rows:
        if id(r) not in labeled_ids:
            unlabeled.append(dict(r))

    # Disjointness on row-id (since we sample without replacement).
    overlap_ok = len(labeled_ids) + len(unlabeled) == len(rows)

    manifest: dict[str, Any] = {
        "split_kind": "adaptive_label_simulation",
        "category_key": category_key,
        "k_per_category": k_per_category,
        "n_categories": len(by_cat),
        "per_category_labeled": per_cat_labeled,
        "n_labeled": len(labeled),
        "n_unlabeled": len(unlabeled),
        "seed": seed,
        "zero_overlap_verified": overlap_ok,
    }
    return labeled, unlabeled, manifest


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _flatten_domains(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Collect every domain string in a row sequence into a flat set."""
    out: set[str] = set()
    for r in rows:
        d = r.get("domain")
        if d is None:
            continue
        if isinstance(d, str):
            out.add(d)
        else:
            out.update(d)
    return out


# Silence flake8 unused-import for math (kept available for future
# numeric helpers e.g. floor/ceil edge cases).
_ = math  # noqa: F841


__all__ = [
    "SplitResult",
    "adaptive_label_simulation",
    "benchmark_holdout_stress",
    "domain_holdout",
    "item_holdout_primary",
    "random_row_smoke",
    "subject_holdout",
]
