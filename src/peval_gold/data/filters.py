"""Binarization filters for the gold-track laboratory.

The HF measurement-db serves 16 per-benchmark response parquets at the
pinned revision ``589ccfdb…``. Six of them store CONTINUOUS scores
(``mtbench`` 1-10, ``ultrafeedback`` 1-5, etc.) and ten store strict
binary correctness in ``{0.0, 1.0}``. Any classifier head trained with
BCE-with-logits needs a per-benchmark binarization decision.

Three modes are exposed:

- :func:`binarize_drop` — **D-7 default**. Drop any row whose response
  is not exactly ``0.0`` or ``1.0``. This is what
  ``the training pipeline:148-159`` defaults to (the ``--binarize``
  default flipped to ``"drop"`` after the the wrapped H100 multi-seed sweep;
  see ``the wrapped submission's ncf_head.meta.json:25``). Drops 6 of 16 benchmarks
  entirely.
- :func:`binarize_median` — explicit ablation only. Per-benchmark
  median threshold (``response > median → 1``, ``response <= median →
  0``). **NOT the default.** Note: the spec for the gold-track
  laboratory uses STRICT ``>`` here, whereas
  ``the training pipeline:265`` uses ``>=``. The two yield identical
  results when no responses equal the exact median; on ties, ``>=``
  marks ties as 1 and strict ``>`` marks ties as 0. The strict form
  matches what a textbook EBM / blueprint TeX ``§Binarization rules``
  derivation produces, and is the gold-track convention going forward.
- :func:`binarize_soft` — explicit ablation only. Per-benchmark min-max
  normalization to ``[0, 1]`` for BCE-style soft training. **NOT the
  default.**

Plus an audit helper:

- :func:`binary_filter_summary` — per-benchmark pre/post-drop counts +
  binary fraction + unique response values, consumed by
  ``scripts/gold_audit_measurement_db.py``.

All filters are pure functions of their inputs — no global state, no
RNG, deterministic. They consume canonical rows produced by
:func:`peval_gold.data.normalize.normalize_row` (so the ``benchmark``
key, not ``benchmark_id``, is the grouping key).
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

CanonicalRow = Mapping[str, Any]
CanonicalRows = Sequence[CanonicalRow]


# ---------------------------------------------------------------------------
# Public filters
# ---------------------------------------------------------------------------


def binarize_drop(rows: CanonicalRows) -> list[dict[str, Any]]:
    """Keep rows where ``response`` is exactly ``0.0`` or ``1.0``.

    NaN is dropped (NaN does not equal 0.0 or 1.0). The returned list is
    fresh dicts copied from inputs — no mutation of caller-owned rows.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        resp = r.get("response")
        if not isinstance(resp, (int, float)):
            continue
        if isinstance(resp, float) and math.isnan(resp):
            continue
        if float(resp) not in (0.0, 1.0):
            continue
        out.append(dict(r))
    return out


def binarize_median(rows: CanonicalRows) -> list[dict[str, Any]]:
    """Per-benchmark median threshold; ``response > median → 1`` else ``0``.

    Two passes:

    1. Group by ``benchmark``; compute the median of the non-NaN
       ``response`` values within each group.
    2. Emit copies of each input row with ``response`` set to
       ``1.0 if r["response"] > median else 0.0``.

    Already-binary benchmarks (all responses in ``{0.0, 1.0}``) pass
    through with ``response`` unchanged — the strict ``>`` would map
    half of them to ``0`` otherwise, which is a known
    binarize-an-already-binary-column footgun.

    NaN-response rows are dropped (they cannot be thresholded).
    """
    return _per_benchmark_apply(
        rows,
        apply_fn=_threshold_strict,
        passthrough_if_binary=True,
    )


def binarize_soft(rows: CanonicalRows) -> list[dict[str, Any]]:
    """Per-benchmark min-max normalization to ``[0, 1]`` (soft targets).

    For each benchmark, compute ``lo = min(responses)`` and ``hi =
    max(responses)``. Emit copies with ``response = (response - lo) /
    (hi - lo)``. When ``hi == lo`` the divisor degenerates to 1.0
    (matching ``the training pipeline:278``'s convention), so a
    constant column maps to all zeros without crashing.

    NaN-response rows are dropped.
    """
    return _per_benchmark_apply(
        rows,
        apply_fn=_min_max_normalize,
        passthrough_if_binary=False,
    )


def binary_filter_summary(rows: CanonicalRows) -> dict[str, dict[str, Any]]:
    """Per-benchmark audit summary used by ``gold_audit_measurement_db.py``.

    Returns a mapping ``benchmark → {n_input, n_binary_kept, n_dropped,
    binary_fraction, unique_responses}`` where:

    - ``n_input``: total rows seen for this benchmark (NaN included).
    - ``n_binary_kept``: rows with ``response ∈ {0.0, 1.0}``.
    - ``n_dropped``: ``n_input - n_binary_kept``.
    - ``binary_fraction``: ``n_binary_kept / n_input``; ``0.0`` for an
      empty benchmark (which cannot happen given the precondition that
      the benchmark key exists, but is defensive).
    - ``unique_responses``: sorted list of distinct non-NaN response
      values for this benchmark, capped at 64 entries to keep the JSON
      audit manageable for high-cardinality continuous benchmarks
      (matharena has hundreds of distinct fractional scores).
    """
    if not rows:
        return {}

    by_bench: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n_input": 0,
            "n_binary_kept": 0,
            "_unique": set(),
        }
    )

    for r in rows:
        bench = str(r.get("benchmark", ""))
        rec = by_bench[bench]
        rec["n_input"] += 1
        resp = r.get("response")
        if isinstance(resp, (int, float)) and not (isinstance(resp, float) and math.isnan(resp)):
            resp_f = float(resp)
            rec["_unique"].add(resp_f)
            if resp_f in (0.0, 1.0):
                rec["n_binary_kept"] += 1

    out: dict[str, dict[str, Any]] = {}
    for bench, rec in by_bench.items():
        n_input = int(rec["n_input"])
        n_kept = int(rec["n_binary_kept"])
        unique = sorted(rec["_unique"])
        out[bench] = {
            "n_input": n_input,
            "n_binary_kept": n_kept,
            "n_dropped": n_input - n_kept,
            "binary_fraction": (n_kept / n_input) if n_input > 0 else 0.0,
            "unique_responses": unique[:64],
            "unique_response_count": len(unique),
        }
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _per_benchmark_apply(
    rows: CanonicalRows,
    apply_fn,
    passthrough_if_binary: bool,
) -> list[dict[str, Any]]:
    """Group rows by benchmark, optionally pass through binary-only groups."""
    if not rows:
        return []
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        bench = str(r.get("benchmark", ""))
        by_bench[bench].append(dict(r))

    out: list[dict[str, Any]] = []
    for bench in sorted(by_bench):
        group = by_bench[bench]
        valid = [
            r
            for r in group
            if isinstance(r.get("response"), (int, float))
            and not (isinstance(r["response"], float) and math.isnan(r["response"]))
        ]
        if not valid:
            continue
        responses = [float(r["response"]) for r in valid]
        if passthrough_if_binary and all(v in (0.0, 1.0) for v in responses):
            out.extend(valid)
            continue
        binarized = apply_fn(valid, responses)
        out.extend(binarized)
    return out


def _threshold_strict(rows: list[dict[str, Any]], responses: list[float]) -> list[dict[str, Any]]:
    """Strict ``> median → 1`` threshold; ties → 0."""
    median = statistics.median(responses)
    binarized: list[dict[str, Any]] = []
    for r in rows:
        new = dict(r)
        new["response"] = 1.0 if float(r["response"]) > median else 0.0
        binarized.append(new)
    return binarized


def _min_max_normalize(rows: list[dict[str, Any]], responses: list[float]) -> list[dict[str, Any]]:
    """Per-benchmark min-max normalize; constant column → zeros."""
    lo = min(responses)
    hi = max(responses)
    scale = (hi - lo) if hi > lo else 1.0
    normalized: list[dict[str, Any]] = []
    for r in rows:
        new = dict(r)
        if hi == lo:
            new["response"] = 0.0
        else:
            new["response"] = (float(r["response"]) - lo) / scale
        normalized.append(new)
    return normalized


__all__ = [
    "binarize_drop",
    "binarize_median",
    "binarize_soft",
    "binary_filter_summary",
]
