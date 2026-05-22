"""TDD red→green tests for ``peval_gold.data.filters``.

Batch 2 deliverable (S1B). Covers the three binarization modes that the
gold-track laboratory exposes:

- ``binarize_drop`` — D-7 default. Drop any row whose ``response`` is not
  exactly ``0.0`` or ``1.0``. This is what ``notebooks/train_ncf.py``
  uses (per ``submission/ncf_head.meta.json:25``) and what the H100
  multi-seed Wave-3c winner trained against. Drops 6 of 16 benchmarks.
- ``binarize_median`` — explicit ablation only. Per-benchmark median
  threshold (``response > median → 1``). NOT the default.
- ``binarize_soft`` — explicit ablation only. Per-benchmark min-max
  normalization to ``[0, 1]`` for BCE-with-logits soft targets. NOT
  the default.

Also covers ``binary_filter_summary`` which is used by the audit script
to emit the per-benchmark counts table.

All tests use small synthetic dicts — no HF access. The HF round-trip is
the audit script's job and lives in ``scripts/gold_audit_measurement_db.py``.
"""

from __future__ import annotations

import math

import pytest

from peval_gold.data.filters import (
    binarize_drop,
    binarize_median,
    binarize_soft,
    binary_filter_summary,
)

# ---------------------------------------------------------------------------
# Tiny synthetic corpora
# ---------------------------------------------------------------------------


def _row(benchmark: str, response: float, **extra) -> dict:
    """Convenience builder for synthetic rows."""
    return {
        "benchmark": benchmark,
        "condition": "none",
        "subject_id": "s",
        "item_id": f"i-{response}-{extra.get('tag', '')}",
        "subject_content": "Name: M",
        "item_content": "Q",
        "response": response,
        **extra,
    }


# ---------------------------------------------------------------------------
# 1. binarize_drop
# ---------------------------------------------------------------------------


def test_binarize_drop_keeps_binary_rows_drops_continuous_and_nan() -> None:
    rows = [
        _row("mmlupro", 0.0),
        _row("mmlupro", 1.0),
        _row("mtbench", 0.5),
        _row("mtbench", 3.0),
        _row("mtbench", float("nan")),
        _row("mmlupro", 0.0),
    ]
    out = binarize_drop(rows)

    assert len(out) == 3
    assert all(r["response"] in (0.0, 1.0) for r in out)
    # No NaN survives — pyflakes / mypy can't catch this so assert explicitly.
    assert not any(math.isnan(r["response"]) for r in out)


def test_binarize_drop_preserves_other_keys_unchanged() -> None:
    rows = [_row("mmlupro", 1.0, foo="bar")]
    out = binarize_drop(rows)
    assert out[0]["foo"] == "bar"
    assert out[0]["benchmark"] == "mmlupro"


def test_binarize_drop_on_all_continuous_benchmark_yields_empty() -> None:
    """Mirrors the meta.json ``mode: drop, n_output: 0`` outcome for cybench /
    livecodebench / matharena / mtbench / rewardbench / ultrafeedback.
    """
    rows = [_row("mtbench", float(i) / 10.0) for i in range(1, 10)]
    out = binarize_drop(rows)
    assert out == []


def test_binarize_drop_empty_input_returns_empty() -> None:
    assert binarize_drop([]) == []


# ---------------------------------------------------------------------------
# 2. binarize_median
# ---------------------------------------------------------------------------


def test_binarize_median_thresholds_per_benchmark_above_strict() -> None:
    """``> median → 1``, ``<= median → 0`` (strict > per S1B spec)."""
    rows = [
        _row("benchA", 1.0),
        _row("benchA", 2.0),
        _row("benchA", 3.0),
        _row("benchA", 4.0),
        _row("benchA", 5.0),
        _row("benchB", 10.0),
        _row("benchB", 20.0),
    ]
    out = binarize_median(rows)
    a_rows = sorted([(r["item_id"], r["response"]) for r in out if r["benchmark"] == "benchA"])
    b_rows = sorted([(r["item_id"], r["response"]) for r in out if r["benchmark"] == "benchB"])

    # benchA median = 3.0 → only 4.0 and 5.0 are > median → 2 ones, 3 zeros.
    assert sum(r[1] for r in a_rows) == 2.0
    assert len(a_rows) == 5
    # benchB median of two values: numpy/statistics median of [10, 20] = 15.
    # Only 20.0 > 15 → 1 one, 1 zero.
    assert sum(r[1] for r in b_rows) == 1.0
    assert len(b_rows) == 2


def test_binarize_median_does_not_mutate_input() -> None:
    rows = [_row("benchA", 1.0), _row("benchA", 2.0), _row("benchA", 3.0)]
    out = binarize_median(rows)
    # Inputs unchanged.
    assert [r["response"] for r in rows] == [1.0, 2.0, 3.0]
    # Output rows are independent dicts with binarized response.
    assert all(r["response"] in (0.0, 1.0) for r in out)
    assert out[0] is not rows[0]


def test_binarize_median_skips_already_binary_benchmark_passthrough() -> None:
    """A benchmark with only binary responses should pass through unchanged."""
    rows = [_row("mmlupro", 0.0), _row("mmlupro", 1.0), _row("mmlupro", 1.0)]
    out = binarize_median(rows)
    responses = sorted(r["response"] for r in out)
    assert responses == [0.0, 1.0, 1.0]


def test_binarize_median_empty_input_returns_empty() -> None:
    assert binarize_median([]) == []


# ---------------------------------------------------------------------------
# 3. binarize_soft
# ---------------------------------------------------------------------------


def test_binarize_soft_min_max_normalizes_per_benchmark_to_unit_interval() -> None:
    rows = [
        _row("benchA", 1.0),
        _row("benchA", 3.0),
        _row("benchA", 5.0),
        _row("benchB", 10.0),
        _row("benchB", 100.0),
    ]
    out = binarize_soft(rows)
    a_resps = sorted(r["response"] for r in out if r["benchmark"] == "benchA")
    b_resps = sorted(r["response"] for r in out if r["benchmark"] == "benchB")

    # benchA: min=1, max=5, scale=4 → 0.0, 0.5, 1.0
    assert a_resps == pytest.approx([0.0, 0.5, 1.0])
    # benchB: min=10, max=100, scale=90 → 0.0, 1.0
    assert b_resps == pytest.approx([0.0, 1.0])
    assert all(0.0 <= r["response"] <= 1.0 for r in out)


def test_binarize_soft_degenerate_constant_benchmark_yields_zero() -> None:
    """When ``min == max``, the min-max normalization would divide by zero;
    the implementation must not crash. Per ``notebooks/train_ncf.py:278``
    the convention is ``scale = (hi - lo) if hi > lo else 1.0``, which
    yields ``(value - lo) / 1.0`` — i.e. all zeros for a constant column.
    """
    rows = [_row("benchA", 7.0), _row("benchA", 7.0)]
    out = binarize_soft(rows)
    assert all(r["response"] == 0.0 for r in out)


def test_binarize_soft_passes_through_already_binary_unchanged() -> None:
    rows = [_row("mmlupro", 0.0), _row("mmlupro", 1.0)]
    out = binarize_soft(rows)
    # min=0, max=1, scale=1 → identity.
    assert sorted(r["response"] for r in out) == [0.0, 1.0]


# ---------------------------------------------------------------------------
# 4. binary_filter_summary
# ---------------------------------------------------------------------------


def test_binary_filter_summary_per_benchmark_counts() -> None:
    rows = [
        _row("mmlupro", 0.0),
        _row("mmlupro", 1.0),
        _row("mmlupro", 1.0),
        _row("mtbench", 0.5),
        _row("mtbench", 3.0),
        _row("mtbench", 1.0),
    ]
    summary = binary_filter_summary(rows)

    # Keys are present.
    assert set(summary.keys()) >= {"mmlupro", "mtbench"}

    # mmlupro: all 3 are binary, 0 dropped.
    assert summary["mmlupro"]["n_input"] == 3
    assert summary["mmlupro"]["n_binary_kept"] == 3
    assert summary["mmlupro"]["n_dropped"] == 0
    assert summary["mmlupro"]["binary_fraction"] == pytest.approx(1.0)

    # mtbench: only 1 of 3 is binary, 2 dropped.
    assert summary["mtbench"]["n_input"] == 3
    assert summary["mtbench"]["n_binary_kept"] == 1
    assert summary["mtbench"]["n_dropped"] == 2
    assert summary["mtbench"]["binary_fraction"] == pytest.approx(1 / 3)


def test_binary_filter_summary_empty_input_returns_empty_dict() -> None:
    assert binary_filter_summary([]) == {}


def test_binary_filter_summary_records_unique_response_values() -> None:
    """Audit-friendly: the summary should list the unique response values
    for each benchmark so a human can sanity-check at a glance whether
    a benchmark is binary, continuous, or coded as integers.
    """
    rows = [
        _row("mmlupro", 0.0),
        _row("mmlupro", 1.0),
        _row("mtbench", 1.0),
        _row("mtbench", 2.5),
        _row("mtbench", 7.3),
    ]
    summary = binary_filter_summary(rows)
    assert sorted(summary["mmlupro"]["unique_responses"]) == [0.0, 1.0]
    assert sorted(summary["mtbench"]["unique_responses"]) == [1.0, 2.5, 7.3]
