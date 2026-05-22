"""TDD red→green tests for the Batch 5 EBPriors model (S2B).

The EB priors model is a hierarchical base-rate lookup that smooths
``P(correct)`` across:

- global
- per-subject (visible via ``parse_subject_name(subject_content)``)
- per-benchmark
- per-condition
- per-(subject, benchmark)
- per-(subject, benchmark, condition) — only if counts justify (>= 5)

Smoothing follows the beta-binomial empirical-Bayes update::

    p_eb(k, n; parent_p, kappa) = (k + kappa * parent_p) / (n + kappa)

where ``kappa`` is estimated per level via method-of-moments on the
sibling variance and clipped to ``[1, 1000]``.

These tests are READ-ONLY against ``submission/``. They construct small
synthetic row sets and verify both the EB formula and the lookup
fallback chain (full-triple → subject×benchmark → subject → benchmark
→ global). The implementation lives at
``src/peval_gold/models/eb.py``; the gold-track Batch 5 walkthrough
``docs/walkthroughs/017_gold_track_eb_blends.md`` reports the
end-to-end candidate results.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    subject: str,
    benchmark: str,
    condition: str,
    response: float,
    item_idx: int,
) -> dict[str, Any]:
    return {
        "subject_content": f"Name: {subject}",
        "benchmark": benchmark,
        "condition": condition,
        "item_content": f"item-{benchmark}-{item_idx}",
        "response": float(response),
    }


def _two_subject_dataset() -> list[dict[str, Any]]:
    """Subject A: 7/10 correct; Subject B: 3/10 correct; one benchmark / condition.

    Hand-derived target values:
    - p_global = 10/20 = 0.5
    - Sibling proportions for the "subject" level: A=0.7, B=0.3.
    - Weighted variance = (10 * 0.04 + 10 * 0.04) / 20 = 0.04
    - Expected within-variance = mean(p_i * (1-p_i) / n_i) = mean(0.021, 0.021) = 0.021
    - Between variance ≈ 0.04 - 0.021 = 0.019
    - kappa_subject = mean*(1-mean)/between - 1 = 0.25/0.019 - 1 ≈ 12.16
    - p_eb(A) = (7 + 12.16*0.5) / (10 + 12.16) ≈ 13.08 / 22.16 ≈ 0.5903
    """
    rows: list[dict[str, Any]] = []
    for i in range(7):
        rows.append(_row("A", "X", "c", 1.0, i))
    for i in range(7, 10):
        rows.append(_row("A", "X", "c", 0.0, i))
    for i in range(3):
        rows.append(_row("B", "X", "c", 1.0, i + 100))
    for i in range(3, 10):
        rows.append(_row("B", "X", "c", 0.0, i + 100))
    return rows


def _balanced_synthetic_dataset(n: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    """Three subjects with distinct base rates; 2 benchmarks, 2 conditions."""
    rng = random.Random(seed)
    p_by_subject = {"A": 0.85, "B": 0.5, "C": 0.15}
    out: list[dict[str, Any]] = []
    for i in range(n):
        subject = rng.choice(list(p_by_subject))
        benchmark = rng.choice(["bench1", "bench2"])
        condition = rng.choice(["c0", "c1"])
        response = 1.0 if rng.random() < p_by_subject[subject] else 0.0
        out.append(_row(subject, benchmark, condition, response, i))
    return out


# ---------------------------------------------------------------------------
# 1. fit produces sensible hierarchical estimates
# ---------------------------------------------------------------------------


def test_eb_fit_returns_sensible_hierarchical_estimates() -> None:
    """High-base-rate subject must map to high p_eb; low-base-rate to low p_eb."""
    from peval_gold.models.eb import EBPriors

    rows = _balanced_synthetic_dataset(n=300, seed=42)
    eb = EBPriors()
    eb.fit(rows)

    high = eb.predict_proba(
        [
            {
                "subject_content": "Name: A",
                "benchmark": "bench1",
                "condition": "c0",
                "item_content": "novel-item",
            }
        ]
    )[0]
    low = eb.predict_proba(
        [
            {
                "subject_content": "Name: C",
                "benchmark": "bench1",
                "condition": "c0",
                "item_content": "novel-item",
            }
        ]
    )[0]
    mid = eb.predict_proba(
        [
            {
                "subject_content": "Name: B",
                "benchmark": "bench1",
                "condition": "c0",
                "item_content": "novel-item",
            }
        ]
    )[0]

    assert 0.0 < low < mid < high < 1.0
    assert high > 0.5
    assert low < 0.5


# ---------------------------------------------------------------------------
# 2. predict_proba matches the EB formula for a known cell
# ---------------------------------------------------------------------------


def test_eb_predict_proba_matches_eb_formula_for_known_subject_cell() -> None:
    """For a row that triggers the subject-level fallback, the predicted
    probability must equal ``(k + kappa * parent_p) / (n + kappa)`` exactly.
    """
    from peval_gold.models.eb import EBPriors

    rows = _two_subject_dataset()
    eb = EBPriors()
    eb.fit(rows)

    info = eb.get_cell_info("subject", "A")
    assert info is not None, "subject A should be in the fitted hierarchy"
    expected = (info["k"] + info["kappa"] * info["parent_p"]) / (info["n"] + info["kappa"])

    # Use unknown benchmark + unknown condition so the lookup chain falls
    # PAST subject_benchmark_condition / subject_benchmark / benchmark and
    # lands on the subject cell.
    pred = eb.predict_proba(
        [
            {
                "subject_content": "Name: A",
                "benchmark": "unknown_bench_zzz",
                "condition": "unknown_cond_zzz",
                "item_content": "novel-item",
            }
        ]
    )[0]
    assert pred == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. Lookup chain: unknown subject → benchmark level
# ---------------------------------------------------------------------------


def test_eb_unknown_subject_falls_back_to_benchmark_level() -> None:
    from peval_gold.models.eb import EBPriors

    rows = _two_subject_dataset()
    eb = EBPriors()
    eb.fit(rows)

    bench_info = eb.get_cell_info("benchmark", "X")
    assert bench_info is not None
    expected = (bench_info["k"] + bench_info["kappa"] * bench_info["parent_p"]) / (
        bench_info["n"] + bench_info["kappa"]
    )

    pred = eb.predict_proba(
        [
            {
                "subject_content": "Name: ZZZ_unknown_subject",
                "benchmark": "X",
                "condition": "c",
                "item_content": "novel-item",
            }
        ]
    )[0]
    assert pred == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Lookup chain: unknown subject AND benchmark → global
# ---------------------------------------------------------------------------


def test_eb_unknown_subject_and_benchmark_falls_back_to_global() -> None:
    from peval_gold.models.eb import EBPriors

    rows = _two_subject_dataset()
    eb = EBPriors()
    eb.fit(rows)

    global_info = eb.get_cell_info("global", None)
    assert global_info is not None
    expected_p = global_info["p_eb"]

    pred = eb.predict_proba(
        [
            {
                "subject_content": "Name: ZZZ_unknown",
                "benchmark": "unknown_bench",
                "condition": "unknown_cond",
                "item_content": "novel-item",
            }
        ]
    )[0]
    assert pred == pytest.approx(expected_p, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Empty input raises ValueError
# ---------------------------------------------------------------------------


def test_eb_fit_empty_train_rows_raises_value_error() -> None:
    from peval_gold.models.eb import EBPriors

    eb = EBPriors()
    with pytest.raises(ValueError):
        eb.fit([])


# ---------------------------------------------------------------------------
# 6. save/load round-trips identical predictions
# ---------------------------------------------------------------------------


def test_eb_save_load_round_trip_preserves_predictions(tmp_path: Path) -> None:
    from peval_gold.models.eb import EBPriors

    rows = _balanced_synthetic_dataset(n=200, seed=7)
    eb = EBPriors()
    eb.fit(rows)

    eval_rows = [
        {
            "subject_content": "Name: A",
            "benchmark": "bench1",
            "condition": "c0",
            "item_content": "z1",
        },
        {
            "subject_content": "Name: B",
            "benchmark": "bench2",
            "condition": "c1",
            "item_content": "z2",
        },
        {
            "subject_content": "Name: ZZZ_unknown",
            "benchmark": "unknown_bench",
            "condition": "unknown_cond",
            "item_content": "z3",
        },
    ]
    pre = eb.predict_proba(eval_rows)

    path = tmp_path / "eb.json"
    eb.save(str(path))
    assert path.exists()
    assert path.stat().st_size > 0

    restored = EBPriors.load(str(path))
    post = restored.predict_proba(eval_rows)
    np.testing.assert_allclose(pre, post, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. kappa estimation is bounded to [1, 1000]
# ---------------------------------------------------------------------------


def test_eb_kappa_clipped_to_range_when_siblings_have_identical_rates() -> None:
    """All subjects have identical ratio (5/10) → between-sibling variance ≈ 0 →
    natural kappa → +inf → must be clipped to 1000.
    """
    from peval_gold.models.eb import EBPriors

    rows = []
    for j in range(5):
        for k in range(5):
            rows.append(_row(f"S{j}", "X", "c", 1.0, j * 100 + k))
        for k in range(5, 10):
            rows.append(_row(f"S{j}", "X", "c", 0.0, j * 100 + k))

    eb = EBPriors()
    eb.fit(rows)

    kappa_subject = eb.get_level_kappa("subject")
    assert 1.0 <= kappa_subject <= 1000.0
    assert kappa_subject == pytest.approx(1000.0, abs=1e-9), (
        f"identical sibling rates should saturate kappa to 1000.0 cap; got {kappa_subject}"
    )


def test_eb_kappa_clipped_below_when_siblings_have_extreme_variance() -> None:
    """All-1 vs all-0 siblings → maximum between variance → natural kappa < 1
    → must be clipped UP to 1.0.
    """
    from peval_gold.models.eb import EBPriors

    rows = []
    for k in range(10):
        rows.append(_row("S_all_correct", "X", "c", 1.0, k))
        rows.append(_row("S_all_wrong", "X", "c", 0.0, k + 100))

    eb = EBPriors()
    eb.fit(rows)

    kappa_subject = eb.get_level_kappa("subject")
    assert 1.0 <= kappa_subject <= 1000.0
    assert kappa_subject == pytest.approx(1.0, abs=1e-9), (
        f"extreme sibling variance should saturate kappa to 1.0 floor; got {kappa_subject}"
    )


# ---------------------------------------------------------------------------
# 8. predict_one mirrors predict_proba for a single row
# ---------------------------------------------------------------------------


def test_eb_predict_one_matches_predict_proba_for_single_row() -> None:
    from peval_gold.models.eb import EBPriors

    rows = _balanced_synthetic_dataset(n=300, seed=42)
    eb = EBPriors()
    eb.fit(rows)

    row = {
        "subject_content": "Name: A",
        "benchmark": "bench1",
        "condition": "c0",
        "item_content": "novel-item",
    }
    one = eb.predict_one(row, labeled=None)
    proba = float(eb.predict_proba([row])[0])
    assert one == pytest.approx(proba, abs=1e-12)
    assert isinstance(one, float)
    assert math.isfinite(one)
    assert 0.0 < one < 1.0


# ---------------------------------------------------------------------------
# 9. Output range and finiteness on unknown rows
# ---------------------------------------------------------------------------


def test_eb_predict_proba_output_is_finite_and_in_unit_interval() -> None:
    from peval_gold.models.eb import EBPriors

    rows = _balanced_synthetic_dataset(n=300, seed=42)
    eb = EBPriors()
    eb.fit(rows)

    eval_rows = [
        {
            "subject_content": f"Name: subj-{i}",
            "benchmark": f"bench-{i % 3}",
            "condition": f"cond-{i % 2}",
            "item_content": "x",
        }
        for i in range(20)
    ]
    p = eb.predict_proba(eval_rows)
    assert isinstance(p, np.ndarray)
    assert p.shape == (len(eval_rows),)
    assert np.all(np.isfinite(p))
    assert np.all((p > 0.0) & (p < 1.0))


# ---------------------------------------------------------------------------
# 10. subject_benchmark_condition cell only created when n >= 5
# ---------------------------------------------------------------------------


def test_eb_subject_benchmark_condition_cell_requires_min_count() -> None:
    """A triple cell with n < 5 must not appear in the lookup; the fallback
    chain should walk to subject_benchmark instead."""
    from peval_gold.models.eb import EBPriors

    rows = []
    rows.append(_row("A", "X", "c1", 1.0, 0))
    rows.append(_row("A", "X", "c1", 0.0, 1))
    for i in range(20):
        rows.append(_row("A", "X", "c2", 1.0 if i % 2 == 0 else 0.0, 100 + i))

    eb = EBPriors()
    eb.fit(rows)

    triple_cell = eb.get_cell_info("subject_benchmark_condition", ("A", "X", "c1"))
    assert triple_cell is None, (
        "subject_benchmark_condition (A,X,c1) has n=2 < 5; cell should be absent"
    )
    triple_cell_c2 = eb.get_cell_info("subject_benchmark_condition", ("A", "X", "c2"))
    assert triple_cell_c2 is not None, (
        "subject_benchmark_condition (A,X,c2) has n=20 >= 5; cell should be present"
    )


# ---------------------------------------------------------------------------
# 11. Visible subject key fallback for missing Name: line
# ---------------------------------------------------------------------------


def test_eb_subject_key_fallback_uses_blake2b_when_name_line_absent() -> None:
    """When subject_content lacks ``Name:``, the EB should fall back to a
    BLAKE2b hash so rows with otherwise-identical content still share a key.
    """
    from peval_gold.models.eb import EBPriors

    rows = []
    for k in range(10):
        rows.append(
            {
                "subject_content": "Organization: org\nFamily: fam-xyz",
                "benchmark": "X",
                "condition": "c",
                "item_content": f"item-{k}",
                "response": 1.0 if k < 7 else 0.0,
            }
        )

    eb = EBPriors()
    eb.fit(rows)

    pred1 = eb.predict_proba(
        [
            {
                "subject_content": "Organization: org\nFamily: fam-xyz",
                "benchmark": "X",
                "condition": "c",
                "item_content": "z",
            }
        ]
    )[0]
    pred2 = eb.predict_proba(
        [
            {
                "subject_content": "Organization: org\nFamily: fam-xyz",
                "benchmark": "X",
                "condition": "c",
                "item_content": "z",
            }
        ]
    )[0]
    assert pred1 == pytest.approx(pred2, abs=1e-12)
    assert 0.0 < pred1 < 1.0


# ---------------------------------------------------------------------------
# 12. EBPriors satisfies the gold-track Predictor protocol
# ---------------------------------------------------------------------------


def test_eb_priors_satisfies_predictor_and_runtime_predictor_protocols() -> None:
    from peval_gold.models.base import Predictor, RuntimePredictor
    from peval_gold.models.eb import EBPriors

    eb = EBPriors()
    assert isinstance(eb, Predictor)
    assert isinstance(eb, RuntimePredictor)
