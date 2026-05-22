"""TDD red→green tests for ``peval_gold.data.splits``.

Batch 2 deliverable (S1B). Covers six splits used by the gold-track
validation harness:

- ``random_row_smoke`` — small random row split for fast smoke tests.
- ``item_holdout_primary`` — PROMOTION-GRADE: group by ``item_id``,
  hold out a fraction of unique items, all rows for those items go to
  val. This is the gate every Batch 4+ challenger must beat.
- ``benchmark_holdout_stress`` — full benchmarks as val; the rest as
  train. Stress test (worst-case generalization).
- ``domain_holdout`` — by benchmark ``domain`` if available; skips
  when the field is absent.
- ``subject_holdout`` — group by ``subject_id``. Diagnostic only.
- ``adaptive_label_simulation`` — simulate the platform's K=5
  revealed-labels-per-data-category policy
  (``starting_kit/README.md:268-271``).

Every split MUST produce zero-overlap on its grouping key. The manifest
returned by each split's helper includes ``zero_overlap_verified: True``
plus the grouping-key counts.

All tests use small synthetic dicts — no HF access.
"""

from __future__ import annotations

import pytest

from peval_gold.data.splits import (
    adaptive_label_simulation,
    benchmark_holdout_stress,
    domain_holdout,
    item_holdout_primary,
    random_row_smoke,
    subject_holdout,
)

# ---------------------------------------------------------------------------
# Tiny synthetic builders
# ---------------------------------------------------------------------------


def _row(
    benchmark: str = "mmlupro",
    item_id: str = "i-1",
    subject_id: str = "s-1",
    condition: str = "none",
    response: float = 1.0,
    **extra,
) -> dict:
    return {
        "benchmark": benchmark,
        "condition": condition,
        "subject_id": subject_id,
        "item_id": item_id,
        "subject_content": f"Name: {subject_id}",
        "item_content": f"content-of-{item_id}",
        "response": response,
        **extra,
    }


def _grid_corpus(n_benchmarks: int = 3, n_items: int = 20, n_subjects: int = 5) -> list[dict]:
    """Cartesian-product synthetic corpus: ``n_benchmarks * n_items * n_subjects`` rows."""
    rows = []
    for b in range(n_benchmarks):
        bench = f"bench{b}"
        for i in range(n_items):
            for s in range(n_subjects):
                rows.append(
                    _row(
                        benchmark=bench,
                        item_id=f"{bench}-i{i}",
                        subject_id=f"s{s}",
                        response=float((b + i + s) % 2),
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# 1. random_row_smoke
# ---------------------------------------------------------------------------


def test_random_row_smoke_returns_disjoint_row_indices() -> None:
    corpus = _grid_corpus()
    train, val, manifest = random_row_smoke(corpus, n_total=200, val_frac=0.1, seed=42)

    # Same-corpus-index-disjointness via id() identity.
    assert {id(r) for r in train}.isdisjoint({id(r) for r in val})
    assert len(train) + len(val) == 200
    assert manifest["zero_overlap_verified"] is True
    assert manifest["n_train"] == len(train)
    assert manifest["n_val"] == len(val)
    # val frac ~= 10%, with floor.
    assert manifest["n_val"] == int(200 * 0.1)


def test_random_row_smoke_subsamples_when_corpus_exceeds_n_total() -> None:
    corpus = _grid_corpus(n_benchmarks=2, n_items=50, n_subjects=4)  # 400 rows
    train, val, _ = random_row_smoke(corpus, n_total=100, val_frac=0.2, seed=7)
    assert len(train) + len(val) == 100
    assert len(val) == 20


def test_random_row_smoke_uses_whole_corpus_when_smaller_than_n_total() -> None:
    corpus = _grid_corpus(n_benchmarks=1, n_items=5, n_subjects=2)  # 10 rows
    train, val, _ = random_row_smoke(corpus, n_total=100, val_frac=0.1, seed=42)
    assert len(train) + len(val) == 10


def test_random_row_smoke_deterministic_across_runs_with_same_seed() -> None:
    corpus = _grid_corpus()
    a_train, a_val, _ = random_row_smoke(corpus, n_total=100, val_frac=0.1, seed=42)
    b_train, b_val, _ = random_row_smoke(corpus, n_total=100, val_frac=0.1, seed=42)
    assert [r["item_id"] for r in a_train] == [r["item_id"] for r in b_train]
    assert [r["item_id"] for r in a_val] == [r["item_id"] for r in b_val]


def test_random_row_smoke_different_seeds_diverge() -> None:
    corpus = _grid_corpus()
    a_train, _, _ = random_row_smoke(corpus, n_total=100, val_frac=0.1, seed=42)
    b_train, _, _ = random_row_smoke(corpus, n_total=100, val_frac=0.1, seed=7)
    assert [r["item_id"] for r in a_train] != [r["item_id"] for r in b_train]


# ---------------------------------------------------------------------------
# 2. item_holdout_primary
# ---------------------------------------------------------------------------


def test_item_holdout_primary_produces_zero_item_id_overlap() -> None:
    corpus = _grid_corpus()  # 3 * 20 * 5 = 300 rows, 60 unique items.
    train, val, manifest = item_holdout_primary(corpus, val_frac=0.1, seed=42)

    train_items = {r["item_id"] for r in train}
    val_items = {r["item_id"] for r in val}
    assert train_items.isdisjoint(val_items)

    assert manifest["zero_overlap_verified"] is True
    assert manifest["group_key"] == "item_id"
    assert manifest["n_train_groups"] == len(train_items)
    assert manifest["n_val_groups"] == len(val_items)
    assert manifest["n_train_groups"] + manifest["n_val_groups"] == 60


def test_item_holdout_primary_val_frac_approximates_target() -> None:
    """Val ITEM count ~ val_frac of unique items. Row count is downstream."""
    corpus = _grid_corpus(n_benchmarks=1, n_items=100, n_subjects=1)  # 100 items.
    _, _, manifest = item_holdout_primary(corpus, val_frac=0.1, seed=42)
    assert manifest["n_val_groups"] == 10


def test_item_holdout_primary_deterministic_across_runs() -> None:
    corpus = _grid_corpus()
    _, a_val, _ = item_holdout_primary(corpus, val_frac=0.1, seed=42)
    _, b_val, _ = item_holdout_primary(corpus, val_frac=0.1, seed=42)
    assert {r["item_id"] for r in a_val} == {r["item_id"] for r in b_val}


def test_item_holdout_primary_keeps_all_rows_of_each_held_out_item() -> None:
    """If item-X goes to val, EVERY row about item-X must be in val
    (not split across train/val)."""
    corpus = _grid_corpus(n_benchmarks=1, n_items=10, n_subjects=4)  # 40 rows.
    train, val, _ = item_holdout_primary(corpus, val_frac=0.3, seed=42)
    val_items = {r["item_id"] for r in val}
    # Every (item, subject) row for a val item must be in val.
    for r in corpus:
        if r["item_id"] in val_items:
            assert r in val
        else:
            assert r in train


# ---------------------------------------------------------------------------
# 3. benchmark_holdout_stress
# ---------------------------------------------------------------------------


def test_benchmark_holdout_stress_puts_all_rows_in_val() -> None:
    corpus = _grid_corpus()  # benches: bench0, bench1, bench2.
    train, val, manifest = benchmark_holdout_stress(corpus, holdout_benchmarks=["bench1"], seed=42)

    train_benches = {r["benchmark"] for r in train}
    val_benches = {r["benchmark"] for r in val}

    assert val_benches == {"bench1"}
    assert "bench1" not in train_benches
    assert train_benches == {"bench0", "bench2"}

    assert manifest["zero_overlap_verified"] is True
    assert manifest["group_key"] == "benchmark"
    assert sorted(manifest["holdout_benchmarks"]) == ["bench1"]
    # All rows accounted for.
    assert len(train) + len(val) == len(corpus)


def test_benchmark_holdout_stress_multiple_benchmarks() -> None:
    corpus = _grid_corpus()
    train, val, manifest = benchmark_holdout_stress(
        corpus, holdout_benchmarks=["bench0", "bench2"], seed=42
    )
    train_benches = {r["benchmark"] for r in train}
    val_benches = {r["benchmark"] for r in val}
    assert train_benches == {"bench1"}
    assert val_benches == {"bench0", "bench2"}
    assert manifest["zero_overlap_verified"] is True


def test_benchmark_holdout_stress_empty_holdout_returns_full_train_empty_val() -> None:
    corpus = _grid_corpus()
    train, val, manifest = benchmark_holdout_stress(corpus, holdout_benchmarks=[], seed=42)
    assert len(train) == len(corpus)
    assert val == []
    assert manifest["zero_overlap_verified"] is True


def test_benchmark_holdout_stress_unknown_benchmark_yields_empty_val() -> None:
    """A holdout benchmark name not present in corpus produces an empty val
    set with zero-overlap still trivially verified."""
    corpus = _grid_corpus()
    train, val, manifest = benchmark_holdout_stress(
        corpus, holdout_benchmarks=["nonexistent"], seed=42
    )
    assert len(train) == len(corpus)
    assert val == []
    assert manifest["zero_overlap_verified"] is True


# ---------------------------------------------------------------------------
# 4. domain_holdout
# ---------------------------------------------------------------------------


def test_domain_holdout_groups_by_domain_when_present() -> None:
    rows = [_row(item_id=f"i-{i}", domain="math") for i in range(5)] + [
        _row(item_id=f"i-{i + 100}", domain="medicine") for i in range(5)
    ]
    train, val, manifest = domain_holdout(rows, holdout_domains=["medicine"], seed=42)
    assert all(r["domain"] == "math" for r in train)
    assert all(r["domain"] == "medicine" for r in val)
    assert manifest["zero_overlap_verified"] is True


def test_domain_holdout_skips_when_no_domain_field_anywhere() -> None:
    """When ``domain`` is absent from every row, return empty val and a
    manifest flagged ``domain_field_present=False``."""
    rows = [_row(item_id=f"i-{i}") for i in range(5)]
    train, val, manifest = domain_holdout(rows, holdout_domains=["math"], seed=42)
    assert train == rows
    assert val == []
    assert manifest["domain_field_present"] is False
    assert manifest["zero_overlap_verified"] is True


def test_domain_holdout_accepts_list_typed_domain_values() -> None:
    """HF ``benchmarks.parquet:domain`` is a list of strings. Honor that
    shape: a row matches if ANY of its domain entries is in the holdout
    set."""
    rows = [
        _row(item_id="i-1", domain=["math", "reasoning"]),
        _row(item_id="i-2", domain=["medicine"]),
        _row(item_id="i-3", domain=["code"]),
    ]
    train, val, _ = domain_holdout(rows, holdout_domains=["math"], seed=42)
    assert {r["item_id"] for r in val} == {"i-1"}
    assert {r["item_id"] for r in train} == {"i-2", "i-3"}


# ---------------------------------------------------------------------------
# 5. subject_holdout
# ---------------------------------------------------------------------------


def test_subject_holdout_zero_overlap_on_subject_id() -> None:
    corpus = _grid_corpus(n_benchmarks=2, n_items=10, n_subjects=10)  # 200 rows.
    train, val, manifest = subject_holdout(corpus, val_frac=0.2, seed=42)

    train_subj = {r["subject_id"] for r in train}
    val_subj = {r["subject_id"] for r in val}

    assert train_subj.isdisjoint(val_subj)
    assert manifest["group_key"] == "subject_id"
    assert manifest["zero_overlap_verified"] is True
    assert manifest["n_val_groups"] == 2  # 20% of 10 subjects.


def test_subject_holdout_deterministic_across_runs() -> None:
    corpus = _grid_corpus()
    _, a_val, _ = subject_holdout(corpus, val_frac=0.2, seed=42)
    _, b_val, _ = subject_holdout(corpus, val_frac=0.2, seed=42)
    assert {r["subject_id"] for r in a_val} == {r["subject_id"] for r in b_val}


# ---------------------------------------------------------------------------
# 6. adaptive_label_simulation
# ---------------------------------------------------------------------------


def test_adaptive_label_simulation_reveals_k_per_category() -> None:
    """K=5 revealed labels per data category, per round."""
    rows = []
    # bench0 has 20 rows; bench1 has 20 rows. No ``category`` field present,
    # so the simulator falls back to ``benchmark`` as the category key.
    for i in range(20):
        rows.append(_row(benchmark="bench0", item_id=f"b0-i{i}"))
        rows.append(_row(benchmark="bench1", item_id=f"b1-i{i}"))

    labeled, unlabeled, manifest = adaptive_label_simulation(rows, k_per_category=5, seed=42)

    # Exactly 5 from each category.
    labeled_by_bench = {}
    for r in labeled:
        labeled_by_bench.setdefault(r["benchmark"], []).append(r)
    assert len(labeled_by_bench["bench0"]) == 5
    assert len(labeled_by_bench["bench1"]) == 5

    # Labeled + unlabeled = full corpus, no overlap.
    assert {id(r) for r in labeled}.isdisjoint({id(r) for r in unlabeled})
    assert len(labeled) + len(unlabeled) == len(rows)

    assert manifest["zero_overlap_verified"] is True
    assert manifest["k_per_category"] == 5
    assert manifest["category_key"] in ("category", "benchmark")


def test_adaptive_label_simulation_prefers_category_field_when_present() -> None:
    rows = [_row(benchmark="bench0", item_id=f"i-{i}", category="sample-A") for i in range(10)] + [
        _row(benchmark="bench0", item_id=f"j-{i}", category="sample-B") for i in range(10)
    ]
    labeled, _, manifest = adaptive_label_simulation(rows, k_per_category=3, seed=42)
    assert manifest["category_key"] == "category"
    labeled_by_cat = {}
    for r in labeled:
        labeled_by_cat.setdefault(r["category"], []).append(r)
    assert len(labeled_by_cat["sample-A"]) == 3
    assert len(labeled_by_cat["sample-B"]) == 3


def test_adaptive_label_simulation_handles_small_categories_gracefully() -> None:
    """When a category has fewer than ``k_per_category`` rows, take all of them
    without raising."""
    rows = [_row(benchmark="bench0", item_id=f"i-{i}") for i in range(3)]
    labeled, unlabeled, _ = adaptive_label_simulation(rows, k_per_category=5, seed=42)
    assert len(labeled) == 3
    assert unlabeled == []


def test_adaptive_label_simulation_deterministic_across_runs() -> None:
    rows = [_row(benchmark="bench0", item_id=f"i-{i}") for i in range(20)]
    a, _, _ = adaptive_label_simulation(rows, k_per_category=5, seed=42)
    b, _, _ = adaptive_label_simulation(rows, k_per_category=5, seed=42)
    assert [r["item_id"] for r in a] == [r["item_id"] for r in b]


# ---------------------------------------------------------------------------
# 7. Cross-cutting: every split returns a 3-tuple with a manifest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name",
    ["random_row_smoke", "item_holdout_primary", "subject_holdout"],
)
def test_all_default_splits_return_three_tuple_with_manifest(fn_name: str) -> None:
    """Every default-style split returns ``(train, val, manifest)`` with
    ``zero_overlap_verified: True``."""
    import peval_gold.data.splits as splits_mod

    fn = getattr(splits_mod, fn_name)
    corpus = _grid_corpus()
    result = fn(corpus, seed=42) if fn_name != "random_row_smoke" else fn(corpus)
    assert len(result) == 3
    train, val, manifest = result
    assert isinstance(train, list)
    assert isinstance(val, list)
    assert isinstance(manifest, dict)
    assert manifest["zero_overlap_verified"] is True
