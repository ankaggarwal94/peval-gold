"""TDD red→green tests for the Batch 3 evaluator + reports + ledger (S1C).

Three modules under test:

- :mod:`peval_gold.eval.evaluator` — ``evaluate`` (single-pass NLL/MLL/
  Brier/ECE/AUC + per-benchmark + per-condition + predict_one timing)
  and ``evaluate_adaptive`` (per-round simulator).
- :mod:`peval_gold.eval.reports` — ``to_json`` / ``to_markdown`` /
  ``compare_reports``.
- :mod:`peval_gold.experiments.ledger` — ``new_run_id`` /
  ``write_manifest`` / ``append_result``. APPEND-ONLY semantics on
  ``results.jsonl`` per gold-track §6.

All evaluator tests use tiny synthetic ``ConstantPredictor`` /
``LinearishPredictor`` shims (no encoder dependency, no @slow marker)
so the suite stays sub-second.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Tiny synthetic shim predictors
# ---------------------------------------------------------------------------


class ConstantRuntime:
    """RuntimePredictor returning a fixed probability per call."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = float(value)
        self.metadata_payload = {"class": "ConstantRuntime", "value": self.value}

    def predict_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
        labeled: list[dict] | None = None,
    ) -> float:
        return self.value

    def metadata(self) -> dict:
        return dict(self.metadata_payload)


class LabelEchoRuntime:
    """Returns ``input['p_target']`` if present, else 0.5.

    Useful for testing AUC: by setting p_target = label * 0.9 + 0.05 we
    get a perfectly-ranking predictor and AUC must come out at 1.0.
    """

    def predict_one(
        self,
        input: dict,  # noqa: A002
        labeled: list[dict] | None = None,
    ) -> float:
        return float(input.get("p_target", 0.5))

    def metadata(self) -> dict:
        return {"class": "LabelEchoRuntime"}


def _synth_rows(n: int = 12) -> list[dict]:
    """Mixed-benchmark / mixed-condition synthetic eval rows."""
    rows: list[dict] = []
    for i in range(n):
        label = float(i % 2)
        rows.append(
            {
                "benchmark": f"bench_{i % 3}",
                "condition": "none" if i % 2 == 0 else "cot",
                "subject_id": f"s-{i % 4}",
                "item_id": f"i-{i}",
                "subject_content": f"Name: model-{i % 4}",
                "item_content": f"q-{i}",
                "response": label,
                # For LabelEchoRuntime: high p when label=1, low when label=0.
                "p_target": 0.9 if label == 1.0 else 0.1,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# evaluator.evaluate — report schema
# ---------------------------------------------------------------------------


def test_evaluate_returns_required_report_schema() -> None:
    from peval_gold.eval.evaluator import evaluate

    pred = ConstantRuntime(value=0.5)
    rows = _synth_rows(12)
    report = evaluate(pred, rows, split_name="unit_test_split")

    assert isinstance(report, dict)
    assert report["split_name"] == "unit_test_split"
    assert report["n_examples"] == 12

    metrics = report["metrics"]
    for key in (
        "ordinary_log_loss",
        "mean_log_likelihood",
        "brier_score",
        "expected_calibration_error",
        "auc",
    ):
        assert key in metrics
        # Every metric is a Python float (json-serializable; no numpy scalars).
        assert isinstance(metrics[key], float) or metrics[key] is None

    timing = report["timing"]
    for key in (
        "predict_one_p50_ms",
        "predict_one_p95_ms",
        "predict_one_max_ms",
        "n_predict_calls",
    ):
        assert key in timing
    assert timing["n_predict_calls"] == 12

    assert "per_benchmark" in report
    assert "per_condition" in report
    assert "artifact" in report
    assert report["artifact"]["predictor_class"] == "ConstantRuntime"
    assert "timestamp_utc" in report


def test_evaluate_metrics_match_metric_module() -> None:
    """Aggregate metrics in the report must match
    ``peval_gold.eval.metrics`` applied directly to the predictions."""
    from peval_gold.eval.evaluator import evaluate
    from peval_gold.eval.metrics import (
        brier_score,
        mean_log_likelihood,
        ordinary_log_loss,
    )

    pred = ConstantRuntime(value=0.6)
    rows = _synth_rows(12)
    report = evaluate(pred, rows, split_name="x")

    y = np.array([r["response"] for r in rows], dtype=float)
    p = np.full(12, 0.6)

    assert report["metrics"]["ordinary_log_loss"] == pytest.approx(
        ordinary_log_loss(y, p), abs=1e-9
    )
    assert report["metrics"]["mean_log_likelihood"] == pytest.approx(
        mean_log_likelihood(y, p), abs=1e-9
    )
    assert report["metrics"]["brier_score"] == pytest.approx(brier_score(y, p), abs=1e-9)


def test_evaluate_auc_is_one_for_perfectly_ranking_predictor() -> None:
    """LabelEchoRuntime produces a perfect ranking; numpy-only AUC must
    be exactly 1.0."""
    from peval_gold.eval.evaluator import evaluate

    pred = LabelEchoRuntime()
    rows = _synth_rows(12)
    report = evaluate(pred, rows, split_name="x")
    assert report["metrics"]["auc"] == pytest.approx(1.0, abs=1e-9)


def test_evaluate_auc_is_none_when_only_one_class_present() -> None:
    """AUC is undefined when y has only one class. Convention: return
    None (not NaN, not 0.5) so downstream report consumers can
    explicitly skip the metric without silent confusion."""
    from peval_gold.eval.evaluator import evaluate

    pred = ConstantRuntime(value=0.5)
    rows = [
        {
            "benchmark": "b",
            "condition": "c",
            "subject_id": "s",
            "item_id": f"i-{i}",
            "subject_content": "x",
            "item_content": "y",
            "response": 1.0,
        }
        for i in range(5)
    ]
    report = evaluate(pred, rows, split_name="single_class")
    assert report["metrics"]["auc"] is None


def test_evaluate_per_benchmark_breakdown_groups_correctly() -> None:
    from peval_gold.eval.evaluator import evaluate

    pred = ConstantRuntime(value=0.5)
    rows = _synth_rows(12)
    report = evaluate(pred, rows, split_name="x")

    pb = report["per_benchmark"]
    # 12 rows / 3 benches => 4 rows per bench, 2 of each class.
    assert set(pb.keys()) == {"bench_0", "bench_1", "bench_2"}
    for stats in pb.values():
        assert stats["n"] == 4
        assert "nll" in stats and isinstance(stats["nll"], float)
        assert "mll" in stats and isinstance(stats["mll"], float)
        assert "auc" in stats  # may be None or a float; just present


def test_evaluate_per_condition_breakdown_groups_correctly() -> None:
    from peval_gold.eval.evaluator import evaluate

    pred = ConstantRuntime(value=0.5)
    rows = _synth_rows(12)
    report = evaluate(pred, rows, split_name="x")

    pc = report["per_condition"]
    assert set(pc.keys()) == {"none", "cot"}
    for stats in pc.values():
        assert stats["n"] == 6


def test_evaluate_handles_empty_dataset_gracefully() -> None:
    from peval_gold.eval.evaluator import evaluate

    pred = ConstantRuntime()
    report = evaluate(pred, [], split_name="empty")
    assert report["n_examples"] == 0
    assert report["timing"]["n_predict_calls"] == 0
    # NLL / MLL / Brier are undefined on empty; convention is None.
    for key in ("ordinary_log_loss", "mean_log_likelihood", "brier_score", "auc"):
        assert report["metrics"][key] is None


# ---------------------------------------------------------------------------
# evaluator.evaluate_adaptive — per-round results
# ---------------------------------------------------------------------------


def test_evaluate_adaptive_returns_per_round_results() -> None:
    from peval_gold.eval.evaluator import evaluate_adaptive

    pred = ConstantRuntime(value=0.5)
    rows = _synth_rows(12)

    # Two rounds: each labeled set is the first 4 rows; unlabeled is rest.
    rounds = [
        (rows[:4], rows[4:]),
        (rows[:6], rows[6:]),
    ]
    out = evaluate_adaptive(pred, rounds, split_name="adaptive_x")
    assert isinstance(out, dict)
    assert "rounds" in out
    assert len(out["rounds"]) == 2
    for round_report in out["rounds"]:
        assert "n_examples" in round_report
        assert "metrics" in round_report


# ---------------------------------------------------------------------------
# reports — JSON + Markdown round-trip + diff
# ---------------------------------------------------------------------------


def test_reports_to_json_round_trips(tmp_path: Path) -> None:
    from peval_gold.eval.evaluator import evaluate
    from peval_gold.eval.reports import to_json

    pred = ConstantRuntime(value=0.5)
    rows = _synth_rows(8)
    report = evaluate(pred, rows, split_name="x")

    out_path = tmp_path / "report.json"
    to_json(report, str(out_path))
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["split_name"] == "x"
    assert loaded["n_examples"] == 8
    # Round-trip key alignment.
    assert loaded["metrics"]["ordinary_log_loss"] == pytest.approx(
        report["metrics"]["ordinary_log_loss"], abs=1e-12
    )


def test_reports_to_markdown_writes_headline_and_breakdown(tmp_path: Path) -> None:
    from peval_gold.eval.evaluator import evaluate
    from peval_gold.eval.reports import to_markdown

    pred = ConstantRuntime(value=0.5)
    rows = _synth_rows(8)
    report = evaluate(pred, rows, split_name="x")

    out_path = tmp_path / "report.md"
    to_markdown(report, str(out_path))
    text = out_path.read_text()

    # Headline metrics must appear.
    assert "Mean log-likelihood" in text or "mean_log_likelihood" in text
    # Per-benchmark + per-condition tables must appear.
    assert "Per-benchmark" in text or "per_benchmark" in text
    assert "Per-condition" in text or "per_condition" in text
    # The split name must appear.
    assert "x" in text


def test_reports_compare_reports_returns_mll_delta() -> None:
    from peval_gold.eval.reports import compare_reports

    report_a = {
        "split_name": "x",
        "metrics": {
            "ordinary_log_loss": 0.7,
            "mean_log_likelihood": -0.7,
            "brier_score": 0.25,
            "expected_calibration_error": 0.1,
            "auc": 0.6,
        },
    }
    report_b = {
        "split_name": "x",
        "metrics": {
            "ordinary_log_loss": 0.6,
            "mean_log_likelihood": -0.6,
            "brier_score": 0.22,
            "expected_calibration_error": 0.08,
            "auc": 0.65,
        },
    }
    diff = compare_reports(report_a, report_b)
    assert "mean_log_likelihood_delta" in diff
    # B is better than A by 0.1 (less negative).
    assert diff["mean_log_likelihood_delta"] == pytest.approx(0.1, abs=1e-12)
    # ordinary_log_loss_delta is the symmetric drop (A_nll - B_nll = +0.1).
    assert diff["ordinary_log_loss_delta"] == pytest.approx(0.1, abs=1e-12)


# ---------------------------------------------------------------------------
# experiments.ledger — run id + manifest + append-only results
# ---------------------------------------------------------------------------


def test_new_run_id_starts_with_prefix_and_is_unique() -> None:
    from peval_gold.experiments.ledger import new_run_id

    a = new_run_id(prefix="eval")
    b = new_run_id(prefix="eval")
    assert a.startswith("eval-")
    assert b.startswith("eval-")
    # Same-second collisions are broken by a monotonic counter so two
    # back-to-back calls produce distinct ids.
    assert a != b


def test_write_manifest_creates_runs_gold_dir_and_returns_path(tmp_path: Path) -> None:
    from peval_gold.experiments import ledger

    run_id = "test-run-abc"
    config = {"family": "current_ncf"}
    files = ["submission/ncf_head.pt"]
    metadata = {"agent": "S1C-test"}

    path = ledger.write_manifest(
        run_id, config=config, files=files, metadata=metadata, root=tmp_path
    )
    assert Path(path).exists()
    payload = json.loads(Path(path).read_text())
    assert payload["run_id"] == run_id
    assert payload["config"] == config
    assert payload["files"] == files
    assert payload["metadata"] == metadata
    # Manifest must be inside runs/gold/<run_id>/.
    assert Path(path).parent.name == run_id


def test_append_result_is_append_only(tmp_path: Path) -> None:
    """``append_result`` must not rewrite the file — repeated calls append
    one JSON line each."""
    from peval_gold.experiments import ledger

    run_id = "test-run-append"
    ledger.write_manifest(run_id, config={}, files=[], metadata={}, root=tmp_path)

    ledger.append_result(run_id, {"k": 1}, root=tmp_path)
    ledger.append_result(run_id, {"k": 2}, root=tmp_path)
    ledger.append_result(run_id, {"k": 3}, root=tmp_path)

    results_path = tmp_path / run_id / "results.jsonl"
    assert results_path.exists()
    lines = results_path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"k": 1}
    assert json.loads(lines[2]) == {"k": 3}
