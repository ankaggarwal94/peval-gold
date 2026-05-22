"""Local evaluator for gold-track ``RuntimePredictor`` candidates.

Two public entry points:

- :func:`evaluate` — single-pass scoring against a flat row sequence.
  Returns the report schema specified in the S1C plan: aggregate
  metrics (NLL/MLL/Brier/ECE/AUC) + per-benchmark + per-condition
  breakdowns + ``predict_one`` timing percentiles + artifact provenance.
- :func:`evaluate_adaptive` — multi-round simulator that mirrors the
  the hosted runtime adaptive-labeling flow. Each round receives a labeled set
  (for the predictor's first-call Platt fit) and an unlabeled set (the
  rows actually scored). Returns per-round reports plus an aggregate.

Conventions
-----------

- AUC: numpy-only rank implementation. Returns ``None`` (NOT NaN, NOT
  0.5) when ``y`` carries only one class. This matches the report
  consumer's expected sentinel for "metric undefined on this slice."
- NLL / MLL / Brier on empty inputs: ``None``.
- Timing: monotonic ``time.perf_counter`` deltas around each call to
  ``predict_one``. Recorded in milliseconds. p50 / p95 / max via
  ``numpy.percentile``.
- All numeric outputs are coerced to native Python ``float`` /
  ``int`` so the report serializes cleanly via :mod:`peval_gold.eval.reports`.

Skipping continuous rows
------------------------

The evaluator only scores rows with ``response ∈ {0.0, 1.0}``. Rows
with continuous ``response`` (e.g. ``mtbench`` 1-10) get counted under
``n_skipped_non_binary`` but excluded from metrics. This matches the
D-7 binarization policy at the data layer
(``peval_gold.data.filters.binarize_drop``) and avoids silently mixing
NLL scales across benchmark families.
"""

from __future__ import annotations

import datetime as _dt
import math
import time
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from peval_gold.eval.metrics import (
    brier_score,
    expected_calibration_error,
    mean_log_likelihood,
    ordinary_log_loss,
)


_METRIC_KEYS_ALL_NONE = {
    "ordinary_log_loss": None,
    "mean_log_likelihood": None,
    "brier_score": None,
    "expected_calibration_error": None,
    "auc": None,
}


def _now_utc_iso() -> str:
    """ISO-8601 UTC timestamp without microseconds. Matches kit log format."""
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_response(value: Any) -> float | None:
    """Return the binary ``response`` ∈ {0.0, 1.0} or ``None`` if non-binary."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return None
    fv = float(value)
    if math.isnan(fv):
        return None
    if fv == 0.0 or fv == 1.0:
        return fv
    return None


def _numpy_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """Rank-based AUC via the Mann-Whitney U identity.

    AUC = (R+ - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    where R+ is the sum of ranks of positive class. Ties get average
    ranks so the formula handles ``rankdata`` correctly.

    Returns ``None`` if ``y`` contains only one class (AUC undefined).
    """
    if y.size == 0:
        return None
    pos_mask = y == 1.0
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # rankdata with method='average' — implement inline to avoid
    # importing scipy for one helper.
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    # Average ties: identify groups of equal values in sorted order.
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    r_pos = float(ranks[pos_mask].sum())
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _safe_metric(fn, y: np.ndarray, p: np.ndarray) -> float | None:
    """Run a metric over ``(y, p)``; return ``None`` if either array is empty."""
    if y.size == 0:
        return None
    return float(fn(y, p))


def _aggregate_metrics(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> dict[str, float | None]:
    """All five top-level metrics in one bundle."""
    if y.size == 0:
        return dict(_METRIC_KEYS_ALL_NONE)
    return {
        "ordinary_log_loss": _safe_metric(ordinary_log_loss, y, p),
        "mean_log_likelihood": _safe_metric(mean_log_likelihood, y, p),
        "brier_score": _safe_metric(brier_score, y, p),
        "expected_calibration_error": float(
            expected_calibration_error(y, p, n_bins=n_bins)
        ),
        "auc": _numpy_auc(y, p),
    }


def _grouped_summary(
    key_fn,
    rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    p: np.ndarray,
    include_auc: bool = True,
) -> dict[str, dict[str, Any]]:
    """Per-group ``{n, nll, mll, auc}`` for either benchmark or condition."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_group[str(key_fn(r))].append(i)

    out: dict[str, dict[str, Any]] = {}
    for key, idxs in by_group.items():
        sub_y = y[idxs]
        sub_p = p[idxs]
        out[key] = {
            "n": len(idxs),
            "nll": _safe_metric(ordinary_log_loss, sub_y, sub_p),
            "mll": _safe_metric(mean_log_likelihood, sub_y, sub_p),
            "auc": _numpy_auc(sub_y, sub_p) if include_auc else None,
        }
    return out


def _percentile_ms(samples_ms: list[float], q: float) -> float:
    if not samples_ms:
        return 0.0
    return float(np.percentile(samples_ms, q))


def _predictor_artifact(predictor: Any) -> dict[str, Any]:
    """Pull provenance from a predictor's ``metadata()`` helper if present.

    Falls back to bare class-name introspection so the report still
    carries a useful identity record for predictors that don't
    implement ``metadata()`` (e.g. the test shims).
    """
    cls_name = type(predictor).__name__
    meta_payload: dict[str, Any] = {}
    if callable(getattr(predictor, "metadata", None)):
        try:
            payload = predictor.metadata()
            if isinstance(payload, dict):
                meta_payload = payload
        except Exception:  # pylint: disable=broad-except
            meta_payload = {}
    return {
        "predictor_class": cls_name,
        "predictor_metadata": meta_payload,
    }


def evaluate(
    predictor: Any,
    dataset: Sequence[Mapping[str, Any]],
    split_name: str,
    labeled: list[Mapping[str, Any]] | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Single-pass evaluation of a ``RuntimePredictor``.

    Parameters
    ----------
    predictor : object
        Must implement ``predict_one(input, labeled=None) -> float``.
        ``metadata() -> dict`` is consulted when present for the
        ``artifact`` block of the report.
    dataset : Sequence[Mapping]
        Canonical rows from :mod:`peval_gold.data.normalize`. Each row
        must carry ``response``; rows with non-binary ``response`` are
        skipped (counted under ``n_skipped_non_binary``).
    split_name : str
        Free-text label echoed into the report.
    labeled : list[Mapping] | None
        Forwarded to ``predict_one(..., labeled=labeled)`` on the
        FIRST scored row only (matches the platform's first-call Platt
        fit semantics; the predictor caches the fit internally).
    ece_bins : int
        Number of equal-width bins for ECE. Default 10.

    Returns
    -------
    dict
        See module docstring for the schema.
    """
    timestamp = _now_utc_iso()

    if not dataset:
        return {
            "split_name": split_name,
            "n_examples": 0,
            "n_skipped_non_binary": 0,
            "metrics": dict(_METRIC_KEYS_ALL_NONE),
            "per_benchmark": {},
            "per_condition": {},
            "timing": {
                "predict_one_p50_ms": 0.0,
                "predict_one_p95_ms": 0.0,
                "predict_one_max_ms": 0.0,
                "n_predict_calls": 0,
            },
            "artifact": _predictor_artifact(predictor),
            "timestamp_utc": timestamp,
        }

    binary_rows: list[Mapping[str, Any]] = []
    skipped = 0
    binary_y: list[float] = []
    for r in dataset:
        y_bin = _coerce_response(r.get("response"))
        if y_bin is None:
            skipped += 1
            continue
        binary_rows.append(r)
        binary_y.append(y_bin)

    preds_p: list[float] = []
    timings_ms: list[float] = []
    first_call = True
    for r in binary_rows:
        t0 = time.perf_counter()
        p_one = predictor.predict_one(
            dict(r), labeled=list(labeled) if (first_call and labeled) else None
        )
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
        preds_p.append(float(p_one))
        first_call = False

    y_arr = np.asarray(binary_y, dtype=float)
    p_arr = np.asarray(preds_p, dtype=float)

    return {
        "split_name": split_name,
        "n_examples": int(len(binary_rows)),
        "n_skipped_non_binary": int(skipped),
        "metrics": _aggregate_metrics(y_arr, p_arr, n_bins=ece_bins),
        "per_benchmark": _grouped_summary(
            lambda r: r.get("benchmark", ""), binary_rows, y_arr, p_arr
        ),
        "per_condition": _grouped_summary(
            lambda r: r.get("condition", ""), binary_rows, y_arr, p_arr
        ),
        "timing": {
            "predict_one_p50_ms": _percentile_ms(timings_ms, 50.0),
            "predict_one_p95_ms": _percentile_ms(timings_ms, 95.0),
            "predict_one_max_ms": float(max(timings_ms)) if timings_ms else 0.0,
            "n_predict_calls": int(len(timings_ms)),
        },
        "artifact": _predictor_artifact(predictor),
        "timestamp_utc": timestamp,
    }


def evaluate_adaptive(
    predictor: Any,
    rounds: Iterable[tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]],
    split_name: str,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Multi-round adaptive-labeling evaluation.

    Each round is a ``(labeled, unlabeled)`` pair:

    - ``labeled`` is passed to ``predict_one(..., labeled=...)`` on the
      first call of the round so the predictor's Platt fit happens
      exactly once per round.
    - ``unlabeled`` is the set of rows actually scored; their NLL /
      MLL / Brier / ECE / AUC are computed per round and aggregated.

    Predictors that hold mutable per-round state (Platt cache,
    SimHash reservoir) get fresh instances per round if needed —
    callers can pass a factory or rely on the predictor's own
    ``reset()`` between rounds. For now, the evaluator does NOT
    auto-reset; the parent harness owns that policy.

    Returns
    -------
    dict
        ``{"split_name", "n_rounds", "n_examples" (aggregate),
        "metrics" (aggregate), "per_benchmark" (aggregate),
        "per_condition" (aggregate), "rounds": list[per-round dict],
        "artifact", "timestamp_utc"}``
    """
    timestamp = _now_utc_iso()
    per_round: list[dict[str, Any]] = []
    all_rows: list[Mapping[str, Any]] = []
    all_y: list[float] = []
    all_p: list[float] = []
    all_timings: list[float] = []
    total_skipped = 0

    for i, (labeled, unlabeled) in enumerate(rounds):
        report = evaluate(
            predictor,
            unlabeled,
            split_name=f"{split_name}::round{i}",
            labeled=list(labeled),
            ece_bins=ece_bins,
        )
        per_round.append(
            {
                "round_index": i,
                "n_labeled": len(labeled),
                "n_unlabeled": int(report["n_examples"]),
                # ``n_examples`` is a per-round alias for ``n_unlabeled``
                # (the predictor scores the unlabeled set; the labeled
                # set is just the Platt-fit input). Kept distinct so a
                # caller can grep either name and downstream code can
                # treat per-round and aggregate reports uniformly.
                "n_examples": int(report["n_examples"]),
                "n_skipped_non_binary": int(report["n_skipped_non_binary"]),
                "metrics": report["metrics"],
                "timing": report["timing"],
            }
        )

        # Re-collect (y, p) for aggregate computation. evaluate() doesn't
        # expose the raw arrays but we can re-derive them cheaply by
        # walking unlabeled with the same coerce / call scheme — except
        # that would double the encoder calls. The per-round report has
        # n / metrics; we reconstruct the aggregate from the raw
        # predictions by re-running predict_one on each unlabeled row
        # AGAIN, which is wasteful. Better: have evaluate() also return
        # y/p arrays under a private key.
        #
        # Trade-off: we keep evaluate()'s public schema clean (no raw
        # arrays leaking out) and instead re-run a cheap loop here that
        # asks the predictor for the same probabilities. The Platt fit
        # is cached after the first call of the round so re-runs only
        # hit the encoder cache, not the fit.
        for r in unlabeled:
            y_bin = _coerce_response(r.get("response"))
            if y_bin is None:
                total_skipped += 1
                continue
            all_rows.append(r)
            all_y.append(y_bin)

        # Pull predictions out of the per-round timing/metrics indirectly
        # by re-running predict_one — but only on the binary subset.
        # Cached encoder + cached Platt → cheap.
        for r in unlabeled:
            y_bin = _coerce_response(r.get("response"))
            if y_bin is None:
                continue
            t0 = time.perf_counter()
            all_p.append(float(predictor.predict_one(dict(r), labeled=None)))
            all_timings.append((time.perf_counter() - t0) * 1000.0)

    y_arr = np.asarray(all_y, dtype=float)
    p_arr = np.asarray(all_p, dtype=float)

    return {
        "split_name": split_name,
        "n_rounds": len(per_round),
        "n_examples": int(y_arr.size),
        "n_skipped_non_binary": int(total_skipped),
        "metrics": _aggregate_metrics(y_arr, p_arr, n_bins=ece_bins),
        "per_benchmark": _grouped_summary(
            lambda r: r.get("benchmark", ""), all_rows, y_arr, p_arr
        ) if all_rows else {},
        "per_condition": _grouped_summary(
            lambda r: r.get("condition", ""), all_rows, y_arr, p_arr
        ) if all_rows else {},
        "timing": {
            "predict_one_p50_ms": _percentile_ms(all_timings, 50.0),
            "predict_one_p95_ms": _percentile_ms(all_timings, 95.0),
            "predict_one_max_ms": float(max(all_timings)) if all_timings else 0.0,
            "n_predict_calls": int(len(all_timings)),
        },
        "rounds": per_round,
        "artifact": _predictor_artifact(predictor),
        "timestamp_utc": timestamp,
    }


__all__ = ["evaluate", "evaluate_adaptive"]
