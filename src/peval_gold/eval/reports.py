"""Report serializers for the gold-track local evaluator.

Three helpers:

- :func:`to_json` — pretty-printed JSON dump of a report dict.
- :func:`to_markdown` — human-readable Markdown summary with the
  headline metrics + per-benchmark + per-condition tables.
- :func:`compare_reports` — diff helper for "current vs challenger"
  comparisons. Returns the MLL / NLL / Brier / AUC / ECE deltas and a
  ``winner`` field (``"a"`` / ``"b"`` / ``"tie"``).

All helpers are pure functions of their input dicts so two calls with
identical input produce byte-identical output (no embedded timestamps
in the serialized payload itself; timestamps come from the report dict).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


def to_json(report: Mapping[str, Any], path: str) -> str:
    """Write ``report`` to ``path`` as pretty-printed JSON. Returns ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_safe)
    return path


def _json_safe(obj: Any) -> Any:
    """Coerce numpy scalars / sets / paths to JSON-friendly types."""
    try:
        import numpy as _np

        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _fmt(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if not math.isfinite(value):
            return "inf"
        return f"{value:.{decimals}f}"
    return str(value)


def to_markdown(report: Mapping[str, Any], path: str) -> str:
    """Write a human-readable Markdown summary of ``report`` to ``path``.

    Sections:

    1. Headline (split name, n_examples, timestamp, predictor class).
    2. Aggregate metrics table.
    3. Per-benchmark breakdown table.
    4. Per-condition breakdown table.
    5. Timing block.
    6. Adaptive per-round section (only when present in the report).
    """
    split = report.get("split_name", "?")
    n = report.get("n_examples", "?")
    ts = report.get("timestamp_utc", "?")
    artifact = report.get("artifact", {}) or {}
    pred_cls = artifact.get("predictor_class", "?")

    metrics = report.get("metrics", {}) or {}
    timing = report.get("timing", {}) or {}
    per_b = report.get("per_benchmark", {}) or {}
    per_c = report.get("per_condition", {}) or {}

    lines: list[str] = []
    lines.append(f"# Evaluation report — `{split}`")
    lines.append("")
    lines.append(f"- Predictor class: `{pred_cls}`")
    lines.append(f"- Examples scored: **{n}**")
    if report.get("n_skipped_non_binary") is not None:
        lines.append(f"- Non-binary rows skipped: {report['n_skipped_non_binary']}")
    if "n_rounds" in report:
        lines.append(f"- Adaptive rounds: {report['n_rounds']}")
    lines.append(f"- Timestamp (UTC): `{ts}`")
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(
        f"| Mean log-likelihood (mll, higher better) | "
        f"`{_fmt(metrics.get('mean_log_likelihood'))}` |"
    )
    lines.append(
        f"| Ordinary log loss (nll, lower better) | `{_fmt(metrics.get('ordinary_log_loss'))}` |"
    )
    lines.append(f"| Brier score (lower better) | `{_fmt(metrics.get('brier_score'))}` |")
    lines.append(
        f"| Expected calibration error (lower better) | "
        f"`{_fmt(metrics.get('expected_calibration_error'))}` |"
    )
    lines.append(f"| ROC AUC (higher better) | `{_fmt(metrics.get('auc'))}` |")
    lines.append("")

    lines.append("## Per-benchmark")
    lines.append("")
    if per_b:
        lines.append("| Benchmark | n | nll | mll | auc |")
        lines.append("|---|---:|---:|---:|---:|")
        for bench, stats in sorted(per_b.items(), key=lambda kv: -int(kv[1].get("n", 0))):
            lines.append(
                f"| `{bench}` | {stats.get('n')} | "
                f"`{_fmt(stats.get('nll'))}` | "
                f"`{_fmt(stats.get('mll'))}` | "
                f"`{_fmt(stats.get('auc'))}` |"
            )
    else:
        lines.append("(per_benchmark: empty)")
    lines.append("")

    lines.append("## Per-condition")
    lines.append("")
    if per_c:
        lines.append("| Condition | n | nll | mll | auc |")
        lines.append("|---|---:|---:|---:|---:|")
        for cond, stats in sorted(per_c.items(), key=lambda kv: -int(kv[1].get("n", 0))):
            lines.append(
                f"| `{cond}` | {stats.get('n')} | "
                f"`{_fmt(stats.get('nll'))}` | "
                f"`{_fmt(stats.get('mll'))}` | "
                f"`{_fmt(stats.get('auc'))}` |"
            )
    else:
        lines.append("(per_condition: empty)")
    lines.append("")

    lines.append("## Timing (`predict_one` wall-clock)")
    lines.append("")
    lines.append("| Percentile | ms |")
    lines.append("|---|---:|")
    lines.append(f"| p50 | `{_fmt(timing.get('predict_one_p50_ms'), 3)}` |")
    lines.append(f"| p95 | `{_fmt(timing.get('predict_one_p95_ms'), 3)}` |")
    lines.append(f"| max | `{_fmt(timing.get('predict_one_max_ms'), 3)}` |")
    lines.append(f"| n_predict_calls | {timing.get('n_predict_calls', 0)} |")
    lines.append("")

    if "rounds" in report and report["rounds"]:
        lines.append("## Adaptive per-round")
        lines.append("")
        lines.append("| Round | n_labeled | n_unlabeled | nll | mll | p50 ms |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in report["rounds"]:
            m = r.get("metrics", {}) or {}
            t = r.get("timing", {}) or {}
            lines.append(
                f"| {r.get('round_index', '?')} | "
                f"{r.get('n_labeled', '?')} | "
                f"{r.get('n_unlabeled', '?')} | "
                f"`{_fmt(m.get('ordinary_log_loss'))}` | "
                f"`{_fmt(m.get('mean_log_likelihood'))}` | "
                f"`{_fmt(t.get('predict_one_p50_ms'), 3)}` |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def compare_reports(
    report_a: Mapping[str, Any],
    report_b: Mapping[str, Any],
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Diff two report dicts on the headline metrics.

    Returns a small dict with per-metric deltas (b - a for "higher is
    better" metrics; a - b for "lower is better" metrics so the sign
    of the delta consistently means "challenger improvement"):

    - ``mean_log_likelihood_delta`` = mll_b - mll_a (positive ⇒ B wins)
    - ``ordinary_log_loss_delta``   = nll_a - nll_b (positive ⇒ B wins)
    - ``brier_score_delta``         = brier_a - brier_b (positive ⇒ B wins)
    - ``auc_delta``                  = auc_b - auc_a (positive ⇒ B wins)
    - ``expected_calibration_error_delta`` = ece_a - ece_b (positive ⇒ B wins)
    - ``winner`` ∈ {``"a"``, ``"b"``, ``"tie"``} based on the sign of
      the MLL delta within ``tol``.
    """
    metrics_a = report_a.get("metrics", {}) or {}
    metrics_b = report_b.get("metrics", {}) or {}

    def _safe(key_a: str, key_b: str = None, invert: bool = False) -> float | None:
        k_a = key_a
        k_b = key_b or key_a
        va = metrics_a.get(k_a)
        vb = metrics_b.get(k_b)
        if va is None or vb is None:
            return None
        return float(va - vb) if invert else float(vb - va)

    mll_delta = _safe("mean_log_likelihood")
    nll_delta = _safe("ordinary_log_loss", invert=True)
    brier_delta = _safe("brier_score", invert=True)
    auc_delta = _safe("auc")
    ece_delta = _safe("expected_calibration_error", invert=True)

    if mll_delta is None or abs(mll_delta) <= tol:
        winner = "tie"
    elif mll_delta > 0:
        winner = "b"
    else:
        winner = "a"

    return {
        "mean_log_likelihood_delta": mll_delta,
        "ordinary_log_loss_delta": nll_delta,
        "brier_score_delta": brier_delta,
        "auc_delta": auc_delta,
        "expected_calibration_error_delta": ece_delta,
        "winner": winner,
        "report_a_split": report_a.get("split_name"),
        "report_b_split": report_b.get("split_name"),
    }


__all__ = ["compare_reports", "to_json", "to_markdown"]
