"""Evaluation primitives for the gold-track laboratory.

Exports:

- :mod:`peval_gold.eval.metrics` — numpy-only metric helpers (clip,
  sigmoid, safe_logit, ordinary_log_loss, mean_log_likelihood,
  brier_score, expected_calibration_error). .
- :mod:`peval_gold.eval.evaluator` — single-pass + adaptive-round
  ``evaluate`` / ``evaluate_adaptive`` over ``RuntimePredictor``
  instances. .
- :mod:`peval_gold.eval.reports` — JSON + Markdown serializers +
  ``compare_reports`` helper for current-vs-challenger diffs. .
"""

from peval_gold.eval.evaluator import evaluate, evaluate_adaptive
from peval_gold.eval.metrics import (
    brier_score,
    clip_probability,
    expected_calibration_error,
    mean_log_likelihood,
    ordinary_log_loss,
    safe_logit,
    sigmoid,
)
from peval_gold.eval.reports import compare_reports, to_json, to_markdown
from peval_gold.eval.transfer_audit import AuditResult, audit_candidate

__all__ = [
    "AuditResult",
    "audit_candidate",
    "brier_score",
    "clip_probability",
    "compare_reports",
    "evaluate",
    "evaluate_adaptive",
    "expected_calibration_error",
    "mean_log_likelihood",
    "ordinary_log_loss",
    "safe_logit",
    "sigmoid",
    "to_json",
    "to_markdown",
]
