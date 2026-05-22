"""peval-gold: a reusable framework for predictive-evaluation ML competitions.

This package exposes Protocol-based abstractions that decouple offline
laboratory experimentation from packaged hosted-runtime submissions:

- :class:`peval_gold.models.Predictor` / :class:`RuntimePredictor` — the two
  predictor shapes (offline batch vs. streaming one-row-at-a-time).
- :class:`peval_gold.calibration.Calibrator` — calibration Protocol with six
  concrete implementations (identity, intercept, temperature, Platt,
  regularized Platt, online).
- :class:`peval_gold.acquisition.AcquisitionPolicy` — adaptive-labeling
  scoring Protocol with seven concrete policies (random, stratified,
  hash-only, SimHash, reservoir-SimHash, uncertainty, current).
- :class:`peval_gold.eval.Evaluator` — single-pass + adaptive-round scoring
  over arbitrary Predictor implementations.
- :func:`peval_gold.eval.audit_candidate` — D-9 pre-submission stress-test
  gate (four sub-gates: probability distribution, packaged-runtime smoke,
  benchmark-heldout stress, calibration probes).

Data dependency
---------------

The default HF dataset reference is ``aims-foundations/measurement-db`` at
the pinned revision in :mod:`peval_gold.data.registry`. The dataset is
publicly licensed; clean checkouts work without authentication.

See the repo README for installation, quickstart, and usage examples.
"""

__version__ = "0.1.0"

# Public API re-exports. Submodule imports are lazy where possible to keep
# `import peval_gold` cheap on machines that only want the Protocol surface.
from peval_gold.acquisition.base import AcquisitionPolicy
from peval_gold.calibration.base import Calibrator
from peval_gold.eval.evaluator import evaluate, evaluate_adaptive
from peval_gold.eval.transfer_audit import AuditResult, audit_candidate
from peval_gold.models.base import Predictor, RuntimePredictor

__all__ = [
    "__version__",
    "AcquisitionPolicy",
    "Calibrator",
    "evaluate",
    "evaluate_adaptive",
    "AuditResult",
    "audit_candidate",
    "Predictor",
    "RuntimePredictor",
]
