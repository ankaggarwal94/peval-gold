"""Calibrator abstractions for the gold-track laboratory.

Re-exports the Batch-1 :class:`Calibrator` Protocol plus the six concrete
implementations landed in  (the calibration grid):

- :class:`IdentityCalibrator` — pass-through baseline.
- :class:`InterceptCalibrator` / :class:`PerCategoryInterceptCalibrator`
  — one-parameter bias calibrators (global or per-category).
- :class:`TemperatureCalibrator` — one-parameter ``sigmoid(logit/T)``.
- :class:`PlattCalibrator` / :class:`RegularizedPlattCalibrator` —
  two-parameter ``sigmoid(a*logit + b)`` (vanilla and L2-regularized).
- :class:`OnlineCalibrator` — per-round adapter that picks the best
  candidate by AIC-style score.
"""

from peval_gold.calibration.base import Calibrator
from peval_gold.calibration.identity import IdentityCalibrator
from peval_gold.calibration.intercept import (
    InterceptCalibrator,
    PerCategoryInterceptCalibrator,
)
from peval_gold.calibration.online import OnlineCalibrator
from peval_gold.calibration.platt import (
    PlattCalibrator,
    RegularizedPlattCalibrator,
)
from peval_gold.calibration.temperature import TemperatureCalibrator

__all__ = [
    "Calibrator",
    "IdentityCalibrator",
    "InterceptCalibrator",
    "OnlineCalibrator",
    "PerCategoryInterceptCalibrator",
    "PlattCalibrator",
    "RegularizedPlattCalibrator",
    "TemperatureCalibrator",
]
