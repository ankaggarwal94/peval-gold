"""TDD red→green tests for the gold-track Protocol surface.

Batch 1 deliverable (S1A). These tests cover the four ``typing.Protocol``
shapes defined under ``src/peval_gold/{models,calibration,acquisition}/``.

Each Protocol is tagged with :func:`typing.runtime_checkable` so that
:func:`isinstance` works against duck-typed shim classes — this is the
load-bearing property that lets the rest of the gold-track pipeline plug
in concrete implementations without inheritance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from peval_gold.acquisition.base import AcquisitionPolicy
from peval_gold.calibration.base import Calibrator
from peval_gold.models.base import Predictor, RuntimePredictor

# ---------------------------------------------------------------------------
# Trivial conforming shims
# ---------------------------------------------------------------------------


class ConstPredictor:
    """Returns 0.5 for every row. Satisfies :class:`Predictor`."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def fit(
        self,
        train_rows: Sequence[dict],
        valid_rows: Sequence[dict] | None = None,
    ) -> None:
        return None

    def predict_proba(self, rows: Sequence[dict]) -> np.ndarray:
        return np.full(len(rows), self.value, dtype=float)

    def save(self, path: str) -> None:
        return None

    @classmethod
    def load(cls, path: str) -> ConstPredictor:
        return cls()


class ConstRuntime:
    """Returns a constant probability. Satisfies :class:`RuntimePredictor`."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def predict_one(self, input: dict, labeled: list[dict] | None = None) -> float:
        return float(self.value)


class ConstCalibrator:
    """Pass-through calibrator. Satisfies :class:`Calibrator`."""

    def fit(self, y_true: np.ndarray, p_pred_or_logits: np.ndarray) -> None:
        return None

    def transform(self, p_pred_or_logits: np.ndarray) -> np.ndarray:
        return np.asarray(p_pred_or_logits, dtype=float)

    def save(self, path: str) -> None:
        return None

    @classmethod
    def load(cls, path: str) -> ConstCalibrator:
        return cls()


class ConstAcquisition:
    """Always returns 1.0. Satisfies :class:`AcquisitionPolicy`."""

    def score_one(self, input: dict) -> float:
        return 1.0

    def reset(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 1. runtime_checkable wiring
# ---------------------------------------------------------------------------


def test_protocols_are_runtime_checkable() -> None:
    """Each Protocol must be tagged ``@runtime_checkable`` so isinstance works."""
    for proto in (Predictor, RuntimePredictor, Calibrator, AcquisitionPolicy):
        assert getattr(proto, "_is_runtime_protocol", False), (
            f"{proto.__name__} must be decorated with @runtime_checkable"
        )


# ---------------------------------------------------------------------------
# 2-5. Positive duck-typing checks
# ---------------------------------------------------------------------------


def test_const_predictor_satisfies_predictor_protocol() -> None:
    assert isinstance(ConstPredictor(), Predictor)


def test_const_runtime_satisfies_runtime_predictor_protocol() -> None:
    assert isinstance(ConstRuntime(), RuntimePredictor)


def test_const_calibrator_satisfies_calibrator_protocol() -> None:
    assert isinstance(ConstCalibrator(), Calibrator)


def test_const_acquisition_satisfies_acquisition_policy_protocol() -> None:
    assert isinstance(ConstAcquisition(), AcquisitionPolicy)


# ---------------------------------------------------------------------------
# 6. Negative case: missing method does NOT satisfy Protocol
# ---------------------------------------------------------------------------


def test_class_missing_predict_proba_does_not_satisfy_predictor() -> None:
    class BrokenPredictor:
        def fit(
            self,
            train_rows: Sequence[dict],
            valid_rows: Sequence[dict] | None = None,
        ) -> None:
            return None

        def save(self, path: str) -> None:
            return None

        @classmethod
        def load(cls, path: str) -> BrokenPredictor:
            return cls()

    assert not isinstance(BrokenPredictor(), Predictor)


def test_class_missing_predict_one_does_not_satisfy_runtime_predictor() -> None:
    class BrokenRuntime:
        def something_else(self) -> None:
            return None

    assert not isinstance(BrokenRuntime(), RuntimePredictor)


def test_class_missing_transform_does_not_satisfy_calibrator() -> None:
    class BrokenCalibrator:
        def fit(self, y_true: np.ndarray, p_pred_or_logits: np.ndarray) -> None:
            return None

        def save(self, path: str) -> None:
            return None

        @classmethod
        def load(cls, path: str) -> BrokenCalibrator:
            return cls()

    assert not isinstance(BrokenCalibrator(), Calibrator)


def test_class_missing_reset_does_not_satisfy_acquisition_policy() -> None:
    class BrokenAcquisition:
        def score_one(self, input: dict) -> float:
            return 0.0

    assert not isinstance(BrokenAcquisition(), AcquisitionPolicy)


# ---------------------------------------------------------------------------
# Cross-cutting: trivial behavioral sanity (no implementation tested, just
# that the shims actually run through their declared method surface).
# ---------------------------------------------------------------------------


def test_const_predictor_predict_proba_returns_expected_shape() -> None:
    pred = ConstPredictor(value=0.42)
    out = pred.predict_proba([{"a": 1}, {"a": 2}, {"a": 3}])
    assert out.shape == (3,)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, np.full(3, 0.42))


def test_const_runtime_predict_one_returns_python_float() -> None:
    rt = ConstRuntime(value=0.7)
    out = rt.predict_one({"input_content": "x"}, labeled=None)
    assert isinstance(out, float)
    assert out == pytest.approx(0.7)


def test_const_calibrator_transform_is_identity() -> None:
    cal = ConstCalibrator()
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(cal.transform(p), p)


def test_const_acquisition_score_one_is_finite() -> None:
    acq = ConstAcquisition()
    score = acq.score_one({"x": 1})
    assert isinstance(score, float)
    assert score == pytest.approx(1.0)
