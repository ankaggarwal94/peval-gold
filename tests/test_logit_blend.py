"""TDD red→green tests for the Batch 5 LogitBlend ensemble (S2B).

The LogitBlend is a logit-space linear combination of constituent
predictors::

    z = w0 + sum_i (w_i * logit(clip(p_i)))
    p_blend = sigmoid(z)

It is fit via L2-regularized logistic regression on a held-out validation
fold using ``scipy.optimize.minimize`` (NOT sklearn) so the offline
laboratory stays import-clean and the implementation could be vendored
into the submission ZIP in a later batch without picking up sklearn.

These tests are READ-ONLY against ``submission/``. They use trivial
constituent shims rather than the heavy CurrentNCF + EBPriors stack so
the test suite stays fast.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Trivial constituent shims
# ---------------------------------------------------------------------------


class _ConstPred:
    """Constituent that always predicts ``value``. Implements the minimal
    surface LogitBlend needs (``predict_proba`` + ``predict_one``).
    """

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def predict_proba(self, rows) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.full(len(rows), self._value, dtype=float)

    def predict_one(self, input, labeled=None) -> float:  # type: ignore[no-untyped-def]  # noqa: A002
        return float(self._value)


class _IdentityProbPred:
    """Constituent that echoes ``row['p_seed']`` so weight-fit tests can
    construct linearly-separable data on a single constituent."""

    def predict_proba(self, rows) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.asarray([float(r.get("p_seed", 0.5)) for r in rows], dtype=float)

    def predict_one(self, input, labeled=None) -> float:  # type: ignore[no-untyped-def]  # noqa: A002
        return float(input.get("p_seed", 0.5))


# ---------------------------------------------------------------------------
# 1. Single constituent at p=0.5 returns ~0.5 after a balanced fit
# ---------------------------------------------------------------------------


def test_logit_blend_one_constituent_identity_at_half_returns_half() -> None:
    """A single constituent that emits 0.5 contributes logit(0.5)=0. With
    balanced labels the fitted intercept should also land near 0, so the
    blend should sit near 0.5 for any input."""
    from peval_gold.models.ensemble import LogitBlend

    rng = np.random.default_rng(0)
    rows = []
    for _ in range(50):
        rows.append({"x": int(rng.integers(0, 999)), "response": 1.0})
    for _ in range(50):
        rows.append({"x": int(rng.integers(0, 999)), "response": 0.0})

    blend = LogitBlend([(_ConstPred(0.5), "id_half")])
    blend.fit(rows)

    pred = blend.predict_one({"x": 12345})
    assert isinstance(pred, float)
    assert math.isfinite(pred)
    assert 0.0 < pred < 1.0
    assert pred == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# 2. Two equally-weighted constituents at logits +x, -x average to 0.5
# ---------------------------------------------------------------------------


def test_logit_blend_two_constituents_equal_weights_averages_logits_not_probs() -> None:
    """With manually-set weights (0, 0.5, 0.5), the blend of constituents
    at p=0.9 and p=0.1 is sigmoid(0.5*logit(0.9) + 0.5*logit(0.1)) =
    sigmoid(0) = 0.5 — averaging in LOGIT space, not probability space.
    (Average of 0.9 and 0.1 in probability space is 0.5 too, but THIS
    test verifies the implementation never falls back to averaging in
    probability space by construction.)"""
    from peval_gold.models.ensemble import LogitBlend

    blend = LogitBlend([(_ConstPred(0.9), "p9"), (_ConstPred(0.1), "p1")])
    blend.set_weights(np.array([0.0, 0.5, 0.5]))

    pred = blend.predict_one({"x": 1})
    assert pred == pytest.approx(0.5, abs=1e-6)

    pred_two_rows = blend.predict_proba([{"x": 1}, {"x": 2}])
    np.testing.assert_allclose(pred_two_rows, np.array([0.5, 0.5]), atol=1e-6)


def test_logit_blend_weights_with_asymmetric_constituents_use_logit_space() -> None:
    """Manual weights (0, 1, 1) on p=0.9 and p=0.7 give sigmoid(logit(0.9)+logit(0.7)).
    Verifying explicitly to lock in logit-space combination semantics."""
    from peval_gold.models.ensemble import LogitBlend

    blend = LogitBlend([(_ConstPred(0.9), "p9"), (_ConstPred(0.7), "p7")])
    blend.set_weights(np.array([0.0, 1.0, 1.0]))

    z = math.log(0.9 / 0.1) + math.log(0.7 / 0.3)
    expected = 1.0 / (1.0 + math.exp(-z))
    assert blend.predict_one({"x": 1}) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. fit_blend_weights on linearly-separable logits recovers a reasonable signal
# ---------------------------------------------------------------------------


def test_fit_blend_weights_recovers_useful_signal_on_synthetic_data() -> None:
    """Predictor 1 is a near-perfect classifier (y = (p1 > 0.5)).
    Predictor 2 is pure noise. Weight for p1 should dominate."""
    from peval_gold.models.ensemble import fit_blend_weights

    rng = np.random.default_rng(0)
    n = 2000
    # Avoid 0/1 exactly so logit is finite.
    p1 = rng.uniform(low=1e-3, high=1.0 - 1e-3, size=n)
    p2 = rng.uniform(low=1e-3, high=1.0 - 1e-3, size=n)
    y = (p1 > 0.5).astype(float)

    p_matrix = np.column_stack([p1, p2])
    weights = fit_blend_weights(p_matrix, y, l2=0.1)

    assert weights.shape == (3,)
    assert math.isfinite(weights[0])
    assert math.isfinite(weights[1])
    assert math.isfinite(weights[2])
    # p1 should pick up positive weight much larger than the noise constituent
    assert weights[1] > 0
    assert abs(weights[1]) > 2 * abs(weights[2])


# ---------------------------------------------------------------------------
# 4. L2 caps weight magnitudes
# ---------------------------------------------------------------------------


def test_fit_blend_weights_l2_keeps_weights_bounded() -> None:
    """Even with a degenerately-separable single predictor, a strong L2
    penalty should keep every weight magnitude under 100."""
    from peval_gold.models.ensemble import fit_blend_weights

    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(low=1e-3, high=1.0 - 1e-3, size=n)
    y = (p > 0.5).astype(float)

    p_matrix = p.reshape(-1, 1)
    weights = fit_blend_weights(p_matrix, y, l2=10.0)

    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert np.all(np.abs(weights) <= 100.0)


# ---------------------------------------------------------------------------
# 5. predict_one matches predict_proba on a single row
# ---------------------------------------------------------------------------


def test_logit_blend_predict_one_matches_predict_proba_single_row() -> None:
    """LogitBlend's per-row path must produce the same number the batched
    path produces (within float tolerance) for any constituent whose
    ``predict_one`` and ``predict_proba`` agree on a single row."""
    from peval_gold.models.ensemble import LogitBlend

    blend = LogitBlend([(_ConstPred(0.7), "p7"), (_ConstPred(0.4), "p4")])
    blend.set_weights(np.array([0.1, 0.8, -0.5]))

    row = {"x": "anything"}
    one = blend.predict_one(row, labeled=None)
    batched = float(blend.predict_proba([row])[0])
    assert one == pytest.approx(batched, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. Empty constituents raises ValueError
# ---------------------------------------------------------------------------


def test_logit_blend_empty_constituents_raises_value_error() -> None:
    from peval_gold.models.ensemble import LogitBlend

    with pytest.raises(ValueError):
        LogitBlend([])


def test_fit_blend_weights_zero_predictors_raises_value_error() -> None:
    from peval_gold.models.ensemble import fit_blend_weights

    rng = np.random.default_rng(0)
    y = (rng.uniform(size=20) > 0.5).astype(float)
    p_matrix = np.zeros((20, 0))
    with pytest.raises(ValueError):
        fit_blend_weights(p_matrix, y, l2=1.0)


# ---------------------------------------------------------------------------
# 7. ensemble_calibration_check returns ece/slope/intercept
# ---------------------------------------------------------------------------


def test_ensemble_calibration_check_returns_ece_slope_intercept() -> None:
    from peval_gold.models.ensemble import ensemble_calibration_check

    rng = np.random.default_rng(0)
    n = 500
    p_blend = rng.uniform(low=1e-3, high=1.0 - 1e-3, size=n)
    y = (rng.uniform(size=n) < p_blend).astype(float)

    out = ensemble_calibration_check(p_blend, y)
    assert isinstance(out, dict)
    for k in ("ece", "slope", "intercept"):
        assert k in out
        assert math.isfinite(out[k])
    assert 0.0 <= out["ece"] <= 1.0


# ---------------------------------------------------------------------------
# 8. predict_proba clips constituents into safe logit range
# ---------------------------------------------------------------------------


def test_logit_blend_clips_constituent_predictions_to_safe_logit_range() -> None:
    """A constituent that emits 0.0 or 1.0 must NOT cause +/-inf logits.
    The blend output should remain finite and in the unit interval."""
    from peval_gold.models.ensemble import LogitBlend

    blend = LogitBlend([(_ConstPred(0.0), "zero"), (_ConstPred(1.0), "one")])
    blend.set_weights(np.array([0.0, 1.0, 1.0]))

    pred = blend.predict_one({"x": 1})
    proba = blend.predict_proba([{"x": 1}])
    assert math.isfinite(pred)
    assert 0.0 < pred < 1.0
    assert math.isfinite(proba[0])
    assert 0.0 < proba[0] < 1.0


# ---------------------------------------------------------------------------
# 9. LogitBlend.fit propagates predict_proba results from constituents
# ---------------------------------------------------------------------------


def test_logit_blend_fit_uses_constituent_predict_proba_on_train_rows() -> None:
    """When valid_rows is None, fit should use train_rows and call each
    constituent's ``predict_proba``. The fitted weight for the
    information-carrying constituent should be positive."""
    from peval_gold.models.ensemble import LogitBlend

    rng = np.random.default_rng(42)
    n = 600
    rows = []
    for _ in range(n):
        p_seed = float(rng.uniform(low=1e-3, high=1.0 - 1e-3))
        y = 1.0 if rng.uniform() < p_seed else 0.0
        rows.append({"p_seed": p_seed, "response": y})

    blend = LogitBlend([(_IdentityProbPred(), "id"), (_ConstPred(0.5), "noise")])
    blend.fit(rows)

    weights = blend.get_weights()
    assert weights.shape == (3,)
    assert math.isfinite(weights[0])
    assert weights[1] > 0  # the IdentityProb constituent is the signal


# ---------------------------------------------------------------------------
# 10. LogitBlend satisfies both protocols
# ---------------------------------------------------------------------------


def test_logit_blend_satisfies_predictor_and_runtime_predictor_protocols() -> None:
    from peval_gold.models.base import Predictor, RuntimePredictor
    from peval_gold.models.ensemble import LogitBlend

    blend = LogitBlend([(_ConstPred(0.5), "id")])
    assert isinstance(blend, Predictor)
    assert isinstance(blend, RuntimePredictor)
