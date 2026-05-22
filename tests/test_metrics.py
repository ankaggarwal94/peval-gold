"""TDD red→green tests for ``peval_gold.eval.metrics``.

Batch 1 deliverable (S1A) for the gold-track workflow. These tests are
written BEFORE any implementation. They are expected to fail with
``ModuleNotFoundError`` during the red phase and to pass once the
minimal numpy-only metric module under ``src/peval_gold/eval/metrics.py``
is created.

NaN policy (documented and tested in :func:`test_clip_probability_nan_pass_through`):
``clip_probability`` propagates NaN unchanged. The downstream log-loss /
ECE helpers all coerce their inputs through ``np.asarray(..., dtype=float)``
and the user is responsible for not passing NaN labels; the metrics
themselves will surface any NaN as NaN in the output rather than silently
imputing a value. This matches the project's documented
"distinguishable defensive fallbacks" principle: NaN should not be
silently coerced to a plausible-but-wrong probability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from peval_gold.eval.metrics import (
    brier_score,
    clip_probability,
    expected_calibration_error,
    mean_log_likelihood,
    ordinary_log_loss,
    safe_logit,
    sigmoid,
)

# ---------------------------------------------------------------------------
# 1. clip_probability
# ---------------------------------------------------------------------------


def test_clip_probability_scalar_clamps_extremes_and_preserves_python_float() -> None:
    out_low = clip_probability(0.0)
    out_high = clip_probability(1.0)
    out_mid = clip_probability(0.5)

    assert isinstance(out_low, float)
    assert isinstance(out_high, float)
    assert isinstance(out_mid, float)
    assert out_low == pytest.approx(1e-4, abs=0.0)
    assert out_high == pytest.approx(1.0 - 1e-4, abs=0.0)
    assert out_mid == pytest.approx(0.5, abs=0.0)


def test_clip_probability_array_clamps_componentwise_and_preserves_ndarray() -> None:
    arr = np.array([-0.5, 0.0, 0.25, 0.75, 1.0, 1.5])
    out = clip_probability(arr, eps=1e-3)

    assert isinstance(out, np.ndarray)
    assert out.shape == arr.shape
    np.testing.assert_allclose(
        out,
        np.array([1e-3, 1e-3, 0.25, 0.75, 1.0 - 1e-3, 1.0 - 1e-3]),
    )


def test_clip_probability_respects_custom_eps() -> None:
    out = clip_probability(0.0, eps=0.05)
    assert out == pytest.approx(0.05)


def test_clip_probability_nan_pass_through() -> None:
    """Document the NaN policy: pass NaN through unchanged.

    The metrics module's contract is that NaN is the caller's problem to
    surface or scrub before passing in. The helper does not silently
    impute a plausible value.
    """
    out_scalar = clip_probability(float("nan"))
    assert math.isnan(out_scalar)

    out_arr = clip_probability(np.array([0.0, float("nan"), 1.0]))
    assert np.isnan(out_arr[1])
    assert out_arr[0] == pytest.approx(1e-4)
    assert out_arr[2] == pytest.approx(1.0 - 1e-4)


# ---------------------------------------------------------------------------
# 2. sigmoid + safe_logit identity
# ---------------------------------------------------------------------------


def test_sigmoid_does_not_overflow_on_large_magnitude_inputs() -> None:
    z = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    out = sigmoid(z)

    assert isinstance(out, np.ndarray)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[2] == pytest.approx(0.5, abs=1e-12)
    assert out[-1] == pytest.approx(1.0, abs=1e-12)


def test_sigmoid_scalar_input_returns_python_float() -> None:
    out = sigmoid(0.0)
    assert isinstance(out, float)
    assert out == pytest.approx(0.5)


def test_sigmoid_of_safe_logit_round_trips_to_clipped_probability() -> None:
    p = np.array([0.0, 1e-6, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    recovered = sigmoid(safe_logit(p))
    expected = clip_probability(p)

    np.testing.assert_allclose(recovered, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. ordinary_log_loss (hand-computed reference)
# ---------------------------------------------------------------------------


def test_ordinary_log_loss_matches_hand_computed_reference() -> None:
    y = np.array([1, 0, 1, 0], dtype=float)
    p = np.array([0.9, 0.1, 0.8, 0.2])
    # Hand-computed: -(log(0.9) + log(0.9) + log(0.8) + log(0.8)) / 4
    expected = -(math.log(0.9) + math.log(0.9) + math.log(0.8) + math.log(0.8)) / 4
    out = ordinary_log_loss(y, p)
    assert isinstance(out, float)
    assert out == pytest.approx(expected, abs=1e-6)
    # And the closed form is roughly 0.16425.
    assert out == pytest.approx(0.16425, abs=1e-4)


def test_ordinary_log_loss_clips_zero_and_one_predictions_to_finite_loss() -> None:
    y = np.array([1, 0], dtype=float)
    p = np.array([0.0, 1.0])
    out = ordinary_log_loss(y, p)
    assert math.isfinite(out)
    expected = -(math.log(1e-4) + math.log(1e-4)) / 2
    assert out == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. mean_log_likelihood sign relation
# ---------------------------------------------------------------------------


def test_mean_log_likelihood_is_negation_of_ordinary_log_loss() -> None:
    y = np.array([1, 0, 1, 0], dtype=float)
    p = np.array([0.9, 0.1, 0.8, 0.2])
    nll = ordinary_log_loss(y, p)
    mll = mean_log_likelihood(y, p)
    assert mll == pytest.approx(-nll, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. brier_score
# ---------------------------------------------------------------------------


def test_brier_score_matches_mean_squared_error_definition() -> None:
    y = np.array([1, 0, 1, 0, 1], dtype=float)
    p = np.array([0.9, 0.1, 0.8, 0.4, 0.6])
    expected = float(np.mean((y - p) ** 2))
    out = brier_score(y, p)
    assert isinstance(out, float)
    assert out == pytest.approx(expected, abs=1e-12)


def test_brier_score_handles_zero_and_one_without_clipping() -> None:
    y = np.array([1, 0], dtype=float)
    p = np.array([1.0, 0.0])
    out = brier_score(y, p)
    assert out == pytest.approx(0.0, abs=0.0)


def test_brier_score_raises_value_error_on_empty_input() -> None:
    with pytest.raises(ValueError):
        brier_score(np.array([], dtype=float), np.array([], dtype=float))


# ---------------------------------------------------------------------------
# 6. expected_calibration_error
# ---------------------------------------------------------------------------


def test_expected_calibration_error_zero_for_perfectly_calibrated_dataset() -> None:
    """All predictions agree exactly with empirical frequency per bin.

    Two bins: 4 predictions at 0.25 with one positive (empirical 0.25),
    4 predictions at 0.75 with three positives (empirical 0.75). Both
    bins contribute 0 to the ECE sum, so ECE = 0.
    """
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=float)
    p = np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75])
    out = expected_calibration_error(y, p, n_bins=10)
    assert isinstance(out, float)
    assert out == pytest.approx(0.0, abs=1e-12)


def test_expected_calibration_error_nonzero_for_miscalibrated_predictions() -> None:
    """Half the data is wrong with full confidence — ECE should be 0.5."""
    y = np.array([1, 1, 0, 0], dtype=float)
    p = np.array([1.0, 1.0, 1.0, 1.0])
    out = expected_calibration_error(y, p, n_bins=10)
    assert out == pytest.approx(0.5, abs=1e-9)


def test_expected_calibration_error_returns_zero_on_empty_input() -> None:
    out = expected_calibration_error(
        np.array([], dtype=float),
        np.array([], dtype=float),
        n_bins=10,
    )
    assert isinstance(out, float)
    assert out == 0.0


# ---------------------------------------------------------------------------
# 7. Length-mismatch validation
# ---------------------------------------------------------------------------


def test_ordinary_log_loss_raises_value_error_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        ordinary_log_loss(np.array([1, 0, 1]), np.array([0.5, 0.5]))


def test_mean_log_likelihood_raises_value_error_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        mean_log_likelihood(np.array([1, 0, 1]), np.array([0.5, 0.5]))


def test_brier_score_raises_value_error_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        brier_score(np.array([1, 0, 1]), np.array([0.5, 0.5]))


def test_expected_calibration_error_raises_value_error_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1, 0, 1]), np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# 8. Sign-convention sanity (matches a hosted-runtime display)
# ---------------------------------------------------------------------------


def test_sign_convention_near_perfect_predictions_drive_mll_close_to_zero() -> None:
    """Near-perfect predictions clipped at 1e-4 produce a tiny positive NLL
    and a near-zero MLL. The kit's display sign convention (higher closer
    to 0 is better) matches our ``mean_log_likelihood`` return value.
    See ``starting_kit/README.md:331-336``.
    """
    y = np.array([1, 0, 1, 0], dtype=float)
    p = np.array([1.0 - 1e-4, 1e-4, 1.0 - 1e-4, 1e-4])
    nll = ordinary_log_loss(y, p)
    mll = mean_log_likelihood(y, p)
    # Closed form: every contribution is -log(1 - 1e-4) ≈ 1.00005e-4,
    # so nll ≈ 1.0001e-4 and mll ≈ -1.0001e-4. The tolerance below admits
    # the +5e-9 second-order term while still pinning the order of
    # magnitude (both values are near zero, not near 1).
    assert nll == pytest.approx(1e-4, abs=1e-6)
    assert mll == pytest.approx(-1e-4, abs=1e-6)
    # Sign convention: mll is in the close-to-zero / negative regime,
    # nll is in the close-to-zero / positive regime; higher (less
    # negative) mll → better, matching the a hosted-runtime display.
    assert mll < 0.0
    assert nll > 0.0
    assert mll == pytest.approx(-nll, abs=1e-12)


# ---------------------------------------------------------------------------
# Cross-cutting: float32 / Python-float input handling
# ---------------------------------------------------------------------------


def test_metrics_handle_float32_inputs_and_return_python_float() -> None:
    y = np.array([1, 0, 1, 0], dtype=np.float32)
    p = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)

    for fn in (ordinary_log_loss, mean_log_likelihood, brier_score):
        out = fn(y, p)
        assert isinstance(out, float)

    out_ece = expected_calibration_error(y, p, n_bins=4)
    assert isinstance(out_ece, float)


def test_metrics_accept_python_lists_via_coercion() -> None:
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.8, 0.2]
    out = ordinary_log_loss(np.asarray(y, dtype=float), np.asarray(p, dtype=float))
    assert isinstance(out, float)
