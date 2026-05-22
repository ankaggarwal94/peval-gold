"""TDD red→green tests for Batch 6 calibration grid (S2C).

Six concrete calibrators are under test:

- :class:`peval_gold.calibration.identity.IdentityCalibrator` — trivial
  baseline; ``transform(p) = np.clip(p, 1e-4, 1-1e-4)``.
- :class:`peval_gold.calibration.intercept.InterceptCalibrator` — one-
  parameter bias on the logit; scipy L-BFGS-B with small L2 penalty.
- :class:`peval_gold.calibration.temperature.TemperatureCalibrator` —
  one-parameter temperature scaler ``sigmoid(logit/T)``; scipy L-BFGS-B
  with internal ``log T`` reparameterization to keep ``T > 0``.
- :class:`peval_gold.calibration.platt.PlattCalibrator` — two-parameter
  ``sigmoid(a*logit+b)``; **torch LBFGS with ``strong_wolfe`` line search,
  ``max_iter=50``**, initial ``(1.0, 0.0)``. Matches
  ``submission/model.py:_fit_platt`` byte-for-byte (regression test).
- :class:`peval_gold.calibration.platt.RegularizedPlattCalibrator` —
  same as Platt but adds an L2 penalty on ``(a - 1)^2 + b^2``.
- :class:`peval_gold.calibration.online.OnlineCalibrator` — adapter that
  picks the lowest-loss candidate after an AIC-style complexity penalty.

Identity-fallback rules (shared across InterceptCalibrator,
TemperatureCalibrator, PlattCalibrator, RegularizedPlattCalibrator):

- ``len(y) < 4`` → identity (b=0 / T=1 / (a,b)=(1,0)).
- All labels are the same class → identity.
- Numerical failure inside the inner solver → identity.

This file is intentionally FAST (~1-2s). No encoder load, no HF cache
needed; every test runs on small synthetic ``(y, p)`` tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Calibration grid imports. They live in modules created in Batch 6.
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
from peval_gold.eval.metrics import ordinary_log_loss

# ---------------------------------------------------------------------------
# Synthetic fixtures (cheap; reused across calibrators).
# ---------------------------------------------------------------------------


def _biased_dataset(n: int = 200, bias: float = 0.2, seed: int = 0) -> tuple:
    """Generate (y, p) where p is systematically shifted by ``bias`` from y.

    Roughly half the rows are positive. Predictions are ``y + bias`` with
    Gaussian noise, clipped to (0, 1). The optimal intercept therefore
    pushes the calibrated probability toward the lower side (negative b).
    """
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) > 0.5).astype(float)
    raw = y * 0.9 + (1.0 - y) * 0.1 + bias
    noise = rng.normal(scale=0.05, size=n)
    p = np.clip(raw + noise, 1e-3, 1.0 - 1e-3)
    return y, p


def _overconfident_dataset(n: int = 200, sharpen: float = 3.0, seed: int = 1) -> tuple:
    """Generate (y, p) where p is overconfident: well-ranked but too sharp.

    True logit is ``z ~ N(0, 1)``; label drawn from sigmoid(z). Predicted
    p uses ``sigmoid(sharpen * z)`` — same rank, sharper than calibrated.
    Optimal T > 1.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p_true = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < p_true).astype(float)
    p = 1.0 / (1.0 + np.exp(-sharpen * z))
    return y, p


def _well_conditioned_dataset(n: int = 1000, seed: int = 2) -> tuple:
    """Generate (y, p) with a known miscalibration that a 2-param Platt fits.

    True logit = z; observed score = 1.4 * z + 0.3 (slope and intercept
    off); label drawn from sigmoid(z). On large n, Platt's best (a, b)
    should be close to (1/1.4, -0.3/1.4) — i.e., a < 1 to undo the
    over-sharpening and b > 0 to undo the negative shift.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p_true = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < p_true).astype(float)
    score_logit = 1.4 * z + 0.3
    p = 1.0 / (1.0 + np.exp(-score_logit))
    return y, p


# ---------------------------------------------------------------------------
# 1. Calibrator Protocol conformance (runtime_checkable isinstance).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        IdentityCalibrator,
        InterceptCalibrator,
        TemperatureCalibrator,
        PlattCalibrator,
        RegularizedPlattCalibrator,
        OnlineCalibrator,
        PerCategoryInterceptCalibrator,
    ],
)
def test_calibrator_satisfies_protocol(cls) -> None:
    """Each calibrator must satisfy the runtime-checkable Protocol."""
    if cls is OnlineCalibrator:
        instance = cls([IdentityCalibrator()])
    elif cls is PerCategoryInterceptCalibrator:
        instance = cls(category_key="benchmark")
    else:
        instance = cls()
    assert isinstance(instance, Calibrator), (
        f"{cls.__name__} must implement the Calibrator Protocol"
    )


# ---------------------------------------------------------------------------
# 2. IdentityCalibrator
# ---------------------------------------------------------------------------


def test_identity_calibrator_is_identity_within_clip() -> None:
    cal = IdentityCalibrator()
    cal.fit(np.array([0, 1, 0, 1], dtype=float), np.array([0.1, 0.9, 0.2, 0.8]))
    p = np.array([0.0, 1e-6, 0.25, 0.5, 0.75, 1.0])
    out = cal.transform(p)
    expected = np.clip(p, 1e-4, 1.0 - 1e-4)
    np.testing.assert_allclose(out, expected, atol=0.0)


def test_identity_calibrator_save_load_round_trip(tmp_path: Path) -> None:
    """save() / load() classmethod must round-trip."""
    cal = IdentityCalibrator()
    pth = tmp_path / "identity.json"
    cal.save(str(pth))
    loaded = IdentityCalibrator.load(str(pth))
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(cal.transform(p), loaded.transform(p))


# ---------------------------------------------------------------------------
# 3. InterceptCalibrator
# ---------------------------------------------------------------------------


def test_intercept_calibrator_reduces_log_loss_on_biased_dataset() -> None:
    """A constant +0.2 bias in p should be largely undone by a single intercept."""
    y, p = _biased_dataset(n=500, bias=0.2, seed=0)
    cal = InterceptCalibrator(l2=0.01)
    cal.fit(y, p)
    p_cal = cal.transform(p)

    nll_before = ordinary_log_loss(y, p)
    nll_after = ordinary_log_loss(y, p_cal)
    assert nll_after < nll_before - 1e-3, (
        f"intercept did not reduce NLL (before={nll_before:.4f}, "
        f"after={nll_after:.4f}, b={cal.b:.3f})"
    )
    # Bias was positive, so optimal b should pull the score down.
    assert cal.b < -0.05


def test_intercept_calibrator_identity_fallback_on_small_label_set() -> None:
    y = np.array([0, 1, 0], dtype=float)
    p = np.array([0.5, 0.5, 0.5])
    cal = InterceptCalibrator()
    cal.fit(y, p)
    assert cal.b == pytest.approx(0.0, abs=0.0)


def test_intercept_calibrator_identity_fallback_on_single_class() -> None:
    y = np.zeros(10, dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    cal = InterceptCalibrator()
    cal.fit(y, p)
    assert cal.b == pytest.approx(0.0, abs=0.0)


# ---------------------------------------------------------------------------
# 4. TemperatureCalibrator
# ---------------------------------------------------------------------------


def test_temperature_calibrator_reduces_log_loss_on_overconfident_dataset() -> None:
    y, p = _overconfident_dataset(n=500, sharpen=3.0, seed=1)
    cal = TemperatureCalibrator(l2=0.01)
    cal.fit(y, p)
    p_cal = cal.transform(p)

    nll_before = ordinary_log_loss(y, p)
    nll_after = ordinary_log_loss(y, p_cal)
    assert nll_after < nll_before - 1e-3, (
        f"temperature did not reduce NLL (before={nll_before:.4f}, "
        f"after={nll_after:.4f}, T={cal.temperature:.3f})"
    )
    # Predictions were sharpened by 3x — optimal T should be > 1.
    assert cal.temperature > 1.5


def test_temperature_calibrator_keeps_temperature_strictly_positive() -> None:
    """Even on a tiny ill-posed dataset, T must stay > 0 to keep sigmoid finite."""
    y = np.array([0, 0, 1, 1, 1], dtype=float)
    p = np.array([0.4, 0.5, 0.5, 0.5, 0.6])
    cal = TemperatureCalibrator(l2=0.01)
    cal.fit(y, p)
    assert cal.temperature > 0.0
    out = cal.transform(np.array([0.5, 0.5, 0.5]))
    assert np.all(np.isfinite(out))


def test_temperature_calibrator_identity_fallback_on_small_label_set() -> None:
    y = np.array([0, 1, 0], dtype=float)
    p = np.array([0.5, 0.5, 0.5])
    cal = TemperatureCalibrator()
    cal.fit(y, p)
    assert cal.temperature == pytest.approx(1.0, abs=0.0)


def test_temperature_calibrator_identity_fallback_on_single_class() -> None:
    y = np.ones(10, dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    cal = TemperatureCalibrator()
    cal.fit(y, p)
    assert cal.temperature == pytest.approx(1.0, abs=0.0)


# ---------------------------------------------------------------------------
# 5. PlattCalibrator
# ---------------------------------------------------------------------------


def test_platt_calibrator_recovers_known_miscalibration() -> None:
    """Platt on a (1.4 * z + 0.3) miscalibration should land a < 1 and b < 0.

    To undo ``score_logit = 1.4 * z + 0.3`` and recover the original ``z``,
    the optimal Platt parameters are ``a = 1/1.4 ≈ 0.71`` and
    ``b = -0.3/1.4 ≈ -0.21`` (matching coefficients of ``a*score_logit+b = z``).
    """
    y, p = _well_conditioned_dataset(n=2000, seed=2)
    cal = PlattCalibrator()
    cal.fit(y, p)
    nll_before = ordinary_log_loss(y, p)
    nll_after = ordinary_log_loss(y, cal.transform(p))
    assert nll_after < nll_before - 1e-3
    assert cal.a < 1.0
    assert cal.b < 0.0


def test_platt_calibrator_matches_submission_model_py_byte_for_byte() -> None:
    """Regression test: PlattCalibrator must produce the same (a, b) as the
    submission/model.py LBFGS+strong_wolfe loop on a known toy dataset.

    The reference implementation lives at submission/model.py:235-276 and
    uses torch.optim.LBFGS with line_search_fn='strong_wolfe', max_iter=50,
    initial (a, b) = (1.0, 0.0). We reproduce that loop inline here so the
    test is self-contained and fails loudly if PlattCalibrator drifts.
    """
    import torch

    rng = np.random.default_rng(123)
    logits = rng.normal(scale=1.5, size=50)
    y = (rng.uniform(size=50) < 1.0 / (1.0 + np.exp(-logits))).astype(float)

    # Reference: byte-for-byte mirror of submission/model.py:_fit_platt.
    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets_t = torch.tensor(y, dtype=torch.float32)
    a_ref = torch.tensor(1.0, requires_grad=True)
    b_ref = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.LBFGS([a_ref, b_ref], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            a_ref * logits_t + b_ref, targets_t
        )
        loss.backward()
        return loss

    opt.step(closure)
    a_expected = float(a_ref.item())
    b_expected = float(b_ref.item())

    # Probabilities = sigmoid(logits); PlattCalibrator consumes probabilities
    # and converts via safe_logit internally. Since |logits| < ~5 the
    # round-trip is exact to float32 precision.
    p = 1.0 / (1.0 + np.exp(-logits))
    cal = PlattCalibrator()
    cal.fit(y, p)

    assert cal.a == pytest.approx(a_expected, abs=1e-5), (
        f"PlattCalibrator a={cal.a:.6f} drifts from submission reference a={a_expected:.6f}"
    )
    assert cal.b == pytest.approx(b_expected, abs=1e-5), (
        f"PlattCalibrator b={cal.b:.6f} drifts from submission reference b={b_expected:.6f}"
    )


def test_platt_calibrator_identity_fallback_on_small_label_set() -> None:
    y = np.array([0, 1, 0], dtype=float)
    p = np.array([0.5, 0.5, 0.5])
    cal = PlattCalibrator()
    cal.fit(y, p)
    assert cal.a == pytest.approx(1.0, abs=0.0)
    assert cal.b == pytest.approx(0.0, abs=0.0)


def test_platt_calibrator_identity_fallback_on_single_class() -> None:
    y = np.zeros(10, dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    cal = PlattCalibrator()
    cal.fit(y, p)
    assert cal.a == pytest.approx(1.0, abs=0.0)
    assert cal.b == pytest.approx(0.0, abs=0.0)


def test_platt_calibrator_save_load_round_trip(tmp_path: Path) -> None:
    y, p = _well_conditioned_dataset(n=200, seed=3)
    cal = PlattCalibrator()
    cal.fit(y, p)
    pth = tmp_path / "platt.json"
    cal.save(str(pth))
    loaded = PlattCalibrator.load(str(pth))
    test_p = np.array([0.1, 0.4, 0.6, 0.9])
    np.testing.assert_allclose(cal.transform(test_p), loaded.transform(test_p))


# ---------------------------------------------------------------------------
# 6. RegularizedPlattCalibrator
# ---------------------------------------------------------------------------


def test_regularized_platt_smaller_deviation_than_vanilla_platt_on_small_noisy_set() -> None:
    """On K=4 noisy labels Platt can swing wildly; the regularized variant
    pulls (a, b) toward (1, 0) so |a-1| + |b| is smaller.
    """
    # K=4 with noisy mapping. Logits and labels chosen to push vanilla
    # Platt away from (1, 0).
    rng = np.random.default_rng(11)
    logits = rng.normal(scale=0.8, size=4)
    # Half flip the labels so the fit is genuinely noisy.
    y = np.array([1, 0, 1, 0], dtype=float)
    p = 1.0 / (1.0 + np.exp(-logits))

    vanilla = PlattCalibrator()
    vanilla.fit(y, p)
    reg = RegularizedPlattCalibrator(l2=0.5)
    reg.fit(y, p)

    vanilla_dist = abs(vanilla.a - 1.0) + abs(vanilla.b)
    reg_dist = abs(reg.a - 1.0) + abs(reg.b)
    assert reg_dist < vanilla_dist, (
        f"regularized Platt should pull toward (1, 0). "
        f"vanilla={vanilla.a:.3f}, {vanilla.b:.3f} (dist={vanilla_dist:.3f}); "
        f"regularized={reg.a:.3f}, {reg.b:.3f} (dist={reg_dist:.3f})"
    )


def test_regularized_platt_identity_fallback_on_small_label_set() -> None:
    y = np.array([0, 1, 0], dtype=float)
    p = np.array([0.5, 0.5, 0.5])
    cal = RegularizedPlattCalibrator()
    cal.fit(y, p)
    assert cal.a == pytest.approx(1.0, abs=0.0)
    assert cal.b == pytest.approx(0.0, abs=0.0)


# ---------------------------------------------------------------------------
# 7. OnlineCalibrator
# ---------------------------------------------------------------------------


def test_online_calibrator_falls_back_to_identity_for_k_less_than_4() -> None:
    """Online adapter must short-circuit to identity when the labeled set is
    smaller than the per-calibrator min-label threshold."""
    y = np.array([0, 1, 0], dtype=float)
    p = np.array([0.3, 0.7, 0.4])
    online = OnlineCalibrator(
        [
            IdentityCalibrator(),
            InterceptCalibrator(),
            PlattCalibrator(),
            RegularizedPlattCalibrator(),
        ]
    )
    online.fit(y, p)
    assert online.last_choice == "IdentityCalibrator"
    np.testing.assert_allclose(
        online.transform(np.array([0.5])),
        np.clip(np.array([0.5]), 1e-4, 1.0 - 1e-4),
    )


def test_online_calibrator_falls_back_to_identity_for_single_class() -> None:
    """All-zeros labels → identity (Platt would blow up)."""
    y = np.zeros(20, dtype=float)
    p = np.linspace(0.01, 0.99, 20)
    online = OnlineCalibrator(
        [
            IdentityCalibrator(),
            InterceptCalibrator(),
            PlattCalibrator(),
        ]
    )
    online.fit(y, p)
    assert online.last_choice == "IdentityCalibrator"


def test_online_calibrator_picks_a_non_identity_on_well_conditioned_k100() -> None:
    """On 100 well-conditioned labels with a real (a, b) miscalibration,
    Platt or RegularizedPlatt should win after the AIC penalty."""
    y, p = _well_conditioned_dataset(n=100, seed=4)
    online = OnlineCalibrator(
        [
            IdentityCalibrator(),
            InterceptCalibrator(),
            TemperatureCalibrator(),
            PlattCalibrator(),
            RegularizedPlattCalibrator(),
        ]
    )
    online.fit(y, p)
    assert online.last_choice != "IdentityCalibrator", (
        "well-conditioned K=100 with a real (a, b) shift should beat identity after the AIC penalty"
    )


def test_online_calibrator_transform_matches_chosen_calibrator() -> None:
    """transform() must route through whichever calibrator won the fit."""
    y, p = _well_conditioned_dataset(n=200, seed=5)
    online = OnlineCalibrator(
        [
            IdentityCalibrator(),
            PlattCalibrator(),
        ]
    )
    online.fit(y, p)
    chosen_name = online.last_choice
    # Find the chosen calibrator in the list, fit it standalone, and
    # verify that the online dispatcher returns the same predictions.
    candidate_map = {type(c).__name__: c for c in online.candidates}
    chosen = candidate_map[chosen_name]
    # The chosen calibrator was fit during online.fit(); refit a fresh
    # instance the same way to verify.
    if chosen_name == "PlattCalibrator":
        ref = PlattCalibrator()
    else:
        ref = IdentityCalibrator()
    ref.fit(y, p)
    test_p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(online.transform(test_p), ref.transform(test_p), atol=1e-9)


# ---------------------------------------------------------------------------
# 8. PerCategoryInterceptCalibrator
# ---------------------------------------------------------------------------


def test_per_category_intercept_uses_global_for_small_categories() -> None:
    """A category with < 4 labels must use the global intercept (not its own)."""
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=float)
    p = np.array([0.6] * 10)
    categories = ["A", "A", "A", "A", "A", "A", "A", "B", "B", "B"]
    cal = PerCategoryInterceptCalibrator(category_key="benchmark")
    cal.fit_with_categories(y, p, categories)

    # A has 7 rows → fit its own intercept; B has 3 → fall back to global.
    assert "A" in cal.per_category_intercept
    assert "B" not in cal.per_category_intercept


def test_per_category_intercept_transform_uses_per_category_when_present() -> None:
    """transform_with_categories() must pick the per-category intercept when
    available, falling back to the global intercept otherwise.
    """
    y = np.concatenate([np.array([1, 1, 1, 0, 0, 0], dtype=float)] * 5)
    p = np.array([0.5] * 30)
    categories = (["A"] * 15) + (["B"] * 15)
    cal = PerCategoryInterceptCalibrator(category_key="benchmark", l2=0.01)
    cal.fit_with_categories(y, p, categories)
    # Apply to a test set that mixes A, B, and a new category C.
    test_p = np.array([0.5, 0.5, 0.5])
    test_cats = ["A", "B", "C"]
    out = cal.transform_with_categories(test_p, test_cats)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    # C should use the global intercept, identical to using global on row 0
    # (an unseen category).
    out_global = cal.transform(np.array([0.5]))
    assert out[2] == pytest.approx(float(out_global[0]), abs=1e-9)


# ---------------------------------------------------------------------------
# 9. Cross-cutting save/load + numerical sanity.
# ---------------------------------------------------------------------------


def test_intercept_calibrator_save_load_round_trip(tmp_path: Path) -> None:
    y, p = _biased_dataset(n=200, bias=0.2, seed=6)
    cal = InterceptCalibrator(l2=0.05)
    cal.fit(y, p)
    pth = tmp_path / "intercept.json"
    cal.save(str(pth))
    loaded = InterceptCalibrator.load(str(pth))
    test_p = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(cal.transform(test_p), loaded.transform(test_p), atol=1e-12)


def test_temperature_calibrator_save_load_round_trip(tmp_path: Path) -> None:
    y, p = _overconfident_dataset(n=200, sharpen=2.0, seed=7)
    cal = TemperatureCalibrator(l2=0.05)
    cal.fit(y, p)
    pth = tmp_path / "temperature.json"
    cal.save(str(pth))
    loaded = TemperatureCalibrator.load(str(pth))
    test_p = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(cal.transform(test_p), loaded.transform(test_p), atol=1e-12)


def test_all_calibrators_return_finite_probabilities_in_unit_interval() -> None:
    """No matter the input or fit state, transform must return finite [eps, 1-eps]."""
    y, p = _biased_dataset(n=100, bias=0.0, seed=8)
    test_p = np.array([0.0, 1e-9, 0.001, 0.5, 0.999, 1.0 - 1e-9, 1.0])

    instances = [
        IdentityCalibrator(),
        InterceptCalibrator(),
        TemperatureCalibrator(),
        PlattCalibrator(),
        RegularizedPlattCalibrator(),
    ]
    for cal in instances:
        cal.fit(y, p)
        out = cal.transform(test_p)
        assert np.all(np.isfinite(out)), f"{type(cal).__name__} produced non-finite"
        assert np.all(out >= 1e-4 - 1e-12), f"{type(cal).__name__} produced p < 1e-4 ({out.min()})"
        assert np.all(out <= 1.0 - 1e-4 + 1e-12), (
            f"{type(cal).__name__} produced p > 1-1e-4 ({out.max()})"
        )
