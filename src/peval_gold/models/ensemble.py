"""Logit-space linear-combination ensemble for gold-track predictors.

This module implements ``LogitBlend``, an L2-regularized logistic-
regression-style ensemble that takes a list of fitted constituent
``Predictor`` / ``RuntimePredictor`` objects and learns scalar blend
weights via :func:`scipy.optimize.minimize`. The blend is::

    z = w0 + sum_i (w_i * logit(clip(p_i)))
    p_blend = sigmoid(z)

The constituents themselves are **frozen** at instantiation: the
blender never retrains them. This is the natural way to combine a
strong-but-uncalibrated NCF (``CurrentNCF``) with a cheap base-rate
prior (``EBPriors``); each owns its own training story, and the blend
just learns how much to trust each one.

Why scipy + not sklearn
-----------------------

The submission ZIP currently ships with only ``torch`` and
``sentence-transformers``. Adding sklearn would be a 30+ MB dependency
that the platform pre-fetches into the container — wasted budget for
a 200-line logistic regression. ``scipy.optimize.minimize`` is part of
the offline laboratory stack already (numpy depends on it transitively
for many sub-modules); using it here keeps the offline + runtime paths
sharing the same minimization machinery if we ever vendor LogitBlend
into ``the wrapped submission's ``.

Calibration safety
------------------

Every constituent probability is clipped to ``[1e-4, 1 - 1e-4]`` before
the logit transform so a buggy constituent that emits exactly 0 or 1
cannot produce ±inf intermediate logits. This mirrors the
``the wrapped predict() module:[1e-4, 1-1e-4]`` clip and is the same
distinguishable-defensive-fallback discipline applied at the API
layer per ``/
distinguishable-defensive-fallbacks-2026-05-18.md``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Match the project-wide clip bound: the wrapped predict() module:_FINAL_CLIP_EPS = 1e-4.
_LOGIT_CLIP_EPS: float = 1e-4

# L2 penalty default. Conservative: with l2=1.0, weights of size ~1 are
# the natural scale; anything much larger pays a meaningful cost.
_DEFAULT_L2: float = 1.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _safe_logit(p: np.ndarray, eps: float = _LOGIT_CLIP_EPS) -> np.ndarray:
    """Clip-then-logit. Returns ``log(p_clip / (1 - p_clip))``."""
    p_arr = np.asarray(p, dtype=float)
    p_clip = np.clip(p_arr, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic; avoids overflow on extreme z."""
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr)
    pos = z_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    neg_exp = np.exp(z_arr[~pos])
    out[~pos] = neg_exp / (1.0 + neg_exp)
    return out


def fit_blend_weights(
    p_matrix: np.ndarray,
    y: np.ndarray,
    l2: float = _DEFAULT_L2,
) -> np.ndarray:
    """L2-regularized logistic regression on logits of ``p_matrix``.

    Parameters
    ----------
    p_matrix : np.ndarray, shape (n_samples, n_predictors)
        Per-row probabilities from each constituent. Values are
        clipped to ``[1e-4, 1 - 1e-4]`` before the logit transform so
        constituents that emit exactly 0/1 don't produce ±inf.
    y : np.ndarray, shape (n_samples,)
        Binary labels in ``{0.0, 1.0}``.
    l2 : float
        Ridge penalty on every coefficient (intercept + per-predictor
        weights). Default ``1.0``. Larger ⇒ smaller weights.

    Returns
    -------
    np.ndarray, shape (n_predictors + 1,)
        ``[intercept, w_1, ..., w_n]``. Every entry is finite. The
        spec test caps ``|w|`` at 100 for the default-l2 path.

    Raises
    ------
    ValueError
        If ``p_matrix`` has no columns (n_predictors == 0) or if
        ``p_matrix`` and ``y`` lengths disagree.
    """
    p_matrix = np.asarray(p_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    if p_matrix.ndim != 2:
        raise ValueError(
            f"p_matrix must be 2-D (n_samples, n_predictors); got shape {p_matrix.shape}"
        )
    n_samples, n_predictors = p_matrix.shape
    if n_predictors == 0:
        raise ValueError("fit_blend_weights requires at least one predictor column")
    if y.shape != (n_samples,):
        raise ValueError(
            f"y shape {y.shape} does not match p_matrix rows {n_samples}"
        )
    if n_samples == 0:
        raise ValueError("fit_blend_weights requires at least one sample")

    logits = _safe_logit(p_matrix)
    # Design matrix: prepend a column of ones for the intercept.
    X = np.hstack([np.ones((n_samples, 1)), logits])

    # scipy.optimize.minimize (BFGS). L-BFGS-B is equally fine but BFGS
    # gives access to the inverse Hessian which can be useful for
    # downstream diagnostics; we don't need that here so any choice works.
    from scipy.optimize import minimize

    def loss_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ w
        # Stable log-loss + gradient (binary cross-entropy with logits).
        # loss = mean(log(1 + exp(-y*z))); using bce form for stability:
        #   loss = mean(log1p(exp(z)) - y*z)
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        p = _sigmoid(z)
        grad = X.T @ (p - y) / n_samples
        # L2 penalty: applies to EVERY coefficient (intercept included).
        # The spec didn't carve out the intercept, and bounding the
        # intercept too is helpful for tightly-bounded predictors.
        loss = loss + 0.5 * l2 * float(np.dot(w, w)) / n_samples
        grad = grad + l2 * w / n_samples
        return loss, grad

    w0 = np.zeros(n_predictors + 1, dtype=float)
    res = minimize(
        loss_and_grad,
        w0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-7},
    )
    weights = np.asarray(res.x, dtype=float)
    if not np.all(np.isfinite(weights)):
        # Defensive: if the optimizer drifts into NaN territory (it
        # shouldn't with BCE + L2), fall back to zeros (intercept-only
        # uninformative blend). Better an honest fallback than a
        # silent NaN propagating into a the hosted runtime submission.
        weights = np.zeros_like(weights)
    return weights


def ensemble_calibration_check(
    p_blend: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Return ``{ece, slope, intercept}`` for an already-blended prediction.

    ``slope`` and ``intercept`` come from a 1-D Platt-style fit:
    ``logit(p_calibrated) = slope * logit(p_blend) + intercept``. A
    perfectly-calibrated blend has slope ≈ 1 and intercept ≈ 0.

    ``ece`` is the equal-width-binning expected calibration error
    (10 bins) — same convention as
    :func:`peval_gold.eval.metrics.expected_calibration_error`.
    """
    p_blend = np.asarray(p_blend, dtype=float)
    y = np.asarray(y, dtype=float)
    if p_blend.shape != y.shape:
        raise ValueError(
            f"shape mismatch: p_blend {p_blend.shape} vs y {y.shape}"
        )
    n = p_blend.size
    if n == 0:
        return {"ece": 0.0, "slope": 1.0, "intercept": 0.0}

    # Reuse fit_blend_weights with a single-predictor column (p_blend) to
    # get (intercept, slope) for free. Use a tiny l2 so this is
    # essentially unregularized Platt.
    weights = fit_blend_weights(
        p_blend.reshape(-1, 1), y, l2=1e-6
    )
    intercept = float(weights[0])
    slope = float(weights[1])

    # ECE via equal-width binning (10 bins).
    edges = np.linspace(0.0, 1.0, 11)
    edges[-1] = np.nextafter(1.0, 2.0)
    bin_ids = np.digitize(p_blend, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, 9)
    ece = 0.0
    for b in range(10):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_p = float(p_blend[mask].mean())
        bin_y = float(y[mask].mean())
        ece += (count / n) * abs(bin_p - bin_y)

    return {
        "ece": float(ece),
        "slope": slope,
        "intercept": intercept,
    }


# ---------------------------------------------------------------------------
# LogitBlend predictor
# ---------------------------------------------------------------------------


class LogitBlend:
    """Logit-space linear blend of frozen constituent predictors.

    Implements both :class:`peval_gold.models.base.Predictor` and
    :class:`peval_gold.models.base.RuntimePredictor`.

    ``constituents`` is a list of ``(predictor, name)`` tuples. The
    blender NEVER retrains the constituents. ``fit`` only learns the
    ``(n_predictors + 1)``-dim weight vector that combines their
    predictions.

    Per-row contract: ``predict_one`` calls each constituent's
    ``predict_one(input, labeled)`` (forwarding the ``labeled`` set so
    a stateful constituent like the current Platt-fitting NCF can do
    its first-call calibration), then blends.
    """

    def __init__(self, constituents: Sequence[tuple[Any, str]]) -> None:
        if not constituents:
            raise ValueError(
                "LogitBlend requires at least one constituent predictor"
            )
        self._constituents: list[tuple[Any, str]] = list(constituents)
        # Default weights: intercept=0, equal-and-positive per-predictor
        # weights summing to 1. This makes the predict_one identity hold
        # immediately after construction without a fit call (useful for
        # the equal-weights tests and for sanity-checking the blend
        # before fit_blend_weights() runs).
        n = len(self._constituents)
        self._weights = np.concatenate([[0.0], np.full(n, 1.0 / n)])

    # ----- Protocol: Predictor -----------------------------------------

    def fit(
        self,
        train_rows: Sequence[Mapping[str, Any]],
        valid_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Learn blend weights via L2-regularized logistic regression.

        When ``valid_rows`` is provided, fit on the validation set (the
        standard out-of-fold pattern that prevents the blender from
        overfitting). When ``valid_rows is None``, fit on ``train_rows``.

        Per-row labels come from ``row["response"]`` and must be in
        ``{0.0, 1.0}``. Non-binary rows are silently dropped (matches
        the D-7 binarize-drop policy at the data layer).
        """
        fit_rows = list(valid_rows) if valid_rows is not None else list(train_rows)
        if not fit_rows:
            raise ValueError("LogitBlend.fit needs at least one training row")

        # Drop non-binary rows.
        binary_rows: list[Mapping[str, Any]] = []
        y_vals: list[float] = []
        for r in fit_rows:
            resp = r.get("response")
            if resp is None:
                continue
            if isinstance(resp, bool):
                y_vals.append(1.0 if resp else 0.0)
                binary_rows.append(r)
                continue
            if not isinstance(resp, (int, float)):
                continue
            fv = float(resp)
            if math.isnan(fv) or fv not in (0.0, 1.0):
                continue
            y_vals.append(fv)
            binary_rows.append(r)

        if not binary_rows:
            raise ValueError(
                "LogitBlend.fit: 0 usable training rows after binary filtering"
            )

        p_matrix = self._collect_constituent_predictions(binary_rows)
        y = np.asarray(y_vals, dtype=float)
        self._weights = fit_blend_weights(p_matrix, y, l2=_DEFAULT_L2)

    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Run each constituent's ``predict_proba`` and blend in logit space."""
        if not rows:
            return np.array([], dtype=np.float64)
        p_matrix = self._collect_constituent_predictions(rows)
        return self._blend(p_matrix)

    def predict_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
        labeled: list[dict] | None = None,
    ) -> float:
        """Per-row blend mirroring the per-call ``predict()`` shape.

        ``labeled`` is forwarded to each constituent's ``predict_one`` so
        a Platt-fitting constituent (like ``CurrentNCF``) can do its
        first-call calibration. The result is the blended probability,
        clipped to ``[1e-4, 1 - 1e-4]``.
        """
        per_pred = np.empty((1, len(self._constituents)), dtype=float)
        for i, (pred, _) in enumerate(self._constituents):
            per_pred[0, i] = float(pred.predict_one(input, labeled=labeled))
        blended = self._blend(per_pred)
        return float(blended[0])

    def save(self, path: str | Path) -> None:
        """Persist the blend weights + constituent names as compact JSON.

        Does NOT serialize the constituents themselves — the consumer of
        :meth:`load` must re-pass them. Constituents are typically
        complex objects (NCFHead + encoder + Platt fit; or a fitted
        EBPriors lookup table) that own their own ``save``/``load``.
        """
        payload = {
            "__version__": 1,
            "weights": self._weights.tolist(),
            "constituent_names": [n for _, n in self._constituents],
            "l2_default": _DEFAULT_L2,
            "logit_clip_eps": _LOGIT_CLIP_EPS,
        }
        Path(path).write_text(json.dumps(payload, separators=(",", ":")))

    @classmethod
    def load(
        cls,
        path: str | Path,
        constituents: Sequence[tuple[Any, str]] | None = None,
    ) -> "LogitBlend":
        """Restore a LogitBlend.

        Because constituents are not serialized (see :meth:`save`), the
        caller must supply ``constituents`` matching the original
        ordering. The classmethod verifies the names line up.
        """
        if constituents is None:
            raise ValueError(
                "LogitBlend.load requires constituents to be passed in; "
                "they are not serialized inside the blend's JSON file."
            )
        payload = json.loads(Path(path).read_text())
        names_on_disk = payload.get("constituent_names", [])
        names_passed = [n for _, n in constituents]
        if names_on_disk != names_passed:
            raise ValueError(
                f"constituent name mismatch: saved {names_on_disk!r} "
                f"vs passed {names_passed!r}"
            )
        obj = cls(constituents)
        obj._weights = np.asarray(payload["weights"], dtype=float)
        return obj

    # ----- Introspection / mutator helpers -----------------------------

    def get_weights(self) -> np.ndarray:
        """Return a copy of the blend weights ``[intercept, w_1, ..., w_n]``."""
        return np.array(self._weights, dtype=float)

    def set_weights(self, weights: np.ndarray) -> None:
        """Replace the blend weights. Validates shape ``(n_constituents + 1,)``."""
        weights = np.asarray(weights, dtype=float)
        expected = len(self._constituents) + 1
        if weights.shape != (expected,):
            raise ValueError(
                f"weights must be shape ({expected},); got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contain non-finite values")
        self._weights = weights

    def metadata(self) -> dict[str, Any]:
        """JSON-serializable provenance snapshot consumed by the evaluator."""
        constituents_meta: list[dict[str, Any]] = []
        for pred, name in self._constituents:
            payload = {"name": name, "class": type(pred).__name__}
            if callable(getattr(pred, "metadata", None)):
                try:
                    inner = pred.metadata()
                    if isinstance(inner, dict):
                        payload["metadata"] = inner
                except Exception:  # pylint: disable=broad-except
                    payload["metadata"] = None
            constituents_meta.append(payload)
        return {
            "class": "LogitBlend",
            "constituents": constituents_meta,
            "weights": self._weights.tolist(),
            "logit_clip_eps": _LOGIT_CLIP_EPS,
        }

    # ----- Internals ---------------------------------------------------

    def _collect_constituent_predictions(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        """Run each constituent's batched ``predict_proba`` over ``rows``.

        Returns a ``(N, n_predictors)`` numpy array suitable for
        :func:`fit_blend_weights` and :meth:`_blend`.
        """
        n = len(rows)
        n_pred = len(self._constituents)
        out = np.empty((n, n_pred), dtype=float)
        for i, (pred, _) in enumerate(self._constituents):
            preds = np.asarray(pred.predict_proba(rows), dtype=float)
            if preds.shape != (n,):
                raise ValueError(
                    f"constituent #{i} predict_proba returned shape "
                    f"{preds.shape}; expected ({n},)"
                )
            out[:, i] = preds
        return out

    def _blend(self, p_matrix: np.ndarray) -> np.ndarray:
        """Apply ``sigmoid(intercept + per-predictor logits @ weights)`` + clip."""
        if p_matrix.shape[1] != len(self._constituents):
            raise ValueError(
                f"p_matrix column count {p_matrix.shape[1]} does not match "
                f"len(constituents) {len(self._constituents)}"
            )
        logits = _safe_logit(p_matrix)
        z = self._weights[0] + logits @ self._weights[1:]
        p = _sigmoid(z)
        return np.clip(p, _LOGIT_CLIP_EPS, 1.0 - _LOGIT_CLIP_EPS)


__all__ = [
    "LogitBlend",
    "ensemble_calibration_check",
    "fit_blend_weights",
]
