"""Platt scaling calibrators (vanilla and L2-regularized).

Two flavors:

- :class:`PlattCalibrator` — the textbook 2-parameter Platt scaler
  ``p_cal = sigmoid(a * logit(p) + b)`` minimizing log-loss via
  ``torch.optim.LBFGS`` with ``line_search_fn="strong_wolfe"`` and
  ``max_iter=50``. **This is a byte-for-byte mirror of
  ``the wrapped predict() module:_fit_platt``** (lines 235-276); a regression
  test in ``tests/test_gold_calibration.py`` locks the equivalence in.
- :class:`RegularizedPlattCalibrator` — same algorithm but adds an
  L2 penalty ``l2 * ((a - 1)^2 + b^2)`` (default ``l2 = 0.1``). The
  regularizer pulls the fit toward identity ``(a, b) = (1, 0)``,
  which is exactly the behavior we want on the tiny K=5 / K=80
  labeled batches the platform reveals each round (per the kit's
  ``starting_kit/README.md:268-271`` K=5-per-data-category policy).

Identity-fallback rules (mirror the wrapped predict() module:_fit_platt):

- ``len(y) < 4`` → ``(a, b) = (1, 0)``.
- All labels are the same class → ``(a, b) = (1, 0)``.
- LBFGS produces ``|a| >= 1e6`` or ``|b| >= 1e6`` or NaN → fallback.

Why torch (and not scipy) here:

- The submission's runtime path uses torch.optim.LBFGS exactly. Using
  the same optimizer keeps the regression test byte-for-byte and
  keeps the offline grid honest about what shipping a calibration
  change would actually compute on-device.
- torch is already an explicit submission dependency; importing it
  here adds zero new runtime weight.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_CLIP_EPS = 1e-4
_MIN_LABELS = 4
_A_BOUND = 1e6
_B_BOUND = 1e6
_MAX_ITER = 50


# ---------------------------------------------------------------------------
# Numpy helpers (used only for the transform path; fit is torch).
# ---------------------------------------------------------------------------


def _safe_logit(p: np.ndarray, eps: float = _CLIP_EPS) -> np.ndarray:
    p_clip = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    pos = z >= 0
    out = np.empty_like(z, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg_exp = np.exp(z[~pos])
    out[~pos] = neg_exp / (1.0 + neg_exp)
    return out


def _identity_fallback_required(y: np.ndarray) -> bool:
    if y.size < _MIN_LABELS:
        return True
    return np.unique(y).size < 2


# ---------------------------------------------------------------------------
# Vanilla Platt (regression test against the wrapped predict() module).
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """2-parameter Platt scaler — exact mirror of the wrapped predict() module logic.

    Parameters
    ----------
    eps : float
        Clip floor. Default ``1e-4``.
    max_iter : int
        Max LBFGS iterations. Default ``50`` (matches submission).
    """

    def __init__(self, eps: float = _CLIP_EPS, max_iter: int = _MAX_ITER) -> None:
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.a: float = 1.0
        self.b: float = 0.0
        self.converged: bool = True

    def fit(self, y_true: np.ndarray, p_pred_or_logits: np.ndarray) -> None:
        """Fit ``(a, b)`` via torch LBFGS+strong_wolfe — mirrors submission.

        ``p_pred_or_logits`` is treated as PROBABILITIES; the inverse logit
        (``safe_logit``) is taken internally so the scaler input matches
        ``the wrapped predict() module:_fit_platt``'s ``logit`` variable. For
        probabilities produced by ``sigmoid(z)`` with ``|z| < ~5``, the
        ``safe_logit(sigmoid(z)) == z`` identity holds to float64
        precision (~1e-12).
        """
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(p_pred_or_logits, dtype=float)
        if y.shape != p.shape:
            raise ValueError(f"y_true shape {y.shape} != p_pred_or_logits shape {p.shape}")
        if _identity_fallback_required(y):
            self.a, self.b = 1.0, 0.0
            self.converged = True
            return

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is project dep
            # Distinguishable defensive fallback: identity (1, 0) rather
            # than a 0.5 sentinel; the caller's transform() will return
            # the un-calibrated probability.
            self.a, self.b = 1.0, 0.0
            self.converged = False
            raise RuntimeError(
                "PlattCalibrator requires torch; it is a project dependency."
            ) from exc

        # Convert probabilities → logits and run the byte-for-byte
        # equivalent of the wrapped predict() module:_fit_platt.
        logits = _safe_logit(p, eps=self.eps)
        logits_t = torch.tensor(logits, dtype=torch.float32)
        targets_t = torch.tensor(y, dtype=torch.float32)

        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        opt = torch.optim.LBFGS(
            [a, b],
            lr=1.0,
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(a * logits_t + b, targets_t)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            a_val, b_val = float(a.item()), float(b.item())
            if not (
                -_A_BOUND < a_val < _A_BOUND
                and -_B_BOUND < b_val < _B_BOUND
                and math.isfinite(a_val)
                and math.isfinite(b_val)
            ):
                self.a, self.b = 1.0, 0.0
                self.converged = False
                return
            self.a, self.b = a_val, b_val
            self.converged = True
        except Exception:  # pylint: disable=broad-except
            # See the NOTE in the wrapped predict() module:226-234 — the inner
            # Platt-fallback returns identity (un-calibrated forward),
            # not a 0.5 sentinel. The outer evaluator's contract is
            # preserved because (a=1, b=0) means the un-calibrated
            # probability is returned via the normal forward.
            self.a, self.b = 1.0, 0.0
            self.converged = False

    def transform(self, p_pred_or_logits: np.ndarray) -> np.ndarray:
        p = np.asarray(p_pred_or_logits, dtype=float)
        z = _safe_logit(p, eps=self.eps)
        return np.clip(_sigmoid(self.a * z + self.b), self.eps, 1.0 - self.eps)

    def save(self, path: str) -> None:
        payload = {
            "class": "PlattCalibrator",
            "a": self.a,
            "b": self.b,
            "eps": self.eps,
            "max_iter": self.max_iter,
            "converged": self.converged,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str) -> PlattCalibrator:
        payload = json.loads(Path(path).read_text())
        cal = cls(
            eps=float(payload.get("eps", _CLIP_EPS)),
            max_iter=int(payload.get("max_iter", _MAX_ITER)),
        )
        cal.a = float(payload.get("a", 1.0))
        cal.b = float(payload.get("b", 0.0))
        cal.converged = bool(payload.get("converged", True))
        return cal


# ---------------------------------------------------------------------------
# Regularized Platt (better for small K=5 labeled sets).
# ---------------------------------------------------------------------------


class RegularizedPlattCalibrator:
    """Platt with L2 shrinkage to identity ``(a, b) = (1, 0)``.

    Loss minimized: ``BCE(a * logit + b, y) + l2 * ((a - 1)^2 + b^2)``.

    Parameters
    ----------
    l2 : float
        Strength of the L2 penalty. Default ``0.1``. Higher values pull
        toward identity (cheap when the labeled set is tiny).
    eps : float
        Clip floor. Default ``1e-4``.
    max_iter : int
        Max LBFGS iterations. Default ``50`` (matches vanilla Platt).
    """

    def __init__(
        self,
        l2: float = 0.1,
        eps: float = _CLIP_EPS,
        max_iter: int = _MAX_ITER,
    ) -> None:
        self.l2 = float(l2)
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.a: float = 1.0
        self.b: float = 0.0
        self.converged: bool = True

    def fit(self, y_true: np.ndarray, p_pred_or_logits: np.ndarray) -> None:
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(p_pred_or_logits, dtype=float)
        if y.shape != p.shape:
            raise ValueError(f"y_true shape {y.shape} != p_pred_or_logits shape {p.shape}")
        if _identity_fallback_required(y):
            self.a, self.b = 1.0, 0.0
            self.converged = True
            return

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is project dep
            self.a, self.b = 1.0, 0.0
            self.converged = False
            raise RuntimeError("RegularizedPlattCalibrator requires torch.") from exc

        logits = _safe_logit(p, eps=self.eps)
        logits_t = torch.tensor(logits, dtype=torch.float32)
        targets_t = torch.tensor(y, dtype=torch.float32)
        l2 = float(self.l2)

        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        opt = torch.optim.LBFGS(
            [a, b],
            lr=1.0,
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(a * logits_t + b, targets_t)
            penalty = l2 * ((a - 1.0) ** 2 + b**2)
            loss = bce + penalty
            loss.backward()
            return loss

        try:
            opt.step(closure)
            a_val, b_val = float(a.item()), float(b.item())
            if not (
                -_A_BOUND < a_val < _A_BOUND
                and -_B_BOUND < b_val < _B_BOUND
                and math.isfinite(a_val)
                and math.isfinite(b_val)
            ):
                self.a, self.b = 1.0, 0.0
                self.converged = False
                return
            self.a, self.b = a_val, b_val
            self.converged = True
        except Exception:  # pylint: disable=broad-except
            self.a, self.b = 1.0, 0.0
            self.converged = False

    def transform(self, p_pred_or_logits: np.ndarray) -> np.ndarray:
        p = np.asarray(p_pred_or_logits, dtype=float)
        z = _safe_logit(p, eps=self.eps)
        return np.clip(_sigmoid(self.a * z + self.b), self.eps, 1.0 - self.eps)

    def save(self, path: str) -> None:
        payload = {
            "class": "RegularizedPlattCalibrator",
            "a": self.a,
            "b": self.b,
            "l2": self.l2,
            "eps": self.eps,
            "max_iter": self.max_iter,
            "converged": self.converged,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str) -> RegularizedPlattCalibrator:
        payload = json.loads(Path(path).read_text())
        cal = cls(
            l2=float(payload.get("l2", 0.1)),
            eps=float(payload.get("eps", _CLIP_EPS)),
            max_iter=int(payload.get("max_iter", _MAX_ITER)),
        )
        cal.a = float(payload.get("a", 1.0))
        cal.b = float(payload.get("b", 0.0))
        cal.converged = bool(payload.get("converged", True))
        return cal


__all__ = ["PlattCalibrator", "RegularizedPlattCalibrator"]
