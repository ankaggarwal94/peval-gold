"""TemperatureCalibrator — single-parameter temperature scaling.

``p_cal = sigmoid(logit(p) / T)`` with ``T > 0``. ``T = 1`` reduces to
identity; ``T > 1`` softens overconfident predictions (the typical case
for neural networks); ``T < 1`` sharpens underconfident predictions.

Why temperature scaling deserves its own slot in the Batch-6 grid:

- On adaptive K=5 labeled sets, vanilla Platt scaling can swing ``a``
  wildly (especially when the small labeled batch is noisy). Temperature
  scaling has only one parameter and cannot rescale the slope
  asymmetrically — that's the same kind of regularization a strong
  prior on Platt's ``a`` would give, but with a clean one-parameter
  story (Guo et al. 2017, "On Calibration of Modern Neural Networks").
- Free to ship in the submission since the runtime already has scipy /
  numpy via torch.

Implementation:

- Internally reparameterize with ``log T`` so the L-BFGS-B optimizer
  works on an unconstrained scalar while we keep ``T = exp(log T) > 0``.
- L2 penalty is applied to ``log T`` (penalizing deviation from
  ``T = 1`` symmetrically in log-space).
- Bounds on ``log T ∈ [log(0.1), log(10)]`` so ``T ∈ [0.1, 10]``
  matches the spec.

Identity-fallback rules (mirror the wrapped predict() module):

- ``len(y) < 4`` → ``T = 1.0``.
- All labels are the same class → ``T = 1.0``.
- Numerical failure → ``T = 1.0``.

This module imports only ``numpy`` and ``scipy``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_CLIP_EPS = 1e-4
_MIN_LABELS = 4
_T_MIN = 0.1
_T_MAX = 10.0


# ---------------------------------------------------------------------------
# Numerically-stable helpers (mirror intercept.py so this module can stand
# alone without the import latency of pulling intercept.py).
# ---------------------------------------------------------------------------


def _bce_with_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))


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
# Public class
# ---------------------------------------------------------------------------


class TemperatureCalibrator:
    """Single-parameter temperature scaler.

    Parameters
    ----------
    l2 : float
        Strength of the L2 penalty on ``log T``. Default ``0.1``.
        Higher values pull the fit toward ``T = 1``.
    eps : float
        Clip floor. Default ``1e-4``.
    t_bounds : tuple[float, float]
        Bounds on ``T``. Default ``(0.1, 10.0)``.
    """

    def __init__(
        self,
        l2: float = 0.1,
        eps: float = _CLIP_EPS,
        t_bounds: tuple[float, float] = (_T_MIN, _T_MAX),
    ) -> None:
        self.l2 = float(l2)
        self.eps = float(eps)
        self.t_bounds = (float(t_bounds[0]), float(t_bounds[1]))
        self.temperature: float = 1.0
        self.converged: bool = True

    def fit(self, y_true: np.ndarray, p_pred_or_logits: np.ndarray) -> None:
        """Fit ``T`` by minimizing mean-BCE + ``l2 * (log T)^2`` via L-BFGS-B."""
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(p_pred_or_logits, dtype=float)
        if y.shape != p.shape:
            raise ValueError(f"y_true shape {y.shape} != p_pred_or_logits shape {p.shape}")
        if _identity_fallback_required(y):
            self.temperature = 1.0
            self.converged = True
            return

        z = _safe_logit(p, eps=self.eps)
        l2 = self.l2
        log_t_bounds = (math.log(self.t_bounds[0]), math.log(self.t_bounds[1]))

        def loss_and_grad(log_t_arr: np.ndarray) -> tuple[float, np.ndarray]:
            log_t = float(log_t_arr[0])
            t = math.exp(log_t)
            z_scaled = z / t
            sig = _sigmoid(z_scaled)
            data_loss = float(np.mean(_bce_with_logits(z_scaled, y)))
            penalty = l2 * log_t * log_t
            loss = data_loss + penalty
            # d/dT BCE = mean((sigmoid(z/T) - y) * (-z / T^2))
            # Chain to log T: d/d(log T) = d/dT * dT/d(log T) = d/dT * T
            # → d/d(log T) BCE = mean((sigmoid(z/T) - y) * (-z / T))
            grad_data = float(np.mean((sig - y) * (-z / t)))
            grad_penalty = 2.0 * l2 * log_t
            return loss, np.array([grad_data + grad_penalty])

        try:
            result = minimize(
                loss_and_grad,
                x0=np.array([0.0]),  # log T = 0 → T = 1 (identity)
                method="L-BFGS-B",
                jac=True,
                bounds=[log_t_bounds],
                options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-7},
            )
            t_val = float(math.exp(float(result.x[0])))
            if not math.isfinite(t_val) or t_val <= 0.0:
                self.temperature = 1.0
                self.converged = False
                return
            self.temperature = t_val
            self.converged = bool(result.success)
        except Exception:  # pylint: disable=broad-except
            self.temperature = 1.0
            self.converged = False

    def transform(self, p_pred_or_logits: np.ndarray) -> np.ndarray:
        p = np.asarray(p_pred_or_logits, dtype=float)
        z = _safe_logit(p, eps=self.eps)
        return np.clip(_sigmoid(z / self.temperature), self.eps, 1.0 - self.eps)

    def save(self, path: str) -> None:
        payload = {
            "class": "TemperatureCalibrator",
            "temperature": self.temperature,
            "l2": self.l2,
            "eps": self.eps,
            "converged": self.converged,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str) -> TemperatureCalibrator:
        payload = json.loads(Path(path).read_text())
        cal = cls(
            l2=float(payload.get("l2", 0.1)),
            eps=float(payload.get("eps", _CLIP_EPS)),
        )
        cal.temperature = float(payload.get("temperature", 1.0))
        cal.converged = bool(payload.get("converged", True))
        return cal


__all__ = ["TemperatureCalibrator"]
