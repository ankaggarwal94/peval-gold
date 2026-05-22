"""Intercept-only calibrators for the Batch-6 calibration grid.

Two flavors:

- :class:`InterceptCalibrator` — global one-parameter bias ``b`` such
  that ``p_cal = sigmoid(logit(p) + b)`` minimizes the L2-regularized
  log-loss. Cheap, one parameter, very hard to overfit. Fits via scipy
  ``L-BFGS-B`` with bounds ``b ∈ [-20, 20]`` and a small L2 penalty
  ``l2 * b^2`` (default ``l2 = 0.1``).

- :class:`PerCategoryInterceptCalibrator` — per-category intercept with
  shared shrinkage toward the global intercept. Categories with
  ``< 4`` labeled examples fall back to the global intercept rather
  than getting their own (the same min-label rule the other calibrators
  use). The category key is configurable via ``category_key`` so the
  grid can A/B per-benchmark vs per-condition vs per-(benchmark,
  condition) intercepts.

Identity-fallback rules (mirror the wrapped predict() module):

- ``len(y) < 4`` → ``b = 0``.
- All labels the same class → ``b = 0`` (otherwise the optimum is at
  ±∞, and even a small L2 produces a numerically large value that the
  evaluator's clip-then-log would distort).
- Numerical failure inside the solver → ``b = 0``.

This module imports only ``numpy`` and ``scipy`` — no ``torch``, no
``sentence-transformers``. Safe to load in the offline evaluator's
calibration loop without paying any model-load cost.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_CLIP_EPS = 1e-4
_LOGIT_CLIP = 20.0  # |logit| above this rarely occurs and would break LBFGS.
_MIN_LABELS = 4


# ---------------------------------------------------------------------------
# Numerically-stable BCE-with-logits in pure numpy.
# ---------------------------------------------------------------------------


def _bce_with_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Elementwise ``max(z, 0) - z*y + log1p(exp(-|z|))``.

    The standard numerically-stable form. Returns an array; the caller
    typically takes ``mean()`` to get the scalar loss.
    """
    return np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))


def _safe_logit(p: np.ndarray, eps: float = _CLIP_EPS) -> np.ndarray:
    """``log(p_clip / (1 - p_clip))`` with clipping to keep finiteness."""
    p_clip = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic for arrays; matches ``eval.metrics.sigmoid``."""
    pos = z >= 0
    out = np.empty_like(z, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg_exp = np.exp(z[~pos])
    out[~pos] = neg_exp / (1.0 + neg_exp)
    return out


def _identity_fallback_required(y: np.ndarray) -> bool:
    """Same gate as the Platt scaler in the wrapped predict() module:_fit_platt."""
    if y.size < _MIN_LABELS:
        return True
    unique = np.unique(y)
    return unique.size < 2


# ---------------------------------------------------------------------------
# 1. Global intercept-only calibrator.
# ---------------------------------------------------------------------------


class InterceptCalibrator:
    """Fit a single bias ``b``: ``p_cal = sigmoid(logit(p) + b)``.

    Parameters
    ----------
    l2 : float
        Strength of the L2 penalty on ``b``. Default ``0.1`` (gentle).
        Higher values pull the fit toward ``b = 0`` (identity).
    eps : float
        Clip floor for the final ``transform`` step. Default ``1e-4``.
    bounds : tuple[float, float]
        Bounds on ``b`` passed to L-BFGS-B. Default ``(-20, 20)``.
    """

    def __init__(
        self,
        l2: float = 0.1,
        eps: float = _CLIP_EPS,
        bounds: tuple[float, float] = (-_LOGIT_CLIP, _LOGIT_CLIP),
    ) -> None:
        self.l2 = float(l2)
        self.eps = float(eps)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.b: float = 0.0
        self.converged: bool = True

    def fit(self, y_true: np.ndarray, p_pred_or_logits: np.ndarray) -> None:
        """Fit ``b`` by minimizing mean-BCE + ``l2 * b^2`` via L-BFGS-B."""
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(p_pred_or_logits, dtype=float)
        if y.shape != p.shape:
            raise ValueError(
                f"y_true shape {y.shape} != p_pred_or_logits shape {p.shape}"
            )
        if _identity_fallback_required(y):
            self.b = 0.0
            self.converged = True
            return

        z = _safe_logit(p, eps=self.eps)
        l2 = self.l2

        def loss_and_grad(b_arr: np.ndarray) -> tuple[float, np.ndarray]:
            b = float(b_arr[0])
            zb = z + b
            sig = _sigmoid(zb)
            loss = float(np.mean(_bce_with_logits(zb, y)) + l2 * b * b)
            # d/db BCE = mean(sigmoid(zb) - y); d/db l2*b^2 = 2*l2*b
            grad = float(np.mean(sig - y) + 2.0 * l2 * b)
            return loss, np.array([grad])

        try:
            result = minimize(
                loss_and_grad,
                x0=np.array([0.0]),
                method="L-BFGS-B",
                jac=True,
                bounds=[self.bounds],
                options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-7},
            )
            b_val = float(result.x[0])
            if not math.isfinite(b_val):
                self.b = 0.0
                self.converged = False
                return
            self.b = b_val
            self.converged = bool(result.success)
        except Exception:  # pylint: disable=broad-except
            # Numerical failures fall back to identity — see
            # distinguishable-defensive-fallbacks pattern: the value is
            # an identity transform (not a defensive constant), so the
            # caller sees the un-calibrated probability rather than a
            # 0.5 sentinel.
            self.b = 0.0
            self.converged = False

    def transform(self, p_pred_or_logits: np.ndarray) -> np.ndarray:
        p = np.asarray(p_pred_or_logits, dtype=float)
        z = _safe_logit(p, eps=self.eps)
        return np.clip(_sigmoid(z + self.b), self.eps, 1.0 - self.eps)

    def save(self, path: str) -> None:
        payload = {
            "class": "InterceptCalibrator",
            "b": self.b,
            "l2": self.l2,
            "eps": self.eps,
            "converged": self.converged,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str) -> "InterceptCalibrator":
        payload = json.loads(Path(path).read_text())
        cal = cls(l2=float(payload.get("l2", 0.1)), eps=float(payload.get("eps", _CLIP_EPS)))
        cal.b = float(payload.get("b", 0.0))
        cal.converged = bool(payload.get("converged", True))
        return cal


# ---------------------------------------------------------------------------
# 2. Per-category intercept with shrinkage to global.
# ---------------------------------------------------------------------------


class PerCategoryInterceptCalibrator:
    """Per-category intercept with shrinkage toward the global intercept.

    Behavior:

    - Always fits the GLOBAL intercept ``b_global`` first (as if all the
      data were one category). That intercept is the fallback for any
      category that has fewer than ``min_per_category`` labels.
    - For each category with ``>= min_per_category`` labels, fits an
      additional per-category intercept ``b_cat`` by minimizing
      mean-BCE on that category's rows PLUS an L2 shrinkage penalty
      ``shrinkage * (b_cat - b_global)^2``. This regularizes per-cat
      intercepts back toward the global intercept rather than letting
      them drift to wild values on small bins.
    - Categories that don't pass the min-label gate use ``b_global``
      directly; the unseen-category branch of ``transform_with_categories``
      also routes to ``b_global``.

    Parameters
    ----------
    category_key : str
        Free-text label of the grouping key (e.g. ``"benchmark"``,
        ``"condition"``). Not used by the fit itself — categories are
        supplied directly to :meth:`fit_with_categories`. Kept on the
        instance for provenance / save / load.
    l2 : float
        L2 penalty on the GLOBAL intercept (default ``0.1``; same scale
        as :class:`InterceptCalibrator`).
    shrinkage : float
        Strength of the per-cat shrinkage to global. Default ``1.0``
        (a single labeled row's worth of "extra" anchoring).
    min_per_category : int
        Min labels in a category to fit its own intercept. Default 4
        (matches the project's ``_PLATT_MIN_LABELS``).
    eps : float
        Clip floor. Default ``1e-4``.
    """

    def __init__(
        self,
        category_key: str = "benchmark",
        l2: float = 0.1,
        shrinkage: float = 1.0,
        min_per_category: int = _MIN_LABELS,
        eps: float = _CLIP_EPS,
    ) -> None:
        self.category_key = str(category_key)
        self.l2 = float(l2)
        self.shrinkage = float(shrinkage)
        self.min_per_category = int(min_per_category)
        self.eps = float(eps)
        self.global_intercept: float = 0.0
        self.per_category_intercept: dict[str, float] = {}

    # ----- Protocol surface (global-only fit when no categories provided) ----

    def fit(self, y_true: np.ndarray, p_pred_or_logits: np.ndarray) -> None:
        """Global-only fit; per-category intercepts stay empty.

        For full per-category fits use :meth:`fit_with_categories`. The
        Protocol surface accepts only ``(y, p)`` so this path treats the
        whole dataset as one category and only fits the global intercept.
        """
        global_fit = InterceptCalibrator(l2=self.l2, eps=self.eps)
        global_fit.fit(y_true, p_pred_or_logits)
        self.global_intercept = float(global_fit.b)
        self.per_category_intercept = {}

    def fit_with_categories(
        self,
        y_true: np.ndarray,
        p_pred_or_logits: np.ndarray,
        categories: Sequence[str],
    ) -> None:
        """Fit global + per-category intercepts.

        Parameters
        ----------
        y_true, p_pred_or_logits : numpy arrays
            Same as :meth:`fit`.
        categories : sequence[str]
            One category label per row; must align with ``y_true`` /
            ``p_pred_or_logits`` (same length, same order).
        """
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(p_pred_or_logits, dtype=float)
        cats = list(categories)
        if y.shape != p.shape or len(cats) != y.size:
            raise ValueError(
                f"shape mismatch: y={y.shape}, p={p.shape}, "
                f"len(categories)={len(cats)}"
            )

        # Global fit first.
        global_fit = InterceptCalibrator(l2=self.l2, eps=self.eps)
        global_fit.fit(y, p)
        self.global_intercept = float(global_fit.b)
        self.per_category_intercept = {}

        if _identity_fallback_required(y):
            # Nothing to do per-category if global was identity-fallback.
            return

        by_cat: dict[str, list[int]] = {}
        for i, c in enumerate(cats):
            by_cat.setdefault(str(c), []).append(i)

        z_all = _safe_logit(p, eps=self.eps)
        for cat, idxs in by_cat.items():
            if len(idxs) < self.min_per_category:
                continue
            idx_arr = np.asarray(idxs, dtype=int)
            y_cat = y[idx_arr]
            if np.unique(y_cat).size < 2:
                continue
            z_cat = z_all[idx_arr]
            b_cat = self._fit_per_cat_intercept(z_cat, y_cat)
            if b_cat is not None:
                self.per_category_intercept[cat] = b_cat

    def _fit_per_cat_intercept(
        self, z: np.ndarray, y: np.ndarray
    ) -> float | None:
        """Fit one per-category intercept with shrinkage to global."""
        b_global = self.global_intercept
        shrinkage = self.shrinkage
        n = float(z.size)

        def loss_and_grad(b_arr: np.ndarray) -> tuple[float, np.ndarray]:
            b = float(b_arr[0])
            zb = z + b
            sig = _sigmoid(zb)
            data_loss = float(np.mean(_bce_with_logits(zb, y)))
            penalty = shrinkage * (b - b_global) ** 2 / n
            loss = data_loss + penalty
            grad_data = float(np.mean(sig - y))
            grad_penalty = 2.0 * shrinkage * (b - b_global) / n
            return loss, np.array([grad_data + grad_penalty])

        try:
            result = minimize(
                loss_and_grad,
                x0=np.array([b_global]),
                method="L-BFGS-B",
                jac=True,
                bounds=[(-_LOGIT_CLIP, _LOGIT_CLIP)],
                options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-7},
            )
            b_val = float(result.x[0])
            if not math.isfinite(b_val):
                return None
            return b_val
        except Exception:  # pylint: disable=broad-except
            return None

    # ----- transform ----------------------------------------------------

    def transform(self, p_pred_or_logits: np.ndarray) -> np.ndarray:
        """Apply the GLOBAL intercept only (Protocol-compatible surface)."""
        p = np.asarray(p_pred_or_logits, dtype=float)
        z = _safe_logit(p, eps=self.eps)
        return np.clip(
            _sigmoid(z + self.global_intercept), self.eps, 1.0 - self.eps
        )

    def transform_with_categories(
        self,
        p_pred_or_logits: np.ndarray,
        categories: Sequence[str],
    ) -> np.ndarray:
        """Apply per-category intercept where available, global otherwise."""
        p = np.asarray(p_pred_or_logits, dtype=float)
        cats = list(categories)
        if len(cats) != p.size:
            raise ValueError(
                f"len(categories)={len(cats)} != p.size={p.size}"
            )
        z = _safe_logit(p, eps=self.eps)
        b_arr = np.full_like(z, fill_value=self.global_intercept)
        for i, c in enumerate(cats):
            b_arr[i] = self.per_category_intercept.get(str(c), self.global_intercept)
        return np.clip(_sigmoid(z + b_arr), self.eps, 1.0 - self.eps)

    def save(self, path: str) -> None:
        payload = {
            "class": "PerCategoryInterceptCalibrator",
            "category_key": self.category_key,
            "l2": self.l2,
            "shrinkage": self.shrinkage,
            "min_per_category": self.min_per_category,
            "eps": self.eps,
            "global_intercept": self.global_intercept,
            "per_category_intercept": self.per_category_intercept,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str) -> "PerCategoryInterceptCalibrator":
        payload = json.loads(Path(path).read_text())
        cal = cls(
            category_key=str(payload.get("category_key", "benchmark")),
            l2=float(payload.get("l2", 0.1)),
            shrinkage=float(payload.get("shrinkage", 1.0)),
            min_per_category=int(payload.get("min_per_category", _MIN_LABELS)),
            eps=float(payload.get("eps", _CLIP_EPS)),
        )
        cal.global_intercept = float(payload.get("global_intercept", 0.0))
        cal.per_category_intercept = dict(
            payload.get("per_category_intercept", {})
        )
        return cal


__all__ = ["InterceptCalibrator", "PerCategoryInterceptCalibrator"]
