"""Numpy-only evaluation metrics for the gold-track laboratory.

Why numpy-only:

- The runtime ``the wrapped submission's `` ZIP must stay tiny and dependency-free
  (currently 1.7 MB with only ``torch`` / ``sentence-transformers`` at
  inference). Adding ``sklearn`` to the offline trainer is fine, but
  building the metrics on numpy keeps the offline scaffolding fast to
  import and easy to reason about.
- Reproducibility: every helper here is a pure function of its inputs,
  with no global config and no RNG. Identical inputs → identical
  output, byte-for-byte.

Sign-convention reminder: the hosted runtime displays a *higher-is-better* number
that is the **negation** of the standard mean negative log-likelihood
(``starting_kit/README.md:331-336``). This module exposes both signs:

- :func:`ordinary_log_loss` is the standard positive NLL (lower is
  better).
- :func:`mean_log_likelihood` is its negation (higher closer to 0 is
  better) and matches the the hosted runtime display direction.

All metrics that consume probabilities clip with ``eps=1e-4`` by default
to avoid ``log(0)``; the Brier score does not clip because it is well-
defined at 0 and 1.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, float, int, "list[float]"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_pair(y_true: ArrayLike, p_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Defensive ``np.asarray`` with length-mismatch validation."""
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(p_pred, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise ValueError(
            f"length mismatch: y_true has shape {y_arr.shape} but p_pred has shape {p_arr.shape}"
        )
    return y_arr, p_arr


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def clip_probability(p: ArrayLike, eps: float = 1e-4) -> np.ndarray | float:
    """Clip probabilities to ``[eps, 1 - eps]``.

    Preserves input shape *and* scalar-ness:

    - Python ``float`` / ``int`` in → Python ``float`` out.
    - Numpy 0-d array in → Python ``float`` out (matches the scalar
      convention so downstream code can do ``p = clip_probability(p)``
      without surprise type changes).
    - Numpy array in → numpy array out with the same shape.

    NaN policy: NaN inputs are passed through unchanged. This matches
    the project's documented "distinguishable defensive fallbacks"
    principle: NaN is the caller's signal that something upstream went
    wrong, and silently coercing it to a plausible probability would
    hide the bug. See
    (project pattern doc).
    """
    is_scalar = isinstance(p, (int, float)) or (isinstance(p, np.ndarray) and p.ndim == 0)
    arr = np.asarray(p, dtype=float)
    nan_mask = np.isnan(arr)
    clipped = np.clip(arr, eps, 1.0 - eps)
    if nan_mask.any():
        clipped = np.where(nan_mask, np.nan, clipped)
    if is_scalar:
        return float(clipped)
    return clipped


def sigmoid(z: ArrayLike) -> np.ndarray | float:
    """Numerically stable logistic function.

    Splits positive and negative branches to avoid overflow:

    - For ``z >= 0``: ``1 / (1 + exp(-z))``.
    - For ``z < 0``:  ``exp(z) / (1 + exp(z))``.

    This keeps every intermediate exponential bounded by 1.

    Preserves scalar-ness exactly like :func:`clip_probability`.
    """
    is_scalar = isinstance(z, (int, float)) or (isinstance(z, np.ndarray) and z.ndim == 0)
    z_arr = np.asarray(z, dtype=float)
    pos = z_arr >= 0
    out = np.empty_like(z_arr)
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    neg_exp = np.exp(z_arr[~pos])
    out[~pos] = neg_exp / (1.0 + neg_exp)
    if is_scalar:
        return float(out)
    return out


def safe_logit(p: ArrayLike, eps: float = 1e-4) -> np.ndarray | float:
    """Logit on clipped probability: ``log(p_clip / (1 - p_clip))``.

    Inverse of :func:`sigmoid` (up to the clipping floor); the identity
    ``sigmoid(safe_logit(p)) == clip_probability(p)`` holds to numpy's
    float64 precision (~1e-12 in practice; tested to 1e-9).
    """
    is_scalar = isinstance(p, (int, float)) or (isinstance(p, np.ndarray) and p.ndim == 0)
    clipped = np.asarray(clip_probability(p, eps=eps), dtype=float)
    out = np.log(clipped / (1.0 - clipped))
    if is_scalar:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def ordinary_log_loss(
    y_true: ArrayLike,
    p_pred: ArrayLike,
    eps: float = 1e-4,
) -> float:
    """Mean negative log-likelihood: ``-(y log p + (1-y) log(1-p)).mean()``.

    Lower is better. Probabilities are clipped to ``[eps, 1 - eps]`` to
    keep the log finite at the extremes — matches the defensive bound
    ``the wrapped predict() module`` enforces on its output and is consistent
    with the kit's ``[1e-4, 1-1e-4]`` clipping convention.

    Raises:
        ValueError: if ``y_true`` and ``p_pred`` have different shapes.
    """
    y_arr, p_arr = _coerce_pair(y_true, p_pred)
    p_clip = np.asarray(clip_probability(p_arr, eps=eps), dtype=float)
    losses = -(y_arr * np.log(p_clip) + (1.0 - y_arr) * np.log(1.0 - p_clip))
    return float(losses.mean())


def mean_log_likelihood(
    y_true: ArrayLike,
    p_pred: ArrayLike,
    eps: float = 1e-4,
) -> float:
    """Negation of :func:`ordinary_log_loss`.

    Higher closer to 0 is better. Matches the sign convention used by
    the hosted runtime's leaderboard display per ``starting_kit/README.md:331-336``.
    """
    return -ordinary_log_loss(y_true, p_pred, eps=eps)


def brier_score(y_true: ArrayLike, p_pred: ArrayLike) -> float:
    """Mean squared error of probabilities: ``mean((y - p) ** 2)``.

    No clipping needed; the Brier score is well-defined at 0 and 1.

    Raises:
        ValueError: on length mismatch, or on empty input (mean of an
            empty array is ill-defined for a scoring metric).
    """
    y_arr, p_arr = _coerce_pair(y_true, p_pred)
    if y_arr.size == 0:
        raise ValueError("brier_score is undefined on empty input")
    return float(np.mean((y_arr - p_arr) ** 2))


def expected_calibration_error(
    y_true: ArrayLike,
    p_pred: ArrayLike,
    n_bins: int = 10,
) -> float:
    """Equal-width-binning ECE.

    Bins the predicted probabilities into ``n_bins`` equal-width buckets
    over ``[0, 1]`` and accumulates the weighted absolute gap between
    bin-mean confidence and bin-mean empirical frequency:

    ``ECE = sum_b (|B_b| / N) * |mean(p in B_b) - mean(y in B_b)|``

    Returns ``0.0`` on empty input (so callers don't have to special-case
    the cold-start ``len == 0`` situation).

    Caveat: equal-width binning is a known approximation. The standard
    alternative is equal-mass (equal-count) binning, which lands in a
    later batch alongside the adaptive simulator. Both have biases —
    see the discussion in the blueprint TeX, ``§Calibration metrics``.

    Raises:
        ValueError: on length mismatch.
    """
    y_arr, p_arr = _coerce_pair(y_true, p_pred)
    n = y_arr.size
    if n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] = np.nextafter(1.0, 2.0)
    bin_ids = np.digitize(p_arr, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_p = float(p_arr[mask].mean())
        bin_y = float(y_arr[mask].mean())
        ece += (count / n) * abs(bin_p - bin_y)

    return float(ece)
