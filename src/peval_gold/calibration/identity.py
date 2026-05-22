"""IdentityCalibrator — the trivial pass-through baseline.

The :class:`IdentityCalibrator` makes no parameter fit; ``transform(p)``
returns ``np.clip(p, 1e-4, 1 - 1e-4)``. It exists so the Batch-6
calibration-grid evaluator can include the "no calibration" condition as
a first-class member of the grid (rather than a special-cased branch).

Per the Batch-1 :class:`peval_gold.calibration.base.Calibrator` Protocol,
``fit`` / ``transform`` / ``save`` / ``load`` are all required. ``fit``
is a no-op; ``save`` writes a tiny tag JSON; ``load`` reconstructs a
fresh instance.

The ``1e-4`` clip floor matches the ``the wrapped predict() module`` post-sigmoid
clip and keeps the log-loss finite at the extremes (matching the
project's ``[1e-4, 1-1e-4]`` convention documented in
(project decision doc)).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_CLIP_EPS = 1e-4


class IdentityCalibrator:
    """No-op calibrator: ``transform(p) = np.clip(p, eps, 1 - eps)``.

    Parameters
    ----------
    eps : float
        Clip floor. Default ``1e-4`` (matches ``the wrapped predict() module``).
    """

    def __init__(self, eps: float = _CLIP_EPS) -> None:
        self.eps = float(eps)

    # ----- Protocol -----------------------------------------------------

    def fit(self, y_true: np.ndarray, p_pred_or_logits: np.ndarray) -> None:
        """No-op. Accepts arrays for Protocol compatibility; never mutates state."""
        return None

    def transform(self, p_pred_or_logits: np.ndarray) -> np.ndarray:
        """Return ``np.clip(p, eps, 1 - eps)`` as a fresh array."""
        arr = np.asarray(p_pred_or_logits, dtype=float)
        return np.clip(arr, self.eps, 1.0 - self.eps)

    def save(self, path: str) -> None:
        """Persist the tag + eps; ~30 bytes on disk."""
        payload = {"class": "IdentityCalibrator", "eps": self.eps}
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str) -> IdentityCalibrator:
        """Reconstruct from a payload written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        return cls(eps=float(payload.get("eps", _CLIP_EPS)))


__all__ = ["IdentityCalibrator"]
