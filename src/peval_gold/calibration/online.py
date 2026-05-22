"""OnlineCalibrator — per-round adapter that picks the best calibrator.

The adapter receives a priority-ordered list of candidate calibrators
and, on each ``fit`` call, picks the one that minimizes the AIC-style
score ``mean_log_loss + k / n_labels`` (where ``k`` is the parameter
count of the candidate and ``n_labels`` is the number of revealed
labels). The chosen calibrator's ``transform`` is then dispatched on
every subsequent ``transform`` call until the next ``fit``.

Rationale (per Batch-6 spec):

- Hosted runtime reveals K=5 labels per data category each round
  (``starting_kit/README.md:268-271``). For many configurations the
  full labeled set is only a few dozen rows. On that sample size the
  vanilla 2-parameter Platt scaler can wildly overfit; a 1-parameter
  intercept-only or temperature scaler often wins after the AIC
  penalty.
- The AIC-style penalty ``k / n`` is a per-sample translation of the
  textbook ``2k - 2 log L`` (with ``log L = -n * NLL``, the textbook
  form simplifies to ``2k/n + 2*NLL`` per sample → dividing both sides
  by 2 keeps the same ordering as ``NLL + k/n``). This keeps the
  selection scale-invariant in ``n`` and matches the comparison the
  grid evaluator does between calibrators with different parameter
  counts.

Identity-fallback rules:

- ``len(y) < 4`` → identity (don't even attempt the candidates).
- All labels are the same class → identity.
- Any candidate that raises during ``fit`` is skipped (recorded in
  ``self.failures`` for debugging) but does not abort the selection.

The chosen calibrator's class name is recorded in ``self.last_choice``
for debugging; if no candidate beats identity (or all candidates
fail), ``last_choice == "IdentityCalibrator"``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from peval_gold.calibration.identity import IdentityCalibrator
from peval_gold.calibration.intercept import InterceptCalibrator
from peval_gold.calibration.platt import (
    PlattCalibrator,
    RegularizedPlattCalibrator,
)
from peval_gold.calibration.temperature import TemperatureCalibrator

_CLIP_EPS = 1e-4
_MIN_LABELS = 4


# Param counts for the AIC-style complexity penalty.
_PARAM_COUNT: dict[str, int] = {
    "IdentityCalibrator": 0,
    "InterceptCalibrator": 1,
    "TemperatureCalibrator": 1,
    "PlattCalibrator": 2,
    "RegularizedPlattCalibrator": 2,
}


def _param_count(cal: Any) -> int:
    """Param count for the AIC penalty. Defaults to 2 if unknown."""
    return _PARAM_COUNT.get(type(cal).__name__, 2)


def _mean_log_loss(y: np.ndarray, p: np.ndarray, eps: float = _CLIP_EPS) -> float:
    """Numpy-only mean BCE. Clipped to ``[eps, 1-eps]`` so log stays finite."""
    p_clip = np.clip(p, eps, 1.0 - eps)
    losses = -(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
    return float(losses.mean())


def _identity_fallback_required(y: np.ndarray) -> bool:
    if y.size < _MIN_LABELS:
        return True
    return np.unique(y).size < 2


class OnlineCalibrator:
    """Per-round adapter that picks among candidate calibrators.

    Parameters
    ----------
    candidates : Sequence[Calibrator]
        Priority-ordered list of calibrator instances. Order matters
        as a tiebreaker: when two candidates tie on the AIC-adjusted
        score (within ``tie_tol``), the earlier one wins (mirrors
        the "simpler model wins" preference at equal cost).
    aic_alpha : float
        Multiplier on the ``k / n`` complexity term. Default ``1.0``
        — that's the per-sample form of textbook AIC after dividing
        out the factor of 2. Set to ``0`` for "lowest NLL wins"
        behavior, set to a larger value to bias more strongly toward
        simpler calibrators.
    eps : float
        Clip floor passed through to calibrators that take an
        ``eps`` constructor. Default ``1e-4``.
    tie_tol : float
        Absolute tolerance on the AIC score for the tiebreak rule.
        Default ``1e-9``.
    """

    def __init__(
        self,
        candidates: Sequence[Any] | None = None,
        aic_alpha: float = 1.0,
        eps: float = _CLIP_EPS,
        tie_tol: float = 1e-9,
    ) -> None:
        if candidates is None:
            candidates = self._default_candidates()
        self.candidates: list[Any] = list(candidates)
        if not self.candidates:
            raise ValueError("OnlineCalibrator requires at least one candidate")
        self.aic_alpha = float(aic_alpha)
        self.eps = float(eps)
        self.tie_tol = float(tie_tol)
        self.last_choice: str = "IdentityCalibrator"
        self._chosen: Any = self._identity_in_candidates()
        self.scores: dict[str, float] = {}
        self.failures: dict[str, str] = {}

    # ----- defaults -----------------------------------------------------

    def _default_candidates(self) -> list[Any]:
        """The default priority list used by the Batch-6 grid."""
        return [
            IdentityCalibrator(),
            InterceptCalibrator(l2=0.1),
            TemperatureCalibrator(l2=0.1),
            PlattCalibrator(),
            RegularizedPlattCalibrator(l2=0.1),
        ]

    def _identity_in_candidates(self) -> Any:
        """Find the first IdentityCalibrator in candidates; create one if missing."""
        for c in self.candidates:
            if isinstance(c, IdentityCalibrator):
                return c
        return IdentityCalibrator(eps=self.eps)

    # ----- Protocol surface ---------------------------------------------

    def fit(self, y_true: np.ndarray, p_pred_or_logits: np.ndarray) -> None:
        """Score every candidate; pick the best by AIC-adjusted log-loss.

        ``self.last_choice`` records the chosen calibrator's class name;
        ``self.scores`` records the AIC-adjusted score for every
        candidate (useful for debugging); ``self.failures`` records the
        exception string for any candidate whose ``fit`` raised.
        """
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(p_pred_or_logits, dtype=float)
        if y.shape != p.shape:
            raise ValueError(f"y_true shape {y.shape} != p_pred_or_logits shape {p.shape}")

        # Hard gate: identity-fallback on tiny / single-class label sets.
        if _identity_fallback_required(y):
            self._chosen = self._identity_in_candidates()
            self.last_choice = "IdentityCalibrator"
            self.scores = {}
            self.failures = {}
            return

        n = float(max(y.size, 1))
        alpha = self.aic_alpha
        self.scores = {}
        self.failures = {}

        best_score = float("inf")
        best_idx = 0  # tiebreak: earlier in priority list wins
        for i, candidate in enumerate(self.candidates):
            cls_name = type(candidate).__name__
            try:
                candidate.fit(y, p)
                p_cal = candidate.transform(p)
                base = _mean_log_loss(y, p_cal, eps=self.eps)
                k = _param_count(candidate)
                score = base + alpha * k / n
                self.scores[cls_name] = score
                if score + self.tie_tol < best_score:
                    best_score = score
                    best_idx = i
            except Exception as exc:  # pylint: disable=broad-except
                # A failing candidate does not abort selection — we
                # record the failure and continue. Distinguishable
                # defensive fallback: the candidate is excluded from
                # the pick rather than masquerading as identity.
                self.failures[cls_name] = f"{type(exc).__name__}: {exc}"

        if not self.scores:
            # All candidates failed — fall back to identity.
            self._chosen = self._identity_in_candidates()
            self.last_choice = "IdentityCalibrator"
            return

        self._chosen = self.candidates[best_idx]
        self.last_choice = type(self._chosen).__name__

    def transform(self, p_pred_or_logits: np.ndarray) -> np.ndarray:
        return self._chosen.transform(p_pred_or_logits)

    def save(self, path: str) -> None:
        """Persist the chosen calibrator's class name + scores + tag.

        Note: this does NOT persist every candidate's fitted params (those
        live on the individual calibrator instances). On :meth:`load`,
        the adapter rebuilds the default candidate list and re-selects
        based on the persisted ``last_choice``. The use case here is
        debugging / provenance — the per-round adapter is meant to be
        re-fit at the start of every the hosted runtime round, so a saved adapter
        is rarely "loaded and used" in production.
        """
        payload = {
            "class": "OnlineCalibrator",
            "last_choice": self.last_choice,
            "scores": self.scores,
            "failures": self.failures,
            "aic_alpha": self.aic_alpha,
            "eps": self.eps,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str) -> OnlineCalibrator:
        payload = json.loads(Path(path).read_text())
        adapter = cls(
            aic_alpha=float(payload.get("aic_alpha", 1.0)),
            eps=float(payload.get("eps", _CLIP_EPS)),
        )
        adapter.last_choice = str(payload.get("last_choice", "IdentityCalibrator"))
        adapter.scores = dict(payload.get("scores", {}))
        adapter.failures = dict(payload.get("failures", {}))
        # Re-bind chosen by name; if not found, stay on identity.
        for c in adapter.candidates:
            if type(c).__name__ == adapter.last_choice:
                adapter._chosen = c  # noqa: SLF001 (intentional re-bind)
                break
        return adapter


__all__ = ["OnlineCalibrator"]
