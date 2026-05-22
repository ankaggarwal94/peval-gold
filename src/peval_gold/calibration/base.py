"""Calibrator Protocol for the gold-track laboratory.

Calibrators map raw predictor outputs (either probabilities ``p`` or
unconstrained logits ``z``) to probabilities better aligned with
empirical frequency. The Protocol intentionally treats the input as
``p_pred_or_logits`` — each concrete implementation documents which
flavor it expects:

- Platt scaling on logits → expects logits, returns probabilities.
- Isotonic regression → expects probabilities, returns probabilities.
- Temperature scaling → expects logits, returns probabilities.

The downstream offline-eval scaffolding () must therefore route
the right input flavor to the right calibrator; the Protocol does not
enforce a single representation because doing so would over-constrain
the design space.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Calibrator(Protocol):
    """Fit on labeled data, then transform new predictions.

    Implementations must provide:

    - :meth:`fit` — train on ``(y_true, p_pred_or_logits)`` arrays. Both
      must have the same length; the implementation should raise
      :class:`ValueError` on length mismatch.
    - :meth:`transform` — apply the fitted mapping to new
      predictions and return a 1-D :class:`numpy.ndarray` of probabilities.
    - :meth:`save` / :meth:`load` — persist a deployable artifact.
      As with :class:`peval_gold.models.base.Predictor`, prefer state-only
      payloads over full-module pickles.
    """

    def fit(
        self,
        y_true: np.ndarray,
        p_pred_or_logits: np.ndarray,
    ) -> None: ...

    def transform(self, p_pred_or_logits: np.ndarray) -> np.ndarray: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> Calibrator: ...
