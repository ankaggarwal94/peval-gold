"""Predictor Protocol shapes used across the gold-track laboratory.

Both Protocols are :func:`~typing.runtime_checkable` so duck-typed shim
classes (and concrete implementations under e.g. ``peval_gold.models.ncf``)
can be validated with :func:`isinstance` without needing to inherit from
an abstract base class.

The two flavors mirror the two execution contexts in the project:

- :class:`Predictor` — the **offline** trainer/inference surface used by
  Slurm sweeps and local experiments. Operates on batches of rows.
- :class:`RuntimePredictor` — the **online** streaming surface that
  mirrors the the hosted runtime ``predict(input, labeled)`` signature exactly,
  including the intentional shadowing of the builtin ``input`` per the
  hosted kit contract (see ``starting_kit/sample_code_the wrapped predict() module``).

Implementations of :class:`Predictor` and :class:`RuntimePredictor` are
allowed to be the *same class* —  of the gold-track plan unifies
them through a Module-level loader. The Protocols are kept separate so
that experimentation in either dimension can proceed without coupling.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Predictor(Protocol):
    """Offline-trainable predictor over rows.

    Implementations must provide:

    - :meth:`fit` — train on a sequence of row dicts (``train_rows``) and
      optionally use ``valid_rows`` for early-stopping / hyperparameter
      tuning. Must mutate the instance in place. Return value is ignored.
    - :meth:`predict_proba` — return a 1-D :class:`numpy.ndarray` of
      probabilities aligned with the input ``rows`` in length and order.
    - :meth:`save` — persist a deployable artifact to ``path``.
      Implementations should save ``state_dict``-style payloads (not full
      module pickles) per the project lesson at
      (project pattern doc).
    - :meth:`load` — classmethod that restores an instance from ``path``.

    Row schema is intentionally left unspecified at the Protocol layer.
     will land a normalized row contract in
    ``peval_gold.data.rows``; concrete predictors should depend on that
    schema, not on the hosted runtime-specific keys directly.
    """

    def fit(
        self,
        train_rows: Sequence[dict],
        valid_rows: Sequence[dict] | None = None,
    ) -> None: ...

    def predict_proba(self, rows: Sequence[dict]) -> np.ndarray: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "Predictor": ...


@runtime_checkable
class RuntimePredictor(Protocol):
    """Streaming inference protocol matching ``predict(input, labeled)``.

    The hosted the hosted runtime container imports the submission once and calls
    ``predict(input: dict, labeled: list[dict] | None) -> float`` ~5000
    times per round. ``predict_one`` is the Python-side mirror of that
    call: a single row in, a single calibrated probability out. The
    output must be a native Python ``float`` in ``[0, 1]``, finite, and
    never NaN — see (project decision doc).

    The ``input`` parameter intentionally shadows the builtin (per kit
    contract). ``labeled`` is optional and may be ``None`` outside the
    adaptive-labeling rounds (e.g. when running unit tests).
    """

    def predict_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
        labeled: list[dict] | None = None,
    ) -> float: ...
