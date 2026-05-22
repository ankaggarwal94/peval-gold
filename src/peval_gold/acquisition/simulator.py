"""Adaptive K=5 acquisition simulator for the Batch-7 shootout (S2D).

Drives the head-to-head benchmark of acquisition policies against the
shipped the wrapped NCF head by simulating the platform's per-round
K-revealed-labels-per-data-category contract.

What this simulator is and is NOT
---------------------------------

This is a STREAMING SCORING simulator. Per round it:

1. Resets the policy's state.
2. Shuffles ``val_rows`` deterministically (seed + round_index) so
   each round sees the rows in a different order. This is what makes
   ``n_rounds=3`` produce three distinct measurements for a
   deterministic policy: ordering-driven state interactions are real
   on the platform too.
3. Streams each row through ``policy.score_one(row)`` and records
   ``(score, row)`` per category.
4. Per category, picks the top-K rows by score → ``labeled`` set.
   Rows NOT in the labeled set form the ``unlabeled`` set.
5. Calls ``predictor.predict_one(row, labeled=labeled)`` on every row
   in ``unlabeled``. The predictor's first call with non-empty
   ``labeled`` fits its Platt scaler once and caches it for the rest
   of the round (matching the platform's first-call Platt semantics).
6. Computes NLL/MLL on ``(y_unlabeled, p_unlabeled)`` and records.

Between rounds the predictor's calibrator state is reset (when the
predictor exposes a ``reset_calibrator`` method — :class:`peval_gold.models.current_ncf.CurrentNCF`
does). Without this reset the round-2 Platt fit would silently inherit
round-1's labels.

This is NOT a training simulator. The predictor is treated as
frozen-and-calibrated; the only variable across rounds is which K rows
per category the policy chose.

NLL convention
--------------

``nll`` is the standard mean negative log-likelihood (lower is better)
computed via :func:`peval_gold.eval.metrics.ordinary_log_loss` with the
``[1e-4, 1-1e-4]`` clipping bound. ``mll`` is its negation (the hosted runtime
display sign per ``starting_kit/README.md:331-336``).

Output schema
-------------

::

    {
      "rounds": int,                       # alias of len(per_round)
      "n_rounds": int,                     # same value, gold-track naming
      "mean_nll": float,
      "mean_mll": float,
      "per_round": list[
        {
          "round_index": int,
          "n_labeled": int,
          "n_unlabeled": int,
          "n_scored": int,                 # rows with binary response
          "nll": float | None,
          "mll": float | None,
          "category_key": str,
          "per_category_labeled": dict[str, int],
        }
      ],
    }
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from peval_gold.eval.metrics import mean_log_likelihood, ordinary_log_loss

_BINARY_VALUES = (0.0, 1.0)


def _coerce_response(value: Any) -> float | None:
    """Return ``response ∈ {0.0, 1.0}`` or ``None`` if non-binary.

    Mirrors the convention in :func:`peval_gold.eval.evaluator._coerce_response`
    so the NLL the simulator computes matches what
    ``scripts/gold_evaluate_current.py`` reports as the baseline.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return None
    fv = float(value)
    if math.isnan(fv):
        return None
    if fv in _BINARY_VALUES:
        return fv
    return None


def _top_k_per_category(
    scored: list[tuple[float, int]],
    rows: list[Mapping[str, Any]],
    category_key: str,
    k_per_category: int,
) -> tuple[list[int], dict[str, int]]:
    """Pick the top-K row indices per category by score.

    ``scored`` is a list of ``(score, row_index)`` pairs in the order
    they were streamed through the policy. The K-per-category selection
    sorts each category's bucket by score descending (ties broken by
    the row's original streaming position to keep selection deterministic
    across runs with the same shuffle seed).

    Returns the set of labeled row indices and the per-category labeled
    count for the round manifest.
    """
    by_cat: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for score, idx in scored:
        cat = str(rows[idx].get(category_key, ""))
        by_cat[cat].append((score, idx))

    labeled_idx: list[int] = []
    per_cat_labeled: dict[str, int] = {}
    for cat in sorted(by_cat):
        bucket = by_cat[cat]
        # Sort by score DESC, then by stream-position ASC (stable tie-break).
        bucket.sort(key=lambda pair: (-pair[0], pair[1]))
        take = bucket[:k_per_category]
        labeled_idx.extend(idx for _score, idx in take)
        per_cat_labeled[cat] = len(take)
    return labeled_idx, per_cat_labeled


def _maybe_reset_calibrator(predictor: Any) -> None:
    """Call ``predictor.reset_calibrator()`` if the predictor has it.

    :class:`peval_gold.models.current_ncf.CurrentNCF` exposes this
    method so the per-round Platt fit re-fires; simple test shims may
    not, and we silently ignore the missing method.
    """
    reset = getattr(predictor, "reset_calibrator", None)
    if callable(reset):
        reset()


def _rows_to_labeled(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert val rows into the ``labeled`` shape ``predict_one`` expects.

    The runtime contract per ``starting_kit/README.md:268-271`` says
    each ``labeled`` entry carries the same four content keys as the
    runtime ``input`` plus a ``label`` field in ``{0, 1}``. Our val
    rows carry ``response`` (canonical schema); the Platt fit on the
    predictor side reads ``ex["label"]`` so we copy the value across.

    Non-binary responses are skipped — they cannot inform a binary
    Platt fit and including them would crash the wrapper's
    ``targets_list.append(float(ex["label"]))`` step.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        y_bin = _coerce_response(r.get("response"))
        if y_bin is None:
            continue
        copy = dict(r)
        copy["label"] = y_bin
        out.append(copy)
    return out


def run_adaptive_simulation(
    policy: Any,
    predictor: Any,
    val_rows: list[Mapping[str, Any]],
    k_per_category: int = 5,
    n_rounds: int = 3,
    category_key: str = "benchmark",
    seed: int = 42,
) -> dict[str, Any]:
    """Run the per-round adaptive K=5 simulation.

    Parameters
    ----------
    policy : object
        Implements ``score_one(input) -> float`` and ``reset() -> None``
        (the :class:`peval_gold.acquisition.base.AcquisitionPolicy`
        Protocol).
    predictor : object
        Implements ``predict_one(input, labeled=None) -> float``. May
        also implement ``reset_calibrator() -> None``; if present, it
        is called between rounds.
    val_rows : list[Mapping]
        Validation rows in canonical schema (must include ``response``,
        ``benchmark``, ``condition``, ``subject_content``, ``item_content``).
    k_per_category : int
        Per-category top-K row count. Default 5 per kit
        ``starting_kit/README.md:268-271``.
    n_rounds : int
        Number of simulated platform rounds. Default 3.
    category_key : str
        Which row field to group on. Default ``"benchmark"`` (matches
        :func:`peval_gold.data.splits.adaptive_label_simulation`'s
        default proxy when no explicit ``category`` field is present).
    seed : int
        Deterministic seed; round ``i`` uses ``Random(seed + i)`` for
        its shuffle.

    Returns
    -------
    dict
        See module docstring for the full schema.
    """
    if not val_rows:
        return {
            "rounds": 0,
            "n_rounds": 0,
            "mean_nll": None,
            "mean_mll": None,
            "per_round": [],
            "category_key": category_key,
            "k_per_category": k_per_category,
            "seed": seed,
        }

    rows = list(val_rows)
    per_round: list[dict[str, Any]] = []

    for round_idx in range(n_rounds):
        rng = random.Random(seed + round_idx)
        order = list(range(len(rows)))
        rng.shuffle(order)
        shuffled = [rows[i] for i in order]

        # Phase 1: stream rows through the policy and capture scores.
        policy.reset()
        _maybe_reset_calibrator(predictor)
        scored: list[tuple[float, int]] = []
        for stream_idx, row in enumerate(shuffled):
            try:
                s = float(policy.score_one(dict(row)))
                if not math.isfinite(s):
                    s = 0.0
            except Exception:  # pylint: disable=broad-except
                # Per the NaN-poisoning policy: a single exception is
                # surfaced as a zero score, not a NaN that would
                # propagate and tank the entire round.
                s = 0.0
            scored.append((s, stream_idx))

        labeled_idx_list, per_cat = _top_k_per_category(
            scored, shuffled, category_key, k_per_category
        )
        labeled_idx = set(labeled_idx_list)
        labeled_rows = [shuffled[i] for i in sorted(labeled_idx)]
        unlabeled_rows = [
            shuffled[i] for i in range(len(shuffled)) if i not in labeled_idx
        ]

        # Phase 2: build the predictor's `labeled` input + score unlabeled.
        labeled_for_predict = _rows_to_labeled(labeled_rows)
        y_list: list[float] = []
        p_list: list[float] = []
        first = True
        for r in unlabeled_rows:
            y_bin = _coerce_response(r.get("response"))
            if y_bin is None:
                continue
            try:
                p = float(
                    predictor.predict_one(
                        dict(r),
                        labeled=labeled_for_predict if first else None,
                    )
                )
            except Exception:  # pylint: disable=broad-except
                # Same distinguishable-fallback discipline: a single
                # predict_one exception scores at 0.5 (midpoint, not
                # 0/1 extreme) so the round still reports a finite NLL.
                p = 0.5
            first = False
            y_list.append(y_bin)
            p_list.append(p)

        nll: float | None
        mll: float | None
        if y_list:
            y_arr = np.asarray(y_list, dtype=float)
            p_arr = np.asarray(p_list, dtype=float)
            nll = float(ordinary_log_loss(y_arr, p_arr))
            mll = float(mean_log_likelihood(y_arr, p_arr))
        else:
            nll = None
            mll = None

        per_round.append(
            {
                "round_index": round_idx,
                "n_labeled": len(labeled_rows),
                "n_unlabeled": len(unlabeled_rows),
                "n_scored": len(y_list),
                "nll": nll,
                "mll": mll,
                "category_key": category_key,
                "per_category_labeled": per_cat,
            }
        )

    finite_nlls = [r["nll"] for r in per_round if r["nll"] is not None]
    finite_mlls = [r["mll"] for r in per_round if r["mll"] is not None]
    mean_nll = float(np.mean(finite_nlls)) if finite_nlls else None
    mean_mll = float(np.mean(finite_mlls)) if finite_mlls else None

    return {
        "rounds": len(per_round),
        "n_rounds": len(per_round),
        "mean_nll": mean_nll,
        "mean_mll": mean_mll,
        "per_round": per_round,
        "category_key": category_key,
        "k_per_category": k_per_category,
        "seed": seed,
    }


__all__ = ["run_adaptive_simulation"]
