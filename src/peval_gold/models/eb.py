"""Empirical-Bayes hierarchical priors for ``P(correct)``.

This module implements ``EBPriors``, a tiny lookup-table predictor that
shrinks per-cell binomial proportions toward their parent-level estimate
via a beta-binomial empirical-Bayes update::

    p_eb(k, n; parent_p, kappa) = (k + kappa * parent_p) / (n + kappa)

Hierarchy (top-down)::

    global ─┬─> subject
            ├─> benchmark
            ├─> condition
            └─> (via subject) subject_benchmark
                                  └─> subject_benchmark_condition (n >= 5)

Lookup chain (most-specific first; tied to ``predict_proba`` / ``predict_one``)::

    subject_benchmark_condition → subject_benchmark → subject → benchmark → global

The ``condition``-only level is computed for completeness (useful for
diagnostics and as a parent of any future ``condition × X`` levels) but
is NOT in the fallback chain — empirically benchmark is a better
fallback than condition because conditions are heterogeneous across
benchmarks.

Why this exists
---------------

The current shipped the hosted runtime submission's ``predict()`` has ZERO
base-rate input: it relies entirely on a 2-layer MLP over frozen
mpnet-base-v2 embeddings of ``subject_content`` and ``item_content``
(per the Batch-0 CoVe audit at
``docs/walkthroughs/012_gold_track_current_state_cove.md`` Claim 1 and
the wrapper at ``src/peval_gold/models/current_ncf.py``). EB priors
add a cheap, strong base-rate signal: when the subject is known and
the item is informationless, knowing that "this subject is correct
70% of the time on this benchmark" is already a 0.7 baseline before
the NCF's encoder ever runs.

Combined with ``peval_gold.models.ensemble.LogitBlend``, an EB-shaped
prior is the standard "kaggle-grade" base for an item-side
collaborative-filtering head — see
a recommendation-blueprint § Empirical
Bayes and shrinkage.

Runtime characteristics
-----------------------

Fit is O(N) over training rows. Predict is O(1) per row — one dict
lookup per level until a hit. The fitted lookup tables are small
(~1-5 MB for the full hierarchy across ~3.13M training rows; ~50 KB
for synthetic test datasets).

The implementation uses ONLY the standard library + numpy. No torch,
no sentence-transformers, no datasets — the EB model is safe to ship
inside the submission ZIP later if the parent decides to promote it
(it would slot in next to or instead of ``the weights file``).

Subject-key handling
--------------------

The hierarchy's "subject" key is parsed via
:func:`peval_gold.data.normalize.parse_subject_name` so the same key is
derivable at training time (from the joined HF row) and at runtime (from
the platform's ``input["subject_content"]``). When the kit-formatted
``Name:`` line is absent, the fallback is a deterministic 16-character
BLAKE2b hash of the first 512 bytes of ``subject_content`` so an
otherwise-content-identical row still shares a key.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from peval_gold.data.normalize import parse_subject_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lookup chain from most-specific to least-specific. The ``condition``
# level is intentionally NOT in this chain — see module docstring for
# rationale. Future ablations can re-enable it by extending the tuple.
_LOOKUP_CHAIN: tuple[str, ...] = (
    "subject_benchmark_condition",
    "subject_benchmark",
    "subject",
    "benchmark",
    "global",
)

# Levels we compute (a superset of _LOOKUP_CHAIN; ``condition`` is
# computed for completeness but not consulted by the fallback chain).
_ALL_LEVELS: tuple[str, ...] = (
    "global",
    "subject",
    "benchmark",
    "condition",
    "subject_benchmark",
    "subject_benchmark_condition",
)

# Parent map (used by both kappa estimation and parent_p lookup).
_PARENT_LEVEL: dict[str, str | None] = {
    "global": None,
    "subject": "global",
    "benchmark": "global",
    "condition": "global",
    "subject_benchmark": "subject",
    "subject_benchmark_condition": "subject_benchmark",
}

# Minimum cell count required for the triple-level cell to be entered
# into the lookup table (per the Batch-5 spec — the triple level is
# only meaningful when there's enough data to overcome shrinkage).
_MIN_COUNT_FOR_TRIPLE: int = 5

# Default kappa when too few siblings to estimate via method-of-moments.
_DEFAULT_KAPPA: float = 10.0

# Hard bounds on kappa per the spec. The lower bound prevents the
# fitted prior from degenerating to "trust the raw count" when sibling
# variance is extreme; the upper bound prevents the prior from
# collapsing to the parent when sibling rates are flat.
_KAPPA_MIN: float = 1.0
_KAPPA_MAX: float = 1000.0

# Subject-key fallback hash size (bytes → 16 hex chars).
_SUBJECT_HASH_BYTES: int = 8


# ---------------------------------------------------------------------------
# Subject key
# ---------------------------------------------------------------------------


def _visible_subject_key(subject_content: str) -> str:
    """Return the EB lookup key for a ``subject_content`` string.

    Per the spec: ``parse_subject_name`` is the primary key. When the
    ``Name:`` line is absent, fall back to a BLAKE2b-8-byte hex of the
    first 512 bytes of ``subject_content`` so two rows with otherwise-
    identical content still share a key.
    """
    name = parse_subject_name(subject_content or "")
    if name:
        return name
    truncated = (subject_content or "")[:512].encode("utf-8", errors="replace")
    return hashlib.blake2b(truncated, digest_size=_SUBJECT_HASH_BYTES).hexdigest()


# ---------------------------------------------------------------------------
# EBPriors
# ---------------------------------------------------------------------------


class EBPriors:
    """Hierarchical EB lookup predictor for ``P(correct)``.

    Implements both :class:`peval_gold.models.base.Predictor` (batched
    offline ``predict_proba``) and
    :class:`peval_gold.models.base.RuntimePredictor` (streaming
    ``predict_one`` for the the hosted runtime-style call shape).

    ``labeled`` is ignored at the EB level — the
    :class:`peval_gold.models.ensemble.LogitBlend` consumer handles
    online updates by blending an EB prediction with a calibrator's
    output. Keeping EB stateless across calls makes the predictor
    trivially safe to share across rounds without any reset bookkeeping.
    """

    def __init__(
        self,
        *,
        kappa_min: float = _KAPPA_MIN,
        kappa_max: float = _KAPPA_MAX,
        default_kappa: float = _DEFAULT_KAPPA,
        min_count_for_triple: int = _MIN_COUNT_FOR_TRIPLE,
    ) -> None:
        self._kappa_min = float(kappa_min)
        self._kappa_max = float(kappa_max)
        self._default_kappa = float(default_kappa)
        self._min_count_for_triple = int(min_count_for_triple)
        self._levels: dict[str, dict[Any, dict[str, float]]] = {lvl: {} for lvl in _ALL_LEVELS}
        # Per-level kappa estimate (set during fit).
        self._kappa_by_level: dict[str, float] = {}
        # Marker for "have we fit yet"; predict_* will use 0.5 fallback
        # if not, but downstream tests construct via fit() first.
        self._fitted: bool = False

    # ----- Protocol: Predictor -----------------------------------------

    def fit(
        self,
        train_rows: Sequence[Mapping[str, Any]],
        valid_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Walk ``train_rows`` once and populate every level of the hierarchy.

        ``valid_rows`` is accepted for API uniformity with other
        ``Predictor``s but is NOT used: EB priors do not need a held-
        out split because there is nothing to "early-stop"; the only
        free parameter is ``kappa`` and that is fit by method-of-moments
        on the same training data per the standard EB recipe.

        Raises
        ------
        ValueError
            If ``train_rows`` is empty.
        """
        del valid_rows  # not used; see docstring

        if not train_rows:
            raise ValueError("EBPriors.fit requires at least one training row")

        # ----- 1. Count (k, n) per cell at every level -----
        counts: dict[str, dict[Any, list[int]]] = {
            lvl: defaultdict(lambda: [0, 0]) for lvl in _ALL_LEVELS
        }

        # Per-row response coercion: tolerate ints, floats, bools; drop
        # NaN / non-binary (matches the D-7 binarization policy at the
        # data layer — see ``peval_gold.data.filters.binarize_drop``).
        n_seen = 0
        for r in train_rows:
            resp = r.get("response")
            if resp is None:
                continue
            if isinstance(resp, bool):
                y = 1 if resp else 0
            elif isinstance(resp, (int, float)):
                fv = float(resp)
                if math.isnan(fv) or fv not in (0.0, 1.0):
                    continue
                y = int(fv)
            else:
                continue

            subj = _visible_subject_key(str(r.get("subject_content", "")))
            bench = str(r.get("benchmark", ""))
            cond = str(r.get("condition", ""))

            # global
            cell = counts["global"][None]
            cell[0] += y
            cell[1] += 1
            # subject
            cell = counts["subject"][subj]
            cell[0] += y
            cell[1] += 1
            # benchmark
            cell = counts["benchmark"][bench]
            cell[0] += y
            cell[1] += 1
            # condition
            cell = counts["condition"][cond]
            cell[0] += y
            cell[1] += 1
            # subject_benchmark
            cell = counts["subject_benchmark"][(subj, bench)]
            cell[0] += y
            cell[1] += 1
            # subject_benchmark_condition
            cell = counts["subject_benchmark_condition"][(subj, bench, cond)]
            cell[0] += y
            cell[1] += 1
            n_seen += 1

        if n_seen == 0:
            raise ValueError("EBPriors.fit found 0 usable rows (no row had response in {0.0, 1.0})")

        # ----- 2. Compute global p_eb (Laplace prior over the entire dataset) -----
        gk, gn = counts["global"][None]
        # Use Laplace-style (alpha=beta=1) for the root level so that a
        # 0/0 corner-case can't happen and the global stays inside (0, 1).
        p_global = (gk + 1.0) / (gn + 2.0)
        self._levels["global"][None] = {
            "k": int(gk),
            "n": int(gn),
            "kappa": float(gn + 2.0),  # documentary; not used in formula
            "parent_p": 0.5,  # implicit Beta(1,1) prior
            "p_eb": float(p_global),
        }
        # Kappa at the root isn't meaningfully an MoM estimate; record
        # the implicit count of pseudo-observations (2) so the level
        # introspection helpers don't trip on a missing key.
        self._kappa_by_level["global"] = 2.0

        # ----- 3. Compute single-key levels: subject, benchmark, condition -----
        # Each shrinks toward p_global. Kappa is estimated from sibling
        # variance via method-of-moments and clipped to [kappa_min,
        # kappa_max]. Default to _default_kappa if too few siblings.
        for lvl in ("subject", "benchmark", "condition"):
            self._fit_single_level(lvl, counts[lvl], parent_p=p_global)

        # ----- 4. Compute subject_benchmark: shrink toward subject -----
        # For each (subj, bench), parent_p comes from subject cell's
        # p_eb. The kappa estimate at this level uses the sibling
        # variance across (subj, bench) cells.
        self._fit_two_key_level(
            "subject_benchmark",
            counts["subject_benchmark"],
            parent_level="subject",
            parent_key_fn=lambda key: key[0],
        )

        # ----- 5. Compute subject_benchmark_condition -----
        # Only retain cells with n >= _min_count_for_triple. Each cell's
        # parent_p is the (subj, bench) p_eb. Kappa estimated on the
        # filtered set.
        self._fit_three_key_level(
            "subject_benchmark_condition",
            counts["subject_benchmark_condition"],
            parent_level="subject_benchmark",
            parent_key_fn=lambda key: (key[0], key[1]),
            min_count=self._min_count_for_triple,
        )

        self._fitted = True

    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return a (N,) numpy array of EB-smoothed probabilities."""
        out = np.empty(len(rows), dtype=np.float64)
        for i, r in enumerate(rows):
            out[i] = self._lookup_one(r)
        return out

    def predict_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
        labeled: list[dict] | None = None,
    ) -> float:
        """Return one EB-smoothed probability for the given input dict.

        ``labeled`` is ignored at the EB level — see the module docstring.
        """
        del labeled
        return float(self._lookup_one(input))

    def save(self, path: str | Path) -> None:
        """Serialize the fitted lookup tables to compact JSON at ``path``."""
        payload: dict[str, Any] = {
            "__version__": 1,
            "kappa_min": self._kappa_min,
            "kappa_max": self._kappa_max,
            "default_kappa": self._default_kappa,
            "min_count_for_triple": self._min_count_for_triple,
            "kappa_by_level": self._kappa_by_level,
            "levels": {},
        }
        for lvl in _ALL_LEVELS:
            payload["levels"][lvl] = [
                # Tuple keys serialize as lists for JSON.
                {"key": _key_to_json(key), "cell": cell}
                for key, cell in self._levels[lvl].items()
            ]
        Path(path).write_text(json.dumps(payload, separators=(",", ":")))

    @classmethod
    def load(cls, path: str | Path) -> EBPriors:
        """Restore an EBPriors instance from a JSON file written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        eb = cls(
            kappa_min=payload.get("kappa_min", _KAPPA_MIN),
            kappa_max=payload.get("kappa_max", _KAPPA_MAX),
            default_kappa=payload.get("default_kappa", _DEFAULT_KAPPA),
            min_count_for_triple=payload.get("min_count_for_triple", _MIN_COUNT_FOR_TRIPLE),
        )
        eb._kappa_by_level = dict(payload.get("kappa_by_level", {}))
        for lvl, entries in payload.get("levels", {}).items():
            eb._levels[lvl] = {}
            for entry in entries:
                eb._levels[lvl][_key_from_json(entry["key"])] = {
                    "k": int(entry["cell"]["k"]),
                    "n": int(entry["cell"]["n"]),
                    "kappa": float(entry["cell"]["kappa"]),
                    "parent_p": float(entry["cell"]["parent_p"]),
                    "p_eb": float(entry["cell"]["p_eb"]),
                }
        eb._fitted = True
        return eb

    # ----- Introspection helpers (used by tests + walkthrough) ---------

    def get_cell_info(self, level: str, key: Any) -> dict[str, float] | None:
        """Return the fitted cell record for ``(level, key)``, or ``None``."""
        cell = self._levels.get(level, {}).get(key)
        if cell is None:
            return None
        return dict(cell)

    def get_level_kappa(self, level: str) -> float:
        """Return the per-level method-of-moments kappa estimate.

        Useful for diagnostics (e.g. "how much did the subject level
        shrink toward global?"). Not part of the Predictor protocol.
        """
        return float(self._kappa_by_level.get(level, self._default_kappa))

    def metadata(self) -> dict[str, Any]:
        """JSON-serializable provenance snapshot consumed by the evaluator."""
        return {
            "class": "EBPriors",
            "fitted": bool(self._fitted),
            "kappa_min": self._kappa_min,
            "kappa_max": self._kappa_max,
            "default_kappa": self._default_kappa,
            "min_count_for_triple": self._min_count_for_triple,
            "kappa_by_level": dict(self._kappa_by_level),
            "n_cells_by_level": {lvl: len(cells) for lvl, cells in self._levels.items()},
        }

    # ----- Internals ---------------------------------------------------

    def _fit_single_level(
        self,
        level: str,
        cells_counts: dict[Any, list[int]],
        *,
        parent_p: float,
    ) -> None:
        kappa = self._estimate_kappa_from_cells(cells_counts, parent_p)
        self._kappa_by_level[level] = kappa
        for key, (k, n) in cells_counts.items():
            p_eb = (k + kappa * parent_p) / (n + kappa)
            self._levels[level][key] = {
                "k": int(k),
                "n": int(n),
                "kappa": float(kappa),
                "parent_p": float(parent_p),
                "p_eb": float(p_eb),
            }

    def _fit_two_key_level(
        self,
        level: str,
        cells_counts: dict[Any, list[int]],
        *,
        parent_level: str,
        parent_key_fn,
    ) -> None:
        # Estimate kappa from sibling variance using the per-cell parent_p
        # (since different cells have different parents at this level).
        kappa = self._estimate_kappa_with_per_cell_parents(
            cells_counts, parent_level=parent_level, parent_key_fn=parent_key_fn
        )
        self._kappa_by_level[level] = kappa
        for key, (k, n) in cells_counts.items():
            parent_p = self._parent_p_for(parent_level, parent_key_fn(key))
            p_eb = (k + kappa * parent_p) / (n + kappa)
            self._levels[level][key] = {
                "k": int(k),
                "n": int(n),
                "kappa": float(kappa),
                "parent_p": float(parent_p),
                "p_eb": float(p_eb),
            }

    def _fit_three_key_level(
        self,
        level: str,
        cells_counts: dict[Any, list[int]],
        *,
        parent_level: str,
        parent_key_fn,
        min_count: int,
    ) -> None:
        filtered = {key: counts for key, counts in cells_counts.items() if counts[1] >= min_count}
        kappa = self._estimate_kappa_with_per_cell_parents(
            filtered, parent_level=parent_level, parent_key_fn=parent_key_fn
        )
        self._kappa_by_level[level] = kappa
        for key, (k, n) in filtered.items():
            parent_p = self._parent_p_for(parent_level, parent_key_fn(key))
            p_eb = (k + kappa * parent_p) / (n + kappa)
            self._levels[level][key] = {
                "k": int(k),
                "n": int(n),
                "kappa": float(kappa),
                "parent_p": float(parent_p),
                "p_eb": float(p_eb),
            }

    def _estimate_kappa_from_cells(
        self,
        cells_counts: dict[Any, list[int]],
        parent_p: float,
    ) -> float:
        """Method-of-moments kappa across siblings, clipped to [min, max]."""
        siblings = [(k, n) for k, n in cells_counts.values() if n > 0]
        if len(siblings) < 2:
            return self._default_kappa

        # Use a weighted mean (sum_k / sum_n) as the shared parent estimate
        # rather than parent_p — at the single-key levels they're equal by
        # construction, but using the within-level mean keeps the moment
        # equations consistent for any subset.
        total_k = sum(k for k, _ in siblings)
        total_n = sum(n for _, n in siblings)
        mu = total_k / total_n if total_n > 0 else parent_p

        # Per-cell observed proportions and variance contributions.
        p_hats = np.array([k / n for k, n in siblings], dtype=float)
        n_arr = np.array([n for _, n in siblings], dtype=float)

        # Weighted sample variance.
        weighted_var = float(np.sum(n_arr * (p_hats - mu) ** 2) / total_n)
        # Mean of the within-sample binomial variance: E[Var(p_hat | p)]
        # = mean(p_i * (1 - p_i) / n_i). Use the OBSERVED p_hat to keep
        # the estimate empirical.
        within = float(np.mean(p_hats * (1.0 - p_hats) / n_arr))

        between = weighted_var - within
        denom_max = mu * (1.0 - mu)

        if denom_max <= 0:
            kappa = self._kappa_max
        elif between <= 0:
            # No detectable between-sibling variance ⇒ siblings are
            # consistent with the parent; shrink heavily.
            kappa = self._kappa_max
        else:
            kappa = denom_max / between - 1.0

        return float(np.clip(kappa, self._kappa_min, self._kappa_max))

    def _estimate_kappa_with_per_cell_parents(
        self,
        cells_counts: dict[Any, list[int]],
        *,
        parent_level: str,
        parent_key_fn,
    ) -> float:
        """Kappa estimator when each cell has a different parent_p.

        We compute the variance of (p_hat - parent_p) instead of (p_hat - mu)
        so the moment equations remain unbiased under per-cell shrinkage.
        """
        siblings = []
        for key, (k, n) in cells_counts.items():
            if n <= 0:
                continue
            parent_p = self._parent_p_for(parent_level, parent_key_fn(key))
            siblings.append((k, n, parent_p))
        if len(siblings) < 2:
            return self._default_kappa

        k_arr = np.array([k for k, _, _ in siblings], dtype=float)
        n_arr = np.array([n for _, n, _ in siblings], dtype=float)
        parent_arr = np.array([pp for _, _, pp in siblings], dtype=float)
        p_arr = k_arr / n_arr

        total_n = float(n_arr.sum())
        weighted_var = float(np.sum(n_arr * (p_arr - parent_arr) ** 2) / total_n)
        within = float(np.mean(p_arr * (1.0 - p_arr) / n_arr))
        between = weighted_var - within

        # Use the weighted mean of parent_p * (1 - parent_p) as the
        # variance scale; this collapses to mu(1-mu) when all parents
        # are equal (matches the single-level estimator).
        denom_max = float(np.sum(n_arr * parent_arr * (1.0 - parent_arr)) / total_n)

        if denom_max <= 0:
            kappa = self._kappa_max
        elif between <= 0:
            kappa = self._kappa_max
        else:
            kappa = denom_max / between - 1.0

        return float(np.clip(kappa, self._kappa_min, self._kappa_max))

    def _parent_p_for(self, level: str, key: Any) -> float:
        """Return p_eb for ``(level, key)``; fall back through the chain if missing."""
        cell = self._levels.get(level, {}).get(key)
        if cell is not None:
            return float(cell["p_eb"])
        # Recurse up the parent chain until we hit global (which is always present).
        parent = _PARENT_LEVEL.get(level)
        if parent is None:
            return 0.5
        if parent == "global":
            global_cell = self._levels.get("global", {}).get(None)
            if global_cell is not None:
                return float(global_cell["p_eb"])
            return 0.5
        # For 2-key levels going up to 1-key, derive the parent key.
        if level == "subject_benchmark":
            return self._parent_p_for("subject", key[0])
        if level == "subject_benchmark_condition":
            return self._parent_p_for("subject_benchmark", (key[0], key[1]))
        # Fallback: global.
        global_cell = self._levels.get("global", {}).get(None)
        return float(global_cell["p_eb"]) if global_cell is not None else 0.5

    def _lookup_one(self, row: Mapping[str, Any]) -> float:
        """Walk the lookup chain and return the most-specific p_eb for ``row``."""
        subj = _visible_subject_key(str(row.get("subject_content", "")))
        bench = str(row.get("benchmark", ""))
        cond = str(row.get("condition", ""))

        keys_by_level: dict[str, Any] = {
            "subject_benchmark_condition": (subj, bench, cond),
            "subject_benchmark": (subj, bench),
            "subject": subj,
            "benchmark": bench,
            "global": None,
        }

        for lvl in _LOOKUP_CHAIN:
            cell = self._levels.get(lvl, {}).get(keys_by_level[lvl])
            if cell is not None:
                return float(cell["p_eb"])
        # Unreachable in practice — fit() always populates the global cell —
        # but keep a final defensive fallback so a partially-loaded EBPriors
        # never raises.
        return 0.5


# ---------------------------------------------------------------------------
# JSON key (de)serialization for tuple-keyed levels
# ---------------------------------------------------------------------------


def _key_to_json(key: Any) -> Any:
    """Convert dict keys (None, str, or tuple of str) into JSON-friendly form."""
    if key is None:
        return None
    if isinstance(key, tuple):
        return list(key)
    return str(key)


def _key_from_json(value: Any) -> Any:
    """Inverse of :func:`_key_to_json`."""
    if value is None:
        return None
    if isinstance(value, list):
        return tuple(value)
    return value


__all__ = ["EBPriors"]
