"""Stratified-bonus acquisition policy ( — S2D).

Combines a SimHash diversity score with a per-stratum coverage bonus,
parameterized over which (benchmark / condition / both) key defines
"stratum". This is a more aggressive version of the shipped
``the wrapped labeling module``'s stratum-coverage term: instead of the
shipped policy's fixed ``0.20 / (1 + count)`` weight, the bonus is
``bonus_weight / (1 + count_in_stratum)`` with a controllable weight.

The intent is to see whether nudging the policy harder toward
underrepresented strata (which directly maps to the K-per-data-category
selection on the platform) improves the adaptive K=5 NLL relative to
the shipped policy's gentler bonus.

Three stratifier choices in the Batch-7 grid:

- ``"benchmark"`` — group by ``input["benchmark"]`` only. Matches what
  ``peval_gold.data.splits.adaptive_label_simulation`` uses as its
  default category proxy.
- ``"condition"`` — group by ``input["condition"]`` only. Useful for
  stress-testing whether the policy can prioritize across the smaller
  condition cardinality.
- ``"benchmark_condition"`` — group by the (benchmark, condition)
  tuple. Closest analog to the kit's hidden "data category" (which
  organizers say is internal but is at minimum benchmark-aware per
  ``starting_kit/README.md:266``).

Stdlib-only: ``hashlib``, ``math``, ``re``. Importing this module
MUST NOT pull in torch / numpy / sentence_transformers / transformers
— locked in by the shootout suite's import-time audit.
"""

from __future__ import annotations

import hashlib
import math
import re

_BITS = 64
_MAX_TOKENS = 256
_MAX_STRATA = 256  # mirrors the shipped policy's bound
_TIE_EPSILON = 0.01
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_VALID_STRATIFIERS = {"benchmark", "condition", "benchmark_condition"}


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _visible_text(ex: dict) -> str:
    return "\n".join(
        (
            _text(ex.get("benchmark")),
            _text(ex.get("condition")),
            _text(ex.get("subject_content")),
            _text(ex.get("item_content")),
        )
    ).lower()


def _tokens(ex: dict) -> list[str]:
    return _TOKEN_RE.findall(_visible_text(ex))[:_MAX_TOKENS]


def _hash_u64(text: str, person: bytes = b"strat-v1") -> int:
    digest = hashlib.blake2b(
        text.encode("utf-8", "replace"),
        digest_size=8,
        person=person,
    ).digest()
    return int.from_bytes(digest, "little")


def _simhash(ex: dict) -> int:
    accum = [0] * _BITS
    toks = _tokens(ex)
    if not toks:
        toks = ["<empty>"]
    for token in toks:
        h = _hash_u64(token, person=b"simhash-v1")
        for bit in range(_BITS):
            accum[bit] += 1 if (h >> bit) & 1 else -1
    signature = 0
    for bit, value in enumerate(accum):
        if value >= 0:
            signature |= 1 << bit
    return signature


class StratifiedBonus:
    """Diversity + parametric stratum-coverage bonus.

    Parameters
    ----------
    stratifier : str
        One of ``"benchmark"`` / ``"condition"`` / ``"benchmark_condition"``.
        Defines the key on which the policy tracks per-stratum counts.
    bonus_weight : float
        Multiplier on the ``1 / (1 + count)`` shape; defaults to 0.5.
        Higher values push harder toward underrepresented strata at the
        expense of diversity-coverage signal.

    Output range
    ------------
    ``[0.0, 1.0 + bonus_weight + tie_epsilon]`` upper bound; clamped to
    ``[0.0, 2.0]`` to match the shipped policy's reporting range.
    """

    def __init__(
        self,
        stratifier: str = "benchmark",
        bonus_weight: float = 0.5,
    ) -> None:
        if stratifier not in _VALID_STRATIFIERS:
            raise ValueError(
                f"stratifier must be one of {sorted(_VALID_STRATIFIERS)!r}, "
                f"got {stratifier!r}"
            )
        if not math.isfinite(bonus_weight) or bonus_weight < 0:
            raise ValueError(
                f"bonus_weight must be a non-negative finite float, "
                f"got {bonus_weight!r}"
            )
        self._stratifier = stratifier
        self._bonus_weight = float(bonus_weight)
        self._reservoir: list[int] = []
        self._reservoir_cap = 128  # fixed; not the parameter under sweep here
        self._stratum_counts: dict[str, int] = {}
        self._candidate_count: int = 0

    def score_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
    ) -> float:
        try:
            ex = dict(input or {})
            signature = _simhash(ex)
            diversity = self._diversity_score(signature)
            bonus = self._stratum_bonus(ex)
            tie_break = self._tie_break(ex)
            score = diversity + bonus + tie_break
            self._candidate_count += 1
            self._update_reservoir(signature, ex)
            self._update_stratum(ex)
            return self._clamp(score)
        except Exception:  # pylint: disable=broad-except
            return 0.0

    def reset(self) -> None:
        self._reservoir.clear()
        self._stratum_counts.clear()
        self._candidate_count = 0

    # ----- internals -------------------------------------------------

    def _stratum_key(self, ex: dict) -> str:
        benchmark = _text(ex.get("benchmark")).lower()[:64]
        condition = _text(ex.get("condition")).lower()[:64]
        if self._stratifier == "benchmark":
            return benchmark
        if self._stratifier == "condition":
            return condition
        return f"{benchmark}|{condition}"

    def _stratum_bonus(self, ex: dict) -> float:
        key = self._stratum_key(ex)
        count = self._stratum_counts.get(key, 0)
        return self._bonus_weight / float(1 + count)

    def _update_stratum(self, ex: dict) -> None:
        key = self._stratum_key(ex)
        if key in self._stratum_counts or len(self._stratum_counts) < _MAX_STRATA:
            self._stratum_counts[key] = self._stratum_counts.get(key, 0) + 1

    def _diversity_score(self, signature: int) -> float:
        if not self._reservoir:
            return 1.0
        nearest = min(
            (signature ^ old).bit_count() for old in self._reservoir
        )
        return nearest / float(_BITS)

    def _update_reservoir(self, signature: int, ex: dict) -> None:
        if len(self._reservoir) < self._reservoir_cap:
            self._reservoir.append(signature)
            return
        if self._candidate_count == 0:
            return
        candidate_key = _visible_text(ex)
        slot = _hash_u64(
            f"{candidate_key}\n{self._candidate_count}",
            person=b"reservoir-v1",
        ) % self._candidate_count
        if slot < self._reservoir_cap:
            self._reservoir[slot] = signature

    @staticmethod
    def _tie_break(ex: dict) -> float:
        """Small deterministic jitter so ties don't collapse the K-top-K
        selection. Uses a different ``person`` from the simhash hash so
        the tie-break signal is statistically independent.
        """
        key = "\n".join(
            (
                _text(ex.get("benchmark")),
                _text(ex.get("condition")),
                _text(ex.get("subject_content")),
                _text(ex.get("item_content")),
            )
        )
        unit = _hash_u64(key, person=b"tie-v1") / float(1 << 64)
        return unit * _TIE_EPSILON

    @staticmethod
    def _clamp(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return float(max(0.0, min(2.0, value)))


__all__ = ["StratifiedBonus"]
