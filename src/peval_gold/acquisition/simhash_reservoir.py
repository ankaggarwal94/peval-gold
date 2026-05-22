"""Parameterized-reservoir SimHash diversity policy ( — S2D).

Generalizes the diversity track of the shipped ``the wrapped labeling module``
to a configurable reservoir size, so the  shootout can sweep
32 / 64 / 128 / 256 and learn whether the shipped ``_MAX_SEEN = 128``
is a tuned local optimum or an arbitrary choice that happens to ship.

Algorithm
---------

1. Compute a 64-bit SimHash signature of the candidate's visible text
   (same tokenizer + bit-accumulator scheme as
   ``the wrapped labeling module:_simhash``).
2. ``diversity_score`` = min over the in-memory reservoir of Hamming
   distance between the candidate signature and each reservoir entry,
   normalized by ``bits`` so the score lands in ``[0, 1]``.
   - Empty reservoir → diversity-floor 1.0 (no comparison possible).
   - Match-in-reservoir → 0.0 (identical signature already seen).
3. Update reservoir: append if under capacity; otherwise pick a slot
   via a deterministic blake2b-keyed modulo so old entries get rolled
   in/out without bias toward early or late candidates.

Differences from the shipped ``the wrapped labeling module``
-------------------------------------------------------

The shipped policy combines diversity + a small stratum-bonus +
tie-break into a clamped score in ``[0, 2]``. This class returns ONLY
the diversity component (in ``[0, 1]``) so the Batch-7 grid can isolate
how much the reservoir-size knob alone affects label quality without
also drifting the stratum-bonus weight or tie-break epsilon.

Stratified bonuses live in a separate :class:`peval_gold.acquisition.stratified.StratifiedBonus`
challenger so the grid can A/B each dimension independently.

Stdlib-only — ``hashlib``, ``math``, ``re``. Importing this module
MUST NOT pull in torch / numpy / sentence_transformers / transformers
(locked in by the shootout suite's import-time audit).
"""

from __future__ import annotations

import hashlib
import math
import re

_DEFAULT_BITS = 64
_DEFAULT_RESERVOIR = 128
_MAX_TOKENS = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _visible_text(ex: dict) -> str:
    """Same 4-field concatenation as ``the wrapped labeling module:_visible_text``."""
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


def _hash_u64(text: str, person: bytes = b"simhash-v1") -> int:
    digest = hashlib.blake2b(
        text.encode("utf-8", "replace"),
        digest_size=8,
        person=person,
    ).digest()
    return int.from_bytes(digest, "little")


def _simhash(ex: dict, bits: int) -> int:
    """SimHash signature over the candidate's visible-text tokens.

    Same bit-accumulator scheme as ``the wrapped labeling module:_simhash``:
    sum +1 / -1 votes per bit across all tokens, then collapse to a
    bit mask where ``accum[bit] >= 0 → 1`` else ``0``.

    The ``<empty>`` fallback for the no-token case keeps the function
    total — important because the contract says ``score_one`` MUST
    return a finite float even on edge inputs like ``{}``.
    """
    accum = [0] * bits
    toks = _tokens(ex)
    if not toks:
        toks = ["<empty>"]

    for token in toks:
        h = _hash_u64(token, person=b"simhash-v1")
        for bit in range(bits):
            accum[bit] += 1 if (h >> bit) & 1 else -1

    signature = 0
    for bit, value in enumerate(accum):
        if value >= 0:
            signature |= 1 << bit
    return signature


class SimHashReservoir:
    """Parameterized SimHash-diversity acquisition policy.

    Parameters
    ----------
    reservoir_size : int
        Maximum number of signatures to keep in the in-memory reservoir.
        The shipped policy uses 128; the Batch-7 grid sweeps 32 / 64 /
        128 / 256.
    bits : int
        SimHash signature width in bits. Defaults to 64 to match the
        shipped ``the wrapped labeling module:_BITS``.

    Output range
    ------------
    ``[0.0, 1.0]``. The diversity score is the Hamming distance from
    the candidate signature to its nearest neighbor in the reservoir,
    normalized by ``bits``. An empty reservoir (cold start) returns
    1.0 by convention.
    """

    def __init__(self, reservoir_size: int = _DEFAULT_RESERVOIR, bits: int = _DEFAULT_BITS) -> None:
        if reservoir_size < 1:
            raise ValueError(f"reservoir_size must be >= 1, got {reservoir_size!r}")
        if bits < 8 or bits % 8 != 0:
            raise ValueError(f"bits must be a positive multiple of 8, got {bits!r}")
        self._reservoir_size = int(reservoir_size)
        self._bits = int(bits)
        self._seen_signatures: list[int] = []
        self._candidate_count: int = 0

    def score_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
    ) -> float:
        try:
            ex = dict(input or {})
            signature = _simhash(ex, self._bits)
            diversity = self._diversity_score(signature)
            self._candidate_count += 1
            self._update_reservoir(signature, ex)
            return self._clamp(diversity)
        except Exception:  # pylint: disable=broad-except
            # NaN poisoning policy: never bubble exceptions; return
            # 0.0 so the round still makes forward progress. The
            # distinguishable-fallback principle is preserved because
            # the score 0.0 means "least diverse" — the platform will
            # rank this candidate at the bottom, which is the right
            # signal for a candidate that could not be processed.
            return 0.0

    def reset(self) -> None:
        self._seen_signatures.clear()
        self._candidate_count = 0

    # ----- internals -------------------------------------------------

    def _diversity_score(self, signature: int) -> float:
        if not self._seen_signatures:
            return 1.0
        nearest = min((signature ^ old).bit_count() for old in self._seen_signatures)
        return nearest / float(self._bits)

    def _update_reservoir(self, signature: int, ex: dict) -> None:
        """Append-or-replace policy keyed by a deterministic hash.

        Below capacity: just append. At capacity: pick a slot in
        ``[0, _candidate_count)`` from a blake2b digest seeded by the
        candidate's visible-text + the current count, and if the slot
        index is inside the reservoir, replace that entry. This is the
        same "reservoir sampling without retaining size estimate"
        scheme used by ``the wrapped labeling module:_update_reservoir`` and
        gives an unbiased random replacement that stays deterministic
        across runs.
        """
        if len(self._seen_signatures) < self._reservoir_size:
            self._seen_signatures.append(signature)
            return
        if self._candidate_count == 0:  # defensive — should never hit
            return
        candidate_key = _visible_text(ex)
        slot = (
            _hash_u64(
                f"{candidate_key}\n{self._candidate_count}",
                person=b"reservoir-v1",
            )
            % self._candidate_count
        )
        if slot < self._reservoir_size:
            self._seen_signatures[slot] = signature

    @staticmethod
    def _clamp(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return float(max(0.0, min(1.0, value)))


__all__ = ["SimHashReservoir"]
