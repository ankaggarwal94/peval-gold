"""Deterministic-random acquisition baseline ( — S2D).

The simplest possible challenger to the shipped SimHash policy: hash the
candidate's visible text together with the per-round call index and map
the digest to a number in ``[0, 1)``. No diversity tracking, no
stratification — just per-call deterministic randomness keyed by both
content and position.

Why this baseline matters:

- It cleanly answers the question "does CurrentSimHash actually buy us
  anything over uniform-random selection of K labels per category?"
  If RandomAcquisition's adaptive K=5 NLL is statistically
  indistinguishable from CurrentSimHash, the SimHash machinery is
  burning runtime budget for no measurable gain and the shipped
  ``the wrapped labeling module`` can be simplified.
- The ``str(_n_seen)`` mix-in makes the policy deterministic across
  runs *but* sensitive to call order, which matches the streaming-API
  reality on the platform (candidates arrive in some platform-chosen
  order — the policy can't see the full set up front).

Stdlib-only: ``hashlib`` is the sole non-builtin import. Importing this
module MUST NOT pull in torch / numpy / sentence_transformers /
transformers — that invariant is locked in by
``tests/test_gold_acquisition.test_policy_module_does_not_import_heavy_ml_stack``.
"""

from __future__ import annotations

import hashlib

_DIGEST_BYTES = 4
_RANGE = float(1 << (8 * _DIGEST_BYTES))  # 2 ** 32


def _visible_text(input_dict: dict) -> str:
    """Concatenate the four visible kit fields into one string.

    Mirrors the convention used by ``the wrapped labeling module:_visible_text``
    so the random baseline keys off the same content surface as the
    shipped SimHash policy. ``None`` values become empty strings; missing
    keys are silently treated as empty.
    """
    parts = (
        input_dict.get("benchmark"),
        input_dict.get("condition"),
        input_dict.get("subject_content"),
        input_dict.get("item_content"),
    )
    return "\n".join("" if p is None else str(p) for p in parts)


class RandomAcquisition:
    """Deterministic per-call random scorer.

    State
    -----
    ``_n_seen`` (int): incremented each :meth:`score_one` call. Mixed
    into the hash so two identical candidates seen at different
    positions in the stream receive different scores. :meth:`reset`
    zeroes it.

    Output range
    ------------
    ``[0.0, 1.0)`` — the 32-bit digest divided by ``2**32``.
    """

    def __init__(self) -> None:
        self._n_seen: int = 0

    def score_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
    ) -> float:
        content = _visible_text(input) + str(self._n_seen)
        digest = hashlib.blake2b(
            content.encode("utf-8", "replace"), digest_size=_DIGEST_BYTES
        ).digest()
        self._n_seen += 1
        return float(int.from_bytes(digest, "big")) / _RANGE

    def reset(self) -> None:
        self._n_seen = 0


__all__ = ["RandomAcquisition"]
