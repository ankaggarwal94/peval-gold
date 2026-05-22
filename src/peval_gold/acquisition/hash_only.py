"""Content-hash-only acquisition baseline ( — S2D).

The maximally-stateless challenger: score = blake2b(visible_text) /
2**32. No reservoir, no stratification, no per-call counter. Two
identical candidates at any position in the stream receive identical
scores; two different candidates produce uniformly-distributed scores
in ``[0, 1)``.

The point of PureHash is to isolate the SIGNAL question from the STATE
question. RandomAcquisition mixes the call index into the hash to make
duplicate candidates produce different scores — which models the
"don't pick the same row twice" intent of the SimHash diversity track
but does it via a positional jitter rather than via actual diversity
measurement. PureHash strips even that, leaving a pure
content-hash-as-priority policy. If PureHash beats CurrentSimHash, it
means the SimHash machinery is OVER-engineering the problem and the
right policy is something close to uniform-random over uniquely-keyed
content.

Stdlib-only: ``hashlib`` is the sole non-builtin import. Importing this
module MUST NOT pull in torch / numpy / sentence_transformers /
transformers — locked in by
``tests/test_gold_acquisition.test_policy_module_does_not_import_heavy_ml_stack``.
"""

from __future__ import annotations

import hashlib

_DIGEST_BYTES = 4
_RANGE = float(1 << (8 * _DIGEST_BYTES))  # 2 ** 32


def _visible_text(input_dict: dict) -> str:
    """Concatenate the four visible kit fields. Matches RandomAcquisition's
    helper to keep the two baselines comparing the same content surface.
    """
    parts = (
        input_dict.get("benchmark"),
        input_dict.get("condition"),
        input_dict.get("subject_content"),
        input_dict.get("item_content"),
    )
    return "\n".join("" if p is None else str(p) for p in parts)


class PureHash:
    """Stateless content-hash priority scorer.

    :meth:`reset` is a no-op (there is no state to clear). It is still
    implemented so :class:`PureHash` satisfies the
    :class:`peval_gold.acquisition.base.AcquisitionPolicy` Protocol.

    Output range
    ------------
    ``[0.0, 1.0)`` — the 32-bit digest divided by ``2**32``.
    """

    def score_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
    ) -> float:
        digest = hashlib.blake2b(
            _visible_text(input).encode("utf-8", "replace"),
            digest_size=_DIGEST_BYTES,
        ).digest()
        return float(int.from_bytes(digest, "big")) / _RANGE

    def reset(self) -> None:
        """No state to clear; provided for Protocol conformance."""


__all__ = ["PureHash"]
