"""Acquisition-policy Protocol for the gold-track laboratory.

The the hosted runtime platform streams candidate inputs to
``acquisition_function(input)`` one at a time. The policy returns a
score; the top-K per data category (currently K=5 per
``starting_kit/README.md:268-271``) are revealed to the next round of
``predict()`` calls as labeled rows.

The :class:`AcquisitionPolicy` Protocol below mirrors that streaming
contract: one row in, one finite score out. Implementations may
maintain hidden state across :meth:`score_one` calls within a round
(e.g. running farthest-point selection) and must clear that state in
:meth:`reset` so the next round (or unit test) starts clean.

NaN poisoning policy: per the documented design rule in
(project pattern doc),
returning NaN/inf triggers a per-round random fallback for *every*
label in that round. Implementations MUST return a finite float and
MUST NOT silently catch and swallow exceptions to a NaN. The Protocol
does not enforce this at the type-system level, but
:func:`peval_gold.eval.metrics.expected_calibration_error` and the
offline simulator () will refuse to operate on non-finite
scores.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AcquisitionPolicy(Protocol):
    """Per-candidate streaming acquisition score.

    - :meth:`score_one` — given a single input row dict, return a finite
      ``float`` score. Higher is "more worth labeling" by the policy's
      definition.
    - :meth:`reset` — clear any per-round hidden state (e.g. the seen
      set in a diversity policy) so the next round is independent.
    """

    def score_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
    ) -> float: ...

    def reset(self) -> None: ...
