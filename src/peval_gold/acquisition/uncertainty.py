"""Uncertainty-proxy acquisition stub ( — S2D).

# RUNTIME_UNSAFE_DO_NOT_SHIP

This module is a placeholder for a future uncertainty-weighted
acquisition policy. The contract slot would be "ask the predictor for
its uncertainty on each candidate; pick the highest-uncertainty rows as
the K=5 to label". That contract is realizable in principle, but on the
hosted the hosted runtime platform it would require either:

1. Running the encoder + NCF head at acquisition time (the path the
   shipped ``the wrapped labeling module`` was REWRITTEN to avoid in the
    the wrapped delta — the prior encoder-per-candidate path was
   too slow to ship and triggered per-round timeouts; see
   (project decision doc)),
   OR
2. Caching a precomputed uncertainty signal across rounds (impossible —
   the kit explicitly states state does NOT persist across rounds per
   ``starting_kit/README.md:235-238`` and the project's CLAUDE.md
   "Critical Constraints" section), OR
3. A cheap stdlib-only uncertainty proxy (e.g., hash-based query-by-
   committee) that has not yet been designed.

Until option 3 has both a design AND a passing
``tests/test_gold_acquisition.test_policy_score_one_p95_under_two_ms``
gate, this module returns a constant ``0.5`` and is documented as
ship-blocked. The stub exists so:

- The Batch-7 shootout has a placeholder row in its grid, making it
  visible to a future agent that an uncertainty challenger is "in the
  plan but not yet built". A future shootout can compare a real
  uncertainty proxy head-to-head with the SimHash / stratified
  baselines without having to re-wire the grid.
- The runtime-safety test
  ``tests/test_gold_acquisition.test_uncertainty_proxy_specifically_does_not_load_encoder``
  pins in place the constraint that even an uncertainty CONTRACT slot
  must not pull in torch / sentence_transformers at import time. Anyone
  attempting a "real" implementation will trip the test before
  shipping a runtime explosion.

DO NOT remove the ``RUNTIME_UNSAFE_DO_NOT_SHIP`` marker above unless
you have replaced this stub with a fully-tested, sub-2 ms p95,
stdlib-only proxy.

Stdlib-only: no imports. Importing this module MUST NOT pull in torch /
numpy / sentence_transformers / transformers.
"""

from __future__ import annotations


class UncertaintyProxy:
    """Placeholder uncertainty-weighted acquisition scorer.

    Returns a constant 0.5 so any K-top-K selection driven by this
    policy degenerates to "stable order from the platform stream" —
    effectively a no-op acquisition signal. This is *intentional*:
    treating the stub as if it were a real challenger would produce a
    misleading "uncertainty beats SimHash" result with NLL == NLL of
    whatever the platform's natural stream order happens to score
    against the predictor.

    See module docstring for the full ship-block rationale.
    """

    def score_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
    ) -> float:
        # Constant 0.5 — sentinel value documenting "no uncertainty
        # signal available yet". The distinguishable-defensive-
        # fallback principle says: prefer a sentinel that's easy to
        # detect post-hoc over a plausible-but-wrong number. 0.5 in
        # ``[0, 1]`` is unambiguously midpoint-of-range and shows up
        # in the comparison-table as a "no signal" cell.
        return 0.5

    def reset(self) -> None:
        """No state to clear."""


__all__ = ["UncertaintyProxy"]
