"""Acquisition policy abstractions for the gold-track laboratory.

Exports:

- :class:`AcquisitionPolicy` Protocol ().
- :class:`CurrentSimHash` — Batch-3 wrapper around the shipped
  ``the wrapped labeling module:acquisition_function`` (stdlib-only;
  importing this submodule does NOT pull in torch / numpy / encoder).

Future challengers (uncertainty, learned diversity, hybrid) land in
sibling modules added by later batches.
"""

from peval_gold.acquisition.base import AcquisitionPolicy
from peval_gold.acquisition.simhash import CurrentSimHash

__all__ = ["AcquisitionPolicy", "CurrentSimHash"]
