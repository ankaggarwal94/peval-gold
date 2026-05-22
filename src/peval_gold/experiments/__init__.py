"""Experiment-tracking helpers for the gold-track laboratory.

Currently exports the :mod:`peval_gold.experiments.ledger` module which
provides:

- ``new_run_id(prefix) -> str`` — timestamp + counter ⇒ unique id.
- ``write_manifest(run_id, config, files, metadata, root=...) -> str`` —
  writes ``<root>/<run_id>/manifest.json``.
- ``append_result(run_id, payload, root=...) -> None`` — append-only
  JSONL line to ``<root>/<run_id>/results.jsonl``.

The parent ``runs/run_ledger.json`` (autonomy envelope, family
caps, anchor SHAs) is intentionally NOT touched by the ledger helpers
— that file is parent-owned per the gold-track  plan.
"""

from peval_gold.experiments.ledger import (
    append_result,
    new_run_id,
    write_manifest,
)

__all__ = ["append_result", "new_run_id", "write_manifest"]
