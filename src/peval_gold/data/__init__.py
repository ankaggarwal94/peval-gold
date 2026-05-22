"""Data substrate for the gold-track laboratory.

This subpackage owns the offline data plane:

- :mod:`peval_gold.data.normalize` — canonical row schema + coercion
  helpers. Accepts both the raw HF response shape (``benchmark_id`` +
  ``test_condition``) and the joined ``predict()``-input shape
  (``benchmark`` + ``condition``).
- :mod:`peval_gold.data.filters` — D-7 ``binarize_drop`` (default) plus
  the explicit-ablation ``binarize_median`` / ``binarize_soft`` modes.
- :mod:`peval_gold.data.hf_loader` — thin wrapper over
  ``aims-foundations/measurement-db`` at the pinned revision
  ``589ccfdb8e82``. Joins responses to subjects/items/benchmarks
  registries and returns canonical rows.
- :mod:`peval_gold.data.registry` — typed lookup-dict helpers for the
  three registry parquets.
- :mod:`peval_gold.data.splits` — six validation-grade splits:
  ``random_row_smoke``, ``item_holdout_primary`` (promotion grade),
  ``benchmark_holdout_stress``, ``domain_holdout``, ``subject_holdout``,
  ``adaptive_label_simulation``.

This subpackage is deliberately disjoint from ``notebooks/load_data.py``
and ``the training pipeline``: the production NCF trainer + the
existing offline loader stay untouched. The gold-track substrate is
parallel infrastructure that future Batch-4+ challengers will use to
pass the item-held-out / adaptive-simulation gates before any new
artifact can ever displace ``the weights file``.
"""

__all__: list[str] = []
