"""Wrapper for a user-supplied ``labeling.py`` SimHash acquisition policy.

This class wraps an EXTERNAL ``labeling.py`` artifact the caller supplies
via constructor arg — it does NOT ship any labeling source. The labeling
module is expected to expose:

- ``acquisition_function(input: dict) -> float`` — the scoring callable.
- ``_seen_signatures`` (list-like, mutable) — per-round seen-signatures state.
- ``_stratum_counts`` (dict-like, mutable) — per-round stratum coverage state.
- ``_candidate_count`` (int) — per-round candidate counter.

The wrapper carries no instance state of its own — it delegates every call
to the loaded labeling module. :meth:`reset` zeroes the three state names
back to their declared defaults so each evaluation round is independent.

Implementation notes
--------------------

- The labeling module is expected to be stdlib-only (``hashlib``, ``math``,
  ``re``) with no network or filesystem side effects, so importing it via
  :mod:`importlib.util` is safe.
- Two ``CurrentSimHash`` instances pointed at the SAME ``labeling_path``
  share the same backing module state by design. Instances pointed at
  DIFFERENT paths are independent.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any

from peval_gold.acquisition.base import AcquisitionPolicy as _AcquisitionPolicy


def _load_labeling_module(labeling_path: Path) -> Any:
    """Load a user-supplied ``labeling.py`` under a private unique name."""
    if not labeling_path.exists():
        raise FileNotFoundError(
            f"labeling.py not found at {labeling_path}; the "
            "CurrentSimHash wrapper requires a user-supplied labeling artifact."
        )
    spec = importlib.util.spec_from_file_location(
        f"peval_gold._loaded_labeling_{hash(str(labeling_path)) & 0xFFFFFFFF:x}",
        labeling_path,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(
            f"could not build spec for {labeling_path}; "
            "Python import machinery returned None"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in (
        "acquisition_function",
        "_seen_signatures",
        "_stratum_counts",
        "_candidate_count",
    ):
        if not hasattr(module, attr):
            raise AttributeError(
                f"loaded labeling.py at {labeling_path} is missing required "
                f"attribute {attr!r}. The wrapper expects acquisition_function() "
                "plus the three per-round state names _seen_signatures, "
                "_stratum_counts, _candidate_count."
            )
    return module


class CurrentSimHash:
    """:class:`AcquisitionPolicy` wrapper for a user-supplied SimHash policy.

    Parameters
    ----------
    labeling_path : str | os.PathLike
        REQUIRED. Path to a Python file that defines
        ``acquisition_function(input: dict) -> float`` plus the three
        per-round state names.
    """

    def __init__(self, labeling_path: str | os.PathLike[str]) -> None:
        self._labeling_path = Path(labeling_path).resolve()
        self._module = _load_labeling_module(self._labeling_path)

    # pylint: disable=redefined-builtin
    def score_one(
        self,
        input: dict,  # noqa: A002 - intentionally shadows builtin to match upstream contract
    ) -> float:
        """Delegate to the loaded labeling module's ``acquisition_function``."""
        score = self._module.acquisition_function(input)
        if not isinstance(score, float) or not math.isfinite(score):
            raise RuntimeError(
                f"acquisition_function returned a non-finite value {score!r}; "
                "this is a regression in the wrapped labeling module."
            )
        return score

    def reset(self) -> None:
        """Zero the loaded module's per-round state."""
        # pylint: disable=protected-access
        self._module._seen_signatures.clear()
        self._module._stratum_counts.clear()
        self._module._candidate_count = 0


__all__ = ["CurrentSimHash"]
