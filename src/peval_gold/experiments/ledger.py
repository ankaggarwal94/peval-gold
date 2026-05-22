"""Append-only run ledger for the gold-track laboratory.

The parent ``runs/run_ledger.json`` (the hosted runtime daily caps, family
counters, anchor SHAs) is **parent-owned** — this module never touches
it. The helpers here own only the per-run subdirectories
``runs/<run_id>/`` containing ``manifest.json`` (config + files +
metadata snapshot at run start) and ``results.jsonl`` (one line per
emitted result; append-only so a fresh agent can replay the run by
reading lines in order).

Design choices
--------------

- ``new_run_id`` uses ISO-style ``YYYYMMDDTHHMMSSZ`` plus a monotonic
  counter so two back-to-back calls in the same wall-clock second
  still produce distinct ids. This avoids collisions when the
  evaluator scripts get auto-batched.
- ``write_manifest`` is a one-shot write (rewrites the whole JSON each
  call); the schema lives entirely inside one file so a fresh reader
  doesn't have to walk a directory tree to learn the run shape.
- ``append_result`` uses ``open(path, 'a')`` with one JSON-encoded
  payload per line — the canonical JSONL pattern. Survives process
  crashes between calls (every appended line is durable once flushed).
- The ``root`` argument defaults to ``RUNS_ROOT`` (``runs/gold`` under
  the repo root) but is callable-injected for the test suite so unit
  tests can use ``tmp_path`` without monkey-patching globals.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPO_ROOT / "runs" / "gold"

# Module-level monotonic counter so same-second back-to-back run_ids
# don't collide. Thread-safe via a tiny lock.
_id_lock = threading.Lock()
_id_counter = 0


def _next_counter() -> int:
    global _id_counter  # noqa: PLW0603 - intentional process-wide counter
    with _id_lock:
        _id_counter += 1
        return _id_counter


def new_run_id(prefix: str = "run") -> str:
    """Return a unique timestamp-based run id like ``run-20260521T223300Z-1``.

    Deterministic given a fixed wall clock + counter starting point.
    Two back-to-back calls always differ because of the appended
    counter.

    Parameters
    ----------
    prefix : str
        Free-text prefix; the convention is one prefix per run family
        (e.g. ``current_ncf``, ``baseline_const``, ``challenger_xgboost``).
    """
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}-{_next_counter()}"


def _run_dir(run_id: str, root: Path | None = None) -> Path:
    """Compute ``<root>/<run_id>`` and ensure parents exist.

    Note: ``root`` can be either the ``runs/`` parent (default
    behavior) OR a tmp_path supplied by tests; in the latter case the
    run dir is ``<tmp_path>/<run_id>`` directly (no extra
    ``runs/`` nesting). This makes the ``root=tmp_path`` injection
    natural to write in unit tests without having to mkdir
    ``tmp_path/runs/`` first.
    """
    base = Path(root) if root is not None else RUNS_ROOT
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(
    run_id: str,
    config: Mapping[str, Any],
    files: list[str],
    metadata: Mapping[str, Any],
    root: Path | str | None = None,
) -> str:
    """Write ``<root>/<run_id>/manifest.json``; return the path as ``str``.

    The manifest is the durable snapshot of what was run:

    - ``run_id`` (echo for grep-ability).
    - ``config`` — the predictor + split + acquisition config used.
    - ``files`` — list of file paths involved (weights, configs).
    - ``metadata`` — agent + git sha + anything provenance-ish.
    - ``created_utc`` — ISO-8601 wall clock at write time.
    """
    run_dir = _run_dir(run_id, root=Path(root) if root is not None else None)
    manifest_path = run_dir / "manifest.json"
    payload = {
        "run_id": run_id,
        "config": dict(config),
        "files": list(files),
        "metadata": dict(metadata),
        "created_utc": _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, default=str))
    return str(manifest_path)


def append_result(
    run_id: str,
    payload: Mapping[str, Any],
    root: Path | str | None = None,
) -> None:
    """Append one JSON-encoded line to ``<root>/<run_id>/results.jsonl``.

    Multiple calls append additional lines — the file is never
    rewritten. Each line is a single JSON object suitable for
    ``json.loads`` per line.

    Crash-safety: the append is followed by ``fh.flush()`` +
    ``os.fsync`` so every committed line survives a process crash
    between calls.
    """
    run_dir = _run_dir(run_id, root=Path(root) if root is not None else None)
    results_path = run_dir / "results.jsonl"
    line = json.dumps(payload, default=str)
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.write("\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            # fsync is best-effort: skip silently on filesystems that
            # don't implement it (tmpfs in some sandboxes).
            pass


# Helper: tiny sleep to break wall-clock ties in tests that need
# distinct timestamps regardless of the counter. Not used by the
# normal API surface; lives here so test code can monkey-patch it if
# needed.
_TIE_BREAK_SLEEP_S = 0.0


def _maybe_tie_break() -> None:  # pragma: no cover - test helper hook
    if _TIE_BREAK_SLEEP_S > 0:
        time.sleep(_TIE_BREAK_SLEEP_S)


__all__ = [
    "RUNS_ROOT",
    "append_result",
    "new_run_id",
    "write_manifest",
]
