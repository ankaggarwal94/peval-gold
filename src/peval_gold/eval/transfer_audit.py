"""D-9 pre-submission stress-test gate (generic auditor).

Evaluates a candidate submission ZIP against four sub-gates derived from the
"transfer-failure audit" pattern in predictive-evaluation competitions:

  (a) Probability distribution — does ``mean(p)`` track the validation positive
      rate (within ±0.05)? Are the extreme tails (``p<1e-3`` and ``p>1-1e-3``)
      smaller than 5% of all predictions?

  (b) Packaged-runtime equivalence (smoke) — does the candidate's ``model.py``
      import cleanly and emit valid finite probabilities on every input row in
      both the main and stress samples?

  (c) Benchmark-heldout stress — is the candidate's NLL on a benchmark-heldout
      stress fold below a configurable threshold (default 0.50)?

  (d) Calibration probes — is the candidate's baked-in calibration near-optimal
      (best in-sample shrink / temperature / intercept probe NLL minus raw NLL
      < 0.01)?

Public surface:
    ``audit_candidate(candidate_zip, validation_rows, stress_rows, out_dir=None)``
    returns an :class:`AuditResult` dataclass with the four sub-gate verdicts.

The audit expects the caller to supply pre-built ``validation_rows.jsonl`` and
``stress_rows.jsonl`` files. Each row is a JSON object with at minimum:
``sample_id`` (int), ``benchmark`` (str), ``condition`` (str),
``subject_content`` (str), ``item_content`` (str), and ``y`` (float, 0 or 1).
Sample-building helpers (stratified sampling from a HF dataset, benchmark
holdout splits) are exposed in :mod:`peval_gold.data.splits`.

This module is intentionally domain-generic: no hosted-runtime-specific paths,
no competition-specific anchors. Downstream callers wire the audit into their
own pipeline by constructing the inputs and interpreting the outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from peval_gold.eval.metrics import (
    brier_score,
    expected_calibration_error,
    ordinary_log_loss,
    safe_logit,
    sigmoid,
)


@dataclass(frozen=True)
class Candidate:
    """A submission candidate to audit.

    Attributes
    ----------
    name : str
        Short identifier for the candidate. Used in output filenames.
    zip_path : Path
        Absolute path to the candidate's flat submission ZIP. The ZIP must
        contain a ``model.py`` exposing
        ``predict(input: dict, labeled: list | None) -> float`` at its root.
    """

    name: str
    zip_path: Path


@dataclass(frozen=True)
class AuditResult:
    """Outcome of running :func:`audit_candidate` against a single candidate.

    Attributes
    ----------
    candidate : str
        Echo of the candidate name.
    sub_gate_a : dict
        Probability-distribution verdict. Keys: ``pass``, ``mean_p``,
        ``positive_rate``, ``abs_diff``, ``frac_extreme_below``,
        ``frac_extreme_above``, ``frac_extreme_total``.
    sub_gate_b : dict
        Packaged-runtime verdict. Keys: ``pass``, ``main_returncode``,
        ``stress_returncode``, ``n_main_valid``, ``n_main_total``,
        ``n_main_errors``, ``n_stress_valid``, ``n_stress_total``.
    sub_gate_c : dict
        Benchmark-heldout-stress verdict. Keys: ``pass``, ``stress_nll``,
        ``threshold_max``.
    sub_gate_d : dict
        Calibration-probe verdict. Keys: ``pass``, ``raw_nll``,
        ``best_probe``, ``best_probe_nll``, ``delta_raw_minus_best``,
        ``threshold_max``.
    overall_pass : bool
        AND of all four sub-gate ``pass`` flags.
    main_metrics, stress_metrics, main_distribution, calibration_probes_full :
        Underlying metric bundles for callers who want more detail.
    zip_sha256 : str
        SHA-256 of the audited ZIP.
    zip_bytes : int
        Size of the audited ZIP in bytes.
    """

    candidate: str
    sub_gate_a: dict
    sub_gate_b: dict
    sub_gate_c: dict
    sub_gate_d: dict
    overall_pass: bool
    main_metrics: dict
    stress_metrics: dict
    main_distribution: dict
    calibration_probes_full: dict
    zip_sha256: str
    zip_bytes: int


# The runner is executed as a separate Python process so module-init
# side-effects of the candidate (torch import, weight loads) do not pollute
# the auditor's process. ``pkg_dir`` is the path to the unpacked candidate
# ZIP; ``import model`` resolves to ``<pkg_dir>/model.py``.
_RUNNER_CODE = r"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

pkg_dir = Path(sys.argv[1]).resolve()
rows_path = Path(sys.argv[2]).resolve()
out_path = Path(sys.argv[3]).resolve()

sys.path.insert(0, str(pkg_dir))
import model  # noqa: E402

rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
out = []
for row in rows:
    inp = {
        "benchmark": row["benchmark"],
        "condition": row["condition"],
        "subject_content": row["subject_content"],
        "item_content": row["item_content"],
    }
    t0 = time.perf_counter()
    try:
        pred = model.predict(inp, None)
        elapsed = time.perf_counter() - t0
        p = float(pred)
        ok = math.isfinite(p) and 0.0 <= p <= 1.0
        out.append({
            "sample_id": row["sample_id"],
            "p": p,
            "ok": ok,
            "elapsed_ms": elapsed * 1000.0,
            "error": None,
        })
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        out.append({
            "sample_id": row["sample_id"],
            "p": None,
            "ok": False,
            "elapsed_ms": elapsed * 1000.0,
            "error": f"{type(exc).__name__}: {exc}",
        })

out_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in out) + "\n", encoding="utf-8")
"""


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_zip(candidate: Candidate, out_dir: Path) -> Path:
    """Unpack the candidate ZIP into ``out_dir/extracted/<candidate.name>/``."""
    if not candidate.zip_path.is_file():
        raise FileNotFoundError(candidate.zip_path)
    extract_dir = out_dir / "extracted" / candidate.name
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(candidate.zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def run_package(
    candidate: Candidate,
    sample_path: Path,
    output_name: str,
    out_dir: Path,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run the candidate's ``model.py`` against ``sample_path`` in a subprocess.

    Returns a dict with ``returncode``, ``predictions_path``, ``log_path``,
    ``zip_sha256``, ``zip_bytes``, plus ``candidate`` + ``sample`` echoes.
    """
    extract_dir = extract_zip(candidate, out_dir)
    runner = out_dir / "_runner.py"
    runner.write_text(_RUNNER_CODE, encoding="utf-8")
    out_path = out_dir / f"predictions_{candidate.name}_{output_name}.jsonl"
    log_path = out_dir / f"predictions_{candidate.name}_{output_name}.log"
    env = os.environ.copy()
    env.pop("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", None)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(runner), str(extract_dir), str(sample_path), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    ended = time.time()
    log_path.write_text(
        "\n".join(
            [
                f"started_unix={started:.0f}",
                f"ended_unix={ended:.0f}",
                f"returncode={proc.returncode}",
                "--- stdout ---",
                proc.stdout,
                "--- stderr ---",
                proc.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return {
        "candidate": candidate.name,
        "sample": output_name,
        "returncode": proc.returncode,
        "predictions_path": str(out_path),
        "log_path": str(log_path),
        "zip_sha256": sha256_file(candidate.zip_path),
        "zip_bytes": candidate.zip_path.stat().st_size,
    }


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load a predictions JSONL file (one row per line)."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def auc_score(y: np.ndarray, p: np.ndarray) -> float | None:
    """ROC-AUC via tied-rank Mann-Whitney-U. Returns ``None`` if degenerate."""
    if y.size == 0:
        return None
    pos = y == 1.0
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=float)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(ranks[pos].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def distribution_stats(p: np.ndarray) -> dict[str, Any]:
    """Summary statistics for a probability distribution."""
    if p.size == 0:
        return {"n": 0}
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logits = np.asarray(safe_logit(clipped, eps=1e-6), dtype=float)
    qs = np.quantile(p, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    return {
        "n": int(len(p)),
        "mean": float(np.mean(p)),
        "std": float(np.std(p)),
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "q01": float(qs[0]),
        "q05": float(qs[1]),
        "q10": float(qs[2]),
        "q50": float(qs[3]),
        "q90": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
        "frac_p_lt_0_001": float(np.mean(p < 0.001)),
        "frac_p_lt_0_01": float(np.mean(p < 0.01)),
        "frac_p_gt_0_99": float(np.mean(p > 0.99)),
        "frac_p_gt_0_999": float(np.mean(p > 0.999)),
        "mean_logit": float(np.mean(logits)),
        "std_logit": float(np.std(logits)),
    }


def metric_bundle(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """NLL + AUC + Brier + ECE + base_rate over (y, p)."""
    if len(y) == 0:
        return {"n": 0, "nll": None, "auc": None, "brier": None, "ece": None, "base_rate": None}
    return {
        "n": int(len(y)),
        "nll": float(ordinary_log_loss(y, p)),
        "auc": auc_score(y, p),
        "brier": float(brier_score(y, p)),
        "ece": float(expected_calibration_error(y, p)),
        "base_rate": float(np.mean(y)),
    }


def _fit_platt(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fit a 1D Platt scaler (slope, intercept) on logit(p) vs. y via IRLS."""
    x = np.asarray(safe_logit(np.clip(p, 1e-6, 1 - 1e-6), eps=1e-6), dtype=float)
    X = np.column_stack([x, np.ones_like(x)])
    beta = np.array([1.0, 0.0], dtype=float)
    for _ in range(50):
        z = X @ beta
        q = np.asarray(sigmoid(z), dtype=float)
        grad = X.T @ (q - y) + 1e-4 * np.array([beta[0] - 1.0, beta[1]])
        w = np.clip(q * (1.0 - q), 1e-6, None)
        hess = X.T @ (X * w[:, None]) + 1e-4 * np.eye(2)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if float(np.linalg.norm(step)) < 1e-6:
            break
        if not np.all(np.isfinite(beta)):
            return 1.0, 0.0
    return float(beta[0]), float(beta[1])


def calibration_probes(
    sample: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    seed: int = 0,
) -> dict[str, Any]:
    """Run shrink / clip / temperature-intercept calibration probes."""
    by_id = {int(r["sample_id"]): r for r in preds if r.get("ok") and r.get("p") is not None}
    ids = [int(r["sample_id"]) for r in sample if int(r["sample_id"]) in by_id]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_cal = max(16, int(0.35 * len(ids)))
    n_cal = min(n_cal, max(0, len(ids) - 16))
    cal_ids = set(ids[:n_cal])
    test_ids = [i for i in ids if i not in cal_ids]
    sample_by_id = {int(r["sample_id"]): r for r in sample}
    cal_y = np.asarray([float(sample_by_id[i]["y"]) for i in cal_ids], dtype=float)
    cal_p = np.asarray([float(by_id[i]["p"]) for i in cal_ids], dtype=float)
    test_y = np.asarray([float(sample_by_id[i]["y"]) for i in test_ids], dtype=float)
    test_p = np.asarray([float(by_id[i]["p"]) for i in test_ids], dtype=float)
    if len(test_y) == 0:
        return {"available": False, "reason": "empty test fold"}
    base_rate = float(np.mean(cal_y)) if len(cal_y) else float(np.mean(test_y))
    probes: dict[str, Any] = {
        "available": True,
        "cal_n": int(len(cal_y)),
        "test_n": int(len(test_y)),
        "cal_base_rate": base_rate,
        "raw": metric_bundle(test_y, test_p),
        "clip_0_001_0_999": metric_bundle(test_y, np.clip(test_p, 0.001, 0.999)),
    }
    for lam in (0.3, 0.5, 0.7, 0.9):
        shrunk = lam * test_p + (1.0 - lam) * base_rate
        probes[f"shrink_lambda_{lam}"] = metric_bundle(test_y, shrunk)
    if len(cal_y) >= 16 and len(set(cal_y.tolist())) == 2:
        slope, intercept = _fit_platt(cal_y, cal_p)
        transformed = np.asarray(
            sigmoid(slope * np.asarray(safe_logit(test_p, eps=1e-6), dtype=float) + intercept),
            dtype=float,
        )
        probes["temperature_intercept_fit"] = {
            "slope": slope,
            "intercept": intercept,
            "metrics": metric_bundle(test_y, transformed),
        }
    else:
        probes["temperature_intercept_fit"] = {
            "available": False,
            "reason": "calibration fold has fewer than 16 rows or one class",
        }
    return probes


def _metrics_for(
    sample: list[dict[str, Any]], preds: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """Align preds to sample by ``sample_id``; return (y, p) numpy arrays."""
    sample_by_id = {int(r["sample_id"]): r for r in sample}
    valid = [
        (int(r["sample_id"]), float(r["p"]))
        for r in preds
        if r.get("ok") and r.get("p") is not None and int(r["sample_id"]) in sample_by_id
    ]
    y = np.asarray([float(sample_by_id[i]["y"]) for i, _ in valid], dtype=float)
    p = np.asarray([pv for _, pv in valid], dtype=float)
    return y, p


def audit_candidate(
    candidate_zip: str | Path,
    validation_rows: str | Path,
    stress_rows: str | Path,
    out_dir: str | Path | None = None,
    *,
    candidate_name: str | None = None,
    timeout_seconds: int = 900,
    seed: int = 20260522,
    stress_nll_threshold: float = 0.50,
    calibration_delta_threshold: float = 0.01,
    distribution_abs_diff_threshold: float = 0.05,
    distribution_tail_threshold: float = 0.05,
) -> AuditResult:
    """Run the four D-9 sub-gates against a candidate submission ZIP.

    Parameters
    ----------
    candidate_zip : str | Path
        Path to a flat submission ZIP containing ``model.py`` at the root.
    validation_rows : str | Path
        Path to a JSONL of validation rows (the "main" sample). Each row has
        ``sample_id``, ``benchmark``, ``condition``, ``subject_content``,
        ``item_content``, and ``y`` (0 or 1).
    stress_rows : str | Path
        Path to a JSONL of benchmark-heldout rows (the "stress" sample).
    out_dir : str | Path | None
        Where to write intermediate files. If ``None``, a tmpdir is created
        and cleaned up on success.
    candidate_name : str | None
        Short identifier. Defaults to the ZIP's stem.
    timeout_seconds : int
        Per-sample subprocess timeout. Defaults to 900s.
    seed : int
        Seed for the calibration-probe fold split.
    stress_nll_threshold : float
        Sub-gate (c) pass threshold. Defaults to 0.50.
    calibration_delta_threshold : float
        Sub-gate (d) pass threshold (``raw_nll - best_probe_nll < threshold``).
    distribution_abs_diff_threshold : float
        Sub-gate (a) pass threshold for ``|mean(p) - positive_rate|``.
    distribution_tail_threshold : float
        Sub-gate (a) pass threshold for combined extreme-tail fraction.

    Returns
    -------
    AuditResult
        Frozen dataclass with the four sub-gate verdicts plus the underlying
        metric bundles.

    Raises
    ------
    FileNotFoundError
        If any of the input files do not exist.
    subprocess.TimeoutExpired
        If the candidate's ``predict()`` exceeds ``timeout_seconds`` on either
        sample.
    """
    candidate_zip = Path(candidate_zip).resolve()
    validation_rows = Path(validation_rows).resolve()
    stress_rows = Path(stress_rows).resolve()
    for p in (candidate_zip, validation_rows, stress_rows):
        if not p.is_file():
            raise FileNotFoundError(p)

    cleanup_dir: Path | None = None
    if out_dir is None:
        cleanup_dir = Path(tempfile.mkdtemp(prefix="peval_gold_audit_"))
        out_dir = cleanup_dir
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = Candidate(
        name=candidate_name or candidate_zip.stem,
        zip_path=candidate_zip,
    )

    main_run = run_package(candidate, validation_rows, "main", out_dir, timeout_seconds)
    stress_run = run_package(candidate, stress_rows, "stress", out_dir, timeout_seconds)

    main_sample = [
        json.loads(line)
        for line in validation_rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stress_sample = [
        json.loads(line)
        for line in stress_rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # If the subprocess crashed (e.g., model.py raises on import) the predictions
    # file may not exist or may be empty. Tolerate that — sub-gate (b) will pick
    # up the non-zero returncode + lack of valid predictions as a failure verdict.
    main_pred_path = Path(main_run["predictions_path"])
    stress_pred_path = Path(stress_run["predictions_path"])
    main_pred = load_predictions(main_pred_path) if main_pred_path.is_file() else []
    stress_pred = load_predictions(stress_pred_path) if stress_pred_path.is_file() else []

    main_y, main_p = _metrics_for(main_sample, main_pred)
    stress_y, stress_p = _metrics_for(stress_sample, stress_pred)

    main_metrics = metric_bundle(main_y, main_p) if main_p.size else {}
    stress_metrics = metric_bundle(stress_y, stress_p) if stress_p.size else {}
    main_dist = distribution_stats(main_p) if main_p.size else {}
    main_probes = calibration_probes(main_sample, main_pred, seed=seed)

    # Sub-gate (a) — probability distribution
    positive_rate_main = float(np.mean(main_y)) if main_y.size else 0.5
    mean_p = float(main_dist.get("mean", 0.0)) if main_dist else 0.0
    frac_lo = float(main_dist.get("frac_p_lt_0_001", 0.0) or 0.0)
    frac_hi = float(main_dist.get("frac_p_gt_0_999", 0.0) or 0.0)
    sub_a_pass = (
        abs(mean_p - positive_rate_main) < distribution_abs_diff_threshold
        and (frac_lo + frac_hi) < distribution_tail_threshold
    )
    sub_a = {
        "pass": sub_a_pass,
        "mean_p": mean_p,
        "positive_rate": positive_rate_main,
        "abs_diff": abs(mean_p - positive_rate_main),
        "frac_extreme_below": frac_lo,
        "frac_extreme_above": frac_hi,
        "frac_extreme_total": frac_lo + frac_hi,
    }

    # Sub-gate (b) — packaged-runtime smoke
    n_main_valid = int(main_p.size)
    n_main_total = len(main_sample)
    n_main_errors = sum(1 for r in main_pred if not r.get("ok"))
    n_stress_valid = int(stress_p.size)
    n_stress_total = len(stress_sample)
    sub_b_pass = (
        main_run["returncode"] == 0
        and stress_run["returncode"] == 0
        and n_main_errors == 0
        and n_main_valid == n_main_total
        and n_stress_valid == n_stress_total
    )
    sub_b = {
        "pass": sub_b_pass,
        "main_returncode": main_run["returncode"],
        "stress_returncode": stress_run["returncode"],
        "n_main_valid": n_main_valid,
        "n_main_total": n_main_total,
        "n_main_errors": n_main_errors,
        "n_stress_valid": n_stress_valid,
        "n_stress_total": n_stress_total,
    }

    # Sub-gate (c) — benchmark-heldout stress NLL
    stress_nll = stress_metrics.get("nll")
    sub_c_pass = stress_nll is not None and float(stress_nll) <= stress_nll_threshold
    sub_c = {
        "pass": sub_c_pass,
        "stress_nll": stress_nll,
        "threshold_max": stress_nll_threshold,
    }

    # Sub-gate (d) — calibration probes baked in
    raw_nll = None
    best_probe = None
    best_probe_nll = None
    if main_probes.get("available"):
        probes_with_nll: dict[str, float] = {}
        for key, val in main_probes.items():
            if not isinstance(val, dict):
                continue
            if "nll" in val and val["nll"] is not None:
                probes_with_nll[key] = float(val["nll"])
            elif (
                "metrics" in val
                and isinstance(val["metrics"], dict)
                and val["metrics"].get("nll") is not None
            ):
                probes_with_nll[key] = float(val["metrics"]["nll"])
        raw_nll = probes_with_nll.get("raw")
        if probes_with_nll:
            best_probe = min(probes_with_nll, key=probes_with_nll.get)
            best_probe_nll = probes_with_nll[best_probe]
    delta = (
        (raw_nll - best_probe_nll)
        if (raw_nll is not None and best_probe_nll is not None)
        else None
    )
    sub_d_pass = delta is not None and delta < calibration_delta_threshold
    sub_d = {
        "pass": sub_d_pass,
        "raw_nll": raw_nll,
        "best_probe": best_probe,
        "best_probe_nll": best_probe_nll,
        "delta_raw_minus_best": delta,
        "threshold_max": calibration_delta_threshold,
    }

    overall_pass = all([sub_a_pass, sub_b_pass, sub_c_pass, sub_d_pass])

    result = AuditResult(
        candidate=candidate.name,
        sub_gate_a=sub_a,
        sub_gate_b=sub_b,
        sub_gate_c=sub_c,
        sub_gate_d=sub_d,
        overall_pass=overall_pass,
        main_metrics=main_metrics,
        stress_metrics=stress_metrics,
        main_distribution=main_dist,
        calibration_probes_full=main_probes,
        zip_sha256=main_run["zip_sha256"],
        zip_bytes=main_run["zip_bytes"],
    )

    if cleanup_dir is not None:
        try:
            shutil.rmtree(cleanup_dir)
        except OSError:
            pass

    return result


__all__ = [
    "Candidate",
    "AuditResult",
    "audit_candidate",
    "extract_zip",
    "run_package",
    "load_predictions",
    "distribution_stats",
    "metric_bundle",
    "calibration_probes",
    "auc_score",
    "sha256_file",
]
