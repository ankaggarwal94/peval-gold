"""TDD red→green tests for Batch 7 acquisition policy shootout (S2D).

Five new acquisition policies (plus a streaming adaptive simulator) under
test here. The CurrentSimHash baseline from S1C is exercised indirectly
through the simulator schema test below; its own unit coverage lives in
``tests/test_gold_current_submission_wrapper.py``.

Discipline:

- Every policy module under ``src/peval_gold/acquisition/`` MUST stay
  stdlib-only at import time. No torch / numpy / sentence-transformers /
  transformers may appear in ``sys.modules`` after importing one of the
  policy modules (UncertaintyProxy is the canonical stress case — it is
  a stub by design and refuses to load any ML library even though its
  contract slot is "uncertainty-based acquisition").
- Every ``score_one`` MUST return a finite Python ``float`` on edge
  inputs (empty content, missing keys). Returning NaN/inf is a contract
  violation per ``docs/solutions/runtime-errors/labeling-py-nan-fallback-bomb-2026-05-17.md``.
- Every ``reset`` MUST clear per-round state. Verified by feeding the
  same input before and after reset and asserting equality.
- Latency budget per spec: each policy's ``score_one`` p95 over 1000
  calls < 2 ms on local CPU. The shipped CurrentSimHash p95 on the
  Schmidt H100 g0381 was 0.194 ms (job 8106); the local laptop bound
  is 2 ms which gives ~10x headroom.
"""

from __future__ import annotations

import importlib
import math
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Shared synthetic inputs
# ---------------------------------------------------------------------------


def _synth_input(idx: int) -> dict:
    return {
        "benchmark": f"bench_{idx % 7}",
        "condition": "none" if idx % 2 == 0 else "cot",
        "subject_content": (
            f"Name: model-{idx % 11}\n"
            f"Organization: org-{idx % 4}\n"
            f"Parameters: {10 ** (6 + (idx % 4))}"
        ),
        "item_content": (
            f"Question {idx}: pick the right option about topic "
            f"{idx % 13} with distractor {idx % 17}."
        ),
    }


SMOKE_INPUT = _synth_input(0)
EDGE_INPUTS = [
    {},
    {"benchmark": None, "condition": None, "subject_content": None, "item_content": None},
    {"benchmark": "", "condition": "", "subject_content": "", "item_content": ""},
    {"benchmark": "x"},  # missing 3 keys
]


# ---------------------------------------------------------------------------
# Runtime-safety regression — heavy ML libs MUST NOT load with a policy
# ---------------------------------------------------------------------------


HEAVY_MODULES = ("torch", "numpy", "sentence_transformers", "transformers")


def _heavy_in_sys_modules_now() -> set[str]:
    """Snapshot which heavy modules are currently loaded."""
    return {m for m in HEAVY_MODULES if m in sys.modules}


@pytest.mark.parametrize(
    "module_path",
    [
        "peval_gold.acquisition.random_baseline",
        "peval_gold.acquisition.hash_only",
        "peval_gold.acquisition.simhash_reservoir",
        "peval_gold.acquisition.stratified",
        "peval_gold.acquisition.uncertainty",
    ],
)
def test_policy_module_does_not_import_heavy_ml_stack(module_path: str) -> None:
    """Importing a policy module MUST NOT pull in torch/numpy/transformers.

    The hosted ``acquisition_function`` runs in a tight per-candidate
    loop (5000+ calls/round). Loading torch at acquisition time was the
    P4-class runtime explosion documented in the project history. This
    is the canonical regression test: snapshot ``sys.modules`` before
    and after the policy import; assert no new heavy entries appeared.

    The ``conftest.py`` shim already loads ``numpy`` indirectly via the
    gold-track Predictor protocol (``peval_gold.models.base``). We only
    fail if the policy import introduces a heavy module that was not
    there BEFORE the import — measured via set difference.
    """
    pre = _heavy_in_sys_modules_now()
    importlib.import_module(module_path)
    post = _heavy_in_sys_modules_now()
    newly_imported = post - pre
    assert newly_imported == set(), (
        f"{module_path} pulled in heavy ML modules at import time: {sorted(newly_imported)}"
    )


def test_uncertainty_proxy_specifically_does_not_load_encoder() -> None:
    """UncertaintyProxy is the stress case: its contract slot WOULD require
    an encoder + NCF head, but the stub MUST defer to a future when an
    uncertainty signal proves itself fast enough. Import-time imports of
    torch / sentence_transformers / transformers are an automatic ship
    blocker.
    """
    from peval_gold.acquisition import uncertainty as _u  # noqa: F401

    for mod in ("sentence_transformers", "transformers"):
        assert mod not in sys.modules, (
            f"UncertaintyProxy import triggered {mod!r} load; "
            "stub must not pre-warm the heavy stack"
        )


def test_uncertainty_proxy_docstring_marks_it_runtime_unsafe() -> None:
    """Per spec the stub carries a ``# RUNTIME_UNSAFE_DO_NOT_SHIP`` marker
    so a future grep audit can find it. The marker must appear in the
    module docstring (or a top-level comment) so reading the file makes
    the intent obvious.
    """
    src = (REPO_ROOT / "src" / "peval_gold" / "acquisition" / "uncertainty.py").read_text()
    assert "RUNTIME_UNSAFE_DO_NOT_SHIP" in src


# ---------------------------------------------------------------------------
# AcquisitionPolicy protocol conformance — all 5 policies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctor",
    [
        ("peval_gold.acquisition.random_baseline", "RandomAcquisition", {}),
        ("peval_gold.acquisition.hash_only", "PureHash", {}),
        ("peval_gold.acquisition.simhash_reservoir", "SimHashReservoir", {}),
        ("peval_gold.acquisition.simhash_reservoir", "SimHashReservoir", {"reservoir_size": 64}),
        ("peval_gold.acquisition.stratified", "StratifiedBonus", {}),
        (
            "peval_gold.acquisition.stratified",
            "StratifiedBonus",
            {"stratifier": "benchmark_condition"},
        ),
        ("peval_gold.acquisition.uncertainty", "UncertaintyProxy", {}),
    ],
)
def test_policy_satisfies_acquisition_policy_protocol(ctor: tuple) -> None:
    from peval_gold.acquisition.base import AcquisitionPolicy

    module_path, cls_name, kwargs = ctor
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    instance = cls(**kwargs)
    assert isinstance(instance, AcquisitionPolicy), (
        f"{module_path}.{cls_name}(**{kwargs}) is not an AcquisitionPolicy"
    )


# ---------------------------------------------------------------------------
# score_one returns finite float / handles edge inputs
# ---------------------------------------------------------------------------


def _construct_all_policies() -> list[tuple[str, object]]:
    from peval_gold.acquisition.hash_only import PureHash
    from peval_gold.acquisition.random_baseline import RandomAcquisition
    from peval_gold.acquisition.simhash_reservoir import SimHashReservoir
    from peval_gold.acquisition.stratified import StratifiedBonus
    from peval_gold.acquisition.uncertainty import UncertaintyProxy

    return [
        ("RandomAcquisition", RandomAcquisition()),
        ("PureHash", PureHash()),
        ("SimHashReservoir(32)", SimHashReservoir(reservoir_size=32)),
        ("SimHashReservoir(128)", SimHashReservoir(reservoir_size=128)),
        ("StratifiedBonus(benchmark)", StratifiedBonus(stratifier="benchmark")),
        (
            "StratifiedBonus(benchmark_condition)",
            StratifiedBonus(stratifier="benchmark_condition"),
        ),
        ("UncertaintyProxy", UncertaintyProxy()),
    ]


def test_every_policy_returns_finite_float_on_smoke_input() -> None:
    for name, pol in _construct_all_policies():
        out = pol.score_one(SMOKE_INPUT)
        assert isinstance(out, float), f"{name}: score_one returned {type(out).__name__}"
        assert math.isfinite(out), f"{name}: score_one returned non-finite {out!r}"


@pytest.mark.parametrize("edge", EDGE_INPUTS)
def test_every_policy_handles_edge_inputs_without_raising(edge: dict) -> None:
    for name, pol in _construct_all_policies():
        try:
            out = pol.score_one(edge)
        except Exception as exc:  # pylint: disable=broad-except
            pytest.fail(f"{name}: score_one raised on edge input {edge!r}: {exc!r}")
        assert isinstance(out, float) and math.isfinite(out), (
            f"{name}: edge input {edge!r} produced {out!r}"
        )


# ---------------------------------------------------------------------------
# reset clears per-round state
# ---------------------------------------------------------------------------


def test_reset_makes_repeated_score_one_calls_idempotent() -> None:
    """For stateful policies, ``reset()`` before scoring a fixed input
    sequence MUST produce the same output sequence both times. RandomAcquisition
    keys its hash on ``_n_seen`` so the per-call output depends on call
    index; reset zeroes ``_n_seen`` and the sequence repeats verbatim.
    """
    for name, pol in _construct_all_policies():
        inputs = [_synth_input(i) for i in range(8)]
        pol.reset()
        first_run = [pol.score_one(dict(x)) for x in inputs]
        pol.reset()
        second_run = [pol.score_one(dict(x)) for x in inputs]
        assert first_run == second_run, (
            f"{name}: reset did not produce idempotent sequence — "
            f"first={first_run!r} second={second_run!r}"
        )


# ---------------------------------------------------------------------------
# SimHashReservoir: reservoir_size knob actually changes behavior
# ---------------------------------------------------------------------------


def test_simhash_reservoir_size_one_collapses_history() -> None:
    """With ``reservoir_size=1`` only ONE signature is kept in the reservoir
    at any moment — so each new candidate's diversity score depends ONLY
    on the most recently added signature, not on the full history. The
    practical consequence: scoring the SAME input twice in a row gives
    a perfect-similarity (score=0) on the second call because the
    reservoir holds an identical signature.
    """
    from peval_gold.acquisition.simhash_reservoir import SimHashReservoir

    pol = SimHashReservoir(reservoir_size=1)
    pol.reset()
    first = pol.score_one(dict(SMOKE_INPUT))
    second = pol.score_one(dict(SMOKE_INPUT))
    # First call has empty reservoir → diversity-floor 1.0.
    assert first == pytest.approx(1.0, abs=1e-9)
    # Second call compares an identical signature → Hamming distance 0 / 64.
    assert second == pytest.approx(0.0, abs=1e-9)


def test_simhash_reservoir_size_large_preserves_diversity_history() -> None:
    """With a large reservoir, the same fixed input scored AFTER many
    distinct candidates have been seen still returns 0 (identical signature
    in the reservoir). The point of this test is the contrast with
    ``reservoir_size=1``: with size=1 the diversity-against-history check
    is destroyed every call; with size=large the SAME input lookup is
    still meaningful (a wide reservoir provides representative coverage).
    """
    from peval_gold.acquisition.simhash_reservoir import SimHashReservoir

    pol = SimHashReservoir(reservoir_size=256)
    pol.reset()
    # Seed the reservoir with our smoke input first.
    seed_score = pol.score_one(dict(SMOKE_INPUT))
    assert seed_score == pytest.approx(1.0, abs=1e-9)
    # Drive 100 distinct candidates through — none collide with SMOKE_INPUT.
    for i in range(1, 100):
        pol.score_one(_synth_input(i))
    # Re-score the smoke input; signature is still in the (256-wide)
    # reservoir, so distance is 0.
    after = pol.score_one(dict(SMOKE_INPUT))
    assert after == pytest.approx(0.0, abs=1e-9)


def test_simhash_reservoir_diversity_score_in_unit_interval() -> None:
    """Hamming distance over a 64-bit signature lands in [0, 1] after
    normalization. Test 200 mixed candidates and assert each score
    stays inside the unit interval.
    """
    from peval_gold.acquisition.simhash_reservoir import SimHashReservoir

    pol = SimHashReservoir(reservoir_size=64, bits=64)
    pol.reset()
    for i in range(200):
        out = pol.score_one(_synth_input(i))
        assert 0.0 <= out <= 1.0, f"score {out!r} for candidate {i} outside [0, 1]"


# ---------------------------------------------------------------------------
# StratifiedBonus: underrepresented strata get a bonus
# ---------------------------------------------------------------------------


def test_stratified_bonus_rewards_underrepresented_categories() -> None:
    """A fresh, never-seen stratum should score higher than the same diversity
    signal in a saturated stratum. Concretely: feed 10 candidates from
    ``benchmark=bench_A``, then score one from ``benchmark=bench_B``.
    The bench_B candidate's bonus term should be ~0.5 (1/(1+0)) vs.
    bench_A's near-zero (1/(1+10)).
    """
    from peval_gold.acquisition.stratified import StratifiedBonus

    pol = StratifiedBonus(stratifier="benchmark", bonus_weight=0.5)
    pol.reset()

    # Saturate bench_A
    for i in range(10):
        pol.score_one(
            {
                "benchmark": "bench_A",
                "condition": "none",
                "subject_content": f"Name: m-{i}",
                "item_content": f"item-{i}",
            }
        )

    saturated = pol.score_one(
        {
            "benchmark": "bench_A",
            "condition": "none",
            "subject_content": "Name: m-X",
            "item_content": "item-saturated",
        }
    )
    fresh = pol.score_one(
        {
            "benchmark": "bench_B",
            "condition": "none",
            "subject_content": "Name: m-X",
            "item_content": "item-fresh",
        }
    )

    # The fresh stratum gets a meaningfully larger bonus contribution.
    # Lower bound on the gap: 0.5 * (1/(1+0) - 1/(1+11)) ~= 0.46.
    # Allow for the diversity term to vary; assert fresh > saturated by
    # at least 0.10 (well below the 0.46 bonus-only delta).
    assert fresh > saturated + 0.10, (
        f"fresh={fresh!r} not meaningfully larger than saturated={saturated!r}"
    )


def test_stratified_bonus_benchmark_condition_stratifier_uses_both() -> None:
    """With ``stratifier='benchmark_condition'`` the bonus tracks the
    (benchmark, condition) tuple, so the same benchmark with a different
    condition counts as a different stratum.
    """
    from peval_gold.acquisition.stratified import StratifiedBonus

    pol = StratifiedBonus(stratifier="benchmark_condition", bonus_weight=0.5)
    pol.reset()

    # Saturate (bench_A, none)
    for i in range(10):
        pol.score_one(
            {
                "benchmark": "bench_A",
                "condition": "none",
                "subject_content": f"Name: m-{i}",
                "item_content": f"item-{i}",
            }
        )

    saturated = pol.score_one(
        {
            "benchmark": "bench_A",
            "condition": "none",
            "subject_content": "Name: m-X",
            "item_content": "item-sat",
        }
    )
    fresh_condition = pol.score_one(
        {
            "benchmark": "bench_A",  # same benchmark
            "condition": "cot",  # different condition
            "subject_content": "Name: m-X",
            "item_content": "item-fresh",
        }
    )

    # Different condition should still register as a fresh stratum.
    assert fresh_condition > saturated + 0.10


# ---------------------------------------------------------------------------
# Adaptive simulator: schema + cross-policy execution
# ---------------------------------------------------------------------------


class _ConstantPredictor:
    """Tiny shim mirroring RuntimePredictor — returns a fixed probability.

    Lives only in this test module so the simulator coverage is encoder-free
    (the production shootout in ``scripts/gold_evaluate_acquisition_shootout.py``
    uses the real CurrentNCF wrapper).
    """

    def __init__(self, value: float = 0.5) -> None:
        self._value = float(value)

    def predict_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
        labeled: list[dict] | None = None,
    ) -> float:
        return self._value


def _synth_val_rows(n: int = 60) -> list[dict]:
    rows: list[dict] = []
    for i in range(n):
        row = _synth_input(i)
        row["response"] = float(i % 2)  # binary labels
        rows.append(row)
    return rows


def test_run_adaptive_simulation_returns_required_schema() -> None:
    """The simulator returns ``{rounds, mean_nll, mean_mll, per_round}``."""
    from peval_gold.acquisition.simhash_reservoir import SimHashReservoir
    from peval_gold.acquisition.simulator import run_adaptive_simulation

    pol = SimHashReservoir(reservoir_size=32)
    pred = _ConstantPredictor(value=0.5)
    rows = _synth_val_rows(60)
    out = run_adaptive_simulation(
        policy=pol,
        predictor=pred,
        val_rows=rows,
        k_per_category=5,
        n_rounds=3,
        category_key="benchmark",
    )
    assert isinstance(out, dict)
    for key in ("rounds", "mean_nll", "mean_mll", "per_round"):
        assert key in out, f"simulator output missing key {key!r}"
    assert isinstance(out["per_round"], list)
    assert len(out["per_round"]) == 3
    assert out["rounds"] == 3
    # NLL of a constant 0.5 predictor on balanced labels is ln(2) ~= 0.693.
    assert isinstance(out["mean_nll"], float)
    assert math.isfinite(out["mean_nll"])
    assert out["mean_nll"] == pytest.approx(-out["mean_mll"], abs=1e-9)
    # Per-round records include the standard keys.
    for r in out["per_round"]:
        for key in ("round_index", "nll", "mll", "n_labeled", "n_unlabeled"):
            assert key in r, f"per_round record missing key {key!r}"


def test_run_adaptive_simulation_resets_policy_between_rounds() -> None:
    """The simulator MUST call ``policy.reset()`` at the start of each
    round; otherwise reservoir state leaks across rounds and the labeled
    set for round-N depends on round-(N-1)'s state. Stress this with a
    tiny ``_ResetCounting`` shim that observes reset() calls.
    """
    from peval_gold.acquisition.simulator import run_adaptive_simulation

    class _Counting:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.score_calls = 0

        def score_one(self, input: dict) -> float:  # noqa: A002
            self.score_calls += 1
            return 0.5

        def reset(self) -> None:
            self.reset_calls += 1

    pol = _Counting()
    pred = _ConstantPredictor()
    rows = _synth_val_rows(40)
    run_adaptive_simulation(policy=pol, predictor=pred, val_rows=rows, k_per_category=5, n_rounds=3)
    assert pol.reset_calls == 3
    # 40 rows × 3 rounds = 120 score_one calls minimum.
    assert pol.score_calls == 120


def test_run_adaptive_simulation_uses_category_key_for_grouping() -> None:
    """The K=5 selection is per-category. With ``category_key='condition'``
    and 60 rows split 30/30 across ``none`` / ``cot``, the per-round
    labeled set must contain at most K × 2 = 10 rows.
    """
    from peval_gold.acquisition.hash_only import PureHash
    from peval_gold.acquisition.simulator import run_adaptive_simulation

    pol = PureHash()
    pred = _ConstantPredictor()
    rows = _synth_val_rows(60)
    out = run_adaptive_simulation(
        policy=pol,
        predictor=pred,
        val_rows=rows,
        k_per_category=5,
        n_rounds=1,
        category_key="condition",
    )
    # Two conditions × K=5 = 10 labeled rows per round.
    assert out["per_round"][0]["n_labeled"] == 10
    assert out["per_round"][0]["n_unlabeled"] == 50


# ---------------------------------------------------------------------------
# Latency: each policy's score_one p95 over 1000 calls < 2 ms
# ---------------------------------------------------------------------------


def _measure_p95_ms(pol: object, n: int = 1000) -> float:
    """Time n score_one calls and return the p95 in milliseconds."""
    inputs = [_synth_input(i) for i in range(n)]
    samples: list[float] = []
    # Warm cache (5 calls excluded).
    for x in inputs[:5]:
        pol.score_one(dict(x))
    for x in inputs:
        t0 = time.perf_counter()
        pol.score_one(dict(x))
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p95_idx = max(0, int(len(samples) * 0.95) - 1)
    return samples[p95_idx]


@pytest.mark.parametrize(
    "ctor",
    [
        ("peval_gold.acquisition.random_baseline", "RandomAcquisition", {}),
        ("peval_gold.acquisition.hash_only", "PureHash", {}),
        ("peval_gold.acquisition.simhash_reservoir", "SimHashReservoir", {"reservoir_size": 128}),
        ("peval_gold.acquisition.stratified", "StratifiedBonus", {"stratifier": "benchmark"}),
        ("peval_gold.acquisition.uncertainty", "UncertaintyProxy", {}),
    ],
)
def test_policy_score_one_p95_under_two_ms(ctor: tuple) -> None:
    """Gold-track §10 runtime constraint: p95 < 2 ms on local CPU.

    The shipped CurrentSimHash p95 on the Schmidt H100 g0381 was 0.194 ms
    (job 8106). The 2 ms local bound gives ~10x headroom for Mac CPU
    variability and noisy laptops.
    """
    module_path, cls_name, kwargs = ctor
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    pol = cls(**kwargs)
    p95 = _measure_p95_ms(pol, n=1000)
    assert p95 < 2.0, f"{cls_name}({kwargs}) p95={p95:.4f}ms exceeds 2 ms budget"


# ---------------------------------------------------------------------------
# Type discipline: score_one return type is native Python float
# ---------------------------------------------------------------------------


def test_score_one_returns_native_python_float_not_numpy() -> None:
    """The kit contract requires native Python float. Same discipline at
    the acquisition layer keeps the JSONL ledger / report serializers
    happy without numpy-scalar coercion.
    """
    for name, pol in _construct_all_policies():
        out = pol.score_one(SMOKE_INPUT)
        assert type(out) is float, (
            f"{name}: score_one returned {type(out).__name__}, expected native float"
        )
