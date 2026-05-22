# peval-gold

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Reusable framework for predictor / calibrator / acquisition / evaluator
abstractions in predictive-evaluation ML competitions.**

`peval-gold` exposes Protocol-based abstractions that decouple offline
laboratory experimentation from packaged hosted-runtime submissions. It
ships with a D-9 pre-submission stress-test gate (`audit_candidate`) that
catches the four main failure modes — overconfident probabilities,
packaged-runtime crashes, benchmark-heldout regressions, and miscalibrated
heads — before you spend a daily submission slot on a candidate that will
embarrass you on the hidden leaderboard.

## Install

```bash
git clone https://github.com/ankaggarwal94/peval-gold.git
cd peval-gold
pip install -e .[test]
```

For the encoder-backed acquisition policies + the CAIMIRA-lite IRT model,
also install the optional `full` extras (pulls in `transformers` +
`sentence-transformers`):

```bash
pip install -e .[test,full]
```

## Run the tests

```bash
pytest -v -m "not network and not slow"
```

The `network` marker gates tests that hit Hugging Face for the dataset;
`slow` gates encoder-load heavy tests. Both are opt-in.

## Quickstart — using the D-9 gate

```python
from pathlib import Path
from peval_gold import audit_candidate

result = audit_candidate(
    candidate_zip=Path("path/to/your/submission.zip"),
    validation_rows=Path("path/to/validation_rows.jsonl"),
    stress_rows=Path("path/to/stress_rows.jsonl"),
    out_dir=Path("path/to/audit_out"),
)

print(f"Overall: {'PASS' if result.overall_pass else 'FAIL'}")
print(f"  (a) probability distribution: {result.sub_gate_a['pass']}")
print(f"  (b) packaged-runtime smoke:   {result.sub_gate_b['pass']}")
print(f"  (c) benchmark-heldout stress: {result.sub_gate_c['pass']}")
print(f"  (d) calibration probes:       {result.sub_gate_d['pass']}")
```

Each row in `validation_rows.jsonl` and `stress_rows.jsonl` is a JSON
object with: `sample_id` (int), `benchmark` (str), `condition` (str),
`subject_content` (str), `item_content` (str), `y` (float, 0 or 1). The
candidate ZIP must contain a flat `model.py` exposing
`predict(input: dict, labeled: list | None) -> float` at its root.

## Quickstart — building a predictor

```python
from peval_gold import Predictor, RuntimePredictor

class MyConstantPredictor:
    """Predicts the empirical positive rate; satisfies Predictor + RuntimePredictor."""

    def fit(self, train_rows, valid_rows=None):
        ys = [float(r["y"]) for r in train_rows]
        self._rate = sum(ys) / max(len(ys), 1)

    def predict_proba(self, rows):
        import numpy as np
        return np.full(len(rows), self._rate, dtype=float)

    def predict_one(self, input, labeled=None):
        return self._rate

    def reset_calibrator(self):
        pass

# isinstance(MyConstantPredictor(), Predictor)  # True (structural)
# isinstance(MyConstantPredictor(), RuntimePredictor)  # True (structural)
```

## Architecture

The package is organized by Protocol:

```text
peval_gold/
├── acquisition/   # AcquisitionPolicy + 7 policies (random, stratified, hash, SimHash, ...)
├── calibration/   # Calibrator + 6 calibrators (identity, intercept, temp, Platt, ...)
├── data/          # HF loader, schema normalization, splits (item/benchmark holdout)
├── eval/          # Evaluator + transfer_audit (D-9 gate)
├── experiments/   # Append-only run ledger
├── features/      # Subject-name parsing helpers
└── models/        # Predictor / RuntimePredictor + 6 reference impls (NCF, EB, IRT, ...)
```

## Data dependency

The default HF dataset is
[`aims-foundations/measurement-db`](https://huggingface.co/datasets/aims-foundations/measurement-db)
at pinned revision `589ccfdb8e82e6e0b5e35e9d23cd83a6df85018f`. The dataset is
publicly licensed; clean checkouts work without authentication. The revision
pin is in `peval_gold.data.registry.DEFAULT_REVISION` — override it via the
`revision` argument on any loader function.

## License

[MIT](LICENSE). Copyright (c) 2026 Ankit Aggarwal.

## Attribution

This framework was extracted from
[`ankaggarwal94/CS321M-coursework`](https://github.com/ankaggarwal94/CS321M-coursework)
where it was developed during a Stanford CS321M (AI Measurement Science)
predictive-evaluation competition entry. The extraction preserves the
domain-generic library code; competition-specific glue (Codabench paths,
submission ZIP names, course-specific narratives) stays in the upstream
repo. Source commits referenced in the initial-commit body:

- `20254c5` test(gold): add 15 gold-track tests
- `ce3df22` fix(.gitignore,src): track src/peval_gold/
- `38a2007` fix(pr1): address ChatGPT-5.5 Pro re-review

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, lint/test commands,
and the project's conventions.
