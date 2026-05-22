# Contributing to peval-gold

Thanks for your interest! Bug reports, feature ideas, and PRs are welcome.

## Dev setup

```bash
git clone https://github.com/ankaggarwal94/peval-gold.git
cd peval-gold
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[test,dev,full]
```

The `dev` extra pulls in `ruff` + `build`. The `full` extra adds
`transformers` + `sentence-transformers` (only needed if you're working on
the encoder-backed acquisition policies or the CAIMIRA-lite IRT model).

## Run the tests

```bash
pytest -v -m "not network and not slow"
```

The full suite (including HF-network tests) requires Hugging Face access:

```bash
pytest -v
```

## Lint + format

```bash
ruff check src/ tests/
ruff format src/ tests/
```

CI runs `ruff check` and `ruff format --check`; both must pass before merge.

## Conventions

- **Python 3.10+** — we use the `X | Y` union syntax and structural Protocol
  classes.
- **Strict pytest markers** — register any new marker in
  `pyproject.toml [tool.pytest.ini_options]` before using it.
- **Repo-relative paths** — never hardcode absolute paths in source or
  tests.
- **No competition leakage** — this is an open-source library; do not add
  references to specific competition platforms, course identifiers, or
  upstream project paths. The `tests/test_transfer_audit.py::test_no_competition_leakage_in_module`
  test guards against this for the audit module; add similar checks for
  other modules as they ship.

## Provenance

This package was extracted from
[`ankaggarwal94/CS321M-coursework`](https://github.com/ankaggarwal94/CS321M-coursework)
in May 2026. The source commits that built the gold-track are:

- `20254c5` test(gold): add 15 gold-track tests (splits, calibration, EB, acquisition, IRT-lite)
- `ce3df22` fix(.gitignore,src): track predictive-eval-competition/src/peval_gold/
- `38a2007` fix(pr1): address ChatGPT-5.5 Pro re-review — 5 claims redressed

A follow-up PR on the upstream repo will refactor its
`scripts/run_d9_gate.py` to import from this package instead of the
in-repo copy.
