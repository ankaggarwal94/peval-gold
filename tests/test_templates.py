"""TDD red→green tests for ``peval_gold.data.templates``.

Three item-text formatters are under test (Batch 4 of the gold-track
workflow, subagent S2A):

- :func:`item_only` — current behavior; returns ``row["item_content"]``
  unchanged. Locks the legacy baseline so a future refactor can prove
  ``canonical_item`` is the only behavioral change.
- :func:`canonical_item` — H1 hypothesis. Prepends ``Benchmark:`` and
  ``Condition:`` lines so the encoder sees benchmark/condition signal
  that the current shipped artifact demonstrably ignores (Batch-0 CoVe
  Claim 1).
- :func:`rich_item` — optional variant that also surfaces ``domain`` and
  ``modality`` when present in the row dict.

All three must be:

- pure (no side effects, no global state mutation);
- deterministic (same input → identical output, byte-for-byte);
- defensive against missing optional keys (``rich_item`` must still
  produce a valid string when ``domain`` / ``modality`` are absent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_row() -> dict:
    """A minimal row carrying only the four required runtime keys."""
    return {
        "benchmark": "mmlupro",
        "condition": "cot",
        "subject_content": "Name: sample-model",
        "item_content": "What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6",
    }


@pytest.fixture
def rich_row(basic_row: dict) -> dict:
    """A row carrying the optional ``domain`` / ``modality`` fields."""
    return {
        **basic_row,
        "domain": "math",
        "modality": "text",
    }


# ---------------------------------------------------------------------------
# item_only
# ---------------------------------------------------------------------------


def test_item_only_returns_item_content_verbatim(basic_row: dict) -> None:
    from peval_gold.data.templates import item_only

    out = item_only(basic_row)
    assert isinstance(out, str)
    assert out == basic_row["item_content"]


def test_item_only_is_deterministic(basic_row: dict) -> None:
    from peval_gold.data.templates import item_only

    a = item_only(basic_row)
    b = item_only(dict(basic_row))
    assert a == b


def test_item_only_does_not_mutate_row(basic_row: dict) -> None:
    from peval_gold.data.templates import item_only

    snapshot = dict(basic_row)
    _ = item_only(basic_row)
    assert basic_row == snapshot


# ---------------------------------------------------------------------------
# canonical_item — the H1 deliverable
# ---------------------------------------------------------------------------


def test_canonical_item_prepends_benchmark_and_condition(basic_row: dict) -> None:
    from peval_gold.data.templates import canonical_item

    out = canonical_item(basic_row)
    assert isinstance(out, str)
    expected = (
        "Benchmark: mmlupro\n"
        "Condition: cot\n"
        "Item:\n"
        "What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6"
    )
    assert out == expected


def test_canonical_item_is_deterministic(basic_row: dict) -> None:
    from peval_gold.data.templates import canonical_item

    a = canonical_item(basic_row)
    b = canonical_item(dict(basic_row))
    assert a == b


def test_canonical_item_does_not_mutate_row(basic_row: dict) -> None:
    from peval_gold.data.templates import canonical_item

    snapshot = dict(basic_row)
    _ = canonical_item(basic_row)
    assert basic_row == snapshot


def test_canonical_item_distinguishes_different_conditions(
    basic_row: dict,
) -> None:
    """Same item under different conditions must yield different encoded text.

    The whole point of H1 is to inject benchmark/condition signal into
    the encoder input. If ``canonical_item`` collapsed across condition
    values the lookup at runtime would degenerate to ``item_only``.
    """
    from peval_gold.data.templates import canonical_item

    cot = canonical_item(basic_row)
    direct = canonical_item({**basic_row, "condition": "direct"})
    assert cot != direct
    assert "Condition: cot" in cot
    assert "Condition: direct" in direct


def test_canonical_item_distinguishes_different_benchmarks(
    basic_row: dict,
) -> None:
    """Same item text under different benchmarks must produce distinct strings."""
    from peval_gold.data.templates import canonical_item

    a = canonical_item(basic_row)
    b = canonical_item({**basic_row, "benchmark": "ai2d_test"})
    assert a != b
    assert "Benchmark: mmlupro" in a
    assert "Benchmark: ai2d_test" in b


# ---------------------------------------------------------------------------
# rich_item — defensive against missing optional keys
# ---------------------------------------------------------------------------


def test_rich_item_includes_domain_and_modality_when_present(
    rich_row: dict,
) -> None:
    from peval_gold.data.templates import rich_item

    out = rich_item(rich_row)
    assert "Benchmark: mmlupro" in out
    assert "Condition: cot" in out
    assert "Domain: math" in out
    assert "Modality: text" in out
    assert "Item:" in out
    assert rich_row["item_content"] in out


def test_rich_item_omits_domain_when_missing(basic_row: dict) -> None:
    from peval_gold.data.templates import rich_item

    out = rich_item(basic_row)
    assert "Domain:" not in out
    assert "Modality:" not in out
    assert "Benchmark: mmlupro" in out
    assert basic_row["item_content"] in out


def test_rich_item_omits_domain_when_empty_string(basic_row: dict) -> None:
    """An empty-string optional key is treated as missing."""
    from peval_gold.data.templates import rich_item

    out = rich_item({**basic_row, "domain": "", "modality": ""})
    assert "Domain:" not in out
    assert "Modality:" not in out


def test_rich_item_handles_list_typed_domain(basic_row: dict) -> None:
    """``benchmarks.parquet:domain`` is a list of strings at the pinned revision.

    ``rich_item`` must produce something stable when the registry-typed
    domain shape (list[str]) is passed; we accept either a flat
    comma-joined rendering or the str() of the list as long as the
    output is deterministic and contains a ``Domain:`` line.
    """
    from peval_gold.data.templates import rich_item

    row = {**basic_row, "domain": ["math", "reasoning"], "modality": ["text"]}
    out = rich_item(row)
    assert "Domain:" in out
    assert "math" in out
    assert "reasoning" in out


def test_rich_item_is_deterministic(rich_row: dict) -> None:
    from peval_gold.data.templates import rich_item

    a = rich_item(rich_row)
    b = rich_item(dict(rich_row))
    assert a == b


def test_rich_item_does_not_mutate_row(rich_row: dict) -> None:
    from peval_gold.data.templates import rich_item

    snapshot = dict(rich_row)
    _ = rich_item(rich_row)
    assert rich_row == snapshot


# ---------------------------------------------------------------------------
# Cross-cutting purity
# ---------------------------------------------------------------------------


def test_templates_are_module_level_callables() -> None:
    """All three templates must be importable from the same module."""
    from peval_gold.data import templates

    for name in ("item_only", "canonical_item", "rich_item"):
        fn = getattr(templates, name)
        assert callable(fn), f"{name} must be a callable in templates module"


def test_templates_source_is_pure_python_no_torch_no_numpy() -> None:
    """The templates module must NOT import torch / numpy / sentence-transformers.

    Future use under ``submission/`` (if H1 promotes) requires the
    templates module to live on the no-network / lightweight side of the
    runtime contract.
    """
    src = (
        REPO_ROOT / "src" / "peval_gold" / "data" / "templates.py"
    ).read_text()
    for forbidden in (
        "import torch",
        "import numpy",
        "SentenceTransformer",
        "from huggingface_hub",
        "from datasets",
    ):
        assert forbidden not in src, f"forbidden fragment {forbidden!r}"
