"""TDD tests for the gold-track Rasch/REEval-lite CPU challenger.

The first-pass model is intentionally small and local-only:

    logit(p) = theta_subject - beta_item_or_text

where ``theta_subject`` is a smoothed subject ability lookup and
``beta_item_or_text`` is either a seen-item difficulty lookup or a
hashed-text ridge-regression difficulty estimate for held-out items.
All tests use synthetic rows only; no HF/cache access and no submission
artifact interaction.
"""

from __future__ import annotations

# pylint: disable=import-error

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _row(
    subject: str,
    item_id: str,
    item_content: str,
    response: float,
    *,
    benchmark: str = "bench",
    condition: str = "none",
) -> dict[str, Any]:
    return {
        "subject_id": subject,
        "item_id": item_id,
        "benchmark": benchmark,
        "condition": condition,
        "subject_content": f"Name: {subject}",
        "item_content": item_content,
        "response": float(response),
    }


def _subject_signal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(30):
        rows.append(_row("strong", f"easy-{i}", "easy clue direct", 1.0))
        rows.append(_row("strong", f"hard-{i}", "hard clue obscure", 1.0 if i < 18 else 0.0))
        rows.append(_row("weak", f"easy-{i}", "easy clue direct", 1.0 if i < 12 else 0.0))
        rows.append(_row("weak", f"hard-{i}", "hard clue obscure", 0.0))
    return rows


def _text_signal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(50):
        subject = f"s{i % 8}"
        rows.append(_row(subject, f"easy-{i}", "easy direct common clue", 1.0))
        rows.append(_row(subject, f"hard-{i}", "hard obscure adversarial clue", 0.0))
    return rows


def test_rasch_lite_subject_ability_orders_known_subjects() -> None:
    from peval_gold.models.rasch_lite import RaschLite

    model = RaschLite(text_dim=64, ridge_lambda=1.0)
    model.fit(_subject_signal_rows())

    strong = model.predict_one(
        {
            "subject_content": "Name: strong",
            "benchmark": "bench",
            "condition": "none",
            "item_id": "novel-neutral",
            "item_content": "neutral unseen clue",
        }
    )
    weak = model.predict_one(
        {
            "subject_content": "Name: weak",
            "benchmark": "bench",
            "condition": "none",
            "item_id": "novel-neutral-2",
            "item_content": "neutral unseen clue",
        }
    )

    assert 0.0 < weak < strong < 1.0
    assert strong - weak > 0.15


def test_rasch_lite_text_regressor_orders_unseen_easy_above_hard_items() -> None:
    from peval_gold.models.rasch_lite import RaschLite

    model = RaschLite(text_dim=128, ridge_lambda=0.5, unseen_text_weight=1.0)
    model.fit(_text_signal_rows())

    easy = model.predict_one(
        {
            "subject_content": "Name: s0",
            "benchmark": "bench",
            "condition": "none",
            "item_id": "never-seen-easy",
            "item_content": "easy direct common clue never seen",
        }
    )
    hard = model.predict_one(
        {
            "subject_content": "Name: s0",
            "benchmark": "bench",
            "condition": "none",
            "item_id": "never-seen-hard",
            "item_content": "hard obscure adversarial clue never seen",
        }
    )

    assert 0.0 < hard < easy < 1.0
    assert easy - hard > 0.25


def test_rasch_lite_seen_item_lookup_overrides_text_regressor() -> None:
    from peval_gold.models.rasch_lite import RaschLite

    rows = []
    for i in range(40):
        rows.append(_row(f"s{i % 5}", "same-text-easy", "ambiguous repeated text", 1.0))
        rows.append(_row(f"s{i % 5}", "same-text-hard", "ambiguous repeated text", 0.0))

    model = RaschLite(text_dim=32)
    model.fit(rows)
    same_subject = {
        "subject_content": "Name: s0",
        "benchmark": "bench",
        "condition": "none",
        "item_content": "ambiguous repeated text",
    }

    p_easy = model.predict_one({**same_subject, "item_id": "same-text-easy"})
    p_hard = model.predict_one({**same_subject, "item_id": "same-text-hard"})
    assert p_easy > p_hard
    assert p_easy - p_hard > 0.4


def test_rasch_lite_save_load_round_trip_preserves_predictions(tmp_path: Path) -> None:
    from peval_gold.models.rasch_lite import RaschLite

    rows = _subject_signal_rows() + _text_signal_rows()
    model = RaschLite(text_dim=96, ridge_lambda=0.75)
    model.fit(rows)

    eval_rows = [
        {
            "subject_content": "Name: strong",
            "benchmark": "bench",
            "condition": "none",
            "item_id": "new-easy",
            "item_content": "easy direct clue",
        },
        {
            "subject_content": "Name: weak",
            "benchmark": "bench",
            "condition": "none",
            "item_id": "new-hard",
            "item_content": "hard obscure clue",
        },
    ]
    before = model.predict_proba(eval_rows)

    path = tmp_path / "rasch_lite.json"
    model.save(str(path))
    restored = RaschLite.load(str(path))
    after = restored.predict_proba(eval_rows)

    np.testing.assert_allclose(before, after, atol=1e-12)


def test_rasch_lite_outputs_finite_probabilities_for_unknown_rows() -> None:
    from peval_gold.models.rasch_lite import RaschLite

    model = RaschLite(text_dim=16)
    model.fit(_text_signal_rows())
    probs = model.predict_proba(
        [
            {"subject_content": "", "benchmark": "zzz", "item_id": "x", "item_content": ""},
            {
                "subject_content": "Name: unknown",
                "benchmark": "zzz",
                "condition": "new",
                "item_id": "y",
                "item_content": "totally unseen vocabulary",
            },
        ]
    )

    assert probs.shape == (2,)
    assert np.all(np.isfinite(probs))
    assert np.all((probs > 0.0) & (probs < 1.0))


def test_rasch_lite_fit_empty_or_nonbinary_rows_raises_value_error() -> None:
    from peval_gold.models.rasch_lite import RaschLite

    with pytest.raises(ValueError):
        RaschLite().fit([])
    with pytest.raises(ValueError):
        RaschLite().fit([_row("s", "i", "x", 0.5)])


def test_rasch_lite_satisfies_gold_predictor_protocols() -> None:
    from peval_gold.models.base import Predictor, RuntimePredictor
    from peval_gold.models.rasch_lite import RaschLite

    model = RaschLite()
    assert isinstance(model, Predictor)
    assert isinstance(model, RuntimePredictor)


def test_rasch_lite_metadata_is_json_serializable_after_fit() -> None:
    from peval_gold.models.rasch_lite import RaschLite

    model = RaschLite(text_dim=32)
    model.fit(_subject_signal_rows())
    metadata = model.metadata()

    assert metadata["class"] == "RaschLite"
    assert metadata["fitted"] is True
    assert metadata["n_subjects"] >= 2
    assert metadata["n_items"] >= 2
    assert math.isfinite(metadata["global_logit"])
