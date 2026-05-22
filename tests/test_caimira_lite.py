from __future__ import annotations

import warnings
from pathlib import Path

import torch


def test_caimira_lite_forward_shapes_and_components() -> None:
    from peval_gold.models.caimira_lite import CaimiraLiteConfig, CaimiraLiteModel

    model = CaimiraLiteModel(
        CaimiraLiteConfig(
            n_subjects=4,
            n_benchmarks=3,
            n_conditions=2,
            item_embed_dim=5,
            latent_dim=3,
        )
    )
    subject_idx = torch.tensor([0, 1, 2, 3])
    benchmark_idx = torch.tensor([0, 1, 2, 0])
    condition_idx = torch.tensor([0, 1, 0, 1])
    item_emb = torch.randn(4, 5)

    logits, parts = model(
        subject_idx,
        item_emb,
        benchmark_idx=benchmark_idx,
        condition_idx=condition_idx,
        return_parts=True,
    )

    assert logits.shape == (4,)
    assert parts["theta"].shape == (4, 3)
    assert parts["difficulty"].shape == (4, 3)
    assert parts["relevance"].shape == (4, 3)
    assert torch.allclose(
        parts["relevance"].sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )
    # D-11: paper-faithful centering uses the corpus-mean `diff_mean`
    # buffer (zero on a fresh model), so at init the centered difficulty
    # equals the raw difficulty. The PRIOR per-item across-latent-dim
    # mean-zero invariant (`difficulty.mean(dim=1) == 0`) is intentionally
    # NOT enforced — that was the bug ChatGPT-5.5 Pro Claim 2 surfaced.
    raw = model.difficulty(item_emb.float())
    assert torch.allclose(parts["difficulty"], raw, atol=1e-6)


def test_caimira_lite_diff_mean_is_registered_buffer_and_zero_at_init() -> None:
    from peval_gold.models.caimira_lite import CaimiraLiteConfig, CaimiraLiteModel

    config = CaimiraLiteConfig(
        n_subjects=2,
        n_benchmarks=1,
        n_conditions=1,
        item_embed_dim=4,
        latent_dim=3,
    )
    model = CaimiraLiteModel(config)
    # The buffer must be present in state_dict so it ships with the
    # artifact (D-11; plans/03_ncf_and_caimira.md:548).
    state = model.state_dict()
    assert "diff_mean" in state
    assert state["diff_mean"].shape == (config.latent_dim,)
    # Fresh model: buffer is zero, so a forward pass leaves difficulty
    # uncentered (raw == centered).
    assert torch.allclose(model.diff_mean, torch.zeros(config.latent_dim), atol=0.0)


def test_caimira_lite_refresh_diff_mean_produces_corpus_centered_difficulty() -> None:
    from peval_gold.models.caimira_lite import CaimiraLiteConfig, CaimiraLiteModel

    torch.manual_seed(0)
    latent_dim = 3
    item_embed_dim = 5
    model = CaimiraLiteModel(
        CaimiraLiteConfig(
            n_subjects=4,
            n_benchmarks=2,
            n_conditions=2,
            item_embed_dim=item_embed_dim,
            latent_dim=latent_dim,
        )
    )
    # Synthetic training item bank.
    all_item_embs = torch.randn(64, item_embed_dim)

    model.refresh_diff_mean(all_item_embs)

    # After refresh: subtracting the buffer from raw difficulties over the
    # same item bank yields zero mean along the item (corpus) axis. This
    # is the paper-faithful CAIMIRA §8.6 invariant.
    raw = model.difficulty(all_item_embs.float())
    centered = raw - model.diff_mean
    assert torch.allclose(
        centered.mean(dim=0),
        torch.zeros(latent_dim),
        atol=1e-6,
    )
    # And the buffer itself equals the mean of raw across items.
    assert torch.allclose(model.diff_mean, raw.mean(dim=0), atol=1e-6)


def test_caimira_lite_legacy_state_dict_load_emits_warning_and_zero_inits() -> None:
    from peval_gold.models.caimira_lite import CaimiraLiteConfig, CaimiraLiteModel

    config = CaimiraLiteConfig(
        n_subjects=3,
        n_benchmarks=2,
        n_conditions=2,
        item_embed_dim=4,
        latent_dim=3,
    )
    model = CaimiraLiteModel(config)
    # Simulate a pre-D-11 legacy artifact: state_dict with diff_mean removed.
    legacy_state = {k: v for k, v in model.state_dict().items() if k != "diff_mean"}

    fresh = CaimiraLiteModel(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # strict=False because the legacy dict is missing one key.
        fresh.load_state_dict(legacy_state, strict=False)
    legacy_warnings = [w for w in caught if "diff_mean" in str(w.message)]
    assert legacy_warnings, "expected a legacy-load warning mentioning diff_mean"
    # Buffer must be zero-initialized so the model still loads cleanly.
    assert torch.allclose(fresh.diff_mean, torch.zeros(config.latent_dim), atol=0.0)


def test_caimira_lite_predict_proba_is_clipped() -> None:
    from peval_gold.models.caimira_lite import CaimiraLiteConfig, CaimiraLiteModel

    model = CaimiraLiteModel(
        CaimiraLiteConfig(
            n_subjects=2,
            n_benchmarks=1,
            n_conditions=1,
            item_embed_dim=4,
            latent_dim=1,
            eps=1e-4,
        )
    )
    with torch.no_grad():
        model.global_intercept.fill_(100.0)

    probs = model.predict_proba_from_tensors(
        subject_idx=torch.tensor([0, 1]),
        item_emb=torch.zeros(2, 4),
        benchmark_idx=torch.zeros(2, dtype=torch.long),
        condition_idx=torch.zeros(2, dtype=torch.long),
    )

    assert probs.shape == (2,)
    assert torch.all(probs <= 1.0 - 1e-4)
    assert torch.all(probs >= 1e-4)


def test_caimira_lite_state_dict_roundtrip(tmp_path: Path) -> None:
    from peval_gold.models.caimira_lite import CaimiraLiteConfig, CaimiraLiteModel

    config = CaimiraLiteConfig(
        n_subjects=3,
        n_benchmarks=2,
        n_conditions=2,
        item_embed_dim=4,
        latent_dim=5,
    )
    model = CaimiraLiteModel(config)
    path = tmp_path / "caimira_lite.pt"
    torch.save(model.state_dict(), path)

    restored = CaimiraLiteModel(config)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    subject_idx = torch.tensor([0, 1, 2])
    benchmark_idx = torch.tensor([0, 1, 0])
    condition_idx = torch.tensor([0, 1, 1])
    item_emb = torch.randn(3, 4)
    with torch.no_grad():
        original_logits = model(subject_idx, item_emb, benchmark_idx, condition_idx)
        restored_logits = restored(subject_idx, item_emb, benchmark_idx, condition_idx)

    assert torch.allclose(original_logits, restored_logits, atol=1e-6)
