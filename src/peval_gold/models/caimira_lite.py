"""Lean content-aware multidimensional IRT model for gold-track experiments.

This module is local-only experiment code. It implements the CAIMIRA-lite
formula used by Device 2 artifact runs:

    logit = sum((theta_subject - difficulty(item)) * relevance(item))
            + subject_bias + benchmark_bias + condition_bias + intercept

Difficulty centering is **paper-faithful corpus-mean** per CAIMIRA §8.6
(``d_j = d'_j - (1/n_q) sum d'_j``), implemented via the registered buffer
``diff_mean`` and the end-of-epoch ``refresh_diff_mean(all_item_embs)``
protocol specified in ``plans/03_ncf_and_caimira.md`` lines 540-547. See
(project decision doc) for the
decision record (supersedes the prior per-item across-latent-dim mean
flagged by the 2026-05-22 ChatGPT-5.5 Pro review Claim 2).

The model is deliberately small and saves cleanly as a ``state_dict``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CaimiraLiteConfig:
    """Configuration for :class:`CaimiraLiteModel`."""

    n_subjects: int
    n_benchmarks: int
    n_conditions: int
    item_embed_dim: int = 768
    latent_dim: int = 3
    eps: float = 1e-4
    use_subject_bias: bool = True
    use_benchmark_bias: bool = True
    use_condition_bias: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CaimiraLiteModel(nn.Module):
    """Small content-aware IRT module with text-amortized item parameters."""

    def __init__(self, config: CaimiraLiteConfig) -> None:
        super().__init__()
        if config.n_subjects < 1:
            raise ValueError("n_subjects must be at least 1")
        if config.n_benchmarks < 1:
            raise ValueError("n_benchmarks must be at least 1")
        if config.n_conditions < 1:
            raise ValueError("n_conditions must be at least 1")
        if config.item_embed_dim < 1:
            raise ValueError("item_embed_dim must be at least 1")
        if config.latent_dim < 1:
            raise ValueError("latent_dim must be at least 1")
        if not (0.0 < config.eps < 0.5):
            raise ValueError("eps must be in (0, 0.5)")

        self.config = config
        self.subject_skill = nn.Embedding(config.n_subjects, config.latent_dim)
        self.relevance = nn.Linear(config.item_embed_dim, config.latent_dim)
        self.difficulty = nn.Linear(config.item_embed_dim, config.latent_dim)
        # Paper-faithful corpus-mean centering buffer (CAIMIRA §8.6;
        # plans/03_ncf_and_caimira.md:540-547). Initialized to zeros so a
        # fresh model is uncentered; `refresh_diff_mean()` must be called
        # at the end of each training epoch to populate this buffer with
        # the mean of the raw difficulty over the full training item bank.
        self.register_buffer("diff_mean", torch.zeros(config.latent_dim))
        self.global_intercept = nn.Parameter(torch.zeros(()))

        self.subject_bias: nn.Embedding | None
        self.benchmark_bias: nn.Embedding | None
        self.condition_bias: nn.Embedding | None

        self.subject_bias = nn.Embedding(config.n_subjects, 1) if config.use_subject_bias else None
        self.benchmark_bias = (
            nn.Embedding(config.n_benchmarks, 1) if config.use_benchmark_bias else None
        )
        self.condition_bias = (
            nn.Embedding(config.n_conditions, 1) if config.use_condition_bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters conservatively for BCE training."""
        nn.init.normal_(self.subject_skill.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.relevance.weight)
        nn.init.zeros_(self.relevance.bias)
        nn.init.xavier_uniform_(self.difficulty.weight)
        nn.init.zeros_(self.difficulty.bias)
        nn.init.zeros_(self.global_intercept)
        if self.subject_bias is not None:
            nn.init.zeros_(self.subject_bias.weight)
        if self.benchmark_bias is not None:
            nn.init.zeros_(self.benchmark_bias.weight)
        if self.condition_bias is not None:
            nn.init.zeros_(self.condition_bias.weight)

    def forward(
        self,
        subject_idx: torch.Tensor,
        item_emb: torch.Tensor,
        benchmark_idx: torch.Tensor | None = None,
        condition_idx: torch.Tensor | None = None,
        *,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return logits for a batch of subject/item observations."""
        if item_emb.ndim != 2:
            raise ValueError("item_emb must have shape (batch, item_embed_dim)")

        theta = self.subject_skill(subject_idx.long())
        relevance = torch.softmax(self.relevance(item_emb.float()), dim=-1)
        difficulty_raw = self.difficulty(item_emb.float())
        # Paper-faithful centering: subtract the corpus-mean buffer (shape
        # (latent_dim,)) computed over the training item bank by
        # `refresh_diff_mean()`. Broadcasts to (batch, latent_dim).
        # Prior implementation centered across the latent axis per item
        # (`difficulty_raw.mean(dim=-1, keepdim=True)`), which is not the
        # CAIMIRA §8.6 protocol; see D-11 for the decision record.
        difficulty = difficulty_raw - self.diff_mean
        logits = ((theta - difficulty) * relevance).sum(dim=-1)
        logits = logits + self.global_intercept

        if self.subject_bias is not None:
            logits = logits + self.subject_bias(subject_idx.long()).squeeze(-1)
        if self.benchmark_bias is not None:
            if benchmark_idx is None:
                raise ValueError("benchmark_idx is required when benchmark bias is enabled")
            logits = logits + self.benchmark_bias(benchmark_idx.long()).squeeze(-1)
        if self.condition_bias is not None:
            if condition_idx is None:
                raise ValueError("condition_idx is required when condition bias is enabled")
            logits = logits + self.condition_bias(condition_idx.long()).squeeze(-1)

        if not return_parts:
            return logits

        parts = {
            "theta": theta,
            "difficulty": difficulty,
            "relevance": relevance,
        }
        return logits, parts

    @torch.no_grad()
    def predict_proba_from_tensors(
        self,
        subject_idx: torch.Tensor,
        item_emb: torch.Tensor,
        benchmark_idx: torch.Tensor | None = None,
        condition_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return clipped probabilities for already-indexed tensors."""
        logits = self.forward(
            subject_idx=subject_idx,
            item_emb=item_emb,
            benchmark_idx=benchmark_idx,
            condition_idx=condition_idx,
            return_parts=False,
        )
        return torch.sigmoid(logits).clamp(
            min=self.config.eps,
            max=1.0 - self.config.eps,
        )

    def regularization_loss(
        self,
        *,
        l2_subject: float = 0.0,
        l2_difficulty: float = 0.0,
        l2_bias: float = 0.0,
    ) -> torch.Tensor:
        """Return a small optional L2 penalty used by training scripts."""
        total = self.global_intercept.new_zeros(())
        if l2_subject:
            total = total + float(l2_subject) * self.subject_skill.weight.square().mean()
        if l2_difficulty:
            total = total + float(l2_difficulty) * self.difficulty.weight.square().mean()
        if l2_bias:
            bias_terms = []
            if self.subject_bias is not None:
                bias_terms.append(self.subject_bias.weight.square().mean())
            if self.benchmark_bias is not None:
                bias_terms.append(self.benchmark_bias.weight.square().mean())
            if self.condition_bias is not None:
                bias_terms.append(self.condition_bias.weight.square().mean())
            if bias_terms:
                total = total + float(l2_bias) * torch.stack(bias_terms).mean()
        return total

    @torch.no_grad()
    def refresh_diff_mean(self, all_item_embs: torch.Tensor) -> None:
        """Refresh the corpus-mean ``diff_mean`` buffer over the item bank.

        Implements paper-faithful CAIMIRA §8.6 centering
        (``d_j = d'_j - (1/n_q) sum d'_j``) per the protocol in
        ``plans/03_ncf_and_caimira.md`` lines 540-547. Call once at the end
        of each training epoch (in inference mode, no grad) so the next
        epoch's forward pass uses a quasi-static reference; call once more
        after loading the best checkpoint before the final val pass.

        Parameters
        ----------
        all_item_embs : torch.Tensor
            ``(n_items, item_embed_dim)`` — embeddings of every item in the
            training bank. Dtype is cast to float internally.

        Notes
        -----
        Runtime is one matmul through ``self.difficulty`` plus a mean
        reduction along the item axis. The buffer is updated in place via
        ``copy_`` so ``state_dict`` ownership and device placement survive.
        """
        if all_item_embs.ndim != 2:
            raise ValueError("all_item_embs must have shape (n_items, item_embed_dim)")
        raw = self.difficulty(all_item_embs.float())
        self.diff_mean.copy_(raw.mean(dim=0))

    def _load_from_state_dict(  # type: ignore[override]
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Soft-fallback for legacy artifacts saved before D-11.

        Pre-D-11 ``state_dict``s lack the ``diff_mean`` buffer (the prior
        implementation centered per-item across the latent axis at forward
        time, with no buffer). To avoid hard-breaking those artifacts, we
        inject a zero buffer and emit a warning instructing the caller to
        either retrain or call ``refresh_diff_mean()`` against the item
        bank before relying on the loaded model.
        """
        buffer_key = f"{prefix}diff_mean"
        if buffer_key not in state_dict:
            state_dict[buffer_key] = torch.zeros_like(self.diff_mean)
            warnings.warn(
                "Loaded a legacy CaimiraLiteModel state_dict without "
                "'diff_mean'; the buffer was zero-initialized. Difficulty "
                "will be uncentered until you call refresh_diff_mean() on "
                "the training item bank, or retrain. See "
                ".",
                stacklevel=2,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
