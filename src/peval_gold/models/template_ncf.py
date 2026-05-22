"""Template-correct NCF wrapper for the gold-track H1 deliverable.

``TemplateNCF`` is the gold-track Batch-4 Predictor that wraps a freshly
trained NCF head (state_dict) and applies a selectable item-text
formatter from :mod:`peval_gold.data.templates` BEFORE encoding the
item text. The subject text is unchanged.

Why this exists
---------------

The current shipped ``the weights file`` was trained on the
``item_only`` text (raw ``item_content``); the encoder therefore never
sees ``benchmark`` or ``condition`` — see CoVe Claim 1 in
``docs/walkthroughs/012_gold_track_current_state_cove.md``. The H1
hypothesis is: re-train the same architecture on the ``canonical_item``
template (``Benchmark: ...\\nCondition: ...\\nItem:\\n...``) and the
encoder will recover that signal.

Relationship to :class:`CurrentNCF`
----------------------------------

This wrapper is intentionally structurally identical to
:class:`peval_gold.models.current_ncf.CurrentNCF` (same Platt
fallback rules, same per-instance encoder + cache discipline, same
``[1e-4, 1 - 1e-4]`` clip). The two only diverge in one place: the
item-side text fed to the encoder. Keeping the structure mirror-image
means the gold-track evaluator can swap one for the other behind the
same :class:`peval_gold.models.base.Predictor` /
:class:`RuntimePredictor` Protocol surface.

Frozen-artifact discipline
--------------------------

Like :class:`CurrentNCF`, this wrapper exposes ``fit()`` that raises
:class:`RuntimeError` — new training belongs in
``notebooks/train_ncf_template.py`` (or its cluster sbatch wrapper),
not in a single-call lab fit. The pointer-JSON save/load pair lets the
laboratory and the eventual submission share the same on-disk weight
blob without copying it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from peval_gold.data.templates import TEMPLATES, get_template

DEFAULT_ENCODER_REPO = "sentence-transformers/all-mpnet-base-v2"

# Platt fallback constants — mirror CurrentNCF / the wrapped predict() module.
_PLATT_MIN_LABELS = 4
_PLATT_A_BOUND = 1e6
_PLATT_B_BOUND = 1e6
_FINAL_CLIP_EPS = 1e-4


# ---------------------------------------------------------------------------
# Module loading helpers (read-only access to the architecture file)
# ---------------------------------------------------------------------------


def _load_ncf_head_class(ncf_head_path: Path) -> type:
    """Load ``NCFHead`` from a path without executing ``the wrapped predict() module``.

    Uses ``importlib.util.spec_from_file_location`` under a private
    unique module name so other imports of ``ncf_head`` (e.g. from the
    submission package) don't interfere with the laboratory copy.
    """
    if not ncf_head_path.exists():
        raise FileNotFoundError(
            f"the architecture file not found at {ncf_head_path}; the "
            "TemplateNCF wrapper requires the shipped architecture."
        )
    spec = importlib.util.spec_from_file_location(
        "peval_gold._loaded_submission_ncf_head_for_template",
        ncf_head_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"could not build spec for {ncf_head_path}; Python import machinery returned None"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "NCFHead"):
        raise AttributeError(
            "loaded the architecture file is missing NCFHead — "
            "TemplateNCF expects the the wrapped-era architecture."
        )
    return module.NCFHead


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class TemplateNCF:
    """NCF inference wrapper that templates the item text before encoding.

    Implements both:

    - :class:`peval_gold.models.base.Predictor` (offline ``predict_proba``
      surface).
    - :class:`peval_gold.models.base.RuntimePredictor` (streaming
      ``predict_one`` surface).

    Parameters
    ----------
    weights_path : str | os.PathLike[str]
        Path to the NCF head state_dict (saved via
        ``torch.save(model.state_dict(), path)``). Must match the
        :class:`NCFHead` architecture in ``the architecture file``.
    template_fn_name : str
        Name of an item-template formatter registered in
        :data:`peval_gold.data.templates.TEMPLATES`. Accepted values:
        ``"item_only"``, ``"canonical_item"``, ``"rich_item"``.
    ncf_head_path : str | os.PathLike[str] | None
        Override path to ``ncf_head.py``. Default: the submission copy.
    encoder_repo : str
        HuggingFace repo id for the SentenceTransformer encoder. Default
        ``sentence-transformers/all-mpnet-base-v2`` (matches the shipped
        artifact's pre-fetch list).
    encoder_batch_size : int
        Batch size for :meth:`predict_proba`'s encoder pass.
    encoder : optional
        Pre-built encoder object. When set, the constructor SKIPS the
        SentenceTransformer load. Used by unit tests to inject a
        deterministic stub that doesn't require the HF cache.
    """

    def __init__(
        self,
        weights_path: str | os.PathLike[str],
        ncf_head_path: str | os.PathLike[str],
        template_fn_name: str = "canonical_item",
        encoder_repo: str = DEFAULT_ENCODER_REPO,
        encoder_batch_size: int = 64,
        encoder: Any | None = None,
    ) -> None:
        self._weights_path = Path(weights_path).resolve()
        self._ncf_head_path = Path(ncf_head_path).resolve()
        self._encoder_repo = encoder_repo
        self._encoder_batch_size = int(encoder_batch_size)
        self._device = "cpu"

        # Validate template name BEFORE the heavy torch / encoder load so a
        # typo in --template fails fast instead of after a 5-second startup.
        self._template_fn_name = str(template_fn_name)
        if self._template_fn_name not in TEMPLATES:
            raise ValueError(
                f"unknown template_fn_name {self._template_fn_name!r}; "
                f"accepted: {sorted(TEMPLATES)}"
            )
        self._template_fn = get_template(self._template_fn_name)

        # Lazy-imports torch + sentence-transformers so importing this
        # module does NOT pay the multi-second torch-load cost on
        # machines that only want the Protocol surface (e.g., a config
        # validator).
        import torch

        self._torch = torch

        if encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self._encoder_repo,
                device=self._device,
                local_files_only=True,
            )
        else:
            # Caller-provided encoder — typically a unit-test stub. We
            # trust the caller's shape; ``predict_one`` only needs an
            # ``.encode(text, convert_to_tensor=True, ...)`` method that
            # returns a torch tensor of shape ``(EMBED_DIM,)``.
            self._encoder = encoder

        ncf_head_cls = _load_ncf_head_class(self._ncf_head_path)
        spec = importlib.util.spec_from_file_location(
            "peval_gold._loaded_submission_ncf_head_for_template_meta",
            self._ncf_head_path,
        )
        meta_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(meta_module)
        self._ncf_arch_version = int(getattr(meta_module, "NCF_ARCH_VERSION", 1))

        state = torch.load(
            self._weights_path,
            map_location=self._device,
            weights_only=True,
        )
        self._head = ncf_head_cls().to(self._device)
        self._head.load_state_dict(state)
        self._head.eval()
        for param in self._head.parameters():
            param.requires_grad_(False)

        self._subject_cache: dict[str, Any] = {}
        self._item_cache: dict[str, Any] = {}

        # Per-instance Platt scaler — never module-level.
        self._calibrator_a: float = 1.0
        self._calibrator_b: float = 0.0
        self._calibrator_fit: bool = False

    # ----- Protocol: RuntimePredictor ---------------------------------

    def predict_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
        labeled: list[dict] | None = None,
    ) -> float:
        """Per-call probability prediction mirroring ``the wrapped predict() module``.

        The KEY difference vs :class:`CurrentNCF`: the encoder input for
        the item side is ``self._template_fn(input)``, not the raw
        ``input["item_content"]``.
        """
        if not self._calibrator_fit and labeled:
            self._calibrator_a, self._calibrator_b = self._fit_platt(labeled)
            self._calibrator_fit = True

        with self._torch.no_grad():
            raw_logit = self._ncf_logit(input)
            calibrated = self._calibrator_a * raw_logit + self._calibrator_b
            prob = self._torch.sigmoid(calibrated).item()

        return float(min(max(prob, _FINAL_CLIP_EPS), 1.0 - _FINAL_CLIP_EPS))

    def reset_calibrator(self) -> None:
        """Drop the cached Platt fit so the next ``predict_one`` re-fits."""
        self._calibrator_a = 1.0
        self._calibrator_b = 0.0
        self._calibrator_fit = False

    # ----- Protocol: Predictor ----------------------------------------

    def fit(
        self,
        train_rows: Sequence[dict],
        valid_rows: Sequence[dict] | None = None,
    ) -> None:
        """Always raises ``RuntimeError``.

        TemplateNCF wraps a frozen artifact; new training belongs in
        ``notebooks/train_ncf_template.py`` or its cluster sbatch
        wrapper. Mirroring the Protocol surface on this method keeps
        type-checkers happy while preventing accidental lab retraining.
        """
        raise RuntimeError(
            "TemplateNCF wraps a frozen artifact "
            f"({self._weights_path.name}); use "
            "notebooks/train_ncf_template.py or "
            "the cluster harness to train a new head."
        )

    def predict_proba(self, rows: Sequence[dict]) -> np.ndarray:
        """Batched offline prediction, identity-Platt only.

        Applies the configured template to each row's item text before
        encoding. Equivalent to running ``predict_one(row, labeled=None)``
        for each row but batched at the encoder level for ~5-10x speedup
        on warm caches.
        """
        n = len(rows)
        if n == 0:
            return np.array([], dtype=np.float64)

        subjects = [r.get("subject_content", "") for r in rows]
        templated_items = [self._template_fn(r) for r in rows]

        u_vecs = self._encode_many(subjects, self._subject_cache)
        v_vecs = self._encode_many(templated_items, self._item_cache)

        with self._torch.no_grad():
            x = self._torch.cat([u_vecs, v_vecs], dim=-1).to(self._device)
            logits = self._head(x)
            probs = self._torch.sigmoid(logits).cpu().numpy().astype(np.float64)

        np.clip(probs, _FINAL_CLIP_EPS, 1.0 - _FINAL_CLIP_EPS, out=probs)
        return probs

    # ----- save / load (pointer JSON, not weight copy) ----------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a small JSON pointer to ``path``.

        Format::

            {
              "weights_path": "/abs/path/to/ncf_head.pt",
              "weights_sha256": "...",
              "ncf_arch_version": 1,
              "encoder_repo": "sentence-transformers/all-mpnet-base-v2",
              "template_fn_name": "canonical_item"
            }
        """
        payload = {
            "weights_path": str(self._weights_path),
            "weights_sha256": _sha256(self._weights_path),
            "ncf_arch_version": self._ncf_arch_version,
            "encoder_repo": self._encoder_repo,
            "template_fn_name": self._template_fn_name,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        encoder: Any | None = None,
    ) -> TemplateNCF:
        """Reconstruct a wrapper from a pointer JSON written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        return cls(
            weights_path=payload["weights_path"],
            template_fn_name=payload.get("template_fn_name", "canonical_item"),
            encoder_repo=payload.get("encoder_repo", DEFAULT_ENCODER_REPO),
            encoder=encoder,
        )

    # ----- Metadata ---------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the wrapped artifact."""
        return {
            "class": "TemplateNCF",
            "weights_path": str(self._weights_path),
            "weights_sha256": _sha256(self._weights_path),
            "ncf_arch_version": self._ncf_arch_version,
            "encoder_repo": self._encoder_repo,
            "template_fn_name": self._template_fn_name,
            "device": self._device,
            "calibrator_fit": bool(self._calibrator_fit),
            "calibrator_a": float(self._calibrator_a),
            "calibrator_b": float(self._calibrator_b),
        }

    # ----- Internals --------------------------------------------------

    def _encode_one(self, text: str, cache: dict[str, Any]) -> Any:
        """Encode one text via the encoder, caching by exact-string key."""
        if text in cache:
            return cache[text]
        vec = self._encoder.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        cache[text] = vec
        return vec

    def _encode_many(self, texts: list[str], cache: dict[str, Any]) -> Any:
        """Batched encoder pass returning a ``(N, EMBED_DIM)`` torch tensor."""
        torch = self._torch
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, t in enumerate(texts):
            if t not in cache:
                missing_idx.append(i)
                missing_texts.append(t)

        if missing_texts:
            for start in range(0, len(missing_texts), self._encoder_batch_size):
                batch = missing_texts[start : start + self._encoder_batch_size]
                vecs = self._encoder.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                for j, t in enumerate(batch):
                    cache[t] = vecs[j]

        out_tensors = [cache[t] for t in texts]
        stacked = torch.stack(out_tensors, dim=0).to(self._device)
        return stacked

    def _ncf_logit(self, input_dict: dict) -> Any:
        """One (subject, templated-item) pair through encoder + NCF head → scalar logit.

        This is the load-bearing difference vs :class:`CurrentNCF`: the
        item side is encoded as ``self._template_fn(input_dict)`` not
        as ``input_dict["item_content"]``.
        """
        torch = self._torch
        u = self._encode_one(input_dict.get("subject_content", ""), self._subject_cache)
        templated = self._template_fn(input_dict)
        v = self._encode_one(templated, self._item_cache)
        with torch.no_grad():
            x = torch.cat([u, v], dim=-1).to(self._device)
            return self._head(x)

    def _fit_platt(self, labeled: list[dict]) -> tuple[float, float]:
        """Mirror ``the wrapped predict() module:_fit_platt`` byte-for-byte.

        Returns ``(1.0, 0.0)`` for the documented identity-fallback cases.
        """
        torch = self._torch
        if len(labeled) < _PLATT_MIN_LABELS:
            return 1.0, 0.0

        # Local exception scope only — same shape as CurrentNCF._fit_platt
        # and the wrapped predict() module:_fit_platt. NOT a swallow of the outer
        # predict() contract.
        try:
            logits_list: list[float] = []
            targets_list: list[float] = []
            for ex in labeled:
                logit = self._ncf_logit(ex)
                logits_list.append(float(logit.detach().cpu().item()))
                targets_list.append(float(ex["label"]))

            if min(targets_list) == max(targets_list):
                return 1.0, 0.0

            logits_t = torch.tensor(logits_list, dtype=torch.float32)
            targets_t = torch.tensor(targets_list, dtype=torch.float32)

            a = torch.tensor(1.0, requires_grad=True)
            b = torch.tensor(0.0, requires_grad=True)
            opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

            def closure():  # type: ignore[no-untyped-def]
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    a * logits_t + b, targets_t
                )
                loss.backward()
                return loss

            opt.step(closure)
            a_val, b_val = float(a.item()), float(b.item())
            if not (
                -_PLATT_A_BOUND < a_val < _PLATT_A_BOUND
                and -_PLATT_B_BOUND < b_val < _PLATT_B_BOUND
                and a_val == a_val
                and b_val == b_val
            ):
                return 1.0, 0.0
            return a_val, b_val
        except Exception:  # pylint: disable=broad-except
            return 1.0, 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """SHA256 of a file's bytes (chunked to bound memory)."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["TemplateNCF"]
