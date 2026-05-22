"""Gold-track wrapper for the shipped the wrapped NCF head + Platt scaler.

The current the hosted runtime submission's runtime entry point
(``the wrapped predict() module``) does module-level loads of the encoder, the
NCF head, the subject / item caches, and the Platt scaler state at
import time. Reusing the same module from the gold-track laboratory
would mean (a) executing the platform-bound module body (which mutates
process-global state), (b) carrying that state across every evaluator
call in the laboratory, and (c) coupling lab-side experimentation to
the production module's import-time side effects.

This wrapper avoids all three coupling points:

- It owns its OWN encoder, NCF head, per-instance subject/item caches,
  and per-instance Platt scaler. The submission module is NEVER
  imported.
- It re-uses the ``NCFHead`` architecture defined in
  ``the architecture file`` via ``importlib.util.spec_from_file_location``
  so the architecture stays in lock-step with the production trainer
  without executing ``the wrapped predict() module``.
- It loads weights via ``torch.load(weights_path, weights_only=True,
  map_location='cpu')`` per the documented safe pattern in
  (project pattern doc).

Mirroring the ``the wrapped predict() module:predict()`` semantics exactly
keeps the offline baseline numbers comparable to the on-platform
leaderboard:

- Encode ``subject_content`` and ``item_content`` with the cached
  encoder (per-instance cache, not module-global).
- Run the concatenated 2*768-d vector through the NCF head to get a
  raw logit.
- Fit a 1D Platt scaler ``sigmoid(a * logit + b)`` on revealed labels
  the first time ``predict_one`` is called with ``labeled`` set;
  cache the (a, b) afterward.
- Apply ``sigmoid(a * logit + b)``, clip to ``[1e-4, 1 - 1e-4]``,
  return a native Python float.

The Platt fallback rules match ``the wrapped predict() module:_fit_platt``
byte-for-byte:

- ``len(labeled) < 4`` → identity (a=1, b=0).
- All labels are the same class → identity.
- LBFGS exception or out-of-range / NaN (a, b) → identity.

Frozen-artifact discipline
--------------------------

The wrapper exposes ``fit()`` that raises ``RuntimeError`` — this is
intentional. The shipped ``the weights file`` is a frozen
deliverable from the the wrapped H100 multi-seed sweep (see
(project decision doc)); any
new training belongs in ``the training pipeline`` or the cluster
sbatch wrappers, not in a single-call lab fit. Mirroring the
``Predictor`` Protocol surface on a frozen wrapper would tempt agents
to "just call ``.fit()`` to retrain" — making it raise loud is the
distinguishable-defensive-fallback discipline applied at the API
layer.

The ``save()`` / ``load()`` pair writes a small JSON pointer (path +
sha256 + arch version) rather than copying the 1.7 MB weight blob.
Two ``CurrentNCF()`` instances both refer to the same on-disk
``ncf_head.pt`` — there is no reason to duplicate it in the offline
laboratory.
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

DEFAULT_ENCODER_REPO = "sentence-transformers/all-mpnet-base-v2"

# Mirrors the wrapped predict() module:_fit_platt thresholds (line 224-269).
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
    unique module name so other imports of ``ncf_head`` (e.g., from
    ``the wrapped predict() module:150`` ``from ncf_head import NCFHead``) don't
    interfere with the laboratory copy.
    """
    if not ncf_head_path.exists():
        raise FileNotFoundError(
            f"the architecture file not found at {ncf_head_path}; the "
            "CurrentNCF wrapper requires the shipped architecture."
        )
    spec = importlib.util.spec_from_file_location(
        "peval_gold._loaded_submission_ncf_head",
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
            "wrapper expects the the wrapped-era architecture."
        )
    return module.NCFHead


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class CurrentNCF:
    """Read-only wrapper for the shipped ``the weights file``.

    Implements both:

    - :class:`peval_gold.models.base.Predictor` (offline ``predict_proba``
      surface). The ``fit`` method raises ``RuntimeError`` because the
      artifact is frozen — see module docstring for rationale.
    - :class:`peval_gold.models.base.RuntimePredictor` (streaming
      ``predict_one`` surface). Mirrors ``the wrapped predict() module:predict()``
      byte-for-byte modulo the LOCAL_SMOKE early-return (which is a
      tooling-only branch and not part of the laboratory contract).

    Per-instance state — NEVER touches any module-level globals:

    - ``self._encoder``: the SentenceTransformer (lazy-init).
    - ``self._head``: the loaded NCFHead with frozen weights.
    - ``self._subject_cache`` / ``self._item_cache``: dict caches over
      encoded text.
    - ``self._calibrator_a`` / ``self._calibrator_b`` / ``self._calibrator_fit``:
      the Platt scaler state.

    Memory rules:

    - ``device='cpu'`` always; the lab does not fight for GPU. The
      a GPU cluster has its own runtime path through ``the training pipeline``.
    - ``local_files_only=True`` so the encoder load resolves from the
      HF cache without a network round-trip (Posture-A-friendly).
    - ``torch.no_grad()`` around every forward pass.
    """

    def __init__(
        self,
        weights_path: str | os.PathLike[str],
        ncf_head_path: str | os.PathLike[str],
        encoder_repo: str = DEFAULT_ENCODER_REPO,
        encoder_batch_size: int = 64,
    ) -> None:
        self._weights_path = Path(weights_path).resolve()
        self._ncf_head_path = Path(ncf_head_path).resolve()
        self._encoder_repo = encoder_repo
        self._encoder_batch_size = int(encoder_batch_size)
        self._device = "cpu"

        # Lazy-imports torch + sentence-transformers so importing this
        # module does NOT pay the ~5s torch-load cost on machines where
        # the user only wants the Protocol surface (e.g., a downstream
        # config validator).
        import torch
        from sentence_transformers import SentenceTransformer

        self._torch = torch  # stashed for use in instance methods
        self._encoder = SentenceTransformer(
            self._encoder_repo,
            device=self._device,
            local_files_only=True,
        )

        ncf_head_cls = _load_ncf_head_class(self._ncf_head_path)
        # Capture the loaded NCFHead module's NCF_ARCH_VERSION constant
        # (lives next to the class definition) for the metadata slot.
        spec = importlib.util.spec_from_file_location(
            "peval_gold._loaded_submission_ncf_head_meta",
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

        # Platt scaler state — instance-level, NOT module-level.
        self._calibrator_a: float = 1.0
        self._calibrator_b: float = 0.0
        self._calibrator_fit: bool = False

    # ----- Protocol: RuntimePredictor ---------------------------------

    def predict_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
        labeled: list[dict] | None = None,
    ) -> float:
        """Per-call probability prediction mirroring ``the wrapped predict() module:predict()``.

        Returns a native Python ``float`` in ``[1e-4, 1 - 1e-4]``,
        finite, never NaN. Identity Platt fallback rules:

        - ``labeled is None`` or empty → identity (a=1, b=0).
        - ``len(labeled) < 4`` → identity.
        - All labels same class → identity.
        - LBFGS NaN / out-of-range → identity.

        Side effect: the FIRST call with a non-empty ``labeled`` set
        fits the Platt scaler and caches (a, b). Subsequent calls
        reuse the cached values until :meth:`reset_calibrator` is
        called.
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
        """Drop the cached Platt fit so the next ``predict_one`` re-fits.

        Required between adaptive-simulation rounds so each round
        produces an independent measurement. The submission module
        resets via container restart; we reset via this method.
        """
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

        The wrapper holds a frozen the wrapped artifact; any retrain
        belongs in ``the training pipeline`` or the cluster sbatch
        wrappers (``the cluster harness``).
        Mirroring the Protocol surface on this method is intentional
        so type-checkers still see a conformant class; the runtime
        raise prevents accidental retraining inside the lab.
        """
        raise RuntimeError(
            "CurrentNCF wraps a frozen the wrapped artifact "
            f"({self._weights_path.name}); use the training pipeline or "
            "the cluster harness to train a new head."
        )

    def predict_proba(self, rows: Sequence[dict]) -> np.ndarray:
        """Batched offline prediction, identity-Platt only.

        Always uses the IDENTITY Platt path (a=1, b=0) regardless of
        the instance's cached calibrator state. This is the contract
        the matching unit test
        (``test_current_ncf_predict_proba_matches_per_row_predict_one``)
        relies on: ``predict_proba(rows)`` must equal
        ``[predict_one(r, labeled=None) for r in rows]`` for a freshly-
        instantiated wrapper.

        Batches encoder forwards by ``self._encoder_batch_size`` (default
        64) and runs NCFHead in a single forward over the concatenated
        2*768-d vectors. The encoder caches are still consulted so
        repeated subject/item content within a batch only pays one
        embed cost.
        """
        n = len(rows)
        if n == 0:
            return np.array([], dtype=np.float64)

        # Batch the encoder calls per text. We collect a unique list of
        # subjects and items in the input order, encode them in batches,
        # then look them back up per row.
        subjects = [r.get("subject_content", "") for r in rows]
        items = [r.get("item_content", "") for r in rows]

        u_vecs = self._encode_many(subjects, self._subject_cache)
        v_vecs = self._encode_many(items, self._item_cache)

        with self._torch.no_grad():
            x = self._torch.cat([u_vecs, v_vecs], dim=-1).to(self._device)
            logits = self._head(x)
            probs = self._torch.sigmoid(logits).cpu().numpy().astype(np.float64)

        # Apply the same [1e-4, 1-1e-4] clip used per-call.
        np.clip(probs, _FINAL_CLIP_EPS, 1.0 - _FINAL_CLIP_EPS, out=probs)
        return probs

    # ----- save / load (pointer JSON, not weight copy) ----------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a small JSON pointer to ``path``.

        Format::

            {
              "weights_path": "/abs/path/to/the weights file",
              "weights_sha256": "...",
              "ncf_arch_version": 1,
              "encoder_repo": "sentence-transformers/all-mpnet-base-v2"
            }

        Does NOT copy the 1.7 MB weight blob; the laboratory and the
        production submission share the same on-disk artifact.
        """
        payload = {
            "weights_path": str(self._weights_path),
            "weights_sha256": _sha256(self._weights_path),
            "ncf_arch_version": self._ncf_arch_version,
            "encoder_repo": self._encoder_repo,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> CurrentNCF:
        """Reconstruct a wrapper from a pointer JSON written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        return cls(
            weights_path=payload.get("weights_path"),
            encoder_repo=payload.get("encoder_repo", DEFAULT_ENCODER_REPO),
        )

    # ----- Metadata --------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the wrapped artifact."""
        return {
            "class": "CurrentNCF",
            "weights_path": str(self._weights_path),
            "weights_sha256": _sha256(self._weights_path),
            "ncf_arch_version": self._ncf_arch_version,
            "encoder_repo": self._encoder_repo,
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
        """Batched encoder pass returning a (N, EMBED_DIM) torch tensor.

        Caches per-text so repeats in a batch only pay one embed cost.
        Returns a torch.float32 tensor on CPU.
        """
        torch = self._torch
        n = len(texts)
        # Resolve cached texts first; build a list of unique missing
        # strings to encode in one call.
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, t in enumerate(texts):
            if t not in cache:
                missing_idx.append(i)
                missing_texts.append(t)

        if missing_texts:
            # Encode missing in batches.
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
        # Stack into a single (N, D) tensor on CPU.
        stacked = torch.stack(out_tensors, dim=0).to(self._device)
        return stacked

    def _ncf_logit(self, input_dict: dict) -> Any:
        """One (subject, item) pair through encoder + NCF head → scalar logit."""
        torch = self._torch
        u = self._encode_one(input_dict.get("subject_content", ""), self._subject_cache)
        v = self._encode_one(input_dict.get("item_content", ""), self._item_cache)
        with torch.no_grad():
            x = torch.cat([u, v], dim=-1).to(self._device)
            return self._head(x)

    def _fit_platt(self, labeled: list[dict]) -> tuple[float, float]:
        """Mirror ``the wrapped predict() module:_fit_platt`` byte-for-byte.

        Returns (1.0, 0.0) for the documented identity-fallback cases.
        """
        torch = self._torch
        if len(labeled) < _PLATT_MIN_LABELS:
            return 1.0, 0.0

        # NOTE: this except mirrors the wrapped predict() module:271 which is
        # the LAST surviving inner Platt-fallback exception handler.
        # It returns the un-calibrated probability via the normal
        # forward path (a=1, b=0); it is NOT a swallow of the outer
        # contract. See the wrapped predict() module:226-234 for the rationale
        # and -
        # fallbacks-2026-05-18.md for the design rule.
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
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["CurrentNCF"]
