"""Shared centering-metadata schema for CAIMIRA artifacts.

Both the local trainer (``scripts/gold_train_caimira_lite.py``) and the
Modal trainer (``modal_train.py::m3_train_caimira_paper_faithful``)
write the same centering audit block into ``caimira_lite.meta.json`` and
``centering_audit.json`` so that fresh agents loading the artifact have
everything they need to verify the D-11 paper-faithful contract end to
end:

  - which axis was centered (train items, per latent dimension)
  - the exact center vector (so runtime can sanity-check the buffer)
  - the latent_dim (so the broadcast shape is unambiguous)
  - the item template version (so encoder text matches at runtime)
  - the encoder id, split id, binarize mode, seed (so reproducibility
    is bounded)
  - the code commit (so the centering rule used at training is the
    same one a fresh checkout would re-implement)

The module is pure stdlib so the Modal container imports it cheaply
(no torch required for the metadata block).
"""

from __future__ import annotations

from typing import Any

#: Keys that the centering metadata block MUST carry. Required by
#: the centering-contract test (``tests/test_caimira_centering_contract.py``
#: Test 6).
REQUIRED_CENTERING_META_KEYS: tuple[str, ...] = (
    "centering_method",
    "difficulty_center",
    "latent_dim",
    "item_template_version",
    "encoder_id",
    "split_id",
    "binarize_mode",
    "seed",
)

#: Keys that are allowed to be ``None`` if the source environment did
#: not provide them (e.g. ``code_git_commit`` is best-effort).
OPTIONAL_CENTERING_META_KEYS: tuple[str, ...] = ("code_git_commit",)

#: Literal value the ``centering_method`` field MUST take for any
#: post-D-11 artifact produced by the corrected-centered Modal sprint.
#: A different value indicates a different (or absent) centering rule
#: and must trigger intake rejection.
CANONICAL_CENTERING_METHOD: str = "train_item_mean_per_latent_dim"


def build_centering_meta_block(
    *,
    difficulty_center: list[float],
    latent_dim: int,
    item_template_version: str,
    encoder_id: str,
    split_id: str,
    binarize_mode: str,
    seed: int,
    code_git_commit: str | None = None,
) -> dict[str, Any]:
    """Build the centering metadata block for a CAIMIRA artifact.

    Parameters
    ----------
    difficulty_center : list[float]
        The train-item-mean centering vector (the ``diff_mean`` buffer)
        as a Python list of floats, length ``latent_dim``.
    latent_dim : int
        The CAIMIRA latent dimension ``m``.
    item_template_version : str
        Versioned identifier for the item template used to encode item
        text before passing to the difficulty projector (e.g.
        ``"canonical_item@2026-05-22"``).
    encoder_id : str
        Identifier for the sentence encoder used to produce item
        embeddings (e.g. ``"sentence-transformers/all-mpnet-base-v2"``).
    split_id : str
        Identifier for the train/val split policy
        (e.g. ``"item_heldout_primary:val_frac=0.1:seed=42"``).
    binarize_mode : str
        Identifier for the row-level binarization policy
        (e.g. ``"drop_nonbinary"`` per D-7).
    seed : int
        Top-level training seed.
    code_git_commit : str | None
        Repo HEAD git commit at training time. ``None`` is allowed in
        environments where the commit is not recoverable, but the block
        will surface that absence to intake.

    Returns
    -------
    dict
        Block in canonical JSON-serializable form.

    Raises
    ------
    ValueError
        If ``len(difficulty_center) != latent_dim``. Silent shape
        disagreement would break the runtime broadcast subtract.
    """
    if len(difficulty_center) != int(latent_dim):
        raise ValueError(
            f"difficulty_center length {len(difficulty_center)} does not "
            f"match latent_dim {latent_dim}. The center vector must have "
            f"one entry per latent skill."
        )
    return {
        "centering_method": CANONICAL_CENTERING_METHOD,
        "difficulty_center": [float(v) for v in difficulty_center],
        "latent_dim": int(latent_dim),
        "item_template_version": str(item_template_version),
        "encoder_id": str(encoder_id),
        "split_id": str(split_id),
        "binarize_mode": str(binarize_mode),
        "seed": int(seed),
        "code_git_commit": str(code_git_commit) if code_git_commit else None,
    }
