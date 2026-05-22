"""CPU-feasible Rasch / REEval-lite baseline for gold-track experiments.

This module is **local-only** and is not imported by ``the wrapped submission's ``.
It implements the first-pass structural psychometric challenger:

    logit(p(correct)) = theta_subject - beta_item_or_text

``theta_subject`` is a smoothed subject ability lookup.  ``beta`` is a
seen-item difficulty lookup when the item was observed during training,
otherwise a tiny hashed-text ridge regressor predicts difficulty from
``item_content`` plus benchmark/condition tokens.

The text regressor deliberately avoids heavyweight embeddings so the
first pass remains CPU-feasible and cache-independent.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from peval_gold.data.normalize import parse_subject_name
from peval_gold.eval.metrics import clip_probability, safe_logit, sigmoid

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+\-.]{1,}", re.IGNORECASE)
_SUBJECT_HASH_BYTES = 8
_ITEM_HASH_BYTES = 8


class RaschLite:
    """Small Rasch-style predictor with held-out text difficulty fallback.

    Parameters are intentionally conservative:

    - ``subject_kappa`` / ``item_kappa`` / ``benchmark_kappa`` smooth raw
      binomial rates toward the global rate.
    - ``text_dim`` controls the hashed feature width.  128-256 is enough
      for first-pass local screening.
    - ``ridge_lambda`` regularizes the text difficulty regressor.
    - ``unseen_text_weight`` blends text-predicted difficulty with a
      benchmark difficulty fallback for unseen items.
    """

    def __init__(
        self,
        *,
        subject_kappa: float = 20.0,
        item_kappa: float = 2.0,
        benchmark_kappa: float = 50.0,
        text_dim: int = 256,
        ridge_lambda: float = 10.0,
        max_text_tokens: int = 64,
        unseen_text_weight: float = 0.75,
        eps: float = 1e-4,
        use_seen_item_lookup: bool = True,
    ) -> None:
        if text_dim < 4:
            raise ValueError("text_dim must be at least 4")
        self.subject_kappa = float(subject_kappa)
        self.item_kappa = float(item_kappa)
        self.benchmark_kappa = float(benchmark_kappa)
        self.text_dim = int(text_dim)
        self.ridge_lambda = float(ridge_lambda)
        self.max_text_tokens = int(max_text_tokens)
        self.unseen_text_weight = float(np.clip(unseen_text_weight, 0.0, 1.0))
        self.eps = float(eps)
        self.use_seen_item_lookup = bool(use_seen_item_lookup)

        self._fitted = False
        self._global_p = 0.5
        self._global_logit = 0.0
        self._subject_theta: dict[str, float] = {}
        self._item_beta: dict[str, float] = {}
        self._benchmark_beta: dict[str, float] = {}
        self._text_weights = np.zeros(self.text_dim, dtype=float)
        self._n_fit_rows = 0
        self._n_text_items = 0

    # ------------------------------------------------------------------
    # Predictor protocol
    # ------------------------------------------------------------------

    def fit(
        self,
        train_rows: Sequence[Mapping[str, Any]],
        valid_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Fit subject ability and item/text difficulty from binary rows."""
        del valid_rows

        subject_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        item_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        benchmark_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        item_text: dict[str, str] = {}
        item_benchmark: dict[str, str] = {}
        item_condition: dict[str, str] = {}
        global_k = 0
        global_n = 0

        for row in train_rows:
            y = _coerce_binary_response(row.get("response"))
            if y is None:
                continue

            subject_key = _subject_key(row)
            item_key = _item_key(row)
            benchmark = str(row.get("benchmark", ""))
            condition = str(row.get("condition", "none"))

            subject_counts[subject_key][0] += y
            subject_counts[subject_key][1] += 1
            item_counts[item_key][0] += y
            item_counts[item_key][1] += 1
            benchmark_counts[benchmark][0] += y
            benchmark_counts[benchmark][1] += 1
            item_text.setdefault(item_key, str(row.get("item_content", "") or ""))
            item_benchmark.setdefault(item_key, benchmark)
            item_condition.setdefault(item_key, condition)
            global_k += y
            global_n += 1

        if global_n == 0:
            raise ValueError(
                "RaschLite.fit requires at least one row with response in {0.0, 1.0}"
            )

        self._n_fit_rows = global_n
        self._global_p = _smoothed_rate(global_k, global_n, 0.5, 2.0)
        self._global_logit = float(safe_logit(self._global_p, eps=self.eps))

        self._subject_theta = {
            key: float(safe_logit(_smoothed_rate(k, n, self._global_p, self.subject_kappa), eps=self.eps))
            for key, (k, n) in subject_counts.items()
        }
        self._benchmark_beta = {
            key: self._difficulty_beta(k, n, self.benchmark_kappa)
            for key, (k, n) in benchmark_counts.items()
        }
        self._item_beta = {
            key: self._difficulty_beta(k, n, self.item_kappa)
            for key, (k, n) in item_counts.items()
        }
        self._text_weights, self._n_text_items = self._fit_text_regressor(
            item_beta=self._item_beta,
            item_text=item_text,
            item_benchmark=item_benchmark,
            item_condition=item_condition,
        )
        self._fitted = True

    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return probabilities aligned with ``rows``."""
        return np.asarray([self.predict_one(dict(row)) for row in rows], dtype=float)

    def save(self, path: str | Path) -> None:
        """Persist a JSON artifact; no pickles or executable code."""
        payload = {
            "config": {
                "subject_kappa": self.subject_kappa,
                "item_kappa": self.item_kappa,
                "benchmark_kappa": self.benchmark_kappa,
                "text_dim": self.text_dim,
                "ridge_lambda": self.ridge_lambda,
                "max_text_tokens": self.max_text_tokens,
                "unseen_text_weight": self.unseen_text_weight,
                "eps": self.eps,
                "use_seen_item_lookup": self.use_seen_item_lookup,
            },
            "fitted": self._fitted,
            "global_p": self._global_p,
            "global_logit": self._global_logit,
            "subject_theta": self._subject_theta,
            "item_beta": self._item_beta,
            "benchmark_beta": self._benchmark_beta,
            "text_weights": self._text_weights.tolist(),
            "n_fit_rows": self._n_fit_rows,
            "n_text_items": self._n_text_items,
        }
        Path(path).write_text(json.dumps(payload, separators=(",", ":")))

    @classmethod
    def load(cls, path: str | Path) -> "RaschLite":
        """Restore a JSON artifact written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        model = cls(**payload["config"])
        model._fitted = bool(payload.get("fitted", False))
        model._global_p = float(payload.get("global_p", 0.5))
        model._global_logit = float(payload.get("global_logit", 0.0))
        model._subject_theta = {
            str(k): float(v) for k, v in payload.get("subject_theta", {}).items()
        }
        model._item_beta = {
            str(k): float(v) for k, v in payload.get("item_beta", {}).items()
        }
        model._benchmark_beta = {
            str(k): float(v) for k, v in payload.get("benchmark_beta", {}).items()
        }
        model._text_weights = np.asarray(payload.get("text_weights", []), dtype=float)
        if model._text_weights.shape != (model.text_dim,):
            model._text_weights = np.zeros(model.text_dim, dtype=float)
        model._n_fit_rows = int(payload.get("n_fit_rows", 0))
        model._n_text_items = int(payload.get("n_text_items", 0))
        return model

    # ------------------------------------------------------------------
    # RuntimePredictor protocol
    # ------------------------------------------------------------------

    def predict_one(
        self,
        input: dict,  # noqa: A002 - kit contract intentionally shadows builtin
        labeled: list[dict] | None = None,
    ) -> float:
        """Predict one probability with the Rasch-lite equation."""
        del labeled
        if not self._fitted:
            return 0.5

        theta = self._subject_theta.get(_subject_key(input), self._global_logit)
        beta = self._difficulty_for_row(input)
        p = sigmoid(theta - beta)
        return float(clip_probability(float(p), eps=self.eps))

    def metadata(self) -> dict[str, Any]:
        """JSON-serializable provenance for evaluator reports."""
        return {
            "class": "RaschLite",
            "fitted": bool(self._fitted),
            "n_fit_rows": int(self._n_fit_rows),
            "n_subjects": len(self._subject_theta),
            "n_items": len(self._item_beta),
            "n_benchmarks": len(self._benchmark_beta),
            "n_text_items": int(self._n_text_items),
            "global_p": float(self._global_p),
            "global_logit": float(self._global_logit),
            "text_dim": int(self.text_dim),
            "ridge_lambda": float(self.ridge_lambda),
            "unseen_text_weight": float(self.unseen_text_weight),
            "use_seen_item_lookup": bool(self.use_seen_item_lookup),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _difficulty_beta(self, k: int, n: int, kappa: float) -> float:
        p = _smoothed_rate(k, n, self._global_p, kappa)
        return float(self._global_logit - float(safe_logit(p, eps=self.eps)))

    def _difficulty_for_row(self, row: Mapping[str, Any]) -> float:
        item_key = _item_key(row)
        if self.use_seen_item_lookup and item_key in self._item_beta:
            return self._item_beta[item_key]

        benchmark = str(row.get("benchmark", ""))
        benchmark_beta = float(self._benchmark_beta.get(benchmark, 0.0))
        text_beta = self._predict_text_beta(row)
        w = self.unseen_text_weight
        return float(w * text_beta + (1.0 - w) * benchmark_beta)

    def _predict_text_beta(self, row: Mapping[str, Any]) -> float:
        features = _hashed_features(
            text=str(row.get("item_content", "") or ""),
            benchmark=str(row.get("benchmark", "")),
            condition=str(row.get("condition", "none")),
            dim=self.text_dim,
            max_tokens=self.max_text_tokens,
        )
        return float(sum(self._text_weights[idx] * value for idx, value in features.items()))

    def _fit_text_regressor(
        self,
        *,
        item_beta: Mapping[str, float],
        item_text: Mapping[str, str],
        item_benchmark: Mapping[str, str],
        item_condition: Mapping[str, str],
    ) -> tuple[np.ndarray, int]:
        xtx = np.zeros((self.text_dim, self.text_dim), dtype=float)
        xty = np.zeros(self.text_dim, dtype=float)
        n_items = 0

        for item_key, beta in item_beta.items():
            text = item_text.get(item_key, "")
            features = _hashed_features(
                text=text,
                benchmark=item_benchmark.get(item_key, ""),
                condition=item_condition.get(item_key, "none"),
                dim=self.text_dim,
                max_tokens=self.max_text_tokens,
            )
            items = list(features.items())
            if not items:
                continue
            for i, xi in items:
                xty[i] += xi * beta
                for j, xj in items:
                    xtx[i, j] += xi * xj
            n_items += 1

        if n_items == 0:
            return np.zeros(self.text_dim, dtype=float), 0

        reg = max(self.ridge_lambda, 0.0)
        xtx += np.eye(self.text_dim, dtype=float) * reg
        # Keep the bias feature lightly regularized; it carries the mean
        # difficulty when text is empty or fully unseen.
        xtx[0, 0] -= reg
        xtx[0, 0] += min(reg, 1e-6)
        try:
            weights = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(xtx) @ xty
        return np.asarray(weights, dtype=float), n_items


def _coerce_binary_response(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if not isinstance(value, (int, float)):
        return None
    fv = float(value)
    if math.isnan(fv) or fv not in (0.0, 1.0):
        return None
    return int(fv)


def _smoothed_rate(k: int, n: int, parent_p: float, kappa: float) -> float:
    return float((k + kappa * parent_p) / (n + kappa))


def _subject_key(row: Mapping[str, Any]) -> str:
    subject_content = str(row.get("subject_content", "") or "")
    name = parse_subject_name(subject_content)
    if name:
        return name
    subject_id = str(row.get("subject_id", "") or "").strip()
    if subject_id:
        return subject_id
    truncated = subject_content[:512].encode("utf-8", errors="replace")
    return hashlib.blake2b(truncated, digest_size=_SUBJECT_HASH_BYTES).hexdigest()


def _item_key(row: Mapping[str, Any]) -> str:
    item_id = str(row.get("item_id", "") or "").strip()
    if item_id:
        return item_id
    content = str(row.get("item_content", "") or "")
    truncated = content[:2048].encode("utf-8", errors="replace")
    return hashlib.blake2b(truncated, digest_size=_ITEM_HASH_BYTES).hexdigest()


def _hashed_features(
    *,
    text: str,
    benchmark: str,
    condition: str,
    dim: int,
    max_tokens: int,
) -> dict[int, float]:
    """Feature hashing with an explicit bias at index 0."""
    feats: dict[int, float] = {0: 1.0}
    tokens = [f"bench:{benchmark}", f"cond:{condition}"]
    tokens.extend(_TOKEN_RE.findall(text.lower())[:max_tokens])
    if not tokens:
        return feats
    scale = 1.0 / math.sqrt(len(tokens))
    for token in tokens:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(h, "big", signed=False)
        idx = 1 + (raw % (dim - 1))
        sign = 1.0 if ((raw >> 63) & 1) == 0 else -1.0
        feats[idx] = feats.get(idx, 0.0) + sign * scale
    return feats
