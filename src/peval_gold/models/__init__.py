"""Model abstractions for the gold-track laboratory.

Exports the offline-trainable :class:`Predictor` and the streaming
:class:`RuntimePredictor` Protocols (), plus the Batch-3
:class:`CurrentNCF` wrapper for the shipped the wrapped NCF artifact.

``CurrentNCF`` is intentionally lazy-imported by string name in the
concrete-class allowlist below — the import touches torch and the
sentence-transformers encoder, which take a few seconds. Modules that
only need the Protocols can ``from peval_gold.models.base import
Predictor`` without paying that cost.
"""

from peval_gold.models.base import Predictor, RuntimePredictor

__all__ = ["CurrentNCF", "Predictor", "RuntimePredictor", "TemplateNCF"]


def __getattr__(name):  # type: ignore[no-untyped-def]
    """Lazy import for the heavy ``CurrentNCF`` / ``TemplateNCF`` wrappers.

    See module docstring for the rationale (avoid torch + encoder
    load at package import time for callers that only want the
    Protocols).
    """
    if name == "CurrentNCF":
        from peval_gold.models.current_ncf import CurrentNCF as _CurrentNCF

        return _CurrentNCF
    if name == "TemplateNCF":
        from peval_gold.models.template_ncf import TemplateNCF as _TemplateNCF

        return _TemplateNCF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
