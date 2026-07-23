"""XGBoost SIRET Matcher public API.

The package deliberately avoids eager imports.  In particular, importing
``features`` pulls in Pandas/PyArrow, whose OpenMP runtime must never leak
into the isolated Torch semantic worker on macOS.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_FEATURE_EXPORTS = {
    "normalize_text",
    "normalize_name",
    "jaro_sim",
    "levenshtein_norm",
    "build_address",
    "extract_street_number",
    "postal_match",
    "city_match",
    "street_number_diff",
    "token_overlap",
    "first_word_match",
    "contains_check",
    "acronym_match",
    "numeric_token_match",
    "make_features",
    "FEATURE_NAMES",
    "semantic_gate_allows",
}
_NAMING_EXPORTS = {
    "build_candidate_names",
    "primary_name",
    "NameSource",
    "CandidateName",
}

__all__ = sorted(_FEATURE_EXPORTS | _NAMING_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _FEATURE_EXPORTS:
        return getattr(import_module(".features", __name__), name)
    if name in _NAMING_EXPORTS:
        return getattr(import_module(".naming", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
