"""XGBoost SIRET Matcher - Shared feature engineering module."""

from .features import (
    # Text normalization
    normalize_text,
    normalize_name,
    # Similarity functions
    jaro_sim,
    levenshtein_norm,
    # Address building
    build_address,
    extract_street_number,
    # Matching functions
    postal_match,
    city_match,
    street_number_diff,
    # Token-based features
    token_overlap,
    first_word_match,
    contains_check,
    acronym_match,
    # Feature computation
    make_features,
    FEATURE_NAMES,
)

from .naming import build_candidate_names, primary_name, NameSource, CandidateName

__all__ = [
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
    "make_features",
    "FEATURE_NAMES",
    "build_candidate_names",
    "primary_name",
    "NameSource",
    "CandidateName",
]
