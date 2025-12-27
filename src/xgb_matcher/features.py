"""
Shared feature engineering functions for XGBoost SIRET matcher.

This module is used by both training and inference scripts to ensure
consistency (no train/serve skew).

Features computed:
  - name_jaro: Jaro-Winkler similarity between normalized names
  - name_levenshtein: Normalized Levenshtein similarity between names
  - addr_jaro: Jaro-Winkler similarity between normalized addresses
  - addr_levenshtein: Normalized Levenshtein similarity between addresses
  - postcode_match: 1 if postcodes match exactly, 0 otherwise
  - city_match: 1 if normalized city names match, 0 otherwise
  - street_number_diff: Absolute difference between street numbers (9999 if unavailable)
  - name_token_overlap: Ratio of common words between names
  - name_first_word_match: 1 if first words match
  - name_contains_crm: 1 if candidate name contains CRM name
  - name_crm_contains_cand: 1 if CRM name contains candidate name
  - acronym_match: 1 if acronym matches expanded form
  - idf_name: Average IDF of overlapping name tokens
  - numeric_token_match: Numeric token overlap between names
  - addr_token_overlap: Ratio of common words in addresses
  - address_density: Number of candidates sharing the same address (per location)
  - street_name_jaro: Jaro similarity on street name only (no number)
  - legal_form_category: Encoded legal form category (PUBLIC/PRIVE/INCONNU)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, Set

import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

try:
    from src.pipe_v6.category_mapping import map_legal_to_category
except ImportError:  # Fallback when PYTHONPATH=src
    from pipe_v6.category_mapping import map_legal_to_category

from .naming import (
    CandidateName,
    NameSource,
    build_candidate_names,
    looks_like_person_name,
    normalize_name,
    normalize_text,
)
from .semantic import max_semantic_similarity


# Thresholds for city-like detection
CITY_OVERLAP_THRESHOLD = 0.7
CITY_MAX_TOKENS = 3
SEMANTIC_GATE_JARO = float(os.getenv("XGB_SEMANTIC_GATE_JARO", "0.90"))

# Stopwords to down-weight generic name overlap (avoid LES/DU false positives)
NAME_STOPWORDS: Set[str] = {
    "LES", "DU", "DE", "LA", "LE", "DES", "D", "L", "AUX", "AU",
}

# Global IDF map for name tokens (set by candidate loader)
_GLOBAL_NAME_IDF: Dict[str, float] = {}
_GLOBAL_NAME_IDF_DEFAULT: float = 0.0


def set_global_name_idf_map(idf_map: Mapping[str, float] | None, default_idf: float | None = None) -> None:
    """Register a global IDF map for name tokens (used by idf_name feature)."""
    global _GLOBAL_NAME_IDF, _GLOBAL_NAME_IDF_DEFAULT
    _GLOBAL_NAME_IDF = dict(idf_map) if idf_map else {}
    if default_idf is None:
        if _GLOBAL_NAME_IDF:
            _GLOBAL_NAME_IDF_DEFAULT = max(_GLOBAL_NAME_IDF.values())
        else:
            _GLOBAL_NAME_IDF_DEFAULT = 0.0
    else:
        _GLOBAL_NAME_IDF_DEFAULT = float(default_idf)


# Feature names in order (used for training and inference)
FEATURE_NAMES: List[str] = [
    # Aggregated name similarities (bag of names)
    "has_any_name",
    "name_count",
    "name_jaro_max",
    "name_jaro_second",
    "name_jaro_gap",
    "name_levenshtein_max",
    "name_token_overlap_max",
    "idf_name",
    "numeric_token_match",
    "name_first_word_match_max",
    "name_contains_crm_max",
    "name_crm_contains_cand_max",
    "acronym_match_max",
    "name_sim_max_etab",
    "name_sim_max_ul",
    "name_sim_max_sigle",
    "name_sim_max_pm_dirigeant",
    "type_of_max_name",
    "is_ul_name_max",
    "is_sigle_max",
    "name_length_max",
    "has_person_name",
    "person_name_jaro_max",
    "name_city_overlap_max",
    "name_is_city_like_max",
    "name_semantic_max",
    "name_semantic_second",
    "name_semantic_gap",
    # Address / location features (unchanged)
    "addr_jaro",
    "addr_levenshtein",
    "postcode_match",
    "city_match",
    "street_number_diff",
    "addr_token_overlap",
    "address_density",
    "street_name_jaro",
    "name_addr_consistency",
    "legal_form_category",
    # New features from reranking rules (Phase 3)
    "is_siege",           # D9: head office indicator
    "is_association",     # B1: association candidate indicator
    "alias_match",        # D6: alias in parentheses match
    "token_overlap_ul",   # D11: token overlap with UL names
    "ul_vs_pm_indicator", # D7: UL vs PM dirigeant source indicator
    "is_crm_school",      # Context: CRM query refers to a school
]


# Association tokens for B1 rule
ASSOCIATION_TOKENS: Set[str] = {
    "ASSOCIATION", "ASSO", "FOYER", "AMICALE", "CONSEIL",
    "PARENTS", "PARENT", "ELEVES", "SPORTIVE", "FSE", "SOCIO",
}

# School tokens for context detection (D11 and B1)
SCHOOL_TOKENS: Set[str] = {
    "COLLEGE", "CLG", "LYCEE", "LYCE", "ECOLE", "INSTITUT", "UNIVERSITE", 
    "UFR", "FACULTE", "CTRE", "CENTRE", "FORMATION", "CFA", "MFR", "ACADEMY",
}


# --------------------------------------------------------------------------- #
# Similarity functions
# --------------------------------------------------------------------------- #


def jaro_sim(a: str, b: str) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.

    Returns 1.0 if both empty, 0.0 if only one is empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return JaroWinkler.similarity(a, b)


def levenshtein_norm(a: str, b: str) -> float:
    """
    Compute normalized Levenshtein similarity (1 - distance/max_len).

    Returns 1.0 if both empty, 0.0 if only one is empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


# --------------------------------------------------------------------------- #
# CRM name cleanup
# --------------------------------------------------------------------------- #


def _remove_tokens(text: str, tokens_to_remove: Set[str]) -> str:
    if not text:
        return ""
    tokens = text.split()
    filtered = [t for t in tokens if t not in tokens_to_remove]
    return " ".join(filtered)


def strip_location_from_crm_name(crm_row: pd.Series, preserve_case: bool = False) -> str:
    """Remove city/CP/INSEE tokens from the CRM name to avoid city bias.

    Keeps the normalized version; if everything is removed, fall back to the original
    normalized name to avoid empty strings.
    """

    original_norm = normalize_name(crm_row.get("crm_name", ""), uppercase=not preserve_case)
    if not original_norm:
        return ""

    city = normalize_text(crm_row.get("crm_city_addr") or crm_row.get("crm_city"))
    postcode = normalize_text(crm_row.get("postcode"))
    insee = normalize_text(crm_row.get("insee"))

    tokens_to_remove: Set[str] = set()
    if city:
        tokens_to_remove.update(city.split())
    if postcode:
        tokens_to_remove.add(postcode)
    if insee:
        tokens_to_remove.add(insee)

    # Use a case-insensitive match for removal if preserving case
    if preserve_case:
        tokens = original_norm.split()
        filtered = [t for t in tokens if t.upper() not in tokens_to_remove]
        cleaned = " ".join(filtered)
    else:
        cleaned = _remove_tokens(original_norm, tokens_to_remove)
        
    return cleaned or original_norm


# --------------------------------------------------------------------------- #
# Address building
# --------------------------------------------------------------------------- #


def build_address(cand: dict) -> str:
    """
    Build normalized address from SIRENE candidate fields.

    Uses: numeroVoie + typeVoie + libelleVoie
    Falls back to complementAdresse if main fields are empty.

    Args:
        cand: Dictionary with SIRENE candidate data

    Returns:
        Normalized address string
    """
    if isinstance(cand, dict):
        cached = cand.get("_xgb_cached_addr_norm")
        if isinstance(cached, str):
            return cached

    parts = [
        cand.get("numeroVoie"),
        cand.get("typeVoie"),
        cand.get("libelleVoie"),
    ]
    addr = " ".join(str(x) for x in parts if x and str(x).strip())
    if not addr and cand.get("complementAdresse"):
        addr = str(cand["complementAdresse"])
    addr_norm = normalize_text(addr)
    if isinstance(cand, dict):
        cand["_xgb_cached_addr_norm"] = addr_norm
    return addr_norm


def extract_street_number(addr: str | None) -> str | None:
    """
    Extract leading street number from address string.

    Uses regex to handle variations like "12", "12B", "12 BIS", etc.

    Args:
        addr: Address string

    Returns:
        Street number as string, or None if not found
    """
    if not addr:
        return None
    addr_str = str(addr).strip()
    if not addr_str:
        return None
    # Match leading digits (optionally followed by letter like "12B")
    match = re.match(r"^(\d+)", addr_str)
    if match:
        return match.group(1)
    return None


def get_street_name(cand: dict) -> str:
    """
    Get only the street name part (typeVoie + libelleVoie), excluding number.

    Used for street_name_jaro feature to compare street names independently.
    """
    if isinstance(cand, dict):
        cached = cand.get("_xgb_cached_street_name_norm")
        if isinstance(cached, str):
            return cached
    parts = [
        cand.get("typeVoie"),
        cand.get("libelleVoie"),
    ]
    norm = normalize_text(" ".join(str(x) for x in parts if x and str(x).strip()))
    if isinstance(cand, dict):
        cand["_xgb_cached_street_name_norm"] = norm
    return norm


def _candidate_city_norm(cand: dict) -> str:
    if isinstance(cand, dict):
        cached = cand.get("_xgb_cached_city_norm")
        if isinstance(cached, str):
            return cached
    norm = normalize_text(cand.get("city"))
    if isinstance(cand, dict):
        cand["_xgb_cached_city_norm"] = norm
    return norm


def extract_street_name_from_address(addr: str | None) -> str:
    """
    Extract street name from full address string (remove leading number).

    Args:
        addr: Full address string

    Returns:
        Street name without the leading number
    """
    if not addr:
        return ""
    addr_str = normalize_text(addr)
    # Remove leading number and any bis/ter suffix
    cleaned = re.sub(r"^\d+\s*(BIS|TER|QUATER|B|T|Q)?\s*", "", addr_str, flags=re.IGNORECASE)
    return cleaned.strip()


# --------------------------------------------------------------------------- #
# Matching functions
# --------------------------------------------------------------------------- #


def postal_match(cp1: str | None, cp2: str | None) -> int:
    """Check if two postal codes match exactly."""
    if not cp1 or not cp2:
        return 0
    return int(str(cp1).strip() == str(cp2).strip())


def city_match(c1: str | None, c2: str | None) -> int:
    """Check if two city names match after normalization."""
    if not c1 or not c2:
        return 0
    return int(normalize_text(c1) == normalize_text(c2))


def street_number_diff(n1: str | None, n2: str | None) -> float:
    """
    Compute absolute difference between street numbers.

    Returns 9999.0 if either number is unavailable or invalid.
    """
    try:
        num1 = int(n1) if n1 else None
        num2 = int(n2) if n2 else None
        if num1 is not None and num2 is not None:
            return float(abs(num1 - num2))
        return 9999.0
    except (TypeError, ValueError):
        return 9999.0


def _address_density_for_candidate(crm: Mapping[str, Any], cand: dict) -> float:
    """Return address density for the candidate, preferring INSEE-based grouping."""
    insee = crm.get("insee")
    if insee:
        density = cand.get("_xgb_addr_density_insee")
        if density is not None:
            return float(density)
    density = cand.get("_xgb_addr_density_cp")
    if density is not None:
        return float(density)
    density = cand.get("address_density")
    if density is not None:
        return float(density)
    return 1.0


# --------------------------------------------------------------------------- #
# Token-based features (new for improved ranking)
# --------------------------------------------------------------------------- #


def _tokenize(
    text: str,
    *,
    stopwords: Set[str] | None = None,
    min_len: int = 1,
) -> Set[str]:
    """Split normalized text into set of tokens (words), with optional filtering."""
    if not text:
        return set()
    tokens = text.split()
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    if min_len > 1:
        tokens = [t for t in tokens if len(t) >= min_len]
    return set(tokens)


def token_overlap(
    a: str,
    b: str,
    *,
    stopwords: Set[str] | None = None,
    min_len: int = 1,
) -> float:
    """
    Compute ratio of common tokens between two strings.

    Returns |A ∩ B| / |A ∪ B| (Jaccard similarity on tokens).
    """
    tokens_a = _tokenize(a, stopwords=stopwords, min_len=min_len)
    tokens_b = _tokenize(b, stopwords=stopwords, min_len=min_len)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def first_word_match(a: str, b: str) -> int:
    """
    Check if the first word of both strings matches.

    Useful for "SARL X" vs "X SARL" patterns.
    """
    if not a or not b:
        return 0
    words_a = a.split()
    words_b = b.split()
    if not words_a or not words_b:
        return 0
    return int(words_a[0] == words_b[0])


def contains_check(container: str, contained: str) -> int:
    """
    Check if container string contains the contained string.

    Useful for detecting "GE FRUITS" in "GE FRUITS DISTRIBUTION".
    """
    if not container or not contained:
        return 0
    return int(contained in container)


def _idf_overlap(a: str, b: str) -> float:
    """Average IDF of overlapping name tokens (0 if no overlap or no IDF map)."""
    if not _GLOBAL_NAME_IDF:
        return 0.0
    tokens_a = _tokenize(a, stopwords=NAME_STOPWORDS, min_len=2)
    tokens_b = _tokenize(b, stopwords=NAME_STOPWORDS, min_len=2)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    if not overlap:
        return 0.0
    return sum(_GLOBAL_NAME_IDF.get(tok, _GLOBAL_NAME_IDF_DEFAULT) for tok in overlap) / len(overlap)


def numeric_token_match(a: str, b: str) -> float:
    """Overlap ratio of numeric tokens between two strings."""
    nums_a = set(re.findall(r"\d+", a or ""))
    nums_b = set(re.findall(r"\d+", b or ""))
    if not nums_a or not nums_b:
        return 0.0
    return len(nums_a & nums_b) / max(len(nums_a), len(nums_b))


def _extract_acronym(text: str) -> str:
    """Extract initials from a multi-word string, ignoring small words and common city names."""
    if not text:
        return ""
    
    # Common French stopwords and specific geography to ignore in acronyms
    STOPWORDS = {"DE", "DU", "LA", "LE", "DES", "ET", "LES", "D", "L", "AUX", "AU"}
    GEOGRAPHY = {"LYON", "CORBAS", "DECINES", "MEYZIEU", "VILLEURBANNE"}
    
    words = text.split()
    if len(words) <= 1:
        return ""
        
    initials = []
    for w in words:
        w_norm = w.strip().upper()
        if not w_norm:
            continue
        if w_norm in STOPWORDS or w_norm in GEOGRAPHY:
            continue
        initials.append(w_norm[0])
        
    return "".join(initials)


def _is_acronym(text: str) -> bool:
    """Check if text looks like an acronym (2-6 uppercase letters, no spaces)."""
    if not text:
        return False
    # After normalization, acronyms are uppercase words with 2-6 chars
    return bool(re.match(r"^[A-Z]{2,6}$", text.replace(" ", "")))


def acronym_match(a: str, b: str) -> int:
    """
    Check if one string is the acronym of the other.

    Handles cases like "SNCF" ↔ "SOCIETE NATIONALE DES CHEMINS DE FER".
    """
    if not a or not b:
        return 0

    # Determine which might be the acronym
    a_is_acronym = _is_acronym(a.replace(" ", ""))
    b_is_acronym = _is_acronym(b.replace(" ", ""))

    if a_is_acronym and not b_is_acronym:
        query_acro = a.replace(" ", "").upper()
        cand_acro = _extract_acronym(b)
        
        # 1. Exact match
        if query_acro == cand_acro:
            return 1
        # 2. Prefix match (e.g. ASL matches Association Syndicale Libre ...)
        if len(query_acro) >= 2 and cand_acro.startswith(query_acro):
            return 1
        return 0
    elif b_is_acronym and not a_is_acronym:
        query_acro = b.replace(" ", "").upper()
        cand_acro = _extract_acronym(a)
        if query_acro == cand_acro or (len(query_acro) >= 2 and cand_acro.startswith(query_acro)):
            return 1

    return 0


# --------------------------------------------------------------------------- #
# Aggregation helpers for bag-of-names strategy
# --------------------------------------------------------------------------- #


_SOURCE_ENCODING = {
    NameSource.NONE: 0,
    NameSource.ETAB_ENSEIGNE: 1,
    NameSource.ETAB_DENOM: 2,
    NameSource.ETAB_ENSEIGNE_X: 3,
    NameSource.PM_DIRIGEANT: 4,
    NameSource.UL_SIGLE: 5,
    NameSource.UL_DENOM_USUELLE: 6,
    NameSource.UL_DENOM: 7,
    NameSource.PERSON_NAME: 8,
}


def _encode_source(source: NameSource | str) -> int:
    """Map NameSource to an ordinal value for model consumption."""
    return _SOURCE_ENCODING.get(source, 0)


def _init_name_feature_defaults() -> Dict[str, float]:
    """Default values when no candidate names are available."""
    return {
        "has_any_name": 0.0,
        "name_count": 0.0,
        "name_jaro_max": 0.0,
        "name_jaro_second": 0.0,
        "name_jaro_gap": 0.0,
        "name_levenshtein_max": 0.0,
        "name_token_overlap_max": 0.0,
        "idf_name": 0.0,
        "numeric_token_match": 0.0,
        "name_first_word_match_max": 0.0,
        "name_contains_crm_max": 0.0,
        "name_crm_contains_cand_max": 0.0,
        "acronym_match_max": 0.0,
        "name_sim_max_etab": 0.0,
        "name_sim_max_ul": 0.0,
        "name_sim_max_sigle": 0.0,
        "name_sim_max_pm_dirigeant": 0.0,
        "type_of_max_name": 0.0,
        "is_ul_name_max": 0.0,
        "is_sigle_max": 0.0,
        "name_length_max": 0.0,
        "has_person_name": 0.0,
        "person_name_jaro_max": 0.0,
        "name_city_overlap_max": 0.0,
        "name_is_city_like_max": 0.0,
        "name_semantic_max": 0.0,
        "name_semantic_second": 0.0,
        "name_semantic_gap": 0.0,
    }


# --------------------------------------------------------------------------- #
# Main feature computation
# --------------------------------------------------------------------------- #

def preprocess_crm_row(crm_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Precompute CRM-side normalizations once per row (for faster inference)."""
    crm_address_raw = crm_row.get("crm_address", "")
    crm_name = strip_location_from_crm_name(crm_row)
    crm_city = crm_row.get("crm_city_addr") or crm_row.get("crm_city", "")

    crm_tokens_all = set(crm_name.split())
    is_crm_school = float(bool(crm_tokens_all & SCHOOL_TOKENS))

    # D6: alias extraction (normalized, filtered)
    crm_raw = crm_row.get("crm_name", "")
    aliases = [normalize_text(m) for m in re.findall(r"\(([^)]+)\)", str(crm_raw))]
    blacklist = {"SIEGE", "BAT", "BATIMENT", "ETAGE", "LOCAL", "AGENCE", "RDC"}
    aliases = [a for a in aliases if len(a) >= 3 and a not in blacklist and not a.isdigit()]

    # D11: CRM token set (normalized + stopword filtered)
    crm_tokens = {tok for tok in crm_name.split() if len(tok) >= 3 and tok not in NAME_STOPWORDS}
    crm_tokens -= {"GROUPE", "SOCIETE", "HOLDING"}

    return {
        "crm_name": crm_name,
        "crm_name_semantic": strip_location_from_crm_name(crm_row, preserve_case=True),
        "crm_addr": normalize_text(crm_address_raw),
        "crm_city": crm_city,
        "crm_city_norm": normalize_text(crm_city),
        "crm_street_num": extract_street_number(crm_address_raw),
        "crm_street_name": extract_street_name_from_address(crm_address_raw),
        "postcode": crm_row.get("postcode"),
        "insee": crm_row.get("insee"),
        "aliases": aliases,
        "crm_tokens": crm_tokens,
        "is_crm_school": is_crm_school,
    }


def build_semantic_name_pool(
    candidate_names: List[CandidateName],
    *,
    crm_city_norm: str,
    cand_city_norm: str,
) -> List[str]:
    """Build the semantic candidate-name pool with the same rules as make_features()."""
    if not candidate_names:
        return []

    pool: List[str] = []
    for nm in candidate_names:
        city_overlap = max(
            token_overlap(nm.text, cand_city_norm, stopwords=NAME_STOPWORDS, min_len=2),
            token_overlap(nm.text, crm_city_norm, stopwords=NAME_STOPWORDS, min_len=2),
        )
        is_city_like = city_overlap > CITY_OVERLAP_THRESHOLD and len(nm.text.split()) <= CITY_MAX_TOKENS
        is_person_nm = nm.source == NameSource.PERSON_NAME or looks_like_person_name(nm.text)
        if not is_city_like and not is_person_nm:
            pool.append(nm.text)

    if pool:
        return pool
    return [nm.text for nm in candidate_names]


def make_features_from_preprocessed(
    crm: Mapping[str, Any],
    cand: dict,
    skip_semantic: bool = False,
) -> Dict[str, float]:
    """
    Compute all similarity features between a CRM row and a SIRENE candidate.

    This function is used by both training and inference to ensure consistency.

    Args:
        crm: Preprocessed CRM dict (see preprocess_crm_row)
        cand: Dictionary with SIRENE candidate data
        skip_semantic: If True, skip expensive semantic features (for stage-1 ranking)

    Returns:
        Dictionary with all feature values
    """
    # Normalize inputs
    crm_name = crm.get("crm_name", "")
    crm_addr = crm.get("crm_addr", "")
    cand_addr = build_address(cand)

    # Bag of names for the candidate
    candidate_names: List[CandidateName] = build_candidate_names(cand)

    # Street components
    crm_street_num = crm.get("crm_street_num")
    cand_street_num = cand.get("numeroVoie")
    if cand_street_num:
        cand_street_num = str(cand_street_num)

    crm_street_name = crm.get("crm_street_name", "")
    cand_street_name = get_street_name(cand)

    crm_city = crm.get("crm_city", "")
    crm_city_norm = crm.get("crm_city_norm", "")
    cand_city_norm = _candidate_city_norm(cand)
    is_crm_school = float(crm.get("is_crm_school", 0.0))

    # ---------------- Name similarities (aggregated) -----------------
    name_features = _init_name_feature_defaults()

    if candidate_names:
        sims = []
        for nm in candidate_names:
            sim_jaro = jaro_sim(crm_name, nm.text)
            sim_lev = levenshtein_norm(crm_name, nm.text)
            tok = token_overlap(crm_name, nm.text, stopwords=NAME_STOPWORDS, min_len=2)
            fw = first_word_match(crm_name, nm.text)
            contains_crm = contains_check(nm.text, crm_name)
            crm_contains = contains_check(crm_name, nm.text)
            acr = acronym_match(crm_name, nm.text)
            city_overlap = max(
                token_overlap(nm.text, cand_city_norm, stopwords=NAME_STOPWORDS, min_len=2),
                token_overlap(nm.text, crm_city_norm, stopwords=NAME_STOPWORDS, min_len=2),
            )
            is_city_like = city_overlap > CITY_OVERLAP_THRESHOLD and len(nm.text.split()) <= CITY_MAX_TOKENS
            is_person_nm = nm.source == NameSource.PERSON_NAME or looks_like_person_name(nm.text)

            sims.append(
                {
                    "nm": nm,
                    "jaro": sim_jaro,
                    "lev": sim_lev,
                    "tok": tok,
                "fw": float(fw),
                "contains_crm": float(contains_crm),
                "crm_contains": float(crm_contains),
                "acronym": float(acr),
                "city_overlap": city_overlap,
                "is_city_like": is_city_like,
                "is_person_nm": is_person_nm,
            }
            )

        # Global maxima
        name_features["has_any_name"] = 1.0
        name_features["name_count"] = float(len(candidate_names))

        sims_sorted = sorted(sims, key=lambda x: x["jaro"], reverse=True)

        # Option hard: prefer non city-like names for the maxima when available
        non_city_sims = [s for s in sims_sorted if not s["is_city_like"]]
        sims_for_max = non_city_sims or sims_sorted

        top = sims_for_max[0]
        second = sims_for_max[1]["jaro"] if len(sims_for_max) > 1 else 0.0

        idf_name = _idf_overlap(crm_name, top["nm"].text)
        numeric_match = numeric_token_match(crm_name, top["nm"].text)

        name_features.update(
            {
                "name_jaro_max": top["jaro"],
                "name_jaro_second": second,
                "name_jaro_gap": top["jaro"] - second,
                "name_levenshtein_max": top["lev"],
                "name_token_overlap_max": max(s["tok"] for s in sims),
                "idf_name": idf_name,
                "numeric_token_match": numeric_match,
                "name_first_word_match_max": max(s["fw"] for s in sims),
                "name_contains_crm_max": max(s["contains_crm"] for s in sims),
                "name_crm_contains_cand_max": max(s["crm_contains"] for s in sims),
                "acronym_match_max": max(s["acronym"] for s in sims),
                "type_of_max_name": float(_encode_source(top["nm"].source)),
                "is_ul_name_max": float(top["nm"].is_ul_name),
                "is_sigle_max": float(top["nm"].is_sigle),
                "name_length_max": float(len(top["nm"].text)),
                "has_person_name": float(any(s["is_person_nm"] for s in sims)),
                "person_name_jaro_max": max([s["jaro"] for s in sims if s["is_person_nm"]] or [0.0]),
                "name_city_overlap_max": max(s["city_overlap"] for s in sims),
                "name_is_city_like_max": float(max(s["is_city_like"] for s in sims)),
            }
        )

        # Source-specific maxima (based on Jaro)
        name_features["name_sim_max_etab"] = max(
            [s["jaro"] for s in sims if not s["nm"].is_ul_name] or [0.0]
        )
        name_features["name_sim_max_ul"] = max(
            [s["jaro"] for s in sims if s["nm"].is_ul_name] or [0.0]
        )
        name_features["name_sim_max_sigle"] = max(
            [s["jaro"] for s in sims if s["nm"].is_sigle] or [0.0]
        )
        name_features["name_sim_max_pm_dirigeant"] = max(
            [s["jaro"] for s in sims if s["nm"].source == NameSource.PM_DIRIGEANT] or [0.0]
        )

        semantic_pool = [
            s["nm"].text
            for s in sims
            if not s["is_city_like"] and not s["is_person_nm"]
        ]
        if not semantic_pool:
            semantic_pool = [s["nm"].text for s in sims]
        
        # Use original case version for semantic embedding (CamelCase split needs it)
        crm_name_semantic = crm.get("crm_name_semantic", "")
        
        # Calculate semantic scores using batch GPU encoding (skip if requested)
        if not skip_semantic:
            from .semantic import top2_semantic_similarities
            sem_max, sem_second, sem_gap = top2_semantic_similarities(crm_name_semantic, semantic_pool)
            name_features["name_semantic_max"] = sem_max
            name_features["name_semantic_second"] = sem_second
            name_features["name_semantic_gap"] = sem_gap

    # ---------------- Address & location features -----------------
    addr_features = {
        "addr_jaro": jaro_sim(crm_addr, cand_addr),
        "addr_levenshtein": levenshtein_norm(crm_addr, cand_addr),
        "postcode_match": float(postal_match(crm.get("postcode"), cand.get("postcode"))),
        "city_match": float(city_match(crm_city, cand.get("city"))),
        "street_number_diff": street_number_diff(crm_street_num, cand_street_num),
        "addr_token_overlap": token_overlap(crm_addr, cand_addr),
        "address_density": _address_density_for_candidate(crm, cand),
        "street_name_jaro": jaro_sim(crm_street_name, cand_street_name),
    }

    features = {**name_features, **addr_features}
    features["name_addr_consistency"] = name_features["name_jaro_max"] * addr_features["addr_jaro"]
    legal_category = map_legal_to_category(cand.get("cj_ul"))
    if legal_category == "PUBLIC":
        features["legal_form_category"] = 1.0
    elif legal_category == "PRIVE":
        features["legal_form_category"] = -1.0
    else:
        features["legal_form_category"] = 0.0

    # ---------------- New features from reranking rules (Phase 3) -----------------
    
    # Get primary candidate name for new features
    from .naming import primary_name
    primary_cand_name = primary_name(cand) or ""
    
    # D9: is_siege - head office indicator
    features["is_siege"] = float(cand.get("is_siege", False))
    
    # B1: is_association - check if candidate name contains association tokens
    cand_name_tokens = set(normalize_text(primary_cand_name).split()) if primary_cand_name else set()
    features["is_association"] = float(bool(cand_name_tokens & ASSOCIATION_TOKENS))
    
    # D6: alias_match - check if aliases in parentheses match candidate name
    aliases = crm.get("aliases") or []
    alias_match_score = 0.0
    if aliases and primary_cand_name:
        cand_norm = normalize_text(primary_cand_name)
        for alias in aliases:
            if jaro_sim(alias, cand_norm) >= 0.80 or alias in cand_norm:
                alias_match_score = 1.0
                break
    features["alias_match"] = alias_match_score
    
    # D11: token_overlap_ul - token overlap with UL names including substring matches
    ul_tokens: Set[str] = set()
    for field in ["denomination_ul", "denomination_usuelle_ul", "sigle_ul"]:
        val = cand.get(field)
        if val:
            ul_tokens.update(normalize_text(str(val)).split())
    
    crm_tokens = crm.get("crm_tokens") or set()
    
    if not crm_tokens:
        features["token_overlap_ul"] = 0.0
    else:
        # Direct token match
        common_tokens = crm_tokens & ul_tokens
        
        # Substring match: check if a candidate token is IN a CRM token 
        # (for CamelCase like DIGITBOXING) or vice-versa.
        substring_matches = set()
        for cand_tok in ul_tokens:
            if len(cand_tok) >= 4:
                for crm_tok in crm_tokens:
                    if (len(crm_tok) >= 6 and cand_tok in crm_tok) or (len(crm_tok) >= 4 and crm_tok in cand_tok):
                        substring_matches.add(cand_tok)
                        break
        
        all_matches = common_tokens | substring_matches
        features["token_overlap_ul"] = len(all_matches) / len(crm_tokens)
    
    # D7: ul_vs_pm_indicator - 1 if best match is from UL, 0 if from PM, 0.5 otherwise
    type_max = features.get("type_of_max_name", 0.0)
    is_ul_max = features.get("is_ul_name_max", 0.0)
    if is_ul_max == 1.0:
        features["ul_vs_pm_indicator"] = 1.0
    elif type_max == 4.0:  # PM_DIRIGEANT
        features["ul_vs_pm_indicator"] = 0.0
    else:
        features["ul_vs_pm_indicator"] = 0.5
        
    # Context features
    features["is_crm_school"] = is_crm_school

    return features


def make_features(crm_row: pd.Series, cand: dict, skip_semantic: bool = False) -> Dict[str, float]:
    """Backward-compatible wrapper (builds CRM preprocessing on the fly)."""
    crm = preprocess_crm_row(crm_row)
    return make_features_from_preprocessed(crm, cand, skip_semantic=skip_semantic)
