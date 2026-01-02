"""
Blocking utilities for multi-strategy candidate retrieval.

Provides lightweight, dependency-free helpers to:
  - derive department codes (for wider pools),
  - compute address hashes,
  - extract numeric tokens,
  - prefilter candidates via TF-IDF,
  - dedupe and attach address density stats.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Dict, Optional, Set, Tuple
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .naming import normalize_text, primary_name
from .features import build_address


STREET_STOPWORDS: Set[str] = {
    "RUE", "AV", "AVE", "AVENUE", "BD", "BOULEVARD", "CHE", "CHEMIN",
    "IMP", "IMPASSE", "ALL", "ALLEE", "PL", "PLACE", "SQ", "SQUARE",
    "ROUTE", "RTE", "QUAI", "SENTIER", "ZA", "ZI", "ZAC",
}


def normalize_code(value: object) -> Optional[str]:
    """Normalize CP/INSEE codes to string, return None if invalid."""
    if value is None:
        return None
    try:
        if str(value).strip() == "" or str(value).lower() == "nan":
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s:
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s


def department_from_code(insee: Optional[str], postcode: Optional[str]) -> Optional[str]:
    """Return department code from INSEE or postcode."""
    code = normalize_code(insee) or normalize_code(postcode)
    if not code:
        return None
    if code.startswith(("97", "98")) and len(code) >= 3:
        return code[:3]
    return code[:2] if len(code) >= 2 else None


def _normalize_street(street: str | None) -> str:
    if not street:
        return ""
    tokens = normalize_text(street).split()
    tokens = [t for t in tokens if t and t not in STREET_STOPWORDS]
    return " ".join(tokens)


def address_hash(street_number: Optional[str], street_name: Optional[str]) -> Optional[str]:
    """Compute a coarse address hash from street number + normalized street."""
    if not street_name:
        return None
    num = str(street_number).strip() if street_number is not None else ""
    street_norm = _normalize_street(street_name)
    if not street_norm:
        return None
    if num:
        return f"{num}|{street_norm}"
    return street_norm


def candidate_address_hash(cand: dict) -> Optional[str]:
    """Compute address hash for a candidate record."""
    num = cand.get("numeroVoie") or cand.get("street_number")
    street_parts = [cand.get("typeVoie") or cand.get("street_type"), cand.get("libelleVoie") or cand.get("street_name")]
    street = " ".join([p for p in street_parts if p])
    return address_hash(str(num) if num is not None else None, street)


def extract_numeric_tokens(text: str | None) -> Set[str]:
    """Extract numeric tokens from text (e.g. 'APAJH 69' -> {'69'})."""
    if not text:
        return set()
    return set(re.findall(r"\b\d{1,6}\b", str(text)))


def candidate_numeric_tokens(cand: dict) -> Set[str]:
    """Extract numeric tokens from candidate name (cached on candidate)."""
    cached = cand.get("_xgb_numeric_tokens")
    if isinstance(cached, set):
        return cached
    name = primary_name(cand) or ""
    tokens = extract_numeric_tokens(name)
    cand["_xgb_numeric_tokens"] = tokens
    return tokens


def filter_candidates_by_numeric_tokens(
    candidates: Iterable[dict],
    numeric_tokens: Set[str],
) -> List[dict]:
    if not numeric_tokens:
        return []
    out = []
    for cand in candidates:
        if candidate_numeric_tokens(cand) & numeric_tokens:
            out.append(cand)
    return out


def filter_candidates_by_address_hash(
    candidates: Iterable[dict],
    addr_hash: Optional[str],
) -> List[dict]:
    if not addr_hash:
        return []
    out = []
    for cand in candidates:
        if candidate_address_hash(cand) == addr_hash:
            out.append(cand)
    return out


def dedupe_candidates(candidates: Iterable[dict]) -> Dict[str, dict]:
    """Deduplicate candidates by SIRET (last wins)."""
    mapping: Dict[str, dict] = {}
    for cand in candidates:
        siret = str(cand.get("siret") or "").strip()
        if not siret:
            continue
        mapping[siret] = cand
    return mapping


def attach_address_density(candidates: Iterable[dict]) -> None:
    """Attach address density counters on candidate dicts."""
    by_insee: Dict[Tuple[str, str], int] = defaultdict(int)
    by_cp: Dict[Tuple[str, str], int] = defaultdict(int)
    for cand in candidates:
        addr = build_address(cand)
        if not addr:
            continue
        insee = str(cand.get("insee") or "")
        cp = str(cand.get("postcode") or "")
        by_insee[(insee, addr)] += 1
        by_cp[(cp, addr)] += 1
    for cand in candidates:
        addr = build_address(cand)
        if not addr:
            cand["_xgb_addr_density_insee"] = 1
            cand["_xgb_addr_density_cp"] = 1
            continue
        insee = str(cand.get("insee") or "")
        cp = str(cand.get("postcode") or "")
        cand["_xgb_addr_density_insee"] = by_insee.get((insee, addr), 1)
        cand["_xgb_addr_density_cp"] = by_cp.get((cp, addr), 1)


def build_tfidf_index(
    candidates: List[dict],
) -> Tuple[Optional[TfidfVectorizer], Optional[any], List[str]]:
    """Build TF-IDF index for candidate primary names."""
    names = [normalize_text(primary_name(c) or "") for c in candidates]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
    )
    try:
        matrix = vectorizer.fit_transform(names)
    except ValueError:
        return None, None, names
    return vectorizer, matrix, names


def prefilter_candidates_tfidf(
    crm_name: str,
    vectorizer: TfidfVectorizer,
    cand_matrix,
    top_k: int,
) -> List[int]:
    """Return candidate indices for top-k TF-IDF similarity."""
    if not crm_name or cand_matrix is None:
        return []
    q = vectorizer.transform([normalize_text(crm_name)])
    sims = q @ cand_matrix.T
    row = sims.getrow(0)
    if row.nnz == 0:
        return []
    idx = row.indices
    data = row.data
    if len(idx) > top_k:
        sel = np.argpartition(data, -top_k)[-top_k:]
        idx = idx[sel]
        data = data[sel]
    order = np.argsort(data)[::-1]
    return idx[order].tolist()

