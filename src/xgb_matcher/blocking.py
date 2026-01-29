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

from .naming import candidate_tfidf_text, normalize_text, primary_name
from .features import build_address


STREET_STOPWORDS: Set[str] = {
    "RUE", "AV", "AVE", "AVENUE", "BD", "BOULEVARD", "CHE", "CHEMIN",
    "IMP", "IMPASSE", "ALL", "ALLEE", "PL", "PLACE", "SQ", "SQUARE",
    "ROUTE", "RTE", "QUAI", "SENTIER", "ZA", "ZI", "ZAC",
}

_TFIDF_PUNCT_RE = re.compile(r"[^\w\s]")


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


def build_siren_index(candidates: List[dict]) -> Dict[str, List[dict]]:
    """Build index mapping SIREN -> list of candidates (siblings) for enrichment."""
    from collections import defaultdict
    index: Dict[str, List[dict]] = defaultdict(list)
    for cand in candidates:
        siren = cand.get("siren") or ""
        if siren:
            index[siren].append(cand)
    return dict(index)


def _candidate_tfidf_name_with_siblings(
    cand: dict,
    siren_index: Dict[str, List[dict]] | None,
) -> str:
    """Build TF-IDF name string enriched with SIREN siblings (Variant C)."""
    from .naming import build_candidate_names
    
    # Start with this candidate's own names
    bag = [cn.text for cn in build_candidate_names(cand)]
    
    # Enrich with sibling names if siren_index is provided
    if siren_index is not None:
        siren = cand.get("siren") or ""
        if siren and siren in siren_index:
            for sibling in siren_index[siren]:
                # Skip self (same siret)
                if sibling.get("siret") == cand.get("siret"):
                    continue
                for cn in build_candidate_names(sibling):
                    bag.append(cn.text)
    
    # Dedupe while preserving order
    seen: Set[str] = set()
    deduped = []
    for name in bag:
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)
    return " ".join(deduped)


def build_tfidf_index(
    candidates: List[dict],
    *,
    name_mode: str = "bag",
    siren_siblings: bool = False,
) -> Tuple[Optional[TfidfVectorizer], Optional[object], List[str]]:
    """Build TF-IDF index for candidate names.
    
    Args:
        candidates: List of candidate dicts.
        name_mode: "bag" for bag-of-names (candidate_tfidf_text), 
                   "primary" for primary_name only (aligned with train config).
        siren_siblings: If True, enrich each candidate's names with SIREN siblings
                        (Variant C from training - recommended for best recall).
    """
    # Build SIREN index if needed for sibling enrichment
    siren_index = build_siren_index(candidates) if siren_siblings else None
    
    if name_mode == "primary":
        names = [normalize_text_for_tfidf(primary_name(c) or "") for c in candidates]
    elif siren_siblings:
        names = [normalize_text_for_tfidf(_candidate_tfidf_name_with_siblings(c, siren_index)) for c in candidates]
    else:
        names = [normalize_text_for_tfidf(candidate_tfidf_text(c) or "") for c in candidates]
    
    # Use (1,3) ngrams for better acronym/partial matching (improves recall)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
        max_df=0.95,  # Ignore very common terms
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
    *,
    cand_names: List[str] | None = None,
    char_top_k: int | None = None,
) -> List[int]:
    """Return candidate indices for top-k TF-IDF similarity.
    
    Uses word TF-IDF first, then char-ngram fallback for acronyms/typos.
    Combines both for improved recall on edge cases.
    """
    if not crm_name or cand_matrix is None:
        return []
    crm_norm = normalize_text_for_tfidf(crm_name)
    q = vectorizer.transform([crm_norm])
    sims = q @ cand_matrix.T
    row = sims.getrow(0)
    
    # Collect word-based matches
    word_indices = set()
    if row.nnz > 0:
        idx = row.indices
        data = row.data
        # Take more candidates from word matching (1.5x top_k) for recall
        word_limit = int(top_k * 1.5)
        if len(idx) > word_limit:
            sel = np.argpartition(data, -word_limit)[-word_limit:]
            idx = idx[sel]
            data = data[sel]
        order = np.argsort(data)[::-1]
        word_indices = set(idx[order].tolist())
    
    # Always try char-ngram fallback for better recall (not just when word fails)
    char_indices = set()
    if cand_names:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=False,
            min_df=1,
        )
        try:
            char_mat = char_vec.fit_transform(cand_names)
            q_char = char_vec.transform([crm_norm])
            char_sims = q_char @ char_mat.T
            char_row = char_sims.getrow(0)
            if char_row.nnz > 0:
                c_idx = char_row.indices
                c_data = char_row.data
                limit = char_top_k or top_k
                if len(c_idx) > limit:
                    sel = np.argpartition(c_data, -limit)[-limit:]
                    c_idx = c_idx[sel]
                    c_data = c_data[sel]
                c_order = np.argsort(c_data)[::-1]
                char_indices = set(c_idx[c_order].tolist())
        except ValueError:
            pass
    
    # Combine word and char indices (union), prioritize word matches
    if not word_indices and not char_indices:
        return []
    
    # If we have word matches, use them as base and add char matches
    if word_indices:
        combined = list(word_indices)
        # Add char-only matches (not already in word matches) up to char_top_k
        char_only = char_indices - word_indices
        char_limit = char_top_k or int(top_k * 0.4)
        combined.extend(list(char_only)[:char_limit])
        return combined[:top_k + char_limit]  # Allow slight overflow for recall
    else:
        # Only char matches available
        return list(char_indices)[:top_k]


def normalize_text_for_tfidf(text: str | None) -> str:
    """Normalize text for TF-IDF with light plural expansion."""
    base = normalize_text(text)
    if not base:
        return ""
    # Remove punctuation for acronym friendliness, then collapse whitespace.
    base = _TFIDF_PUNCT_RE.sub(" ", base)
    base = " ".join(base.split())
    tokens = base.split()
    # Add compact acronyms from single-letter sequences (e.g., J.N.C -> JNC).
    acronyms: List[str] = []
    buf: List[str] = []
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            buf.append(tok)
        else:
            if len(buf) >= 2:
                acronyms.append("".join(buf))
            buf = []
    if len(buf) >= 2:
        acronyms.append("".join(buf))
    if acronyms:
        tokens = tokens + acronyms
    out: List[str] = []
    seen = set()
    for tok in tokens:
        if not tok:
            continue
        if tok not in seen:
            out.append(tok)
            seen.add(tok)
        # Add light singular variants to improve recall on plurals.
        if tok.isalpha() and len(tok) >= 5:
            if tok.endswith("AUX") and len(tok) >= 6:
                sing = tok[:-3] + "AL"
            elif tok.endswith("S") and not tok.endswith("SS"):
                sing = tok[:-1]
            elif tok.endswith("X"):
                sing = tok[:-1]
            else:
                sing = None
            if sing and sing not in seen:
                out.append(sing)
                seen.add(sing)
    return " ".join(out)
