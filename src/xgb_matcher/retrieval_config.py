"""Canonical retrieval configuration and signature (Variant B / SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class RetrievalConfigV1:
    version: str = "v1"

    # Pool construction
    pool_mode: str = "insee_then_postcode"
    include_closed: bool = True
    drop_unnamed: bool = True

    # TF-IDF name
    tfidf_name_mode: str = "bag"
    tfidf_name_ngrams: Tuple[int, int] = (1, 2)
    tfidf_name_token_pattern: str = r"(?u)\b\w+\b"
    tfidf_name_lowercase: bool = False
    tfidf_name_norm: Optional[str] = None
    tfidf_name_min_df: int = 1

    # TF-IDF address
    tfidf_addr_ngrams: Tuple[int, int] = (1, 2)
    tfidf_addr_token_pattern: str = r"(?u)\b\w+\b"
    tfidf_addr_lowercase: bool = False
    tfidf_addr_norm: Optional[str] = None
    tfidf_addr_min_df: int = 1
    tfidf_addr_max_df: float = 0.95

    # Char TF-IDF fallback
    char_ngrams: Tuple[int, int] = (3, 5)
    char_analyzer: str = "char_wb"
    char_min_df: int = 1
    char_top_k: int = 200

    # Prefilter + padding
    prefilter_k: int = 500
    prefilter_k_mode: str = "per_channel"
    prefilter_union_cap: Optional[int] = None
    min_candidates: int = 100

    # Stage 1 pruning
    stage1_top_n: int = 50

    # SIREN siblings
    siren_siblings: bool = False
    max_siren_siblings: Optional[int] = None
    max_names_per_candidate: Optional[int] = None

    # Rescue
    rescue_addr_hash: bool = True
    rescue_numeric_tokens: bool = True

    # Mega-communes policy
    mega_insee_max_rows: int = 100_000
    mega_insee_policy: str = "full_insee"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "pool_mode": self.pool_mode,
            "include_closed": self.include_closed,
            "drop_unnamed": self.drop_unnamed,
            "tfidf_name_mode": self.tfidf_name_mode,
            "tfidf_name_ngrams": list(self.tfidf_name_ngrams),
            "tfidf_name_token_pattern": self.tfidf_name_token_pattern,
            "tfidf_name_lowercase": self.tfidf_name_lowercase,
            "tfidf_name_norm": self.tfidf_name_norm,
            "tfidf_name_min_df": self.tfidf_name_min_df,
            "tfidf_addr_ngrams": list(self.tfidf_addr_ngrams),
            "tfidf_addr_token_pattern": self.tfidf_addr_token_pattern,
            "tfidf_addr_lowercase": self.tfidf_addr_lowercase,
            "tfidf_addr_norm": self.tfidf_addr_norm,
            "tfidf_addr_min_df": self.tfidf_addr_min_df,
            "tfidf_addr_max_df": self.tfidf_addr_max_df,
            "char_ngrams": list(self.char_ngrams),
            "char_analyzer": self.char_analyzer,
            "char_min_df": self.char_min_df,
            "char_top_k": self.char_top_k,
            "prefilter_k": self.prefilter_k,
            "prefilter_k_mode": self.prefilter_k_mode,
            "prefilter_union_cap": self.prefilter_union_cap,
            "min_candidates": self.min_candidates,
            "stage1_top_n": self.stage1_top_n,
            "siren_siblings": self.siren_siblings,
            "max_siren_siblings": self.max_siren_siblings,
            "max_names_per_candidate": self.max_names_per_candidate,
            "rescue_addr_hash": self.rescue_addr_hash,
            "rescue_numeric_tokens": self.rescue_numeric_tokens,
            "mega_insee_max_rows": self.mega_insee_max_rows,
            "mega_insee_policy": self.mega_insee_policy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalConfigV1":
        def _tuple(val, default: Tuple[int, int]) -> Tuple[int, int]:
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return int(val[0]), int(val[1])
            return default

        return cls(
            version=str(data.get("version", "v1")),
            pool_mode=data.get("pool_mode", cls().pool_mode),
            include_closed=bool(data.get("include_closed", cls().include_closed)),
            drop_unnamed=bool(data.get("drop_unnamed", cls().drop_unnamed)),
            tfidf_name_mode=data.get("tfidf_name_mode", cls().tfidf_name_mode),
            tfidf_name_ngrams=_tuple(data.get("tfidf_name_ngrams"), cls().tfidf_name_ngrams),
            tfidf_name_token_pattern=data.get("tfidf_name_token_pattern", cls().tfidf_name_token_pattern),
            tfidf_name_lowercase=bool(data.get("tfidf_name_lowercase", cls().tfidf_name_lowercase)),
            tfidf_name_norm=data.get("tfidf_name_norm", cls().tfidf_name_norm),
            tfidf_name_min_df=int(data.get("tfidf_name_min_df", cls().tfidf_name_min_df)),
            tfidf_addr_ngrams=_tuple(data.get("tfidf_addr_ngrams"), cls().tfidf_addr_ngrams),
            tfidf_addr_token_pattern=data.get("tfidf_addr_token_pattern", cls().tfidf_addr_token_pattern),
            tfidf_addr_lowercase=bool(data.get("tfidf_addr_lowercase", cls().tfidf_addr_lowercase)),
            tfidf_addr_norm=data.get("tfidf_addr_norm", cls().tfidf_addr_norm),
            tfidf_addr_min_df=int(data.get("tfidf_addr_min_df", cls().tfidf_addr_min_df)),
            tfidf_addr_max_df=float(data.get("tfidf_addr_max_df", cls().tfidf_addr_max_df)),
            char_ngrams=_tuple(data.get("char_ngrams"), cls().char_ngrams),
            char_analyzer=data.get("char_analyzer", cls().char_analyzer),
            char_min_df=int(data.get("char_min_df", cls().char_min_df)),
            char_top_k=int(data.get("char_top_k", cls().char_top_k)),
            prefilter_k=int(data.get("prefilter_k", cls().prefilter_k)),
            prefilter_k_mode=data.get("prefilter_k_mode", cls().prefilter_k_mode),
            prefilter_union_cap=data.get("prefilter_union_cap", cls().prefilter_union_cap),
            min_candidates=int(data.get("min_candidates", cls().min_candidates)),
            stage1_top_n=int(data.get("stage1_top_n", cls().stage1_top_n)),
            siren_siblings=bool(data.get("siren_siblings", cls().siren_siblings)),
            max_siren_siblings=data.get("max_siren_siblings", cls().max_siren_siblings),
            max_names_per_candidate=data.get("max_names_per_candidate", cls().max_names_per_candidate),
            rescue_addr_hash=bool(data.get("rescue_addr_hash", cls().rescue_addr_hash)),
            rescue_numeric_tokens=bool(data.get("rescue_numeric_tokens", cls().rescue_numeric_tokens)),
            mega_insee_max_rows=int(data.get("mega_insee_max_rows", cls().mega_insee_max_rows)),
            mega_insee_policy=data.get("mega_insee_policy", cls().mega_insee_policy),
        )

    def signature(self) -> "RetrievalSignatureV1":
        return RetrievalSignatureV1.from_config(self)


@dataclass(frozen=True)
class RetrievalSignatureV1:
    version: str
    hash: str
    config: Dict[str, Any]

    @staticmethod
    def _hash_payload(config: Dict[str, Any]) -> str:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_config(cls, config: RetrievalConfigV1) -> "RetrievalSignatureV1":
        config_dict = config.to_dict()
        return cls(
            version=str(config.version),
            hash=cls._hash_payload(config_dict),
            config=config_dict,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalSignatureV1":
        config = data.get("config") or {}
        version = str(data.get("version", "v1"))
        hash_val = data.get("hash") or cls._hash_payload(config)
        return cls(version=version, hash=str(hash_val), config=dict(config))

    def matches(self, config: RetrievalConfigV1) -> bool:
        return self.hash == self._hash_payload(config.to_dict())


__all__ = [
    "RetrievalConfigV1",
    "RetrievalSignatureV1",
]
