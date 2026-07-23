from __future__ import annotations

# CRITICAL: Set semantic env BEFORE any imports
import os
import sys

def _peek_mode(argv: list[str]) -> str:
    for arg in argv:
        if arg.startswith("--mode="):
            value = arg.split("=", 1)[1].strip()
            return value or "ranker"
    if "--mode" not in argv:
        return "ranker"
    try:
        idx = argv.index("--mode")
        return argv[idx + 1]
    except Exception:
        return "ranker"


_mode = _peek_mode(sys.argv)
os.environ["XGB_SEMANTIC_ENABLED"] = "1" if _mode == "decider" else "0"

import argparse
import gc
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import xgboost as xgb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    FAST_RANKER_FEATURE_NAMES,
    build_address,
    build_semantic_name_pool,
    inject_semantic_features_batch,
    make_features_from_preprocessed,
    normalize_text,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.blocking import (
    normalize_text_for_tfidf,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.dense_retrieval import PartitionEmbeddingStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.naming import primary_name, build_candidate_names
from src.xgb_matcher.semantic import clear_cache as clear_semantic_cache
from src.xgb_matcher.siren_retrieval import SirenGlobalIndex, SirenToGeoIndex
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache
from src.xgb_matcher.timing import PipelineTimer


DEFAULT_OUTPUT = Path("data/samples_v5.parquet")
TRAINING_DATA = Path("data/crm_ok_gt.csv")
PARTITIONS_DIR = Path("data/candidates_v7_all")
ETAB_PARQUET = Path("data/StockEtablissement_utf8.parquet")
MODEL_DIR = Path("models")

MAX_NEGATIVES = 50
HARD_RATIO = 0.5
SAME_ADDR_NEG_MAX = 20  # Increased from 10 to better capture colocataire negatives
PREFILTER_TOP_K = 500
MIN_CANDIDATES_SUBSET = 100
SEED = 42

# Fast-path knobs
FAST_INSEE_MAX = 200_000
FAST_SAMPLE_FALLBACK = 10_000
FAST_CANDIDATE_COLUMNS = [
    "siret", "siren", "denomination", "enseigne1", "enseigne2", "enseigne3",
    "etablissementSiege", "numeroVoie", "typeVoie", "libelleVoie",
    "complementAdresse", "postcode", "city", "cj_ul", "etat_admin",
    "sigle_ul", "denomination_ul", "denomination_usuelle_ul", "nom_ul",
    "nom_usage_ul", "prenom_usuel_ul", "pseudonyme_ul", "is_siege",
    "pm_dirigeant_names", "insee",
]

DEFAULT_FLUSH_EVERY = 500
MEGA_POOL_THRESHOLD = 100_000

# SSOT configuration
VARIANT_KNOBS = {
    "B": {
        "tfidf_name_mode": "bag",
        "char_top_k": 200,
        "siren_siblings": False,
        "description": "Variant B (bag-of-names + char fallback)",
    },
}

SEMANTIC_ENABLED = os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"

SAMPLE_SCHEMA = pa.schema([
    *[(fn, pa.float32()) for fn in FEATURE_NAMES],
    ("label", pa.int32()),
    ("query_id", pa.int64()),
    ("siret", pa.string()),
    ("siren", pa.string()),
    ("etat_admin", pa.string()),
    ("semantic_enabled", pa.int32()),
    ("split", pa.string()),
])


class StreamingParquetWriter:
    def __init__(self, output_path: Path, schema: pa.Schema, flush_every: int = DEFAULT_FLUSH_EVERY, *, append_from: Optional[Path] = None):
        self.output_path = output_path
        self.schema = schema
        self.flush_every = flush_every
        self._writer: Optional[pq.ParquetWriter] = None
        self._append_from = append_from
        self._buffers: Dict[str, List[Any]] = {field.name: [] for field in schema}
        self._count = 0
        self._total_written = 0
        self._semantic_nonzero_count = 0
        self._total_samples = 0
        if self._append_from is not None:
            self._initialize_with_existing(self._append_from)
    
    def add_sample(self, sample: Dict[str, Any]) -> None:
        for field in self.schema:
            val = sample.get(field.name)
            if val is None:
                if field.type == pa.float32(): val = 0.0
                elif field.type in (pa.int32(), pa.int64()): val = 0
                else: val = ""
            self._buffers[field.name].append(val)
        self._count += 1
        self._total_samples += 1
        sem_max = sample.get("name_semantic_max", 0.0)
        if sem_max and sem_max > 0: self._semantic_nonzero_count += 1
        if self._count >= self.flush_every: self.flush()
    
    def flush(self) -> None:
        if self._count == 0: return
        arrays = []
        for field in self.schema:
            data = self._buffers[field.name]
            if field.type == pa.float32(): arr = pa.array(data, type=pa.float32())
            elif field.type == pa.int32(): arr = pa.array(data, type=pa.int32())
            elif field.type == pa.int64(): arr = pa.array(data, type=pa.int64())
            else: arr = pa.array(data, type=pa.string())
            arrays.append(arr)
        table = pa.Table.from_arrays(arrays, schema=self.schema)
        if self._writer is None: self._writer = pq.ParquetWriter(str(self.output_path), self.schema)
        self._writer.write_table(table)
        self._total_written += self._count
        for key in self._buffers: self._buffers[key] = []
        self._count = 0

    def _initialize_with_existing(self, source_path: Path) -> None:
        if not source_path.exists(): return
        parquet_file = pq.ParquetFile(source_path)
        if parquet_file.schema_arrow != self.schema:
            raise RuntimeError("Resume schema mismatch")
        self._writer = pq.ParquetWriter(str(self.output_path), self.schema)
        for idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(idx)
            self._writer.write_table(table)
            self._total_written += table.num_rows
            self._total_samples += table.num_rows
            if "name_semantic_max" in table.schema.names:
                col = table.column("name_semantic_max")
                nonzero = pc.sum(pc.greater(col, 0)).as_py() or 0
                self._semantic_nonzero_count += int(nonzero)
    
    def close(self) -> Tuple[int, float]:
        self.flush()
        if self._writer is not None: self._writer.close()
        rate = self._semantic_nonzero_count / self._total_samples if self._total_samples > 0 else 0.0
        return self._total_written, rate


class ProgressTracker:
    def __init__(self, progress_path: Path):
        self.progress_path = progress_path
        self.processed: Set[str] = set()
        if self.progress_path.exists():
            with open(self.progress_path) as f:
                for line in f: self.processed.add(line.strip())
    
    def is_processed(self, loc_key: str) -> bool: return loc_key in self.processed
    def mark_processed(self, loc_key: str) -> None:
        self.processed.add(loc_key)
        with open(self.progress_path, "a") as f: f.write(f"{loc_key}\n")
    def clear(self) -> None:
        self.processed.clear()
        if self.progress_path.exists(): self.progress_path.unlink()


def _norm_code(x: object) -> str | None:
    if x is None or pd.isna(x): return None
    s = str(x).strip()
    if not s or s.lower() == "nan": return None
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s or None

def load_training_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    df = df.rename(columns={"crm_cp": "postcode", "crm_insee": "insee", "crm_adresse": "crm_address", "crm_commune": "crm_city", "gt_siret": "ground_truth_siret"})
    df["ground_truth_siret"] = df["ground_truth_siret"].astype(str).str.strip()
    df = df[df["ground_truth_siret"].str.len() == 14]
    df["postcode"] = df["postcode"].apply(_norm_code)
    df["insee"] = df["insee"].apply(_norm_code)
    df["crm_id"] = range(len(df))
    df["siren"] = df["ground_truth_siret"].str[:9]
    df["loc_key"] = df["insee"].fillna("") + "|" + df["postcode"].fillna("")
    return df

def create_siren_split(df: pd.DataFrame, seed: int = SEED):
    sirens = df["siren"].unique().tolist()
    random.seed(seed); random.shuffle(sirens)
    n = len(sirens)
    train_s = set(sirens[:int(n * 0.70)]); dev_s = set(sirens[int(n * 0.70):int(n * 0.85)]); test_s = set(sirens[int(n * 0.85):])
    return df[df["siren"].isin(train_s)].copy(), df[df["siren"].isin(dev_s)].copy(), df[df["siren"].isin(test_s)].copy()

def find_latest_ranker(models_dir: Path) -> Tuple[Optional[Path], Optional[Path], bool]:
    fast = sorted(models_dir.glob("xgbranker_fast_*.json"), reverse=True)
    if fast:
        p = fast[0]; ts = p.stem.replace("xgbranker_fast_", ""); meta = models_dir / f"xgb_two_stage_meta_{ts}.json"
        return p, (meta if meta.exists() else None), True
    reg = sorted(models_dir.glob("xgbranker_*.json"), reverse=True)
    if reg:
        p = reg[0]; ts = p.stem.replace("xgbranker_", ""); meta = models_dir / f"xgb_two_stage_meta_{ts}.json"
        return p, (meta if meta.exists() else None), False
    return None, None, False

def load_ranker_meta(meta_path: Optional[Path], *, is_fast: bool) -> Tuple[List[str], List[str]]:
    if not meta_path or not meta_path.exists():
        return (FAST_RANKER_FEATURE_NAMES if is_fast else FEATURE_NAMES), ([f for f in FEATURE_NAMES if f.startswith("name_semantic_")] if is_fast else [])
    with open(meta_path) as f: meta = json.load(f)
    f_order = meta.get("ranker_fast_feature_order" if is_fast else "ranker_feature_order") or meta.get("feature_order") or FEATURE_NAMES
    return f_order, (meta.get("semantic_features_zeroed_for_ranker_fast") or [])

def score_with_ranker(feats: List[dict], ranker: xgb.Booster, feature_order: List[str], zero_features: List[str] | None = None) -> np.ndarray:
    X = np.array([[f.get(fn, 0.0) for fn in feature_order] for f in feats], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    if zero_features:
        idx = [feature_order.index(f) for f in zero_features if f in feature_order]
        if idx: X[:, idx] = 0.0
    return ranker.predict(xgb.DMatrix(X, feature_names=feature_order))

def _inject_semantic_features_for_items(crm_pre: dict, items: List[Dict[str, Any]], candidates: List[dict], *, siret_key: str) -> None:
    if not SEMANTIC_ENABLED: return
    crm_name_sem = crm_pre.get("crm_name_semantic", "")
    if not crm_name_sem: return
    siret_to_cand = {c.get("siret"): c for c in candidates}
    name_lists = []; targets = []
    for item in items:
        cand = siret_to_cand.get(item.get(siret_key, ""))
        if not cand: continue
        pool = build_semantic_name_pool(build_candidate_names(cand), crm_city_norm=crm_pre.get("crm_city_norm", ""), cand_city_norm=normalize_text(cand.get("city")))
        if pool: name_lists.append(pool); targets.append(item)
    if not name_lists: return
    inject_semantic_features_batch(crm_name_sem, targets, name_lists)

def generate_samples_for_query(
    crm_row: pd.Series,
    crm_pre: dict,
    candidates: List[dict],
    prefilter_sirets: Optional[List[str]] = None,
    max_neg: int = MAX_NEGATIVES,
    ranker: xgb.Booster | None = None,
    ranker_feature_order: Optional[List[str]] = None,
    ranker_zero_features: Optional[List[str]] = None,
    is_ranker_fast: bool = False,
    semantic_expected: bool = False,
    use_prefilter_sampling: bool = False,
) -> List[dict]:
    gt_siret = crm_row.get("ground_truth_siret", ""); q_id = crm_row["crm_id"]
    if not candidates: return []
    def _norm_siret(val: object) -> str:
        return str(val or "").zfill(14)

    if use_prefilter_sampling and not ranker:
        cand_map = {_norm_siret(c.get("siret")): c for c in candidates if c.get("siret")}
        gt_norm = _norm_siret(gt_siret)
        if gt_norm not in cand_map:
            return []
        ordered = []
        if prefilter_sirets:
            for s in prefilter_sirets:
                norm = _norm_siret(s)
                if norm in cand_map:
                    ordered.append(norm)
        if not ordered:
            ordered = list(cand_map.keys())

        neg_ordered = [s for s in ordered if s != gt_norm]
        num_hard = int(max_neg * HARD_RATIO)
        hard = neg_ordered[:num_hard]
        rest = neg_ordered[num_hard:]
        rand = random.sample(rest, min(max_neg - len(hard), len(rest))) if rest else []
        selected = hard + rand

        samples = []
        for siret_norm in [gt_norm] + selected:
            cand = cand_map.get(siret_norm)
            if not cand:
                continue
            f = make_features_from_preprocessed(crm_pre, cand, skip_semantic=True)
            f.update({"_siret": cand["siret"], "_is_pos": (siret_norm == gt_norm), "_siren": cand.get("siren"), "_etat_admin": cand.get("etat_admin")})
            s = {fn: f.get(fn, 0.0) for fn in FEATURE_NAMES}
            s.update({
                "label": int(f["_is_pos"]),
                "query_id": q_id,
                "siret": f["_siret"],
                "siren": f["_siren"],
                "etat_admin": f["_etat_admin"],
                "semantic_enabled": int(semantic_expected),
            })
            samples.append(s)
        return samples

    all_feats = []
    for cand in candidates:
        f = make_features_from_preprocessed(crm_pre, cand, skip_semantic=True)
        f.update({"_siret": cand["siret"], "_is_pos": (cand["siret"] == gt_siret), "_siren": cand.get("siren"), "_etat_admin": cand.get("etat_admin")})
        all_feats.append(f)
    if ranker:
        scores = score_with_ranker(all_feats, ranker, ranker_feature_order or FEATURE_NAMES, ranker_zero_features)
        for f, s in zip(all_feats, scores): f["_score"] = float(s)
    else:
        for f in all_feats: f["_score"] = 0.6 * f.get("name_jaro_max", 0.0) + 0.4 * f.get("addr_jaro", 0.0)
    pos = [f for f in all_feats if f["_is_pos"]]
    neg = [f for f in all_feats if not f["_is_pos"]]
    if not pos: return []

    gt_siret = crm_row.get("ground_truth_siret", "")
    gt_siren = str(gt_siret[:9]) if gt_siret and len(gt_siret) >= 9 else ""

    num_hard = int(max_neg * HARD_RATIO)
    neg_s = sorted(neg, key=lambda x: x["_score"], reverse=True)
    hard = neg_s[:num_hard]; rest = neg_s[num_hard:]
    rand = random.sample(rest, min(max_neg - num_hard, len(rest))) if rest else []

    # Colocataire negatives: widened threshold to capture partial address matches
    same_addr = sorted(
        [f for f in neg
         if float(f.get("addr_jaro", 0.0)) >= 0.80
         and float(f.get("name_jaro_max", 0.0)) < 0.65],
        key=lambda x: x.get("addr_jaro", 0.0), reverse=True
    )[:SAME_ADDR_NEG_MAX]

    # Geographic homonyme negatives are selected after semantic injection below.
    geo_homo_neg = sorted(
        [f for f in neg
         if float(f.get("name_semantic_max", 0.0)) > 0.85
         and float(f.get("postcode_match", 0.0)) == 0.0
         and float(f.get("name_jaro_max", 0.0)) > 0.75],
        key=lambda x: x.get("name_semantic_max", 0.0), reverse=True
    )[:5]

    # Sibling-SIREN negatives: same SIREN, different SIRET (multi-site disambiguation)
    sibling_neg = []
    if gt_siren:
        sibling_neg = sorted(
            [f for f in neg
             if str(f.get("_siren", ""))[:9] == gt_siren],
            key=lambda x: x.get("_score", 0.0), reverse=True
        )[:5]

    if semantic_expected:
        geo_homo_candidates = [
            feature_row
            for feature_row in neg
            if float(feature_row.get("postcode_match", 0.0)) == 0.0
            and float(feature_row.get("name_jaro_max", 0.0)) > 0.75
        ]
        semantic_items = []
        for feature_row in geo_homo_candidates:
            semantic_items.append(
                {
                    **{fn: feature_row.get(fn, 0.0) for fn in FEATURE_NAMES},
                    "siret": feature_row["_siret"],
                }
            )
        _inject_semantic_features_for_items(
            crm_pre,
            semantic_items,
            candidates,
            siret_key="siret",
        )
        semantic_by_siret = {item["siret"]: item for item in semantic_items}
        for feature_row in geo_homo_candidates:
            semantic_row = semantic_by_siret.get(feature_row["_siret"])
            if semantic_row:
                for feature_name in (
                    "name_semantic_max",
                    "name_semantic_second",
                    "name_semantic_gap",
                ):
                    feature_row[feature_name] = semantic_row.get(feature_name, 0.0)

        geo_homo_neg = sorted(
            [f for f in neg
             if float(f.get("name_semantic_max", 0.0)) > 0.85
             and float(f.get("postcode_match", 0.0)) == 0.0
             and float(f.get("name_jaro_max", 0.0)) > 0.75],
            key=lambda x: x.get("name_semantic_max", 0.0), reverse=True
        )[:5]

    selected = []; seen = set()
    for group in (hard, same_addr, geo_homo_neg, sibling_neg, rand):
        for f in group:
            if f["_siret"] not in seen:
                selected.append(f); seen.add(f["_siret"])
                if len(selected) >= max_neg: break
        if len(selected) >= max_neg: break
    samples = []
    for f in pos + selected:
        s = {fn: f.get(fn, 0.0) for fn in FEATURE_NAMES}
        s.update({
            "label": int(f["_is_pos"]),
            "query_id": q_id,
            "siret": f["_siret"],
            "siren": f["_siren"],
            "etat_admin": f["_etat_admin"],
            "semantic_enabled": int(semantic_expected),
        })
        samples.append(s)
    if semantic_expected:
        _inject_semantic_features_for_items(crm_pre, samples, candidates, siret_key="siret")
    return samples

def _process_loc_key(
    loc_key: str,
    group_records: List[dict],
    partitions_dir: Path,
    config: RetrievalConfigV1,
    max_negatives: int,
    ranker_path: Optional[Path],
    ranker_f_order: List[str],
    ranker_z_features: List[str],
    is_fast: bool,
    semantic_expected: bool,
    use_prefilter_sampling: bool,
    split_name: str,
    persistent_cache: Optional[TfidfPersistentCache],
    dense_store_dir: Optional[Path],
    siren_index_path: Optional[Path] = None,
) -> Tuple[List[dict], List[str]]:
    """Process a single loc_key group. Designed for multiprocess execution.

    Returns (samples_list, lost_gt_lines).
    """
    store = PartitionedCandidateStore(partitions_dir)

    ranker = None
    if ranker_path:
        ranker = xgb.Booster()
        ranker.load_model(str(ranker_path))

    # Load SIREN indices independently:
    # - global index for Route B (name-only SIREN retrieval)
    # - geo index for SIREN expansion (can work in --geo-only mode)
    siren_global_index = None
    siren_to_geo = None
    if siren_index_path and siren_index_path.exists():
        try:
            siren_global_index = SirenGlobalIndex(siren_index_path)
        except Exception:
            siren_global_index = None
        try:
            siren_to_geo = SirenToGeoIndex(siren_index_path / "siren_to_geo.parquet")
        except Exception:
            siren_to_geo = None
        # Route B requires both global and geo indices. Keep global disabled if geo is missing.
        if siren_global_index is not None and siren_to_geo is None:
            siren_global_index = None

    tfidf_cache: Dict = {}
    dense_store = PartitionEmbeddingStore(dense_store_dir) if config.dense_retrieval_enabled else None
    samples: List[dict] = []
    lost_lines: List[str] = []

    for row_dict in group_records:
        crm_pre = preprocess_crm_row(row_dict)
        gt_siret = row_dict.get("ground_truth_siret")
        res = build_candidate_pool(
            store, row_dict, crm_pre, config, tfidf_cache, gt_siret,
            persistent_cache=persistent_cache,
            dense_store=dense_store,
            siren_global_index=siren_global_index,
            siren_to_geo=siren_to_geo,
        )
        if not res.candidates:
            continue
        set_global_name_idf_map(res.idf_map, res.default_idf)
        if gt_siret and not res.gt_in_tfidf_pool:
            lost_lines.append(
                f'{row_dict.get("crm_id")},"{str(row_dict.get("crm_name")).replace(chr(34),chr(34)*2)}",'
                f'{gt_siret},{res.loss_reason or "UNKNOWN"},{split_name},{loc_key}'
            )
        batch = generate_samples_for_query(
            crm_row=pd.Series(row_dict),
            crm_pre=crm_pre,
            candidates=res.candidates,
            prefilter_sirets=res.prefilter_sirets,
            max_neg=max_negatives,
            ranker=ranker,
            ranker_feature_order=ranker_f_order,
            ranker_zero_features=ranker_z_features,
            is_ranker_fast=is_fast,
            semantic_expected=semantic_expected,
            use_prefilter_sampling=use_prefilter_sampling,
        )
        for s in batch:
            s["split"] = split_name
            samples.append(s)
    return samples, lost_lines


def generate_split(
    df: pd.DataFrame,
    partitions_dir: Path,
    prefilter_k: int,
    max_negatives: int,
    ranker: xgb.Booster | None,
    ranker_f_order: List[str],
    ranker_z_features: List[str],
    split_name: str,
    writer: StreamingParquetWriter,
    log_base: Path,
    drop_unnamed: bool,
    tfidf_mode: str,
    exclude_closed: bool,
    char_top_k: int,
    is_fast: bool,
    semantic_expected: bool,
    use_prefilter_sampling: bool,
    dense_retrieval_enabled: bool,
    dense_top_k: int,
    dense_store_dir: Optional[Path],
    siren_index_path: Optional[Path] = None,
    siren_global_index: Optional[Any] = None,
    siren_to_geo: Optional[Any] = None,
    siren_expansion_enabled: bool = False,
    *,
    ranker_path: Optional[Path] = None,
    max_workers: int = 0,
) -> Tuple[int, int]:
    total_samples = 0; total_pos = 0

    # If indices are not provided, load them from path.
    # Keep loading independent to support --geo-only (expansion without global index).
    if siren_index_path and siren_index_path.exists():
        if siren_global_index is None:
            try:
                siren_global_index = SirenGlobalIndex(siren_index_path)
            except Exception:
                siren_global_index = None
        if siren_to_geo is None:
            try:
                siren_to_geo = SirenToGeoIndex(siren_index_path / "siren_to_geo.parquet")
            except Exception:
                siren_to_geo = None
        # Route B requires both global and geo indices. Keep global disabled if geo is missing.
        if siren_global_index is not None and siren_to_geo is None:
            siren_global_index = None

    config = RetrievalConfigV1(
        pool_mode="insee_then_postcode",
        include_closed=not exclude_closed,
        drop_unnamed=drop_unnamed,
        tfidf_name_mode=tfidf_mode,
        prefilter_k=prefilter_k,
        char_top_k=char_top_k,
        min_candidates=MIN_CANDIDATES_SUBSET,
        dense_retrieval_enabled=dense_retrieval_enabled,
        dense_top_k=dense_top_k,
        siren_global_index_path=str(siren_index_path) if siren_index_path else None,
        siren_expansion_enabled=siren_expansion_enabled,
    )
    lost_gt_log = log_base.parent / f"lost_gt_analysis_{log_base.stem}.csv"
    header_written = lost_gt_log.exists()

    # P0a: Persistent TF-IDF cache
    persistent_cache = TfidfPersistentCache(config.signature().hash)
    timer = PipelineTimer()
    dense_store = PartitionEmbeddingStore(dense_store_dir) if dense_retrieval_enabled else None

    grouped = df.groupby("loc_key")
    loc_keys = list(grouped.groups.keys())

    # Decide: parallel or sequential
    n_workers = max_workers if max_workers > 0 else int(os.environ.get("XGB_SAMPLE_WORKERS", "0"))

    if n_workers > 1 and not semantic_expected:
        # P0b: Parallel execution by loc_key (no semantic — safe for multiprocess)
        from concurrent.futures import ProcessPoolExecutor, as_completed

        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for loc_key in loc_keys:
                group = grouped.get_group(loc_key)
                records = group.to_dict("records")
                fut = executor.submit(
                    _process_loc_key,
                    loc_key, records, partitions_dir, config,
                    max_negatives, ranker_path, ranker_f_order, ranker_z_features,
                    is_fast, semantic_expected, use_prefilter_sampling,
                    split_name, persistent_cache, dense_store_dir,
                    siren_index_path,
                )
                futures[fut] = loc_key

            pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Generating {split_name} (parallel)")
            for fut in pbar:
                samples, lost_lines = fut.result()
                for s in samples:
                    writer.add_sample(s)
                    total_samples += 1
                    total_pos += (1 if s["label"] == 1 else 0)
                if lost_lines:
                    with open(lost_gt_log, "a") as f:
                        if not header_written:
                            f.write("crm_id,crm_name,gt_siret,loss_reason,split,loc_key\n")
                            header_written = True
                        f.write("\n".join(lost_lines) + "\n")
    else:
        # Sequential (original path, with P0a cache + P0c timing)
        store = PartitionedCandidateStore(partitions_dir)
        pbar = tqdm(grouped, desc=f"Generating {split_name}")
        for loc_key, group in pbar:
            tfidf_cache: Dict = {}
            for row_dict in group.to_dict("records"):
                crm_pre = preprocess_crm_row(row_dict); gt_siret = row_dict.get("ground_truth_siret")
                with timer.stage("build_candidate_pool"):
                    res = build_candidate_pool(
                        store, row_dict, crm_pre, config, tfidf_cache, gt_siret,
                        persistent_cache=persistent_cache, dense_store=dense_store, timer=timer,
                        siren_global_index=siren_global_index, siren_to_geo=siren_to_geo,
                    )
                if not res.candidates: continue
                set_global_name_idf_map(res.idf_map, res.default_idf)
                if gt_siret and not res.gt_in_tfidf_pool:
                    with open(lost_gt_log, "a") as f:
                        if not header_written: f.write("crm_id,crm_name,gt_siret,loss_reason,split,loc_key\n"); header_written = True
                        f.write(f'{row_dict.get("crm_id")},"{str(row_dict.get("crm_name")).replace(chr(34),chr(34)*2)}",{gt_siret},{res.loss_reason or "UNKNOWN"},{split_name},{loc_key}\n')
                with timer.stage("generate_samples"):
                    samples = generate_samples_for_query(
                        crm_row=pd.Series(row_dict),
                        crm_pre=crm_pre,
                        candidates=res.candidates,
                        prefilter_sirets=res.prefilter_sirets,
                        max_neg=max_negatives,
                        ranker=ranker,
                        ranker_feature_order=ranker_f_order,
                        ranker_zero_features=ranker_z_features,
                        is_ranker_fast=is_fast,
                        semantic_expected=semantic_expected,
                        use_prefilter_sampling=use_prefilter_sampling,
                    )
                for s in samples:
                    s["split"] = split_name; writer.add_sample(s)
                    total_samples += 1; total_pos += (1 if s["label"] == 1 else 0)
            gc.collect()

        # P0c: Log timing summary
        import logging
        logging.basicConfig(level=logging.INFO)
        timer.log_summary(prefix=split_name)
        cache_stats = persistent_cache.stats()
        print(f"[TfidfCache] {split_name}: hits={cache_stats['hits']} misses={cache_stats['misses']}")

    return total_samples, total_pos

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-csv", type=Path, default=TRAINING_DATA)
    parser.add_argument("--partitions-dir", type=Path, default=PARTITIONS_DIR)
    parser.add_argument("--prefilter-k", type=int, default=PREFILTER_TOP_K)
    parser.add_argument("--char-top-k", type=int, default=200)
    parser.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES)
    parser.add_argument("--tfidf-name-mode", choices=["bag"], default="bag")
    parser.add_argument("--exclude-closed-candidates", action="store_true", default=False)
    parser.add_argument("--mode", choices=["ranker", "decider"], default="ranker")
    parser.add_argument("--enable-ranker-hard-negatives", action="store_true", default=False)
    parser.add_argument("--enable-dense-retrieval", action="store_true", default=False)
    parser.add_argument("--dense-top-k", type=int, default=500)
    parser.add_argument("--dense-store-dir", type=Path, default=None)
    parser.add_argument("--siren-index", type=Path, default=None, help="Path to Route B SIREN global index (data/siren_index)")
    parser.add_argument("--enable-siren-expansion", action="store_true", default=False, help="Enable SIREN expansion (V7 + local + cross-partition)")
    args = parser.parse_args()
    semantic_expected = args.mode == "decider"
    if semantic_expected != SEMANTIC_ENABLED:
        raise RuntimeError(
            f"semantic_expected mismatch: mode={args.mode} env={int(SEMANTIC_ENABLED)}"
        )
    
    df = load_training_data(args.training_csv)
    train_df, dev_df, test_df = create_siren_split(df, SEED)
    ranker = None; r_fo = []; r_zf = []; is_fast = False; r_info = None
    use_hard_negatives = args.enable_ranker_hard_negatives or args.mode == "decider"
    use_prefilter_sampling = args.mode == "ranker" and not use_hard_negatives
    if use_hard_negatives:
        rp, mp, is_fast = find_latest_ranker(MODEL_DIR)
        if rp:
            ranker = xgb.Booster(); ranker.load_model(str(rp))
            r_fo, r_zf = load_ranker_meta(mp, is_fast=is_fast); r_info = {"path": str(rp), "is_fast": is_fast}
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = StreamingParquetWriter(args.output, SAMPLE_SCHEMA)

    # Load SIREN indices independently:
    # - Route B needs global + geo
    # - Expansion only needs geo (supports --geo-only build artifact)
    siren_global_index = None
    siren_to_geo = None
    if args.siren_index and args.siren_index.exists():
        try:
            siren_global_index = SirenGlobalIndex(args.siren_index)
        except Exception as e:
            print(f"[WARNING] Failed to load SIREN global index: {e}.")
            siren_global_index = None
        try:
            siren_to_geo = SirenToGeoIndex(args.siren_index / "siren_to_geo.parquet")
        except Exception as e:
            print(f"[WARNING] Failed to load siren_to_geo index: {e}.")
            siren_to_geo = None

        if siren_global_index is not None and siren_to_geo is not None:
            print(f"[Route B] Loaded SIREN global + geo index from {args.siren_index}")
        elif siren_to_geo is not None:
            print(f"[SIREN Expansion] Loaded geo index from {args.siren_index} (Route B global disabled)")
            siren_global_index = None
        else:
            print("[WARNING] No usable SIREN indices loaded. Falling back to V7 without expansion.")

    counts = {}
    for name, sdf in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        c, p = generate_split(
            sdf,
            args.partitions_dir,
            args.prefilter_k,
            args.max_negatives,
            ranker,
            r_fo,
            r_zf,
            name,
            writer,
            args.output,
            True,
            args.tfidf_name_mode,
            args.exclude_closed_candidates,
            args.char_top_k,
            is_fast,
            semantic_expected,
            use_prefilter_sampling,
            args.enable_dense_retrieval,
            args.dense_top_k,
            args.dense_store_dir,
            args.siren_index,
            siren_global_index,
            siren_to_geo,
            args.enable_siren_expansion,
        )
        counts[name] = (c, p); clear_semantic_cache(); gc.collect()
    
    written, sem_rate = writer.close()
    if semantic_expected and sem_rate < 0.20:
        raise RuntimeError(f"Semantic sanity check failed: {sem_rate:.2%}")
    
    rc = RetrievalConfigV1(
        pool_mode="insee_then_postcode",
        include_closed=not args.exclude_closed_candidates,
        drop_unnamed=True,
        tfidf_name_mode=args.tfidf_name_mode,
        prefilter_k=args.prefilter_k,
        char_top_k=args.char_top_k,
        min_candidates=MIN_CANDIDATES_SUBSET,
        dense_retrieval_enabled=args.enable_dense_retrieval,
        dense_top_k=args.dense_top_k,
        siren_global_index_path=str(args.siren_index) if args.siren_index else None,
        siren_expansion_enabled=args.enable_siren_expansion,
        siren_expansion_pool_cap=1500,
    )
    sig = rc.signature()
    meta = {
        "generated": datetime.now().isoformat(),
        "counts": counts,
        "total": written,
        "semantic_rate": sem_rate,
        "semantic_expected": semantic_expected,
        "mode": args.mode,
        "sampling_strategy": "prefilter" if use_prefilter_sampling else "full",
        "dense_retrieval_enabled": args.enable_dense_retrieval,
        "dense_top_k": args.dense_top_k,
        "dense_store_dir": str(args.dense_store_dir) if args.dense_store_dir else None,
        "retrieval_signature_v1": {"version": sig.version, "hash": sig.hash, "config": sig.config},
    }
    with open(args.output.with_suffix(".json"), "w") as f: json.dump(meta, f, indent=2)

if __name__ == "__main__": main()
