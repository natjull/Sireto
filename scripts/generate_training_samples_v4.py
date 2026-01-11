#!/usr/bin/env python3
"""
Generate training samples v4 - partitioned candidates + TF-IDF prefilter.

Key ideas:
  - Load candidates per commune (partitioned store)
  - Build TF-IDF index on candidate names for that commune
  - Prefilter top-K per CRM using TF-IDF similarity
  - Compute full features only on top-K
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    make_features,
    normalize_text,
    set_global_name_idf_map,
    build_address,
)
from src.xgb_matcher.blocking import normalize_text_for_tfidf
from src.xgb_matcher.naming import primary_name, build_candidate_names


DEFAULT_OUTPUT = Path("data/samples_aligned_v4.parquet")
TRAINING_DATA = Path("data/entrainements.csv")
PARTITIONS_DIR = Path("data/candidates_v4")
ETAB_PARQUET = Path("data/StockEtablissement_utf8.parquet")
MODEL_DIR = Path("models")

MAX_NEGATIVES = 50
HARD_RATIO = 0.5
SAME_ADDR_NEG_MAX = 10
PREFILTER_TOP_K = 500
MIN_CANDIDATES_SUBSET = 100  # Guarantee at least this many candidates after TF-IDF prefilter
SEED = 42

SEMANTIC_ENABLED = os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"


def _norm_code(x: object) -> str | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_training_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    df = df.rename(columns={
        "SITE": "crm_name",
        "SITE_CLI_ADRESSE": "crm_address",
        "SITE_CLI_COMMUNE": "crm_city",
        "CODE_POSTAL": "postcode",
        "CODE_INSEE": "insee",
        "SIRET": "ground_truth_siret",
    })
    df["ground_truth_siret"] = df["ground_truth_siret"].astype(str).str.strip()
    df = df[df["ground_truth_siret"].notna() & (df["ground_truth_siret"].str.len() == 14)]
    df["postcode"] = df["postcode"].apply(_norm_code)
    df["insee"] = df["insee"].apply(_norm_code)
    df["crm_id"] = range(len(df))
    df["siren"] = df["ground_truth_siret"].str[:9]
    df["loc_key"] = df["insee"].fillna("") + "|" + df["postcode"].fillna("")
    return df


def create_siren_split(df: pd.DataFrame, seed: int = SEED):
    sirens = df["siren"].unique().tolist()
    random.seed(seed)
    random.shuffle(sirens)
    n = len(sirens)
    train_sirens = set(sirens[:int(n * 0.70)])
    dev_sirens = set(sirens[int(n * 0.70):int(n * 0.85)])
    test_sirens = set(sirens[int(n * 0.85):])
    return (
        df[df["siren"].isin(list(train_sirens))].copy(),
        df[df["siren"].isin(list(dev_sirens))].copy(),
        df[df["siren"].isin(list(test_sirens))].copy(),
    )


def _is_closed_status(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().upper().startswith("F")


def _load_etat_admin_map(etab_path: Path, sirets: List[str]) -> Dict[str, str]:
    dataset = ds.dataset(etab_path, format="parquet")
    out: Dict[str, str] = {}
    chunk_size = 1000
    for i in range(0, len(sirets), chunk_size):
        chunk = [s for s in sirets[i : i + chunk_size] if s]
        if not chunk:
            continue
        filt = ds.field("siret").isin(chunk)
        table = dataset.to_table(filter=filt, columns=["siret", "etatAdministratifEtablissement"])
        if table.num_rows == 0:
            try:
                chunk_int = [int(s) for s in chunk]
            except Exception:
                chunk_int = []
            if chunk_int:
                filt = ds.field("siret").isin(chunk_int)
                table = dataset.to_table(filter=filt, columns=["siret", "etatAdministratifEtablissement"])
        if table.num_rows > 0:
            for siret, etat in table.to_pandas().values.tolist():
                out[str(siret)] = etat
    return out


def find_latest_ranker(models_dir: Path) -> Tuple[Optional[Path], Optional[Path], bool]:
    fast = sorted(models_dir.glob("xgbranker_fast_*.json"), reverse=True)
    if fast:
        ranker_path = fast[0]
        ts = ranker_path.stem.replace("xgbranker_fast_", "")
        meta_path = models_dir / f"xgb_matcher_features_{ts}.json"
        return ranker_path, (meta_path if meta_path.exists() else None), True
    regular = sorted(models_dir.glob("xgbranker_*.json"), reverse=True)
    if regular:
        ranker_path = regular[0]
        ts = ranker_path.stem.replace("xgbranker_", "")
        meta_path = models_dir / f"xgb_matcher_features_{ts}.json"
        return ranker_path, (meta_path if meta_path.exists() else None), False
    return None, None, False


def load_ranker_meta(meta_path: Optional[Path]) -> Tuple[List[str], List[str]]:
    if not meta_path or not meta_path.exists():
        return FEATURE_NAMES, [f for f in FEATURE_NAMES if f.startswith("name_semantic_")]
    with open(meta_path) as f:
        meta = json.load(f)
    feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
    semantic_zero = meta.get("semantic_features_zeroed_for_ranker_fast") or []
    return feature_order, semantic_zero


def score_with_ranker(
    feats: List[dict],
    *,
    ranker: xgb.Booster,
    feature_order: List[str],
    zero_features: List[str] | None = None,
) -> np.ndarray:
    X = np.array([[f.get(fn, 0.0) for fn in feature_order] for f in feats], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if zero_features:
        idx = [feature_order.index(f) for f in zero_features if f in feature_order]
        if idx:
            X[:, idx] = 0.0
    dmat = xgb.DMatrix(X, feature_names=feature_order)
    return ranker.predict(dmat)


def _compute_idf_map(candidates: List[dict]) -> Tuple[Dict[str, float], float]:
    doc_freq: Dict[str, int] = defaultdict(int)
    doc_count = 0
    for cand in candidates:
        name = primary_name(cand)
        if not name:
            continue
        tokens = {t for t in normalize_text(name).split() if len(t) >= 2}
        if not tokens:
            continue
        doc_count += 1
        for tok in tokens:
            doc_freq[tok] += 1
    if doc_count == 0:
        return {}, 0.0
    idf_map = {tok: np.log((doc_count + 1) / (df + 1)) + 1.0 for tok, df in doc_freq.items()}
    default_idf = np.log((doc_count + 1) / 1) + 1.0
    return idf_map, default_idf


def _attach_address_density(candidates: List[dict], key: str, target_field: str) -> None:
    buckets: Dict[Tuple[str, str], int] = defaultdict(int)
    for cand in candidates:
        addr = build_address(cand)
        if addr:
            buckets[(cand.get(key) or "", addr)] += 1
    for cand in candidates:
        addr = build_address(cand)
        if addr:
            cand[target_field] = buckets.get((cand.get(key) or "", addr), 1)
        else:
            cand[target_field] = 1


def load_candidates_for_loc(
    partitions_dir: Path,
    insee: Optional[str],
    postcode: Optional[str],
    dataset_insee: Optional[ds.Dataset] = None,
    dataset_cp: Optional[ds.Dataset] = None,
) -> List[dict]:
    if insee:
        dataset = dataset_insee or ds.dataset(partitions_dir / "insee", format="parquet", partitioning="hive")
        # Convert insee to int to match the partition column type
        try:
            insee_int = int(insee)
            table = dataset.to_table(filter=ds.field("insee") == insee_int)
        except (ValueError, TypeError):
            return []
    elif postcode:
        dataset = dataset_cp or ds.dataset(partitions_dir / "cp", format="parquet", partitioning="hive")
        # Convert postcode to int to match the partition column type
        try:
            postcode_int = int(postcode)
            table = dataset.to_table(filter=ds.field("postcode") == postcode_int)
        except (ValueError, TypeError):
            return []
    else:
        return []
    candidates = table.to_pylist()
    # Ensure type consistency
    for cand in candidates:
        cand["siret"] = str(cand.get("siret") or "")
        cand["siren"] = str(cand.get("siren") or "")
    return candidates


def _dedupe_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _candidate_tfidf_name(cand: dict, name_mode: str) -> str:
    if name_mode == "bag":
        bag = [cn.text for cn in build_candidate_names(cand)]
        return " ".join(_dedupe_preserve(bag))
    return primary_name(cand) or ""


def build_tfidf_index(
    candidates: List[dict],
    *,
    name_mode: str = "primary",
) -> Tuple[Optional[TfidfVectorizer], Optional[any], List[str]]:
    names = [normalize_text_for_tfidf(_candidate_tfidf_name(cand, name_mode) or "") for cand in candidates]
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
        # Empty vocabulary - all names are empty or stopwords
        return None, None, names
    return vectorizer, matrix, names


def prefilter_candidates_tfidf(
    crm_names: List[str],
    vectorizer: TfidfVectorizer,
    cand_matrix,
    top_k: int,
) -> List[List[int]]:
    crm_norm = [normalize_text_for_tfidf(n or "") for n in crm_names]
    q = vectorizer.transform(crm_norm)
    sims = q @ cand_matrix.T

    top_indices: List[List[int]] = []
    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz == 0:
            top_indices.append([])
            continue
        idx = row.indices
        data = row.data
        if len(idx) > top_k:
            sel = np.argpartition(data, -top_k)[-top_k:]
            idx = idx[sel]
            data = data[sel]
        order = np.argsort(data)[::-1]
        top_indices.append(idx[order].tolist())
    return top_indices


def generate_samples_for_query(
    crm_row: pd.Series,
    candidates: List[dict],
    max_neg: int = MAX_NEGATIVES,
    ranker: xgb.Booster | None = None,
    ranker_feature_order: Optional[List[str]] = None,
    ranker_zero_features: Optional[List[str]] = None,
    max_same_siren_negatives: int | None = None,
) -> List[dict]:
    gt_siret = crm_row.get("ground_truth_siret", "")
    gt_siren = gt_siret[:9] if gt_siret else ""
    query_id = crm_row["crm_id"]

    if not candidates:
        return []

    all_feats = []
    for cand in candidates:
        feat = make_features(crm_row, cand, skip_semantic=not SEMANTIC_ENABLED)
        feat["_siret"] = cand["siret"]
        feat["_is_pos"] = (cand["siret"] == gt_siret)
        all_feats.append(feat)

    if ranker is not None:
        feature_order = ranker_feature_order or FEATURE_NAMES
        scores = score_with_ranker(
            all_feats,
            ranker=ranker,
            feature_order=feature_order,
            zero_features=ranker_zero_features or [],
        )
        for f, s in zip(all_feats, scores):
            f["_score"] = float(s)
    else:
        for f in all_feats:
            f["_score"] = 0.6 * f.get("name_jaro_max", 0.0) + 0.4 * f.get("addr_jaro", 0.0)

    positives = [f for f in all_feats if f["_is_pos"]]
    negatives = [f for f in all_feats if not f["_is_pos"]]
    if not positives:
        return []

    if gt_siren and max_same_siren_negatives is not None and max_same_siren_negatives >= 0:
        same_siren = []
        other_negs = []
        for f in negatives:
            siret = f.get("_siret") or ""
            if siret.startswith(gt_siren):
                same_siren.append(f)
            else:
                other_negs.append(f)
        if max_same_siren_negatives == 0:
            negatives = other_negs
        else:
            same_siren = sorted(same_siren, key=lambda x: x.get("_score", 0.0), reverse=True)
            negatives = other_negs + same_siren[:max_same_siren_negatives]

    num_hard = int(max_neg * HARD_RATIO)
    neg_sorted = sorted(negatives, key=lambda x: x["_score"], reverse=True)
    hard = neg_sorted[:num_hard]
    rest = neg_sorted[num_hard:]
    rand = random.sample(rest, min(max_neg - num_hard, len(rest))) if rest else []

    same_addr = [
        f for f in negatives
        if float(f.get("addr_jaro", 0.0)) >= 0.95 and float(f.get("name_jaro_max", 0.0)) < 0.50
    ]
    same_addr = sorted(same_addr, key=lambda x: x.get("addr_jaro", 0.0), reverse=True)
    same_addr = same_addr[: min(SAME_ADDR_NEG_MAX, max_neg)]

    selected = []
    seen = set()
    for group in (hard, same_addr, rand):
        for f in group:
            siret = f.get("_siret")
            if siret in seen:
                continue
            selected.append(f)
            if siret:
                seen.add(siret)
            if len(selected) >= max_neg:
                break
        if len(selected) >= max_neg:
            break

    samples = []
    for f in positives:
        s = {fn: f.get(fn, 0.0) for fn in FEATURE_NAMES}
        s["label"] = 1
        s["query_id"] = query_id
        s["siret"] = f["_siret"]
        s["semantic_enabled"] = int(SEMANTIC_ENABLED)
        samples.append(s)

    for f in selected:
        s = {fn: f.get(fn, 0.0) for fn in FEATURE_NAMES}
        s["label"] = 0
        s["query_id"] = query_id
        s["siret"] = f["_siret"]
        s["semantic_enabled"] = int(SEMANTIC_ENABLED)
        samples.append(s)

    return samples


def generate_split(
    df: pd.DataFrame,
    partitions_dir: Path,
    prefilter_k: int,
    max_negatives: int,
    ranker: xgb.Booster | None,
    ranker_feature_order: Optional[List[str]],
    ranker_zero_features: Optional[List[str]],
    split_name: str,
    writer: pq.ParquetWriter | None,
    output_path: Path,
    *,
    drop_unnamed_candidates: bool,
    tfidf_name_mode: str,
    max_same_siren_negatives: int | None,
    exclude_closed_candidates: bool,
) -> Tuple[int, int, pq.ParquetWriter | None]:
    total_samples = 0
    total_pos = 0

    dataset_insee = ds.dataset(partitions_dir / "insee", format="parquet", partitioning="hive")
    dataset_cp = ds.dataset(partitions_dir / "cp", format="parquet", partitioning="hive")

    # Group by location key
    grouped = df.groupby("loc_key")
    for loc_key, group in tqdm(grouped, total=len(grouped), desc=f"Generating {split_name}"):
        insee = None
        postcode = None
        if loc_key and "|" in loc_key:
            insee, postcode = loc_key.split("|", 1)
            insee = insee or None
            postcode = postcode or None

        candidates = load_candidates_for_loc(
            partitions_dir,
            insee,
            postcode,
            dataset_insee=dataset_insee,
            dataset_cp=dataset_cp,
        )
        if not candidates:
            continue
        candidates_pool = candidates

        if exclude_closed_candidates:
            candidates_pool = [
                c for c in candidates_pool
                if str(c.get("etat_admin") or "").strip().upper() != "F"
            ]
            if not candidates_pool:
                continue

        if drop_unnamed_candidates:
            candidates = [c for c in candidates_pool if build_candidate_names(c)]
            if not candidates:
                candidates = candidates_pool
        else:
            candidates = candidates_pool

        # Per-location IDF + address density
        idf_map, default_idf = _compute_idf_map(candidates)
        set_global_name_idf_map(idf_map, default_idf)
        key = "insee" if insee else "postcode"
        field = "_xgb_addr_density_insee" if insee else "_xgb_addr_density_cp"
        _attach_address_density(candidates, key=key, target_field=field)

        # TF-IDF prefilter
        vectorizer, cand_matrix, _ = build_tfidf_index(candidates, name_mode=tfidf_name_mode)
        crm_names = group["crm_name"].tolist()
        if vectorizer is not None and cand_matrix is not None:
            top_indices = prefilter_candidates_tfidf(crm_names, vectorizer, cand_matrix, prefilter_k)
        else:
            # Fallback: no TF-IDF available, use empty indices to trigger random sampling
            top_indices = [[] for _ in crm_names]

        # Build siret -> candidate map to ensure GT inclusion
        siret_index = {cand.get("siret"): cand for cand in candidates_pool}

        samples_batch: List[dict] = []
        for row, idx_list in zip(group.itertuples(index=False, name="Row"), top_indices):
            row_dict = row._asdict()  # type: ignore[union-attr]
            gt_siret = row_dict.get("ground_truth_siret")
            # FIX: Guarantee at least MIN_CANDIDATES_SUBSET candidates
            # TF-IDF prefilter may return very few candidates for specific proper nouns
            if idx_list and len(idx_list) >= MIN_CANDIDATES_SUBSET:
                subset = [candidates[i] for i in idx_list if i < len(candidates)]
            else:
                # Combine TF-IDF matches with random samples to reach MIN_CANDIDATES_SUBSET
                if candidates:
                    tfidf_set = set(idx_list) if idx_list else set()
                    tfidf_cands = [candidates[i] for i in tfidf_set if i < len(candidates)]
                    remaining_idx = [i for i in range(len(candidates)) if i not in tfidf_set]
                    needed = max(0, min(MIN_CANDIDATES_SUBSET, prefilter_k) - len(tfidf_cands))
                    random_extra_idx = random.sample(remaining_idx, min(needed, len(remaining_idx))) if remaining_idx else []
                    random_extra = [candidates[i] for i in random_extra_idx]
                    subset = tfidf_cands + random_extra
                else:
                    subset = []
            if gt_siret and gt_siret in siret_index:
                gt_cand = siret_index[gt_siret]
                if gt_cand not in subset:
                    subset.append(gt_cand)
            samples = generate_samples_for_query(
                pd.Series(row_dict),
                subset,
                max_neg=max_negatives,
                ranker=ranker,
                ranker_feature_order=ranker_feature_order,
                ranker_zero_features=ranker_zero_features,
                max_same_siren_negatives=max_same_siren_negatives,
            )
            samples_batch.extend(samples)

        if samples_batch:
            df_tmp = pd.DataFrame(samples_batch)
            df_tmp["split"] = split_name
            table = pa.Table.from_pandas(df_tmp, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
            total_samples += len(df_tmp)
            total_pos += int(df_tmp["label"].sum())

    return total_samples, total_pos, writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training samples v4 (partitioned + TF-IDF).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-csv", type=Path, default=TRAINING_DATA)
    parser.add_argument("--partitions-dir", type=Path, default=PARTITIONS_DIR)
    parser.add_argument("--etab-parquet", type=Path, default=ETAB_PARQUET)
    parser.add_argument("--prefilter-k", type=int, default=PREFILTER_TOP_K)
    parser.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES)
    parser.add_argument(
        "--tfidf-name-mode",
        choices=["primary", "bag"],
        default="primary",
        help="TF-IDF name source: primary (default) or bag-of-names.",
    )
    parser.add_argument(
        "--drop-unnamed-candidates",
        dest="drop_unnamed_candidates",
        action="store_true",
        default=True,
        help="Drop candidates with no usable names (default: enabled).",
    )
    parser.add_argument(
        "--keep-unnamed-candidates",
        dest="drop_unnamed_candidates",
        action="store_false",
        help="Keep candidates with no usable names.",
    )
    parser.add_argument(
        "--max-same-siren-negatives",
        type=int,
        default=0,
        help="Cap same-SIREN negatives per query (0 exclude, -1 unlimited).",
    )
    parser.add_argument(
        "--exclude-closed-gt",
        dest="exclude_closed_gt",
        action="store_true",
        default=True,
        help="Exclude GT with etatAdministratifEtablissement == 'F'.",
    )
    parser.add_argument(
        "--include-closed-gt",
        dest="exclude_closed_gt",
        action="store_false",
        help="Keep closed GT in training samples.",
    )
    parser.add_argument(
        "--exclude-closed-candidates",
        dest="exclude_closed_candidates",
        action="store_true",
        default=True,
        help="Exclude candidates with etat_admin == 'F'.",
    )
    parser.add_argument(
        "--include-closed-candidates",
        dest="exclude_closed_candidates",
        action="store_false",
        help="Include candidates with etat_admin == 'F'.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ranker-model", type=Path, default=None)
    parser.add_argument("--ranker-meta", type=Path, default=None)
    parser.add_argument("--disable-ranker-hard-negatives", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if not args.exclude_closed_gt and args.exclude_closed_candidates:
        print("[WARN] include_closed_gt with exclude_closed_candidates may drop closed GT queries.")

    print("=" * 60)
    print("Sample Generation v4 (Partitioned + TF-IDF)")
    print("=" * 60)

    print("\n1. Loading CRM data...")
    df = load_training_data(args.training_csv)
    print(f"   Rows: {len(df)}, SIRENs: {df['siren'].nunique()}")
    if args.exclude_closed_gt:
        if not args.etab_parquet.exists():
            raise FileNotFoundError(f"Establishment parquet not found: {args.etab_parquet}")
        gt_sirets = df["ground_truth_siret"].astype(str).tolist()
        status_map = _load_etat_admin_map(args.etab_parquet, gt_sirets)
        closed_mask = df["ground_truth_siret"].map(
            lambda s: _is_closed_status(status_map.get(str(s)))
        )
        missing = df["ground_truth_siret"].map(lambda s: str(s) not in status_map).sum()
        before = len(df)
        df = df[~closed_mask].copy()
        print(
            f"   Excluded closed GT: {before - len(df)} removed (missing status: {missing})"
        )

    print("\n2. SIREN split...")
    train_df, dev_df, test_df = create_siren_split(df, args.seed)
    print(f"   Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    ranker = None
    ranker_feature_order: Optional[List[str]] = None
    ranker_zero_features: List[str] = []
    is_fast = False
    ranker_info = None
    if not args.disable_ranker_hard_negatives:
        ranker_path = args.ranker_model
        meta_path = args.ranker_meta
        if ranker_path is None:
            ranker_path, meta_path, is_fast = find_latest_ranker(MODEL_DIR)
        else:
            is_fast = "fast" in ranker_path.name.lower()
        if ranker_path and ranker_path.exists():
            ranker = xgb.Booster()
            ranker.load_model(str(ranker_path))
            ranker_feature_order, semantic_zero = load_ranker_meta(meta_path)
            ranker_zero_features = semantic_zero if is_fast else []
            ranker_info = {
                "path": str(ranker_path),
                "meta": str(meta_path) if meta_path else None,
                "is_fast": is_fast,
            }
            print(f"\n[HardNeg] Using ranker: {ranker_path} (fast={is_fast})")
        else:
            print("\n[HardNeg] No ranker model found → fallback to heuristic scoring.")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    total_counts = {}

    train_count, train_pos, writer = generate_split(
        train_df,
        args.partitions_dir,
        args.prefilter_k,
        args.max_negatives,
        ranker,
        ranker_feature_order,
        ranker_zero_features,
        "train",
        writer,
        output_path,
        drop_unnamed_candidates=args.drop_unnamed_candidates,
        tfidf_name_mode=args.tfidf_name_mode,
        max_same_siren_negatives=(None if args.max_same_siren_negatives < 0 else args.max_same_siren_negatives),
        exclude_closed_candidates=args.exclude_closed_candidates,
    )
    total_counts["train"] = train_count

    dev_count, dev_pos, writer = generate_split(
        dev_df,
        args.partitions_dir,
        args.prefilter_k,
        args.max_negatives,
        ranker,
        ranker_feature_order,
        ranker_zero_features,
        "dev",
        writer,
        output_path,
        drop_unnamed_candidates=args.drop_unnamed_candidates,
        tfidf_name_mode=args.tfidf_name_mode,
        max_same_siren_negatives=(None if args.max_same_siren_negatives < 0 else args.max_same_siren_negatives),
        exclude_closed_candidates=args.exclude_closed_candidates,
    )
    total_counts["dev"] = dev_count

    test_count, test_pos, writer = generate_split(
        test_df,
        args.partitions_dir,
        args.prefilter_k,
        args.max_negatives,
        ranker,
        ranker_feature_order,
        ranker_zero_features,
        "test",
        writer,
        output_path,
        drop_unnamed_candidates=args.drop_unnamed_candidates,
        tfidf_name_mode=args.tfidf_name_mode,
        max_same_siren_negatives=(None if args.max_same_siren_negatives < 0 else args.max_same_siren_negatives),
        exclude_closed_candidates=args.exclude_closed_candidates,
    )
    total_counts["test"] = test_count

    if writer is not None:
        writer.close()

    meta = {
        "generated": datetime.now().isoformat(),
        "seed": args.seed,
        "train": train_count,
        "dev": dev_count,
        "test": test_count,
        "hard_negative_ranker": ranker_info,
        "prefilter_k": args.prefilter_k,
        "tfidf_name_mode": args.tfidf_name_mode,
        "drop_unnamed_candidates": args.drop_unnamed_candidates,
        "max_same_siren_negatives": args.max_same_siren_negatives,
        "exclude_closed_gt": args.exclude_closed_gt,
        "exclude_closed_candidates": args.exclude_closed_candidates,
        "etab_parquet": str(args.etab_parquet),
        "partitions_dir": str(args.partitions_dir),
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved samples to {output_path}")


if __name__ == "__main__":
    main()
