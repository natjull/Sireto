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
from src.xgb_matcher.naming import primary_name


DEFAULT_OUTPUT = Path("data/samples_aligned_v4.parquet")
DEFAULT_SPLITS_DIR = Path("data/splits")
TRAINING_DATA = Path("data/entrainements.csv")
PARTITIONS_DIR = Path("data/candidates_v4")
MODEL_DIR = Path("models")

MAX_NEGATIVES = 50
HARD_RATIO = 0.5
SAME_ADDR_NEG_MAX = 10
PREFILTER_TOP_K = 500
SEED = 42

SEMANTIC_ENABLED = os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"


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
    df = df[df["ground_truth_siret"].notna() & (df["ground_truth_siret"].str.len() == 14)]
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
        df[df["siren"].isin(train_sirens)].copy(),
        df[df["siren"].isin(dev_sirens)].copy(),
        df[df["siren"].isin(test_sirens)].copy(),
    )


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
    buckets: Dict[str, int] = defaultdict(int)
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


def build_tfidf_index(candidates: List[dict]) -> Tuple[Optional[TfidfVectorizer], Optional[any], List[str]]:
    names = [normalize_text(primary_name(cand) or "") for cand in candidates]
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
    crm_norm = [normalize_text(n or "") for n in crm_names]
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
) -> List[dict]:
    gt_siret = crm_row.get("ground_truth_siret", "")
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
        samples.append(s)

    for f in selected:
        s = {fn: f.get(fn, 0.0) for fn in FEATURE_NAMES}
        s["label"] = 0
        s["query_id"] = query_id
        s["siret"] = f["_siret"]
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
    writer,
) -> Tuple[int, int, any]:
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

        # Per-location IDF + address density
        idf_map, default_idf = _compute_idf_map(candidates)
        set_global_name_idf_map(idf_map, default_idf)
        key = "insee" if insee else "postcode"
        field = "_xgb_addr_density_insee" if insee else "_xgb_addr_density_cp"
        _attach_address_density(candidates, key=key, target_field=field)

        # TF-IDF prefilter
        vectorizer, cand_matrix, _ = build_tfidf_index(candidates)
        crm_names = group["crm_name"].tolist()
        if vectorizer is not None and cand_matrix is not None:
            top_indices = prefilter_candidates_tfidf(crm_names, vectorizer, cand_matrix, prefilter_k)
        else:
            # Fallback: no TF-IDF available, use empty indices to trigger random sampling
            top_indices = [[] for _ in crm_names]

        # Build siret -> candidate index map to ensure GT inclusion
        siret_index = {cand.get("siret"): i for i, cand in enumerate(candidates)}

        samples_batch: List[dict] = []
        for row, idx_list in zip(group.itertuples(index=False), top_indices):
            row_dict = row._asdict()
            gt_siret = row_dict.get("ground_truth_siret")
            if idx_list:
                subset = [candidates[i] for i in idx_list if i < len(candidates)]
            else:
                if candidates:
                    k = min(prefilter_k, len(candidates))
                    subset = random.sample(candidates, k)
                else:
                    subset = []
            if gt_siret and gt_siret in siret_index:
                gt_cand = candidates[siret_index[gt_siret]]
                if gt_cand not in subset:
                    subset.append(gt_cand)
            samples = generate_samples_for_query(
                pd.Series(row_dict),
                subset,
                max_neg=max_negatives,
                ranker=ranker,
                ranker_feature_order=ranker_feature_order,
                ranker_zero_features=ranker_zero_features,
            )
            samples_batch.extend(samples)

        if samples_batch:
            df_tmp = pd.DataFrame(samples_batch)
            df_tmp["split"] = split_name
            table = pa.Table.from_pandas(df_tmp, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(generate_split.output_path), table.schema)
            writer.write_table(table)
            total_samples += len(df_tmp)
            total_pos += int(df_tmp["label"].sum())

    return total_samples, total_pos, writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training samples v4 (partitioned + TF-IDF).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--training-csv", type=Path, default=TRAINING_DATA)
    parser.add_argument("--partitions-dir", type=Path, default=PARTITIONS_DIR)
    parser.add_argument("--prefilter-k", type=int, default=PREFILTER_TOP_K)
    parser.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ranker-model", type=Path, default=None)
    parser.add_argument("--ranker-meta", type=Path, default=None)
    parser.add_argument("--disable-ranker-hard-negatives", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Sample Generation v4 (Partitioned + TF-IDF)")
    print("=" * 60)

    print("\n1. Loading CRM data...")
    df = load_training_data(args.training_csv)
    print(f"   Rows: {len(df)}, SIRENs: {df['siren'].nunique()}")

    print("\n2. SIREN split...")
    train_df, dev_df, test_df = create_siren_split(df, args.seed)
    print(f"   Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.splits_dir / "train.csv", index=False)
    dev_df.to_csv(args.splits_dir / "dev.csv", index=False)
    test_df.to_csv(args.splits_dir / "test.csv", index=False)

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
    generate_split.output_path = output_path

    writer = None
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
        "partitions_dir": str(args.partitions_dir),
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved samples to {output_path}")


if __name__ == "__main__":
    main()
