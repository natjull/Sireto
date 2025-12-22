"""
Train XGBoost and LightGBM matchers for SIRET matching.

Trains two models:
  - XGBClassifier (binary classification baseline)
  - XGBRanker (learning-to-rank with pairwise loss)

Data:
 - CRM sample: data/entrainements.csv (delimiter=';')
 - SIRET data: data/StockEtablissement_utf8.parquet

Features are computed using the shared module: src.xgb_matcher.features
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRanker

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    make_features,
    normalize_text,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# LightGBM disabled in this streamlined version
HAS_LIGHTGBM = False

# Paths
DATA_DIR = Path("data")
CRM_PATH = DATA_DIR / "entrainements.csv"
HISTO_PATH = DATA_DIR / "StockEtablissement_utf8.parquet"
UNITE_LEGALE_PATH = DATA_DIR / "StockUniteLegale_utf8.parquet"
HARVEST_DB = DATA_DIR / "harvest_full.sqlite"
MODEL_DIR = Path("models")

# Training config
N_NEGATIVES = 10  # Negative samples per positive


# --------------------------------------------------------------------------- #
# GPU detection
# --------------------------------------------------------------------------- #


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available for XGBoost."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_crm() -> pd.DataFrame:
    """Load and clean CRM training data."""
    df = pd.read_csv(CRM_PATH, sep=";", dtype=str).rename(
        columns={
            "SITE": "crm_name",
            "CODE_POSTAL": "postcode",
            "CODE_INSEE": "insee",
            "COMMUNE": "crm_city",
            "SIRET": "siret",
            "SITE_CLI_ADRESSE": "crm_address",
            "SITE_CLI_COMMUNE": "crm_city_addr",
        }
    )
    df["siret"] = df["siret"].str.strip()
    df = df.dropna(subset=["siret"])
    df = df[df["siret"] != ""]
    return df


def whitelist_sirets_from_parquet(sirets: Iterable[str]) -> set[str]:
    """Stream parquet and keep only SIRETs present in CRM."""
    target = set(sirets)
    keep: set[str] = set()
    pf = pq.ParquetFile(HISTO_PATH)
    for batch in pf.iter_batches(columns=["siret"], batch_size=100_000):
        for s in batch.column("siret").to_pylist():
            if s in target:
                keep.add(s)
        if len(keep) == len(target):
            break
    return keep


def load_unite_legale_names(sirens: set[str]) -> Dict[str, dict]:
    """Load Unite Legale names for given sirens if parquet is available."""
    if not UNITE_LEGALE_PATH.exists() or not sirens:
        return {}
    cols = [
        "siren",
        "sigleUniteLegale",
        "denominationUniteLegale",
        "denominationUsuelle1UniteLegale",
        "denominationUsuelle2UniteLegale",
        "denominationUsuelle3UniteLegale",
        "nomUniteLegale",
        "nomUsageUniteLegale",
        "prenomUsuelUniteLegale",
        "pseudonymeUniteLegale",
    ]
    pf = pq.ParquetFile(UNITE_LEGALE_PATH)
    mapping: Dict[str, dict] = {}
    for batch in pf.iter_batches(columns=cols, batch_size=100_000):
        pdf = batch.to_pandas()
        pdf = pdf[pdf["siren"].isin(sirens)]
        for _, r in pdf.iterrows():
            mapping[r["siren"]] = {
                "sigle_ul": r.get("sigleUniteLegale"),
                "denomination_ul": r.get("denominationUniteLegale"),
                "denomination_usuelle_ul": " ".join(
                    filter(
                        None,
                        [
                            r.get("denominationUsuelle1UniteLegale"),
                            r.get("denominationUsuelle2UniteLegale"),
                            r.get("denominationUsuelle3UniteLegale"),
                        ],
                    )
                ),
                "nom_ul": r.get("nomUniteLegale"),
                "nom_usage_ul": r.get("nomUsageUniteLegale"),
                "prenom_usuel_ul": r.get("prenomUsuelUniteLegale"),
                "pseudonyme_ul": r.get("pseudonymeUniteLegale"),
            }
    return mapping


def _chunked(seq: List[str], size: int = 900) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def load_pm_dirigeant_names(sirens: set[str]) -> Dict[str, List[str]]:
    """Load PM dirigeant names (personne morale) from harvest_full.sqlite."""
    if not HARVEST_DB.exists() or not sirens:
        return {}
    con = sqlite3.connect(HARVEST_DB)
    cur = con.cursor()
    out: Dict[str, set[str]] = {s: set() for s in sirens}
    for chunk in _chunked(sorted(sirens)):
        q = ",".join("?" for _ in chunk)
        cur.execute(
            f"""
            SELECT siren, denomination
            FROM dirigeants
            WHERE siren IN ({q})
              AND type_dirigeant = 'personne morale'
              AND denomination IS NOT NULL
            """,
            chunk,
        )
        for siren, denom in cur.fetchall():
            if denom and siren in out:
                out[siren].add(denom)
    con.close()
    return {k: sorted(v) for k, v in out.items() if v}


def load_parquet_subset(keep_sirets: set[str]) -> Dict[str, dict]:
    """
    Load SIRENE establishment data for given SIRETs and enrich with Unite Legale names when available.
    """
    columns = [
        "siret",
        "siren",
        "enseigne1Etablissement",
        "enseigne2Etablissement",
        "enseigne3Etablissement",
        "denominationUsuelleEtablissement",
        "numeroVoieEtablissement",
        "typeVoieEtablissement",
        "libelleVoieEtablissement",
        "complementAdresseEtablissement",
        "codePostalEtablissement",
        "libelleCommuneEtablissement",
        "codeCommuneEtablissement",
        "categorieJuridiqueUniteLegale",
    ]
    pf = pq.ParquetFile(HISTO_PATH)
    mapping: Dict[str, dict] = {}
    sirens: set[str] = set()

    for batch in pf.iter_batches(columns=columns, batch_size=100_000):
        pdf = batch.to_pandas()
        pdf = pdf[pdf["siret"].isin(keep_sirets)]
        for _, r in pdf.iterrows():
            cand = {
                "siret": r.get("siret"),
                "siren": r.get("siren"),
                "enseigne1": r.get("enseigne1Etablissement"),
                "enseigne2": r.get("enseigne2Etablissement"),
                "enseigne3": r.get("enseigne3Etablissement"),
                "denomination": r.get("denominationUsuelleEtablissement"),
                "numeroVoie": r.get("numeroVoieEtablissement"),
                "typeVoie": r.get("typeVoieEtablissement"),
                "libelleVoie": r.get("libelleVoieEtablissement"),
                "complementAdresse": r.get("complementAdresseEtablissement"),
                "postcode": r.get("codePostalEtablissement"),
                "city": r.get("libelleCommuneEtablissement"),
                "insee": r.get("codeCommuneEtablissement"),
                "cj_ul": r.get("categorieJuridiqueUniteLegale"),
            }
            if cand.get("siren"):
                sirens.add(str(cand["siren"]))
            mapping[r["siret"]] = cand

    # Enrich with Unite Legale names
    ul_names = load_unite_legale_names(sirens)
    if ul_names:
        for siret, cand in mapping.items():
            siren = cand.get("siren")
            if siren and siren in ul_names:
                cand.update(ul_names[siren])

    # Enrich with PM dirigeant names (from harvest_full.sqlite)
    pm_names = load_pm_dirigeant_names(sirens)
    if pm_names:
        for siret, cand in mapping.items():
            siren = cand.get("siren")
            if siren and siren in pm_names:
                cand["pm_dirigeant_names"] = pm_names[siren]

    return mapping


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #


@dataclass
class RankingDataset:
    """Dataset structured for ranking models with group information."""
    X: pd.DataFrame
    y: pd.Series
    groups: List[int] = field(default_factory=list)  # Size of each query group
    query_ids: List[int] = field(default_factory=list)  # Query ID for each sample


def build_location_index(crm: pd.DataFrame) -> Dict[str, List[int]]:
    """Build index of CRM rows by location for fast negative sampling."""
    loc_index: Dict[str, List[int]] = {}
    for idx, row in crm.iterrows():
        insee = row.get("insee", "")
        postcode = row.get("postcode", "")
        for loc_key in [f"insee:{insee}", f"postcode:{postcode}"]:
            if loc_key not in loc_index:
                loc_index[loc_key] = []
            loc_index[loc_key].append(idx)
    return loc_index


def prepare_ranking_dataset() -> RankingDataset:
    """
    Prepare dataset with group structure for ranking models.

    Returns RankingDataset with:
      - X: Feature matrix
      - y: Labels (1 for positive, 0 for negative)
      - groups: List of group sizes (for ranker)
      - query_ids: Query ID for each sample
    """
    print("Loading CRM data...")
    crm = load_crm()
    print(f"CRM rows loaded: {len(crm)}")

    print("Filtering SIRET against parquet whitelist...")
    whitelist = whitelist_sirets_from_parquet(crm["siret"].tolist())
    crm = crm[crm["siret"].isin(whitelist)].reset_index(drop=True)
    print(f"CRM rows after parquet whitelist: {len(crm)}")

    if len(crm) == 0:
        return RankingDataset(pd.DataFrame(), pd.Series(dtype=int))

    print("Loading parquet data for matching SIRETs...")
    parquet_data = load_parquet_subset(whitelist)

    print("Building location index...")
    loc_index = build_location_index(crm)

    feature_rows: List[dict] = []
    labels: List[int] = []
    groups: List[int] = []
    query_ids: List[int] = []

    rng = np.random.default_rng(42)

    print("Generating training pairs with group structure...")
    iterator = crm.iterrows()
    if tqdm:
        iterator = tqdm(iterator, total=len(crm), desc="Building pairs")

    query_id = 0
    for idx, row in iterator:
        siret = row["siret"]
        cand_data = parquet_data.get(siret)
        if not cand_data:
            continue

        group_size = 0

        # Positive pair
        feature_rows.append(make_features(row, cand_data))
        labels.append(1)
        query_ids.append(query_id)
        group_size += 1

        # Collect negative candidates (same location = hard negatives)
        insee = row.get("insee", "")
        postcode = row.get("postcode", "")
        neg_candidates = set()
        for loc_key in [f"insee:{insee}", f"postcode:{postcode}"]:
            if loc_key in loc_index:
                neg_candidates.update(loc_index[loc_key])
        neg_candidates.discard(idx)

        # Filter to those with parquet data
        neg_candidates_list = [
            i for i in neg_candidates
            if crm.loc[i, "siret"] in parquet_data
        ]

        # Sample negatives
        if len(neg_candidates_list) < N_NEGATIVES:
            all_indices = [
                i for i in crm.index
                if i != idx and crm.loc[i, "siret"] in parquet_data
            ]
            rng.shuffle(all_indices)
            neg_candidates_list = all_indices[:N_NEGATIVES]
        else:
            neg_candidates_list = list(
                rng.choice(
                    neg_candidates_list,
                    size=min(N_NEGATIVES, len(neg_candidates_list)),
                    replace=False,
                )
            )

        # Generate negative pairs
        for neg_idx in neg_candidates_list:
            neg_siret = crm.loc[neg_idx, "siret"]
            neg_data = parquet_data.get(neg_siret)
            if neg_data:
                feature_rows.append(make_features(row, neg_data))
                labels.append(0)
                query_ids.append(query_id)
                group_size += 1

        groups.append(group_size)
        query_id += 1

    X = pd.DataFrame(feature_rows)[FEATURE_NAMES]
    y = pd.Series(labels, dtype=int)

    print(f"Dataset: {len(X)} samples, {sum(labels)} positives, {len(groups)} queries")
    return RankingDataset(X, y, groups, query_ids)


# --------------------------------------------------------------------------- #
# Train/Val Split with Group Awareness
# --------------------------------------------------------------------------- #


@dataclass
class SplitData:
    """Train/validation split preserving group structure."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    groups_train: List[int]
    groups_val: List[int]
    query_ids_train: List[int]
    query_ids_val: List[int]


def split_by_groups(dataset: RankingDataset, test_size: float = 0.2) -> SplitData:
    """
    Split dataset by query groups (not by samples).

    This ensures all samples from a query stay together in train or val.
    """
    n_groups = len(dataset.groups)
    group_indices = list(range(n_groups))

    # Split groups
    train_groups, val_groups = train_test_split(
        group_indices, test_size=test_size, random_state=42
    )
    train_groups_set = set(train_groups)
    val_groups_set = set(val_groups)

    # Map samples to their groups
    sample_to_group = []
    current_idx = 0
    for g_idx, g_size in enumerate(dataset.groups):
        for _ in range(g_size):
            sample_to_group.append(g_idx)
        current_idx += g_size

    # Split samples
    train_mask = [sample_to_group[i] in train_groups_set for i in range(len(dataset.X))]
    val_mask = [sample_to_group[i] in val_groups_set for i in range(len(dataset.X))]

    X_train = dataset.X[train_mask].reset_index(drop=True)
    X_val = dataset.X[val_mask].reset_index(drop=True)
    y_train = dataset.y[train_mask].reset_index(drop=True)
    y_val = dataset.y[val_mask].reset_index(drop=True)

    groups_train = [dataset.groups[i] for i in train_groups]
    groups_val = [dataset.groups[i] for i in val_groups]

    query_ids_train = [dataset.query_ids[i] for i, m in enumerate(train_mask) if m]
    query_ids_val = [dataset.query_ids[i] for i, m in enumerate(val_mask) if m]

    return SplitData(
        X_train, X_val, y_train, y_val,
        groups_train, groups_val,
        query_ids_train, query_ids_val
    )


# --------------------------------------------------------------------------- #
# Model Training
# --------------------------------------------------------------------------- #


def train_xgb_classifier(
    split: SplitData,
    scaler: StandardScaler,
    use_gpu: bool = False,
) -> Tuple[XGBClassifier, int | None]:
    """Train XGBoost binary classifier."""
    print("\n" + "=" * 60)
    print("Training XGBClassifier (binary classification)")
    print("=" * 60)

    X_train_scaled = scaler.fit_transform(split.X_train)
    X_val_scaled = scaler.transform(split.X_val)

    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    if use_gpu:
        params["device"] = "cuda"
        params["tree_method"] = "hist"
    else:
        params["tree_method"] = "hist"

    clf = XGBClassifier(**params)
    clf.fit(
        X_train_scaled,
        split.y_train,
        eval_set=[(X_val_scaled, split.y_val)],
        verbose=50,
    )

    best_iter = getattr(clf, "best_iteration", None)
    if best_iter is None:
        best_iter = params.get("n_estimators")
    print(f"Best iteration (approx): {best_iter}")

    return clf, best_iter


def train_xgb_ranker(
    split: SplitData,
    scaler: StandardScaler,
    use_gpu: bool = False,
) -> Tuple[XGBRanker, int | None]:
    """Train XGBoost Ranker with pairwise loss."""
    print("\n" + "=" * 60)
    print("Training XGBRanker (learning-to-rank)")
    print("=" * 60)

    X_train_scaled = scaler.transform(split.X_train)
    X_val_scaled = scaler.transform(split.X_val)

    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "rank:pairwise",
        "random_state": 42,
        "n_jobs": -1,
    }

    if use_gpu:
        params["device"] = "cuda"
        params["tree_method"] = "hist"
    else:
        params["tree_method"] = "hist"

    ranker = XGBRanker(**params)
    ranker.fit(
        X_train_scaled,
        split.y_train,
        group=split.groups_train,
        eval_set=[(X_val_scaled, split.y_val)],
        eval_group=[split.groups_val],
        verbose=50,
    )

    best_iter = getattr(ranker, "best_iteration", None)
    if best_iter is None:
        best_iter = params.get("n_estimators")
    print(f"Best iteration (approx): {best_iter}")

    return ranker, best_iter


def train_lgbm_ranker(
    split: SplitData,
    scaler: StandardScaler,
) -> Tuple["lgb.LGBMRanker", int | None]:
    """Train LightGBM Ranker with lambdarank."""
    if not HAS_LIGHTGBM:
        print("LightGBM not available, skipping")
        return None, None

    print("\n" + "=" * 60)
    print("Training LGBMRanker (lambdarank)")
    print("=" * 60)

    X_train_scaled = scaler.transform(split.X_train)
    X_val_scaled = scaler.transform(split.X_val)

    ranker = lgb.LGBMRanker(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="lambdarank",
        random_state=42,
        n_jobs=-1,
        importance_type="gain",
    )

    ranker.fit(
        X_train_scaled,
        split.y_train,
        group=split.groups_train,
        eval_set=[(X_val_scaled, split.y_val)],
        eval_group=[split.groups_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = getattr(ranker, "best_iteration_", None)
    print(f"Best iteration: {best_iter}")

    return ranker, best_iter


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def compute_recall_at_k(
    split: SplitData,
    model,
    scaler: StandardScaler,
    k_values: List[int] = [1, 3, 5],
    is_ranker: bool = False,
) -> Dict[int, float]:
    """
    Compute Recall@K for ranking evaluation.

    For classifiers, uses predict_proba. For rankers, uses predict.
    """
    X_val_scaled = scaler.transform(split.X_val)

    if is_ranker:
        scores = model.predict(X_val_scaled)
    else:
        scores = model.predict_proba(X_val_scaled)[:, 1]

    # Reconstruct groups from query_ids
    query_ids = split.query_ids_val
    y_val = split.y_val.values

    # Group samples by query
    from collections import defaultdict
    groups = defaultdict(lambda: {"scores": [], "labels": []})
    for i, (score, label, qid) in enumerate(zip(scores, y_val, query_ids)):
        groups[qid]["scores"].append(score)
        groups[qid]["labels"].append(label)

    # Filter groups with at least one positive
    valid_groups = [g for g in groups.values() if sum(g["labels"]) > 0]

    recall_at_k = {}
    for k in k_values:
        hits = 0
        for group in valid_groups:
            sorted_indices = np.argsort(group["scores"])[::-1][:k]
            if any(group["labels"][i] == 1 for i in sorted_indices):
                hits += 1
        recall_at_k[k] = hits / len(valid_groups) if valid_groups else 0.0

    return recall_at_k


def compute_accuracy(
    split: SplitData,
    model,
    scaler: StandardScaler,
    is_ranker: bool = False,
) -> float:
    """Compute binary classification accuracy."""
    X_val_scaled = scaler.transform(split.X_val)

    if is_ranker:
        # For rankers, use threshold on scores
        scores = model.predict(X_val_scaled)
        threshold = np.median(scores)
        predictions = (scores > threshold).astype(int)
    else:
        predictions = model.predict(X_val_scaled)

    return (predictions == split.y_val.values).mean()


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #


def plot_feature_importance(
    model,
    feature_names: List[str],
    model_name: str,
    output_path: Path,
):
    """Plot and save feature importance."""
    if not HAS_MATPLOTLIB:
        return

    # Get importance based on model type
    if hasattr(model, "get_booster"):
        importance = model.get_booster().get_score(importance_type="gain")
        importance_named = {}
        for feat, score in importance.items():
            if feat.startswith("f"):
                idx = int(feat[1:])
                if idx < len(feature_names):
                    importance_named[feature_names[idx]] = score
            else:
                importance_named[feat] = score
    elif hasattr(model, "feature_importances_"):
        importance_named = dict(zip(feature_names, model.feature_importances_))
    else:
        return

    if not importance_named:
        return

    sorted_features = sorted(importance_named.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_features)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), values, align="center")
    plt.yticks(range(len(names)), names)
    plt.xlabel("Gain")
    plt.title(f"Feature Importance - {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_model_comparison(results: Dict[str, Dict]) -> str:
    """Print comparison table and return best model name."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    headers = ["Model", "Recall@1", "Recall@3", "Recall@5", "Accuracy", "Best Iter"]
    print(f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10}")
    print("-" * 70)

    best_model = None
    best_recall1 = 0

    for name, metrics in results.items():
        r1 = metrics.get("recall@1", 0)
        r3 = metrics.get("recall@3", 0)
        r5 = metrics.get("recall@5", 0)
        acc = metrics.get("accuracy", 0)
        best_iter = metrics.get("best_iteration", "N/A")

        print(f"{name:<20} {r1*100:>8.1f}% {r3*100:>8.1f}% {r5*100:>8.1f}% {acc*100:>8.1f}% {str(best_iter):>10}")

        if r1 > best_recall1:
            best_recall1 = r1
            best_model = name

    print("-" * 70)
    print(f"Best model (by Recall@1): {best_model}")
    return best_model


# --------------------------------------------------------------------------- #
# Model Saving
# --------------------------------------------------------------------------- #


def save_models(
    models: Dict[str, Tuple],
    scaler: StandardScaler,
    feature_names: List[str],
    best_model_name: str,
):
    """Save all trained models and metadata with timestamp suffix."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp suffix for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Model version: {timestamp}")

    # Save scaler (with timestamp)
    scaler_path = MODEL_DIR / f"xgb_matcher_scaler_{timestamp}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    # Save each model (with timestamp)
    for name, (model, best_iter) in models.items():
        if model is None:
            continue

        safe_name = name.lower().replace(" ", "_")

        if "lgbm" in name.lower():
            model_path = MODEL_DIR / f"{safe_name}_{timestamp}.txt"
            model.booster_.save_model(str(model_path))
        else:
            model_path = MODEL_DIR / f"{safe_name}_{timestamp}.json"
            model.get_booster().save_model(str(model_path))

        print(f"Saved {name} to {model_path}")

    # Save best model with timestamp
    best_model, best_iter = models[best_model_name]
    if "lgbm" in best_model_name.lower():
        default_path = MODEL_DIR / f"xgb_matcher_{timestamp}.txt"
        best_model.booster_.save_model(str(default_path))
    else:
        default_path = MODEL_DIR / f"xgb_matcher_{timestamp}.json"
        best_model.get_booster().save_model(str(default_path))
    print(f"Saved best model ({best_model_name}) as: {default_path}")

    # Save feature metadata (with timestamp)
    meta = {
        "timestamp": timestamp,
        "feature_order": feature_names,
        "n_features": len(feature_names),
        "best_model": best_model_name,
        "best_iteration": best_iter,
    }
    meta_path = MODEL_DIR / f"xgb_matcher_features_{timestamp}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved feature metadata to {meta_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    use_gpu = check_gpu_available()
    if use_gpu:
        print("GPU detected - will use CUDA acceleration")
    else:
        print("No GPU detected - will use CPU")

    # Prepare dataset with group structure
    dataset = prepare_ranking_dataset()
    if dataset.X.empty:
        raise SystemExit("No training data after filtering; check inputs.")

    # Split by groups
    split = split_by_groups(dataset, test_size=0.2)
    print(f"\nTraining set: {len(split.X_train)} samples ({len(split.groups_train)} groups)")
    print(f"Validation set: {len(split.X_val)} samples ({len(split.groups_val)} groups)")

    # Fit scaler once
    scaler = StandardScaler()
    scaler.fit(split.X_train)

    # Train all models
    models = {}
    results = {}

    # 1. XGBClassifier
    clf, clf_iter = train_xgb_classifier(split, scaler, use_gpu)
    models["XGBClassifier"] = (clf, clf_iter)
    recall_clf = compute_recall_at_k(split, clf, scaler, is_ranker=False)
    acc_clf = compute_accuracy(split, clf, scaler, is_ranker=False)
    results["XGBClassifier"] = {
        "recall@1": recall_clf[1],
        "recall@3": recall_clf[3],
        "recall@5": recall_clf[5],
        "accuracy": acc_clf,
        "best_iteration": clf_iter,
    }
    print(f"XGBClassifier: Recall@1={recall_clf[1]*100:.1f}%, Recall@3={recall_clf[3]*100:.1f}%, Recall@5={recall_clf[5]*100:.1f}%")

    # 2. XGBRanker
    ranker, ranker_iter = train_xgb_ranker(split, scaler, use_gpu)
    models["XGBRanker"] = (ranker, ranker_iter)
    recall_ranker = compute_recall_at_k(split, ranker, scaler, is_ranker=True)
    acc_ranker = compute_accuracy(split, ranker, scaler, is_ranker=True)
    results["XGBRanker"] = {
        "recall@1": recall_ranker[1],
        "recall@3": recall_ranker[3],
        "recall@5": recall_ranker[5],
        "accuracy": acc_ranker,
        "best_iteration": ranker_iter,
    }
    print(f"XGBRanker: Recall@1={recall_ranker[1]*100:.1f}%, Recall@3={recall_ranker[3]*100:.1f}%, Recall@5={recall_ranker[5]*100:.1f}%")

    # 3. LGBMRanker
    if HAS_LIGHTGBM:
        lgbm_ranker, lgbm_iter = train_lgbm_ranker(split, scaler)
        if lgbm_ranker is not None:
            models["LGBMRanker"] = (lgbm_ranker, lgbm_iter)
            recall_lgbm = compute_recall_at_k(split, lgbm_ranker, scaler, is_ranker=True)
            acc_lgbm = compute_accuracy(split, lgbm_ranker, scaler, is_ranker=True)
            results["LGBMRanker"] = {
                "recall@1": recall_lgbm[1],
                "recall@3": recall_lgbm[3],
                "recall@5": recall_lgbm[5],
                "accuracy": acc_lgbm,
                "best_iteration": lgbm_iter,
            }
            print(f"LGBMRanker: Recall@1={recall_lgbm[1]*100:.1f}%, Recall@3={recall_lgbm[3]*100:.1f}%, Recall@5={recall_lgbm[5]*100:.1f}%")

    # Compare and select best
    best_model_name = print_model_comparison(results)

    # Plot feature importance for each model
    feature_names = list(split.X_train.columns)
    for name, (model, _) in models.items():
        if model is not None:
            safe_name = name.lower().replace(" ", "_")
            plot_feature_importance(
                model, feature_names, name,
                MODEL_DIR / f"feature_importance_{safe_name}.png"
            )

    # Save all models
    save_models(models, scaler, feature_names, best_model_name)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: {best_model_name}")
    print(f"Target Recall@1 > 85%: {'ACHIEVED' if results[best_model_name]['recall@1'] > 0.85 else 'NOT YET'}")


if __name__ == "__main__":
    main()
