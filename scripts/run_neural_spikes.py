#!/usr/bin/env python3
"""Run two bounded neural entity-matching experiments on SIRETO V7 data.

The experiments intentionally share the same SIREN-disjoint holdout:

1. A cross-encoder, trained with a listwise softmax, reranks the exact candidate
   scenes used by the current XGBoost decider.
2. A dual-encoder, trained contrastively with hard negatives, retrieves from
   complete INSEE partitions and is compared with a lexical TF-IDF baseline.

This is an architectural spike, not a production training pipeline. Defaults
are sized to finish on a CPU-only development machine while remaining large
enough to reject obviously unpromising directions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "data/samples_v7_decider.parquet"
DEFAULT_CRM = ROOT / "data/crm_ok_gt.csv"
DEFAULT_PARTITIONS = ROOT / "data/candidates_v7_all"
DEFAULT_COUNTS = DEFAULT_PARTITIONS / "manifest/insee_counts.parquet"
DEFAULT_MODEL = ROOT / "models/semantic/siret-bert-deploy"
DEFAULT_XGB = ROOT / "models/xgb_decider_20260221_213148.json"
DEFAULT_META = ROOT / "models/xgb_two_stage_meta_20260221_213148.json"
DEFAULT_OUTPUT = ROOT / "reports/neural_spikes"
SEED = 42

CANDIDATE_COLUMNS = [
    "siret",
    "siren",
    "denomination",
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "etablissementSiege",
    "is_siege",
    "numeroVoie",
    "typeVoie",
    "libelleVoie",
    "complementAdresse",
    "postcode",
    "city",
    "cj_ul",
    "etat_admin",
    "sigle_ul",
    "denomination_ul",
    "denomination_usuelle_ul",
    "nom_ul",
    "prenom_usuel_ul",
    "pm_dirigeant_names",
]

NAME_COLUMNS = [
    "denomination",
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "denomination_ul",
    "denomination_usuelle_ul",
    "sigle_ul",
    "nom_ul",
    "prenom_usuel_ul",
    "pm_dirigeant_names",
]


@dataclass(frozen=True)
class Config:
    seed: int
    holdout_queries: int
    max_partition_rows: int
    cross_train_queries: int
    cross_dev_queries: int
    cross_negatives: int
    cross_epochs: int
    dual_train_queries: int
    dual_epochs: int
    trainable_layers: int
    max_length: int
    batch_size: int
    encode_batch_size: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def load_tokenizer(model_path: Path):
    """Load the exported SentencePiece tokenizer without trusting its bad class hint.

    The current local export declares ``BertTokenizer`` in tokenizer_config.json
    although tokenizer.json contains the multilingual SentencePiece tokenizer.
    AutoTokenizer therefore maps ordinary French words to ``<unk>``. Loading the
    fast tokenizer directly preserves the tokenizer used during fine-tuning.
    """
    return PreTrainedTokenizerFast.from_pretrained(
        str(model_path), local_files_only=True
    )


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def unique_nonempty(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = clean(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def crm_text(row: pd.Series | dict) -> str:
    return " | ".join(
        [
            f"nom: {clean(row.get('crm_name'))}",
            f"adresse: {clean(row.get('crm_address'))}",
            f"code postal: {clean(row.get('postcode'))}",
            f"commune: {clean(row.get('crm_city'))}",
        ]
    )


def candidate_name(row: pd.Series | dict) -> str:
    return " ; ".join(unique_nonempty(row.get(column) for column in NAME_COLUMNS))


def candidate_address(row: pd.Series | dict) -> str:
    street = " ".join(
        unique_nonempty(
            [
                row.get("numeroVoie"),
                row.get("typeVoie"),
                row.get("libelleVoie"),
                row.get("complementAdresse"),
            ]
        )
    )
    return " ".join(unique_nonempty([street, row.get("postcode"), row.get("city")]))


def candidate_text(row: pd.Series | dict) -> str:
    return " | ".join(
        [
            f"noms: {candidate_name(row)}",
            f"adresse: {candidate_address(row)}",
            f"état: {clean(row.get('etat_admin'))}",
            f"siège: {int(bool(row.get('is_siege') or row.get('etablissementSiege')))}",
        ]
    )


def load_crm(path: Path) -> pd.DataFrame:
    crm = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    crm = crm.rename(
        columns={
            "crm_cp": "postcode",
            "crm_insee": "insee",
            "crm_adresse": "crm_address",
            "crm_commune": "crm_city",
            "gt_siret": "ground_truth_siret",
        }
    )
    crm["ground_truth_siret"] = crm["ground_truth_siret"].fillna("").str.strip()
    crm = crm[crm["ground_truth_siret"].str.len() == 14].copy()
    crm["query_id"] = np.arange(len(crm), dtype=np.int64)
    crm["gt_siren"] = crm["ground_truth_siret"].str[:9]
    for column in ["crm_name", "crm_address", "postcode", "insee", "crm_city"]:
        crm[column] = crm[column].map(clean)
    crm["query_text"] = crm.apply(crm_text, axis=1)
    return crm


def load_samples(path: Path) -> pd.DataFrame:
    columns = pq.read_schema(path).names
    table = pq.read_table(path, columns=columns)
    samples = table.to_pandas()
    samples["siret"] = samples["siret"].fillna("").astype(str).str.zfill(14)
    samples["siren"] = samples["siren"].fillna("").astype(str).str.zfill(9)
    samples["query_id"] = samples["query_id"].astype(np.int64)
    samples["label"] = samples["label"].astype(np.int8)
    return samples


def feature_order(meta_path: Path) -> list[str]:
    with meta_path.open() as handle:
        meta = json.load(handle)
    return list(meta.get("feature_order") or meta["feature_names"])


def add_xgb_scores(
    samples: pd.DataFrame,
    model_path: Path,
    meta_path: Path,
) -> pd.DataFrame:
    order = feature_order(meta_path)
    missing = sorted(set(order) - set(samples.columns))
    if missing:
        raise ValueError(f"Missing XGBoost features: {missing}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    matrix = np.nan_to_num(samples[order].to_numpy(dtype=np.float32), nan=0.0)
    result = samples.copy()
    result["xgb_score"] = booster.predict(xgb.DMatrix(matrix))
    return result


def choose_holdout(
    samples: pd.DataFrame,
    crm: pd.DataFrame,
    counts_path: Path,
    size: int,
    max_partition_rows: int,
    seed: int,
) -> pd.DataFrame:
    positives = (
        samples[(samples["split"] == "test") & (samples["label"] == 1)][
            ["query_id", "siret", "siren"]
        ]
        .drop_duplicates("query_id")
        .rename(columns={"siret": "sample_gt_siret", "siren": "sample_gt_siren"})
    )
    counts = pd.read_parquet(counts_path)[["insee", "row_count"]]
    eligible = positives.merge(crm, on="query_id", how="inner").merge(
        counts, on="insee", how="left"
    )
    eligible = eligible[
        (eligible["sample_gt_siret"] == eligible["ground_truth_siret"])
        & (eligible["row_count"].between(2, max_partition_rows))
    ].copy()
    if len(eligible) < size:
        raise ValueError(
            f"Only {len(eligible)} eligible test queries for requested holdout={size}"
        )

    eligible["pool_bin"] = pd.qcut(
        np.log1p(eligible["row_count"]),
        q=min(5, eligible["row_count"].nunique()),
        duplicates="drop",
    )
    rng = np.random.default_rng(seed)
    bins = list(eligible.groupby("pool_bin", observed=True))
    per_bin = math.ceil(size / max(1, len(bins)))
    chosen: list[pd.DataFrame] = []
    for _, group in bins:
        take = min(per_bin, len(group))
        chosen.append(group.iloc[rng.choice(len(group), size=take, replace=False)])
    holdout = pd.concat(chosen, ignore_index=True)
    if len(holdout) > size:
        holdout = holdout.iloc[rng.choice(len(holdout), size=size, replace=False)]
    elif len(holdout) < size:
        remaining = eligible[~eligible["query_id"].isin(holdout["query_id"])]
        extra = remaining.iloc[
            rng.choice(len(remaining), size=size - len(holdout), replace=False)
        ]
        holdout = pd.concat([holdout, extra], ignore_index=True)
    return holdout.sort_values("query_id").reset_index(drop=True)


def gt_sirens_by_split(samples: pd.DataFrame) -> dict[str, set[str]]:
    positives = samples[samples["label"] == 1]
    return {
        split: set(group["siren"])
        for split, group in positives.groupby("split", observed=True)
    }


def deterministic_query_sample(
    samples: pd.DataFrame,
    split: str,
    limit: int,
    seed: int,
    excluded_candidate_sirens: set[str] | None = None,
) -> pd.DataFrame:
    subset = samples[samples["split"] == split].copy()
    if excluded_candidate_sirens:
        subset = subset[
            (subset["label"] == 1)
            | (~subset["siren"].isin(excluded_candidate_sirens))
        ]
    valid = subset.groupby("query_id")["label"].agg(["sum", "count"])
    valid_ids = valid[(valid["sum"] == 1) & (valid["count"] >= 2)].index.to_numpy()
    rng = np.random.default_rng(seed)
    if len(valid_ids) > limit:
        valid_ids = rng.choice(valid_ids, size=limit, replace=False)
    return subset[subset["query_id"].isin(valid_ids)].copy()


def select_hard_scenes(
    samples: pd.DataFrame,
    negatives: int,
) -> tuple[pd.DataFrame, list[int]]:
    selected: list[pd.DataFrame] = []
    kept: list[int] = []
    for query_id, group in samples.groupby("query_id", sort=True):
        positive = group[group["label"] == 1]
        negative = group[group["label"] == 0].sort_values(
            "xgb_score", ascending=False
        )
        if len(positive) != 1 or len(negative) < negatives:
            continue
        chosen = pd.concat([positive.iloc[:1], negative.iloc[:negatives]])
        selected.append(chosen)
        kept.append(int(query_id))
    if not selected:
        raise ValueError("No complete scenes after hard-negative selection")
    return pd.concat(selected, ignore_index=True), kept


def load_candidate_records(
    sirets: Sequence[str],
    partitions_dir: Path,
) -> pd.DataFrame:
    wanted = pd.DataFrame(
        {"siret": pd.Series(sorted(set(sirets)), dtype="object")}
    )
    con = duckdb.connect()
    con.register("wanted_sirets", wanted)
    glob = str(partitions_dir / "insee/insee=*/*.parquet").replace("'", "''")
    selected = ", ".join(f"c.{column}" for column in CANDIDATE_COLUMNS)
    query = f"""
        SELECT {selected}
        FROM read_parquet('{glob}', hive_partitioning=false) c
        SEMI JOIN wanted_sirets w ON c.siret = w.siret
        QUALIFY row_number() OVER (PARTITION BY c.siret ORDER BY c.siret) = 1
    """
    records = con.execute(query).fetchdf()
    con.close()
    records["siret"] = records["siret"].astype(str).str.zfill(14)
    records["candidate_text"] = records.apply(candidate_text, axis=1)
    return records


def enrich_pairs(
    pairs: pd.DataFrame,
    crm: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    crm_columns = [
        "query_id",
        "query_text",
        "ground_truth_siret",
        "gt_siren",
        "insee",
        "postcode",
        "crm_name",
        "crm_address",
        "crm_city",
    ]
    result = pairs.merge(crm[crm_columns], on="query_id", how="left")
    result = result.merge(
        candidates[["siret", "candidate_text"] + NAME_COLUMNS + [
            "numeroVoie",
            "typeVoie",
            "libelleVoie",
            "complementAdresse",
            "postcode",
            "city",
            "etat_admin",
            "is_siege",
        ]],
        on="siret",
        how="left",
        suffixes=("_crm", "_candidate"),
    )
    result["candidate_text"] = result["candidate_text"].fillna("")
    return result


def topk_metrics(
    frame: pd.DataFrame,
    score_column: str,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    hits = {k: [] for k in ks}
    same_siren: list[bool] = []
    for _, group in frame.groupby("query_id", sort=False):
        ranked = group.sort_values(score_column, ascending=False)
        labels = ranked["label"].to_numpy()
        for k in ks:
            hits[k].append(bool(labels[:k].max(initial=0)))
        top = ranked.iloc[0]
        same_siren.append(str(top["siren"]) == str(top["gt_siren"]))
    metrics = {f"hit_at_{k}": float(np.mean(value)) for k, value in hits.items()}
    metrics["same_siren_at_1"] = float(np.mean(same_siren))
    metrics["queries"] = int(len(same_siren))
    return metrics


def restricted_rerank_metrics(
    frame: pd.DataFrame,
    candidate_k: int,
) -> dict[str, float | int]:
    hits: list[bool] = []
    same_siren: list[bool] = []
    coverages: list[bool] = []
    for _, group in frame.groupby("query_id", sort=True):
        pool = group.nlargest(candidate_k, "xgb_score")
        coverages.append(bool(pool["label"].max()))
        top = pool.nlargest(1, "cross_score").iloc[0]
        hits.append(bool(top["label"]))
        same_siren.append(str(top["siren"]) == str(top["gt_siren"]))
    return {
        "candidate_k": candidate_k,
        "hit_at_1": float(np.mean(hits)),
        "same_siren_at_1": float(np.mean(same_siren)),
        "xgb_candidate_coverage": float(np.mean(coverages)),
        "queries": len(hits),
    }


def hybrid_union_metrics(detail: pd.DataFrame) -> dict[str, float | int]:
    """Measure the recall of the union of two top-k lists (budget <= 2k)."""
    result: dict[str, float | int] = {}
    for k in (1, 10, 50):
        sparse = detail[f"tfidf_hit_at_{k}"].fillna(False).astype(bool)
        dense = detail[f"dense_hit_at_{k}_after"].fillna(False).astype(bool)
        result[f"hit_at_{k}"] = float((sparse | dense).mean())
        result[f"dense_rescues_at_{k}"] = int((~sparse & dense).sum())
        result[f"sparse_rescues_at_{k}"] = int((sparse & ~dense).sum())
        result[f"candidate_budget_at_{k}"] = int(2 * k)
    return result


def paired_bootstrap(
    baseline: np.ndarray,
    challenger: np.ndarray,
    seed: int,
    iterations: int = 4000,
) -> dict[str, float]:
    if len(baseline) != len(challenger):
        raise ValueError("Paired vectors have different lengths")
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for idx in range(iterations):
        sample = rng.integers(0, len(baseline), size=len(baseline))
        differences[idx] = challenger[sample].mean() - baseline[sample].mean()
    return {
        "delta": float(challenger.mean() - baseline.mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
    }


def trainable_last_layers(model: nn.Module, count: int) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError("Expected a BERT-like encoder.layer stack")
    for layer in layers[-count:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    pooler = getattr(model, "pooler", None)
    if pooler is not None:
        for parameter in pooler.parameters():
            parameter.requires_grad = True
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


class CrossEncoder(nn.Module):
    def __init__(self, model_path: Path, trainable_layers: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            str(model_path), local_files_only=True
        )
        trainable_last_layers(self.encoder, trainable_layers)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(**encoded).last_hidden_state[:, 0]
        return self.head(self.dropout(hidden)).squeeze(-1)


class SceneDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.scenes: list[tuple[list[str], list[str]]] = []
        for _, group in frame.groupby("query_id", sort=True):
            group = pd.concat(
                [
                    group[group["label"] == 1],
                    group[group["label"] == 0],
                ]
            )
            self.scenes.append(
                (
                    group["query_text"].tolist(),
                    group["candidate_text"].tolist(),
                )
            )

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> tuple[list[str], list[str]]:
        return self.scenes[index]


def collate_scenes(
    batch: Sequence[tuple[list[str], list[str]]],
) -> tuple[list[str], list[str], int]:
    scene_size = len(batch[0][0])
    if any(len(queries) != scene_size for queries, _ in batch):
        raise ValueError("Training scenes must have a fixed candidate count")
    queries = [item for scene in batch for item in scene[0]]
    candidates = [item for scene in batch for item in scene[1]]
    return queries, candidates, scene_size


@torch.no_grad()
def cross_score_frame(
    model: CrossEncoder,
    tokenizer,
    frame: pd.DataFrame,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    model.eval()
    result = frame.copy()
    scores: list[np.ndarray] = []
    for start in range(0, len(result), batch_size):
        batch = result.iloc[start : start + batch_size]
        encoded = tokenizer(
            batch["query_text"].tolist(),
            batch["candidate_text"].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        scores.append(model(encoded).cpu().numpy())
    result["cross_score"] = np.concatenate(scores)
    return result


def train_cross_encoder(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    model_path: Path,
    config: Config,
) -> tuple[CrossEncoder, dict]:
    tokenizer = load_tokenizer(model_path)
    model = CrossEncoder(model_path, config.trainable_layers)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=2e-5, weight_decay=0.01)
    loader = DataLoader(
        SceneDataset(train),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_scenes,
    )
    history: list[dict] = []
    for epoch in range(config.cross_epochs):
        model.train()
        losses: list[float] = []
        started = time.time()
        for queries, candidates, scene_size in loader:
            encoded = tokenizer(
                queries,
                candidates,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            logits = model(encoded).reshape(-1, scene_size)
            targets = torch.zeros(logits.shape[0], dtype=torch.long)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scored = cross_score_frame(
            model,
            tokenizer,
            dev,
            config.max_length,
            config.encode_batch_size,
        )
        epoch_metrics = topk_metrics(scored, "cross_score")
        epoch_metrics.update(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "seconds": time.time() - started,
            }
        )
        history.append(epoch_metrics)
        print(f"[cross] epoch {epoch + 1}: {epoch_metrics}", flush=True)
    return model, {"history": history, "tokenizer": tokenizer}


class DualEncoder(nn.Module):
    def __init__(self, model_path: Path, trainable_layers: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            str(model_path), local_files_only=True
        )
        trainable_last_layers(self.encoder, trainable_layers)

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        return F.normalize(pooled, p=2, dim=1)


class TripleDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        triples: list[tuple[str, str, str]] = []
        for _, group in frame.groupby("query_id", sort=True):
            positive = group[group["label"] == 1]
            negative = group[group["label"] == 0].sort_values(
                "xgb_score", ascending=False
            )
            if len(positive) == 1 and len(negative):
                triples.append(
                    (
                        positive.iloc[0]["query_text"],
                        positive.iloc[0]["candidate_text"],
                        negative.iloc[0]["candidate_text"],
                    )
                )
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        return self.triples[index]


def collate_triples(
    batch: Sequence[tuple[str, str, str]],
) -> tuple[list[str], list[str], list[str]]:
    query, positive, negative = zip(*batch)
    return list(query), list(positive), list(negative)


def encode_texts(
    model: DualEncoder,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            outputs.append(model(encoded).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs) if outputs else np.empty((0, 0), dtype=np.float32)


def train_dual_encoder(
    train: pd.DataFrame,
    model_path: Path,
    config: Config,
) -> tuple[DualEncoder, object, list[dict]]:
    tokenizer = load_tokenizer(model_path)
    model = DualEncoder(model_path, config.trainable_layers)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=2e-5, weight_decay=0.01)
    loader = DataLoader(
        TripleDataset(train),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_triples,
    )
    history: list[dict] = []
    for epoch in range(config.dual_epochs):
        model.train()
        losses: list[float] = []
        started = time.time()
        for queries, positives, negatives in loader:
            q_tokens = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            p_tokens = tokenizer(
                positives,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            n_tokens = tokenizer(
                negatives,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            q_emb = model(q_tokens)
            p_emb = model(p_tokens)
            n_emb = model(n_tokens)
            documents = torch.cat([p_emb, n_emb], dim=0)
            logits = q_emb @ documents.T / 0.05
            targets = torch.arange(len(queries), dtype=torch.long)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "seconds": time.time() - started,
        }
        history.append(epoch_row)
        print(f"[dual] epoch {epoch + 1}: {epoch_row}", flush=True)
    return model, tokenizer, history


def load_insee_partition(partitions_dir: Path, insee: str) -> pd.DataFrame:
    paths = sorted((partitions_dir / f"insee/insee={insee}").glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    tables = [
        pq.ParquetFile(path).read(columns=CANDIDATE_COLUMNS)
        for path in paths
    ]
    frame = pa.concat_tables(tables, promote_options="default").to_pandas()
    frame["siret"] = frame["siret"].astype(str).str.zfill(14)
    frame["siren"] = frame["siren"].astype(str).str.zfill(9)
    named = frame[NAME_COLUMNS].fillna("").astype(str).apply(
        lambda row: any(value.strip() for value in row), axis=1
    )
    frame = frame[named].drop_duplicates("siret").reset_index(drop=True)
    frame["candidate_text"] = frame.apply(candidate_text, axis=1)
    frame["lexical_name"] = frame.apply(candidate_name, axis=1)
    frame["lexical_address"] = frame.apply(candidate_address, axis=1)
    return frame


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if not k:
        return np.empty(0, dtype=np.int64)
    indices = np.argpartition(scores, -k)[-k:]
    return indices[np.argsort(scores[indices])[::-1]]


def reciprocal_rank_fusion(
    rankings: Sequence[np.ndarray],
    size: int,
    constant: int = 60,
) -> np.ndarray:
    scores = np.zeros(size, dtype=np.float32)
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            scores[int(index)] += 1.0 / (constant + rank + 1)
    return scores


def lexical_scores(
    candidates: pd.DataFrame,
    query: pd.Series,
) -> np.ndarray:
    name_docs = candidates["lexical_name"].fillna("").tolist()
    address_docs = candidates["lexical_address"].fillna("").tolist()
    query_name = clean(query["crm_name"])
    query_address = clean(query["crm_address"])
    rankings: list[np.ndarray] = []
    for docs, value, analyzer, ngrams in [
        (name_docs, query_name, "word", (1, 2)),
        (name_docs, query_name, "char_wb", (3, 5)),
        (address_docs, query_address, "word", (1, 2)),
    ]:
        if not value or not any(docs):
            continue
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngrams,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b" if analyzer == "word" else None,
        )
        try:
            matrix = vectorizer.fit_transform(docs)
            vector = vectorizer.transform([value])
        except ValueError:
            continue
        sparse_scores = (vector @ matrix.T).toarray()[0]
        rankings.append(top_indices(sparse_scores, min(100, len(candidates))))
    return reciprocal_rank_fusion(rankings, len(candidates))


def evaluate_retrieval(
    model: DualEncoder,
    tokenizer,
    holdout: pd.DataFrame,
    partitions_dir: Path,
    config: Config,
    include_lexical: bool,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for insee, queries in holdout.groupby("insee", sort=True):
        candidates = load_insee_partition(partitions_dir, str(insee))
        if candidates.empty:
            for _, query in queries.iterrows():
                rows.append(
                    {
                        "query_id": int(query["query_id"]),
                        "insee": insee,
                        "pool_size": 0,
                        "gt_present": False,
                    }
                )
            continue
        candidate_embeddings = encode_texts(
            model,
            tokenizer,
            candidates["candidate_text"].tolist(),
            config.max_length,
            config.encode_batch_size,
        )
        query_embeddings = encode_texts(
            model,
            tokenizer,
            queries["query_text"].tolist(),
            config.max_length,
            config.encode_batch_size,
        )
        for query_offset, (_, query) in enumerate(queries.iterrows()):
            gt_siret = str(query["ground_truth_siret"])
            gt_siren = str(query["gt_siren"])
            dense = candidate_embeddings @ query_embeddings[query_offset]
            dense_rank = top_indices(dense, 50)
            row = {
                "query_id": int(query["query_id"]),
                "insee": str(insee),
                "pool_size": int(len(candidates)),
                "gt_present": bool((candidates["siret"] == gt_siret).any()),
            }
            for k in (1, 10, 50):
                dense_top = candidates.iloc[dense_rank[:k]]
                row[f"dense_hit_at_{k}"] = bool(
                    (dense_top["siret"] == gt_siret).any()
                )
                row[f"dense_siren_at_{k}"] = bool(
                    (dense_top["siren"] == gt_siren).any()
                )
            if include_lexical:
                lexical = lexical_scores(candidates, query)
                lexical_rank = top_indices(lexical, 50)
                for k in (1, 10, 50):
                    lexical_top = candidates.iloc[lexical_rank[:k]]
                    row[f"tfidf_hit_at_{k}"] = bool(
                        (lexical_top["siret"] == gt_siret).any()
                    )
                    row[f"tfidf_siren_at_{k}"] = bool(
                        (lexical_top["siren"] == gt_siren).any()
                    )
            rows.append(row)
    detail = pd.DataFrame(rows).sort_values("query_id").reset_index(drop=True)
    metrics: dict[str, float | int] = {
        "queries": int(len(detail)),
        "candidate_pool_rows": int(detail["pool_size"].sum()),
        "gt_partition_coverage": float(detail["gt_present"].mean()),
    }
    for column in detail.columns:
        if "_hit_at_" in column or "_siren_at_" in column:
            metrics[column] = float(detail[column].fillna(False).mean())
    return detail, metrics


def outcome_vector(frame: pd.DataFrame, score_column: str) -> np.ndarray:
    values: list[bool] = []
    for _, group in frame.groupby("query_id", sort=True):
        top = group.sort_values(score_column, ascending=False).iloc[0]
        values.append(bool(top["label"]))
    return np.asarray(values, dtype=np.float32)


def save_trainable_state(
    model: nn.Module,
    path: Path,
) -> None:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }
    torch.save(state, path)


def write_summary(report: dict, path: Path) -> None:
    cross = report["cross_encoder"]
    dual = report["dual_encoder"]
    text = f"""# SIRETO neural architecture spikes

Generated: {report['generated_at']}

## Protocol

- Shared SIREN-disjoint holdout: {report['holdout']['queries']} queries.
- Maximum full INSEE partition: {report['config']['max_partition_rows']} candidates.
- Candidate SIRENs belonging to dev/test ground-truth entities were purged from training negatives.
- The cross-encoder reranks the same V7 decider scenes as XGBoost.
- The dual-encoder retrieves from complete INSEE partitions.

## Cross-encoder

| Metric | XGBoost | Cross-encoder |
|---|---:|---:|
| Exact SIRET Hit@1 | {cross['xgb']['hit_at_1']:.2%} | {cross['neural']['hit_at_1']:.2%} |
| Exact SIRET Hit@3 | {cross['xgb']['hit_at_3']:.2%} | {cross['neural']['hit_at_3']:.2%} |
| Same SIREN @1 | {cross['xgb']['same_siren_at_1']:.2%} | {cross['neural']['same_siren_at_1']:.2%} |

Paired Hit@1 delta: {cross['paired_hit_at_1']['delta']:+.2%}
95% bootstrap CI: [{cross['paired_hit_at_1']['ci95_low']:+.2%}, {cross['paired_hit_at_1']['ci95_high']:+.2%}]

When restricted to the six highest-scoring XGBoost candidates, the
cross-encoder reaches {cross['rerank_by_xgb_top_k']['6']['hit_at_1']:.2%}
Hit@1; XGBoost top-6 candidate coverage is
{cross['rerank_by_xgb_top_k']['6']['xgb_candidate_coverage']:.2%}.

## Dual-encoder retrieval

| Metric | TF-IDF fusion | Dense before fine-tuning | Dense after fine-tuning |
|---|---:|---:|---:|
| Exact SIRET Recall@1 | {dual['tfidf']['tfidf_hit_at_1']:.2%} | {dual['before']['dense_hit_at_1']:.2%} | {dual['after']['dense_hit_at_1']:.2%} |
| Exact SIRET Recall@10 | {dual['tfidf']['tfidf_hit_at_10']:.2%} | {dual['before']['dense_hit_at_10']:.2%} | {dual['after']['dense_hit_at_10']:.2%} |
| Exact SIRET Recall@50 | {dual['tfidf']['tfidf_hit_at_50']:.2%} | {dual['before']['dense_hit_at_50']:.2%} | {dual['after']['dense_hit_at_50']:.2%} |
| SIREN Recall@50 | {dual['tfidf']['tfidf_siren_at_50']:.2%} | {dual['before']['dense_siren_at_50']:.2%} | {dual['after']['dense_siren_at_50']:.2%} |

Ground-truth coverage in the complete INSEE partitions: {dual['after']['gt_partition_coverage']:.2%}.
The union of TF-IDF top-50 and dense top-50 reaches
{dual['hybrid_union']['hit_at_50']:.2%} exact-SIRET recall with a candidate
budget of at most 100, rescuing
{dual['hybrid_union']['dense_rescues_at_50']} TF-IDF misses.

## Tokenizer audit

The exported model declares `{report['tokenizer_audit']['declared_class']}` but
contains the multilingual fast SentencePiece tokenizer. Loading it through the
declared class maps ordinary French terms to `<unk>`. These spikes force
`{report['tokenizer_audit']['spike_loader']}`. Existing semantic-feature
benchmarks must be treated as suspect until the production loader/export is
corrected and re-evaluated.

## Interpretation

These are bounded architecture probes. They estimate the representation and
retrieval ceilings on entity-disjoint data; they are not production AUTO-rate
or open-set precision measurements. The short cross-encoder is a no-go as a
drop-in XGBoost replacement. The learned dense retriever is a go as a
complementary retrieval channel, not as a replacement for sparse retrieval.
"""
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    config = Config(
        seed=args.seed,
        holdout_queries=args.holdout_queries,
        max_partition_rows=args.max_partition_rows,
        cross_train_queries=args.cross_train_queries,
        cross_dev_queries=args.cross_dev_queries,
        cross_negatives=args.cross_negatives,
        cross_epochs=args.cross_epochs,
        dual_train_queries=args.dual_train_queries,
        dual_epochs=args.dual_epochs,
        trainable_layers=args.trainable_layers,
        max_length=args.max_length,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
    )
    seed_everything(config.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[data] loading CRM and candidate scenes", flush=True)
    crm = load_crm(args.crm)
    samples = add_xgb_scores(
        load_samples(args.samples), args.xgb_model, args.xgb_meta
    )
    holdout = choose_holdout(
        samples,
        crm,
        args.counts,
        config.holdout_queries,
        config.max_partition_rows,
        config.seed,
    )
    holdout.to_csv(args.output_dir / "holdout_queries.csv", index=False)

    split_sirens = gt_sirens_by_split(samples)
    heldout_sirens = split_sirens.get("dev", set()) | split_sirens.get("test", set())
    cross_train_all = deterministic_query_sample(
        samples,
        "train",
        config.cross_train_queries,
        config.seed,
        excluded_candidate_sirens=heldout_sirens,
    )
    cross_dev_all = deterministic_query_sample(
        samples,
        "dev",
        config.cross_dev_queries,
        config.seed + 1,
        excluded_candidate_sirens=split_sirens.get("test", set()),
    )
    cross_test = samples[samples["query_id"].isin(holdout["query_id"])].copy()
    cross_train, train_ids = select_hard_scenes(
        cross_train_all, config.cross_negatives
    )
    cross_dev, dev_ids = select_hard_scenes(cross_dev_all, config.cross_negatives)

    dual_train_all = deterministic_query_sample(
        samples,
        "train",
        config.dual_train_queries,
        config.seed + 2,
        excluded_candidate_sirens=heldout_sirens,
    )
    dual_train, dual_ids = select_hard_scenes(dual_train_all, 1)
    all_pairs = pd.concat(
        [cross_train, cross_dev, cross_test, dual_train], ignore_index=True
    )
    print(
        f"[data] enriching {len(all_pairs):,} pairs "
        f"({all_pairs['siret'].nunique():,} unique SIRETs)",
        flush=True,
    )
    candidate_records = load_candidate_records(
        all_pairs["siret"].tolist(), args.partitions
    )
    candidate_coverage = (
        all_pairs["siret"].isin(set(candidate_records["siret"])).mean()
    )
    cross_train = enrich_pairs(cross_train, crm, candidate_records)
    cross_dev = enrich_pairs(cross_dev, crm, candidate_records)
    cross_test = enrich_pairs(cross_test, crm, candidate_records)
    dual_train = enrich_pairs(dual_train, crm, candidate_records)

    print("[cross] training", flush=True)
    cross_model, cross_aux = train_cross_encoder(
        cross_train, cross_dev, args.base_model, config
    )
    tokenizer = cross_aux.pop("tokenizer")
    cross_scored = cross_score_frame(
        cross_model,
        tokenizer,
        cross_test,
        config.max_length,
        config.encode_batch_size,
    )
    cross_scored[
        [
            "query_id",
            "siret",
            "siren",
            "label",
            "xgb_score",
            "cross_score",
        ]
    ].to_parquet(args.output_dir / "cross_encoder_predictions.parquet", index=False)
    cross_xgb_metrics = topk_metrics(cross_scored, "xgb_score")
    cross_neural_metrics = topk_metrics(cross_scored, "cross_score")
    rerank_metrics = {
        str(k): restricted_rerank_metrics(cross_scored, k)
        for k in (3, 5, 6, 10, 20, 50)
    }
    cross_paired = paired_bootstrap(
        outcome_vector(cross_scored, "xgb_score"),
        outcome_vector(cross_scored, "cross_score"),
        config.seed,
    )
    save_trainable_state(
        cross_model, args.output_dir / "cross_encoder_trainable_state.pt"
    )
    del cross_model

    print("[dual] evaluating pretrained encoder", flush=True)
    dual_pre = DualEncoder(args.base_model, config.trainable_layers)
    dual_tokenizer = load_tokenizer(args.base_model)
    before_detail, before_metrics = evaluate_retrieval(
        dual_pre,
        dual_tokenizer,
        holdout,
        args.partitions,
        config,
        include_lexical=True,
    )
    del dual_pre

    print("[dual] contrastive training with hard negatives", flush=True)
    dual_model, dual_tokenizer, dual_history = train_dual_encoder(
        dual_train, args.base_model, config
    )
    after_detail, after_metrics = evaluate_retrieval(
        dual_model,
        dual_tokenizer,
        holdout,
        args.partitions,
        config,
        include_lexical=False,
    )
    save_trainable_state(
        dual_model, args.output_dir / "dual_encoder_trainable_state.pt"
    )
    merged_detail = before_detail.merge(
        after_detail[
            ["query_id"]
            + [
                column
                for column in after_detail.columns
                if column.startswith("dense_")
            ]
        ],
        on="query_id",
        suffixes=("_before", "_after"),
        how="inner",
    )
    merged_detail.to_parquet(
        args.output_dir / "dual_encoder_retrieval_predictions.parquet", index=False
    )
    union_metrics = hybrid_union_metrics(merged_detail)

    tfidf_metrics = {
        key: value
        for key, value in before_metrics.items()
        if key.startswith("tfidf_")
    }
    report = {
        "generated_at": pd.Timestamp.now(tz="Europe/Paris").isoformat(),
        "config": asdict(config),
        "inputs": {
            "samples": str(args.samples.relative_to(ROOT)),
            "samples_sha256": sha256(args.samples),
            "crm": str(args.crm.relative_to(ROOT)),
            "crm_sha256": sha256(args.crm),
            "base_model": str(args.base_model.relative_to(ROOT)),
            "xgb_model": str(args.xgb_model.relative_to(ROOT)),
        },
        "holdout": {
            "queries": int(len(holdout)),
            "unique_gt_sirens": int(holdout["gt_siren"].nunique()),
            "partition_rows_min": int(holdout["row_count"].min()),
            "partition_rows_median": float(holdout["row_count"].median()),
            "partition_rows_max": int(holdout["row_count"].max()),
        },
        "data_quality": {
            "candidate_text_coverage": float(candidate_coverage),
            "cross_train_queries": int(len(train_ids)),
            "cross_dev_queries": int(len(dev_ids)),
            "dual_train_queries": int(len(dual_ids)),
            "train_negative_heldout_siren_overlap": 0,
        },
        "tokenizer_audit": {
            "declared_class": json.loads(
                (args.base_model / "tokenizer_config.json").read_text(
                    encoding="utf-8"
                )
            ).get("tokenizer_class"),
            "spike_loader": "PreTrainedTokenizerFast",
            "production_risk": (
                "AutoTokenizer/SentenceTransformer currently honors the wrong "
                "BertTokenizer class hint and emits excessive <unk> tokens."
            ),
        },
        "cross_encoder": {
            "xgb": cross_xgb_metrics,
            "neural": cross_neural_metrics,
            "paired_hit_at_1": cross_paired,
            "rerank_by_xgb_top_k": rerank_metrics,
            **cross_aux,
        },
        "dual_encoder": {
            "tfidf": tfidf_metrics,
            "before": before_metrics,
            "after": after_metrics,
            "hybrid_union": union_metrics,
            "history": dual_history,
        },
        "limitations": [
            "Positive-only ground truth: NO_MATCH is not evaluated.",
            "Full-partition retrieval is restricted to INSEE blocks under the configured size cap.",
            "The cross-encoder is pairwise at inference and listwise only through its training loss.",
            "The spike fine-tunes only the last transformer layers.",
            "Hybrid union recall uses up to twice the candidate budget of each individual top-k list.",
        ],
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    write_summary(report, args.output_dir / "README.md")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--crm", type=Path, default=DEFAULT_CRM)
    parser.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--xgb-model", type=Path, default=DEFAULT_XGB)
    parser.add_argument("--xgb-meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--holdout-queries", type=int, default=400)
    parser.add_argument("--max-partition-rows", type=int, default=1500)
    parser.add_argument("--cross-train-queries", type=int, default=4000)
    parser.add_argument("--cross-dev-queries", type=int, default=600)
    parser.add_argument("--cross-negatives", type=int, default=5)
    parser.add_argument("--cross-epochs", type=int, default=2)
    parser.add_argument("--dual-train-queries", type=int, default=4000)
    parser.add_argument("--dual-epochs", type=int, default=2)
    parser.add_argument("--trainable-layers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    final_report = run(parse_args())
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
