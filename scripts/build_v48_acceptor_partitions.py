#!/usr/bin/env python3
"""Freeze leak-resistant V4.8 acceptor partitions without scoring a model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.8-acceptor-partitions-1"
EXPECTED_HASHES = {
    "contract": "d26669a0214c11a0b45d4f505f2aba76f45d9088e48c838e133c2ecb5855c5f6",
    "current_labels": "e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2",
    "adjudications": "c3ceb30e0186c58c6dc9957658935eb7d5a557e75cae4e83c6f6f2cabfb80b74",
    "historical_scenes": "8f3bc4633ada9eb6347e47a1029f0e69fa8946b1c3c1df38c72232f572088dc9",
    "historical_assignments": "33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193",
    "historical_queries": "6a12f1c4ca9ec33636ebcf7748c208595c6168d7cdb8c068e1434af3fe22abb0",
    "historical_labels": "69032b745817959422ef26e4c0c1228686260c1daa272ca5d6aba1d7be087b04",
    "acceptor_metadata": "73199451b2de6ae383c9c0c58b10ab9c7393994a4efdec45f9c8e1e9f150691c",
}
EXPECTED_COUNTS = {
    "historical": 7003,
    "historical_fit": 5547,
    "historical_fit_eligible": 5545,
    "historical_dev": 1456,
    "current": 172,
    "random": 57,
    "random_reliable": 52,
    "targeted": 115,
    "targeted_reliable": 98,
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _siren(value: Any) -> str:
    digits = "".join(character for character in _text(value) if character.isdigit())
    if len(digits) == 14:
        return digits[:9]
    return digits if len(digits) == 9 else ""


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _load_inputs(
    *,
    current_labels_path: Path,
    adjudications_path: Path,
    historical_scenes_path: Path,
    historical_assignments_path: Path,
    historical_queries_path: Path,
    historical_labels_path: Path,
    acceptor_metadata_path: Path,
    contract_path: Path,
    enforce_canonical: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = {
        "current_labels": Path(current_labels_path).resolve(),
        "adjudications": Path(adjudications_path).resolve(),
        "historical_scenes": Path(historical_scenes_path).resolve(),
        "historical_assignments": Path(historical_assignments_path).resolve(),
        "historical_queries": Path(historical_queries_path).resolve(),
        "historical_labels": Path(historical_labels_path).resolve(),
        "acceptor_metadata": Path(acceptor_metadata_path).resolve(),
        "contract": Path(contract_path).resolve(),
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if enforce_canonical:
        mismatches = {
            name: (EXPECTED_HASHES[name], actual)
            for name, actual in hashes.items()
            if actual != EXPECTED_HASHES[name]
        }
        if mismatches:
            raise ValueError(f"V4.8 input hash mismatch: {mismatches}")

    current = pd.read_parquet(paths["current_labels"])
    adjudications = pd.read_parquet(paths["adjudications"])
    scenes = pd.read_parquet(paths["historical_scenes"])
    assignments = pd.read_parquet(paths["historical_assignments"])
    queries = pd.read_parquet(paths["historical_queries"])
    labels = pd.read_parquet(paths["historical_labels"])

    _assert_columns(
        current,
        [
            "audit_case_id",
            "query_id",
            "sampling_stratum",
            "input_siret",
            "current_top1_siret",
            "current_top1_siren",
            "current_adjudication_label",
            "current_evidence_validated",
            "current_acceptor_target",
            "current_training_eligible",
            "replayed_top1_siret",
        ],
        "current_labels",
    )
    _assert_columns(
        adjudications,
        ["query_id", "validated_correct_siret"],
        "adjudications",
    )
    _assert_columns(
        scenes,
        [
            "query_id",
            "predicted_siren",
            "ground_truth_siren",
            "is_exact_siret_correct",
            "ranker_prediction_is_out_of_sample",
            "acceptor_eligible",
        ],
        "historical_scenes",
    )
    _assert_columns(
        assignments,
        ["query_id", "siren_component_id", "split", "oof_fold"],
        "historical_assignments",
    )
    _assert_columns(queries, ["query_id", "input_siret", "input_siren"], "queries")
    _assert_columns(
        labels,
        ["query_id", "ground_truth_siret", "ground_truth_siren"],
        "labels",
    )

    frames = [scenes, assignments, queries, labels]
    for frame in frames:
        frame["query_id"] = frame["query_id"].astype(str)
        if frame["query_id"].duplicated().any():
            raise ValueError("Historical V4.1 inputs require unique query IDs")
    current["query_id"] = current["query_id"].astype(str)
    current["audit_case_id"] = current["audit_case_id"].astype(str)
    if current["query_id"].duplicated().any() or current["audit_case_id"].duplicated().any():
        raise ValueError("V4.7 current labels require unique IDs")
    if set(current["query_id"]) & set(scenes["query_id"]):
        raise ValueError("V4.7 and historical V4.1 query IDs overlap")

    metadata = json.loads(paths["acceptor_metadata"].read_text(encoding="utf-8"))
    feature_order = [str(value) for value in metadata.get("feature_order", [])]
    if len(feature_order) != 80 or len(set(feature_order)) != 80:
        raise ValueError("V4.8 requires the 80 unique V4.1 acceptor features")
    _assert_columns(current, feature_order, "current_labels features")
    reliable = current["current_evidence_validated"].astype(bool)
    if not current.loc[reliable, "current_training_eligible"].astype(bool).all():
        raise ValueError("A reliable current label is not training eligible")
    expected_targets = current["current_adjudication_label"].map(
        {"TOP1_CORRECT": 1.0, "TOP1_WRONG": 0.0, "AMBIGUOUS": 0.0}
    )
    observed_targets = pd.to_numeric(
        current["current_acceptor_target"], errors="coerce"
    )
    if (
        expected_targets.loc[reliable].isna().any()
        or observed_targets.loc[reliable].isna().any()
        or not observed_targets.loc[reliable]
        .eq(expected_targets.loc[reliable])
        .all()
    ):
        raise ValueError("A reliable current acceptor target is incoherent")
    current_top1 = current["current_top1_siret"].map(
        lambda value: "".join(x for x in _text(value) if x.isdigit())
    )
    replayed_top1 = current["replayed_top1_siret"].map(
        lambda value: "".join(x for x in _text(value) if x.isdigit())
    )
    if (
        current_top1.loc[reliable].str.len().ne(14).any()
        or not current_top1.loc[reliable].eq(replayed_top1.loc[reliable]).all()
    ):
        raise ValueError("A reliable current label is not bound to replayed top1")
    feature_matrix = current.loc[reliable, feature_order].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(feature_matrix.to_numpy(dtype=float)).all():
        raise ValueError("A reliable current scene contains a non-finite feature")

    historical = (
        scenes.merge(assignments, on="query_id", validate="one_to_one")
        .merge(
            queries[["query_id", "input_siret", "input_siren"]],
            on="query_id",
            validate="one_to_one",
        )
        .merge(
            labels[["query_id", "ground_truth_siret", "ground_truth_siren"]],
            on="query_id",
            validate="one_to_one",
            suffixes=("", "_label"),
        )
    )
    if "ground_truth_siren_label" in historical:
        left = historical["ground_truth_siren"].map(_siren)
        right = historical["ground_truth_siren_label"].map(_siren)
        if not left.eq(right).all():
            raise ValueError("Historical scene and label SIREN disagree")

    adjudicated_exact = (
        adjudications.assign(query_id=adjudications["query_id"].astype(str))
        .groupby("query_id", sort=False)["validated_correct_siret"]
        .agg(lambda values: next((_text(value) for value in values if _text(value)), ""))
    )
    current["validated_correct_siret"] = current["query_id"].map(adjudicated_exact).fillna("")

    if enforce_canonical:
        actual_counts = {
            "historical": len(historical),
            "historical_fit": int(historical["split"].eq("fit").sum()),
            "historical_fit_eligible": int(
                (
                    historical["split"].eq("fit")
                    & historical["acceptor_eligible"].astype(bool)
                ).sum()
            ),
            "historical_dev": int(historical["split"].eq("dev").sum()),
            "current": len(current),
            "random": int(current["sampling_stratum"].eq("RANDOM_POPULATION").sum()),
            "random_reliable": int(
                (
                    current["sampling_stratum"].eq("RANDOM_POPULATION")
                    & current["current_evidence_validated"].astype(bool)
                ).sum()
            ),
            "targeted": int(current["sampling_stratum"].ne("RANDOM_POPULATION").sum()),
            "targeted_reliable": int(
                (
                    current["sampling_stratum"].ne("RANDOM_POPULATION")
                    & current["current_evidence_validated"].astype(bool)
                ).sum()
            ),
        }
        if actual_counts != EXPECTED_COUNTS:
            raise ValueError(
                f"V4.8 canonical count mismatch: {actual_counts} != {EXPECTED_COUNTS}"
            )
    if not historical["ranker_prediction_is_out_of_sample"].astype(bool).all():
        raise ValueError("Historical acceptor scenes must be out of sample")
    if not historical["split"].isin({"fit", "dev"}).all():
        raise ValueError("Historical split contains a forbidden value")
    return historical, current, hashes


def _row_sirens(row: pd.Series, *, historical: bool) -> list[str]:
    if historical:
        values = [
            row.get("input_siren"),
            row.get("input_siret"),
            row.get("ground_truth_siren"),
            row.get("ground_truth_siret"),
        ]
    else:
        values = [
            row.get("input_siret"),
            row.get("current_top1_siren"),
            row.get("current_top1_siret"),
            row.get("validated_correct_siret"),
        ]
    return sorted({value for raw in values if (value := _siren(raw))})


def _build_components(
    historical: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    union = UnionFind()
    row_sirens: dict[str, list[str]] = {}
    row_metadata: list[dict[str, Any]] = []

    def add_row(row_key: str, sirens: list[str], atom: str = "") -> None:
        row_node = f"row:{row_key}"
        union.find(row_node)
        if atom:
            union.union(row_node, f"atom:{atom}")
        for siren in sirens:
            union.union(row_node, f"siren:{siren}")
        row_sirens[row_key] = sirens

    for row in historical.to_dict("records"):
        query_id = _text(row["query_id"])
        row_key = f"historical:{query_id}"
        sirens = _row_sirens(pd.Series(row), historical=True)
        legacy_component = _text(row["siren_component_id"])
        add_row(row_key, sirens, atom=f"historical:{legacy_component}")
        row_metadata.append(
            {
                "row_key": row_key,
                "population": "historical",
                "query_id": query_id,
                "audit_case_id": "",
                "sampling_stratum": "",
                "legacy_split": _text(row["split"]),
                "legacy_component_id": legacy_component,
                "acceptor_eligible": bool(row["acceptor_eligible"]),
                "label_visible": True,
                "acceptor_target": int(row["is_exact_siret_correct"]),
                "adjudication_label": "",
                "evidence_validated": True,
            }
        )

    for row in current.to_dict("records"):
        query_id = _text(row["query_id"])
        row_key = f"current:{query_id}"
        sirens = _row_sirens(pd.Series(row), historical=False)
        add_row(row_key, sirens)
        is_random = _text(row["sampling_stratum"]) == "RANDOM_POPULATION"
        reliable = bool(row["current_evidence_validated"])
        raw_target = row.get("current_acceptor_target")
        target = (
            int(float(raw_target))
            if reliable and not is_random and _text(raw_target)
            else pd.NA
        )
        row_metadata.append(
            {
                "row_key": row_key,
                "population": "current",
                "query_id": query_id,
                "audit_case_id": _text(row["audit_case_id"]),
                "sampling_stratum": _text(row["sampling_stratum"]),
                "legacy_split": "",
                "legacy_component_id": "",
                "acceptor_eligible": bool(reliable),
                "label_visible": bool(reliable and not is_random),
                "acceptor_target": target,
                "adjudication_label": (
                    _text(row["current_adjudication_label"])
                    if reliable and not is_random
                    else ""
                ),
                "evidence_validated": reliable,
            }
        )

    members: dict[str, list[str]] = defaultdict(list)
    for node in union.parent:
        members[union.find(node)].append(node)
    component_id_by_root = {
        root: hashlib.sha256("|".join(sorted(nodes)).encode("utf-8")).hexdigest()[:16]
        for root, nodes in members.items()
    }
    component_by_row = {
        row_key: component_id_by_root[union.find(f"row:{row_key}")]
        for row_key in row_sirens
    }
    rows = pd.DataFrame(row_metadata)
    rows["component_id"] = rows["row_key"].map(component_by_row)
    rows["linked_sirens_json"] = rows["row_key"].map(
        lambda value: json.dumps(row_sirens[value], ensure_ascii=False)
    )

    edges = pd.DataFrame(
        [
            {
                "row_key": row_key,
                "component_id": component_by_row[row_key],
                "siren": siren,
            }
            for row_key, sirens in sorted(row_sirens.items())
            for siren in sirens
        ],
        columns=["row_key", "component_id", "siren"],
    )
    return rows, edges


def _assign_partitions(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    random_components = set(
        output.loc[
            output["population"].eq("current")
            & output["sampling_stratum"].eq("RANDOM_POPULATION"),
            "component_id",
        ]
    )
    dev_components = set(
        output.loc[
            output["population"].eq("historical")
            & output["legacy_split"].eq("dev"),
            "component_id",
        ]
    ) - random_components
    hard_components = set(
        output.loc[
            output["population"].eq("current")
            & output["sampling_stratum"].ne("RANDOM_POPULATION")
            & output["evidence_validated"].astype(bool),
            "component_id",
        ]
    ) - random_components - dev_components

    component_partition: dict[str, str] = {}
    for component in output["component_id"].unique():
        if component in random_components:
            component_partition[component] = "random_sealed"
        elif component in dev_components:
            component_partition[component] = "historical_dev"
        elif component in hard_components:
            component_partition[component] = "hard_oof"
        else:
            component_partition[component] = "historical_fit"
    output["partition"] = output["component_id"].map(component_partition)

    reliable_hard = output[
        output["population"].eq("current")
        & output["sampling_stratum"].ne("RANDOM_POPULATION")
        & output["evidence_validated"].astype(bool)
        & output["partition"].eq("hard_oof")
    ].copy()
    component_stats = (
        reliable_hard.assign(
            wrong=reliable_hard["adjudication_label"].eq("TOP1_WRONG").astype(int),
            negative=reliable_hard["acceptor_target"].eq(0).astype(int),
        )
        .groupby("component_id", sort=False)
        .agg(wrong=("wrong", "sum"), negative=("negative", "sum"), cases=("row_key", "size"))
        .reset_index()
    )
    component_stats["order_key"] = component_stats["component_id"].map(
        lambda value: hashlib.sha256(
            f"v4.8-hard-oof:42:{value}".encode("utf-8")
        ).hexdigest()
    )
    component_stats = component_stats.sort_values("order_key")
    loads = [{"wrong": 0, "negative": 0, "cases": 0} for _ in range(5)]
    folds: dict[str, int] = {}
    for record in component_stats.to_dict("records"):
        fold = min(
            range(5),
            key=lambda index: (
                loads[index]["wrong"],
                loads[index]["negative"],
                loads[index]["cases"],
                index,
            ),
        )
        folds[_text(record["component_id"])] = fold
        for key in ("wrong", "negative", "cases"):
            loads[fold][key] += int(record[key])
    output["hard_fold"] = output["component_id"].map(folds).astype("Int64")

    def role(row: pd.Series) -> str:
        if row["population"] == "historical":
            if row["partition"] == "random_sealed":
                return "historical_random_excluded"
            if row["partition"] == "hard_oof":
                return "historical_hard_support"
            if row["legacy_split"] == "dev":
                return "historical_dev"
            if row["partition"] == "historical_dev":
                return "historical_fit_dev_excluded"
            return "historical_fit"
        if row["sampling_stratum"] == "RANDOM_POPULATION":
            return "random_sealed"
        if row["partition"] == "random_sealed":
            return "hard_random_locked"
        if row["partition"] == "historical_dev":
            return "hard_dev_locked"
        if bool(row["evidence_validated"]):
            return "hard_oof"
        return "targeted_unresolved"

    output["role"] = output.apply(role, axis=1)
    output = output.sort_values(["population", "query_id"]).reset_index(drop=True)
    return output


def _validate_partitions(rows: pd.DataFrame, *, enforce_canonical: bool) -> None:
    if rows.groupby("component_id")["partition"].nunique().max() != 1:
        raise ValueError("A V4.8 component crosses partitions")
    random_rows = rows[
        rows["population"].eq("current")
        & rows["sampling_stratum"].eq("RANDOM_POPULATION")
    ]
    if not random_rows["partition"].eq("random_sealed").all():
        raise ValueError("A V4.8 random case is not sealed")
    if random_rows["label_visible"].astype(bool).any():
        raise ValueError("A V4.8 random target leaked into partition assignments")
    if random_rows["acceptor_target"].notna().any():
        raise ValueError("A V4.8 random target is visible")
    hard_rows = rows[
        rows["population"].eq("current")
        & rows["sampling_stratum"].ne("RANDOM_POPULATION")
        & rows["evidence_validated"].astype(bool)
    ]
    allowed = {"hard_oof", "hard_dev_locked", "hard_random_locked"}
    if not set(hard_rows["role"]).issubset(allowed):
        raise ValueError("A reliable targeted case has no valid V4.8 role")
    hard_oof = hard_rows[hard_rows["role"].eq("hard_oof")]
    if not hard_oof.empty and hard_oof["hard_fold"].isna().any():
        raise ValueError("A V4.8 hard OOF case has no fold")
    if enforce_canonical:
        if len(rows[rows["population"].eq("historical")]) != EXPECTED_COUNTS["historical"]:
            raise ValueError("Historical V4.8 row count changed")
        if len(random_rows) != EXPECTED_COUNTS["random"]:
            raise ValueError("Random V4.8 row count changed")
        if len(hard_rows) != EXPECTED_COUNTS["targeted_reliable"]:
            raise ValueError("Targeted reliable V4.8 row count changed")


def build_partitions(
    *,
    current_labels_path: Path,
    adjudications_path: Path,
    historical_scenes_path: Path,
    historical_assignments_path: Path,
    historical_queries_path: Path,
    historical_labels_path: Path,
    acceptor_metadata_path: Path,
    contract_path: Path,
    output_root: Path,
    enforce_canonical: bool = True,
) -> Path:
    historical, current, hashes = _load_inputs(
        current_labels_path=current_labels_path,
        adjudications_path=adjudications_path,
        historical_scenes_path=historical_scenes_path,
        historical_assignments_path=historical_assignments_path,
        historical_queries_path=historical_queries_path,
        historical_labels_path=historical_labels_path,
        acceptor_metadata_path=acceptor_metadata_path,
        contract_path=contract_path,
        enforce_canonical=enforce_canonical,
    )
    rows, edges = _build_components(historical, current)
    rows = _assign_partitions(rows)
    _validate_partitions(rows, enforce_canonical=enforce_canonical)

    summary = {
        "historical_rows": int(rows["population"].eq("historical").sum()),
        "current_rows": int(rows["population"].eq("current").sum()),
        "component_count": int(rows["component_id"].nunique()),
        "partition_counts": {
            str(key): int(value)
            for key, value in rows.groupby("partition").size().to_dict().items()
        },
        "role_counts": {
            str(key): int(value)
            for key, value in rows.groupby("role").size().to_dict().items()
        },
        "targeted_reliable_role_counts": {
            str(key): int(value)
            for key, value in rows.loc[
                rows["population"].eq("current")
                & rows["sampling_stratum"].ne("RANDOM_POPULATION")
                & rows["evidence_validated"].astype(bool)
            ]
            .groupby("role")
            .size()
            .to_dict()
            .items()
        },
        "random_case_count": int(
            (
                rows["population"].eq("current")
                & rows["sampling_stratum"].eq("RANDOM_POPULATION")
            ).sum()
        ),
        "random_targets_exposed": 0,
        "model_loaded": False,
        "model_scoring_performed": False,
        "test_opened": False,
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "input_hashes": hashes,
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "component_assignments_sha256": hashlib.sha256(
            rows[
                ["row_key", "component_id", "partition", "role", "hard_fold"]
            ].to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "summary": summary,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    target = Path(output_root).resolve() / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.8 partition artifact exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent))
    try:
        assignments_path = staging / "partition_assignments.parquet"
        edges_path = staging / "component_edges.parquet"
        summary_path = staging / "summary.json"
        rows.to_parquet(assignments_path, index=False)
        edges.sort_values(["component_id", "row_key", "siren"]).to_parquet(
            edges_path, index=False
        )
        _json_dump(summary_path, summary)
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "builder": str(Path(__file__).resolve()),
                "current_labels": str(Path(current_labels_path).resolve()),
                "adjudications": str(Path(adjudications_path).resolve()),
                "historical_scenes": str(Path(historical_scenes_path).resolve()),
                "historical_assignments": str(
                    Path(historical_assignments_path).resolve()
                ),
                "historical_queries": str(Path(historical_queries_path).resolve()),
                "historical_labels": str(Path(historical_labels_path).resolve()),
                "acceptor_metadata": str(Path(acceptor_metadata_path).resolve()),
                "contract": str(Path(contract_path).resolve()),
            },
            "outputs": {
                assignments_path.name: file_sha256(assignments_path),
                edges_path.name: file_sha256(edges_path),
                summary_path.name: file_sha256(summary_path),
            },
            "invariants": {
                "candidate_pool_sirens_used_as_edges": False,
                "random_targets_exposed": False,
                "model_loaded": False,
                "model_scoring_performed": False,
                "test_opened": False,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    root = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-labels",
        type=Path,
        default=root
        / "audits/v4_7_current_adjudications/4cc5420fb5da0683/current_labels.parquet",
    )
    parser.add_argument(
        "--adjudications",
        type=Path,
        default=root
        / "audits/v4_7_current_adjudications/4cc5420fb5da0683/adjudications.parquet",
    )
    parser.add_argument(
        "--historical-scenes",
        type=Path,
        default=root
        / "models/v4_1/f938abf6b8a87155/acceptor_scenes.parquet",
    )
    parser.add_argument(
        "--historical-assignments",
        type=Path,
        default=root
        / "models/v4_1/f938abf6b8a87155/split_assignments.parquet",
    )
    parser.add_argument(
        "--historical-queries",
        type=Path,
        default=root / "datasets/v4_1/f938abf6b8a87155/queries.parquet",
    )
    parser.add_argument(
        "--historical-labels",
        type=Path,
        default=root / "datasets/v4_1/f938abf6b8a87155/labels.parquet",
    )
    parser.add_argument(
        "--acceptor-metadata",
        type=Path,
        default=root / "models/v4_1/f938abf6b8a87155/acceptor/metadata.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("docs/v4_8_current_acceptor_feasibility_contract.md"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "datasets/v4_8_acceptor_partitions",
    )
    parser.add_argument("--no-canonical-checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = build_partitions(
        current_labels_path=args.current_labels,
        adjudications_path=args.adjudications,
        historical_scenes_path=args.historical_scenes,
        historical_assignments_path=args.historical_assignments,
        historical_queries_path=args.historical_queries,
        historical_labels_path=args.historical_labels,
        acceptor_metadata_path=args.acceptor_metadata,
        contract_path=args.contract,
        output_root=args.output_root,
        enforce_canonical=not args.no_canonical_checks,
    )
    print(target)


if __name__ == "__main__":
    main()
