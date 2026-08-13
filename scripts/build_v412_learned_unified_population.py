#!/usr/bin/env python3
"""Build the V4.12-L learned-development population without retrieval data.

The population is the 17,054 historical CRM rows qualified by V3, amended by
the 279 locally audited labels.  Audited rows already in the historical CRM
replace their V3 label; audited fresh rows are appended.  Control-label
corrections are applied only when the query is already in that population.
Fresh controls stay outside training as a regression guard.

This builder deliberately has no candidate/retrieval input.  Its five folds
are connected by every historical or corrected SIREN known for a query, so a
legal entity (including a corrected cross-SIREN label) cannot cross folds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.features import normalize_text
from src.xgb_matcher.v9_dataset import file_sha256, normalize_siret


SCHEMA_VERSION = "sireto-v4.12-learned-unified-population-1"
SEED = 42
FOLD_COUNT = 5
DEFAULT_QUALIFICATION_DIRS = (
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v3/a76eebf6a8b157ea"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v3/ab8343817551c0a5"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v3/72cc411a916c4814"),
)
DEFAULT_FRESH_QUERIES = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_11_input_blind/ec4326ec57e4411d/queries.parquet"
)
ALLOWED_LABELS = {"MATCH_EXACT", "NO_MATCH", "AMBIGUOUS", "UNRESOLVED"}


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_fold(group_id: str, seed: int, fold_count: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def _read_qualification(
    directories: Iterable[Path],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    input_records: list[dict[str, str]] = []
    for directory in directories:
        benchmark_path = directory / "benchmark.parquet"
        manifest_path = directory / "manifest.json"
        if not benchmark_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete V3 qualification: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "sireto-benchmark-v3-evidence-1":
            raise ValueError(f"Unexpected V3 schema in {manifest_path}")
        expected_hash = manifest.get("outputs", {}).get("benchmark.parquet")
        observed_hash = file_sha256(benchmark_path)
        if expected_hash != observed_hash:
            raise ValueError(f"V3 benchmark hash mismatch: {benchmark_path}")
        frames.append(pd.read_parquet(benchmark_path))
        manifests.append(manifest)
        input_records.extend(
            [
                {"path": str(benchmark_path), "sha256": observed_hash},
                {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            ]
        )
    policy_versions = {
        str(manifest.get("schema_version")) for manifest in manifests
    }
    establishment_hashes = {
        str(manifest.get("establishment_snapshot_sha256")) for manifest in manifests
    }
    legal_unit_hashes = {
        str(manifest.get("legal_unit_snapshot_sha256")) for manifest in manifests
    }
    if len(policy_versions) != 1 or len(establishment_hashes) != 1 or len(legal_unit_hashes) != 1:
        raise ValueError("V3 qualification manifests are not mutually compatible")
    frame = pd.concat(frames, ignore_index=True)
    frame["query_id"] = frame["query_id"].astype(str)
    if frame["query_id"].duplicated().any():
        raise ValueError("V3 qualification query_id values must be unique")
    return frame, manifests, input_records


def _validate_crm_binding(crm_path: Path, historical: pd.DataFrame) -> None:
    crm = pd.read_csv(crm_path, sep=";", dtype=str, keep_default_na=False)
    crm = crm.reset_index(names="query_id")
    crm["query_id"] = crm["query_id"].astype(str)
    if len(crm) != len(historical):
        raise ValueError(
            f"CRM/V3 row-count mismatch: {len(crm)} != {len(historical)}"
        )
    joined = crm.merge(
        historical,
        on="query_id",
        validate="one_to_one",
        suffixes=("_crm", "_v3"),
    )
    comparisons = (
        ("gt_siret", "historical_ground_truth_siret"),
        ("crm_name_crm", "crm_name_v3"),
        ("crm_adresse", "crm_address"),
        ("crm_cp", "postcode"),
        ("crm_insee", "insee"),
    )
    for crm_column, benchmark_column in comparisons:
        left = joined[crm_column].map(_clean_text)
        right = joined[benchmark_column].map(_clean_text)
        mismatch = left.ne(right)
        if mismatch.any():
            example_ids = joined.loc[mismatch, "query_id"].head(5).tolist()
            raise ValueError(
                f"CRM/V3 mismatch for {crm_column}: {int(mismatch.sum())} rows; "
                f"examples={example_ids}"
            )


def _query_projection(frame: pd.DataFrame) -> pd.DataFrame:
    queries = pd.DataFrame(
        {
            "query_id": frame["query_id"].astype(str),
            "crm_record_id": frame.get("crm_record_id", "").map(_clean_text),
            "crm_name": frame["crm_name"].map(_clean_text),
            "crm_address": frame["crm_address"].map(_clean_text),
            "crm_postcode": frame.get("postcode", "").map(_clean_text),
            "crm_city": frame.get("crm_city", "").map(_clean_text),
            "crm_insee": frame.get("insee", "").map(_clean_text),
            "reference_date": frame.get("reference_date", "").map(_clean_text),
        }
    )
    queries["crm_name_norm"] = queries["crm_name"].map(normalize_text)
    queries["crm_address_norm"] = queries["crm_address"].map(normalize_text)
    queries["crm_city_norm"] = queries["crm_city"].map(normalize_text)
    return queries


def _fresh_query_projection(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "query_id",
        "crm_record_id",
        "crm_name",
        "crm_address",
        "crm_postcode",
        "crm_city",
        "crm_insee",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fresh query source is missing: {sorted(missing)}")
    output = frame[list(required)].copy()
    for column in output.columns:
        output[column] = output[column].map(_clean_text)
    output["reference_date"] = ""
    output["crm_name_norm"] = output["crm_name"].map(normalize_text)
    output["crm_address_norm"] = output["crm_address"].map(normalize_text)
    output["crm_city_norm"] = output["crm_city"].map(normalize_text)
    return output


def _snapshot_states(snapshot_path: Path, sirets: pd.Series) -> dict[str, str]:
    wanted = pd.DataFrame(
        {"siret": sorted({str(value) for value in sirets.dropna() if value})}
    )
    connection = duckdb.connect()
    try:
        connection.register("wanted_sirets", wanted)
        rows = connection.execute(
            """
            SELECT CAST(s.siret AS VARCHAR) AS siret,
                   CAST(s.etatAdministratifEtablissement AS VARCHAR) AS state
            FROM read_parquet(?) AS s
            INNER JOIN wanted_sirets AS w
              ON CAST(s.siret AS VARCHAR) = w.siret
            """,
            [str(snapshot_path)],
        ).fetch_df()
    finally:
        connection.close()
    rows["siret"] = rows["siret"].astype(str).str.zfill(14)
    if rows["siret"].duplicated().any():
        raise ValueError("Establishment snapshot contains duplicate SIRET values")
    states = dict(zip(rows["siret"], rows["state"], strict=True))
    missing = sorted(set(wanted["siret"]) - set(states))
    if missing:
        raise ValueError(
            f"{len(missing)} exact labels are absent from the SIRENE snapshot; "
            f"examples={missing[:5]}"
        )
    return states


def _build_folds(labels: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    union = _UnionFind()
    query_sirens: dict[str, list[str]] = {}
    for row in labels.itertuples(index=False):
        sirens = sorted(
            {
                str(value)
                for value in (
                    row.historical_ground_truth_siren,
                    row.ground_truth_siren,
                )
                if _clean_text(value)
            }
        )
        query_sirens[str(row.query_id)] = sirens
        for siren in sirens[1:]:
            union.union(sirens[0], siren)
        if sirens:
            union.find(sirens[0])

    records: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        sirens = query_sirens[str(row.query_id)]
        group_id = union.find(sirens[0]) if sirens else f"QUERY:{row.query_id}"
        records.append(
            {
                "query_id": str(row.query_id),
                "siren_component_id": group_id,
                "oof_fold": _stable_fold(group_id, seed, FOLD_COUNT),
                "legacy_split": str(row.legacy_split),
            }
        )
    assignments = pd.DataFrame(records)
    if assignments.groupby("siren_component_id")["oof_fold"].nunique().max() != 1:
        raise AssertionError("A SIREN component crosses OOF folds")
    if set(assignments["oof_fold"]) != set(range(FOLD_COUNT)):
        raise ValueError("The unified population did not populate all five folds")
    return assignments.sort_values("query_id").reset_index(drop=True)


def build(args: argparse.Namespace) -> Path:
    qualification_dirs = [Path(value) for value in args.qualification_dir]
    historical, qualification_manifests, inputs = _read_qualification(
        qualification_dirs
    )
    _validate_crm_binding(args.crm_source, historical)

    audited = pd.read_csv(
        args.audited_labels, dtype=str, keep_default_na=False
    )
    audited_canonical = pd.read_csv(
        args.audited_canonical, dtype=str, keep_default_na=False
    )
    controls = pd.read_csv(
        args.control_corrections, dtype=str, keep_default_na=False
    )
    fresh_source = pd.read_parquet(args.fresh_queries)
    for name, frame in (
        ("audited labels", audited),
        ("audited canonical", audited_canonical),
        ("control corrections", controls),
        ("fresh queries", fresh_source),
    ):
        if frame["query_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} query_id values must be unique")
    if set(audited["query_id"]) != set(audited_canonical["query_id"]):
        raise ValueError("Audited local and canonical label IDs differ")
    if set(audited["label_kind"]) - ALLOWED_LABELS:
        raise ValueError("Audited labels contain an unsupported label kind")

    historical = historical.copy()
    historical["query_id"] = historical["query_id"].astype(str)
    queries = _query_projection(historical)
    labels = pd.DataFrame(
        {
            "query_id": historical["query_id"],
            "label_kind": historical["label_kind"].astype(str),
            "ground_truth_siret": historical["ground_truth_siret"].map(normalize_siret),
            "historical_ground_truth_siret": historical[
                "historical_ground_truth_siret"
            ].map(normalize_siret),
            "legacy_split": historical["split"].astype(str),
            "label_source": "qualification_v3_direct_evidence",
            "validator": "historical_unreviewed_v3_policy",
            "reliability": "WEAK_POLICY",
            "evidence_reference": "docs/benchmark_v3_evidence_policy.md",
            "label_is_human_validated": False,
        }
    )

    base_ids = set(labels["query_id"])
    audited_ids = set(audited["query_id"].astype(str))
    fresh_ids = audited_ids - base_ids
    fresh_by_id = _fresh_query_projection(fresh_source).set_index("query_id")
    missing_fresh = sorted(fresh_ids - set(fresh_by_id.index))
    if missing_fresh:
        raise ValueError(f"Audited fresh queries are missing: {missing_fresh}")
    if len(fresh_ids) != args.expected_fresh_count:
        raise ValueError(
            f"Expected {args.expected_fresh_count} fresh audited queries, "
            f"found {len(fresh_ids)}"
        )
    queries = pd.concat(
        [queries, fresh_by_id.loc[sorted(fresh_ids)].reset_index()],
        ignore_index=True,
    )

    canonical_by_id = audited_canonical.set_index("query_id")
    audited_by_id = audited.set_index("query_id")
    labels = labels.set_index("query_id")
    for query_id, row in audited_by_id.iterrows():
        kind = str(row["label_kind"])
        siret = normalize_siret(row["ground_truth_siret"])
        if kind == "MATCH_EXACT" and not siret:
            raise ValueError(f"Audited MATCH_EXACT lacks SIRET: {query_id}")
        if kind != "MATCH_EXACT" and siret:
            raise ValueError(f"Audited open label carries SIRET: {query_id}")
        historical_siret = normalize_siret(
            canonical_by_id.loc[query_id, "ground_truth_siret"]
        )
        labels.loc[query_id, :] = {
            "label_kind": kind,
            "ground_truth_siret": siret,
            "historical_ground_truth_siret": historical_siret,
            "legacy_split": (
                labels.loc[query_id, "legacy_split"]
                if query_id in base_ids
                else "fresh_consumed_development"
            ),
            "label_source": f"audited_279:{row['cohort']}",
            "validator": "human_audit",
            "reliability": str(row["reliability"]),
            "evidence_reference": str(row["evidence_reference"]),
            "label_is_human_validated": True,
        }
    labels = labels.reset_index()

    external_controls: list[dict[str, Any]] = []
    labels = labels.set_index("query_id")
    for row in controls.to_dict("records"):
        query_id = str(row["query_id"])
        corrected = normalize_siret(row["corrected_ground_truth_siret"])
        if query_id not in labels.index:
            external_controls.append(row)
            continue
        labels.loc[query_id, "label_kind"] = "MATCH_EXACT"
        labels.loc[query_id, "ground_truth_siret"] = corrected
        labels.loc[query_id, "historical_ground_truth_siret"] = normalize_siret(
            row["historical_ground_truth_siret"]
        )
        labels.loc[query_id, "label_source"] = "control_label_counteraudit"
        labels.loc[query_id, "validator"] = "human_audit"
        labels.loc[query_id, "reliability"] = str(row["reliability"])
        labels.loc[query_id, "evidence_reference"] = str(row["evidence_reference"])
        labels.loc[query_id, "label_is_human_validated"] = True
    labels = labels.reset_index()

    if len(historical) != args.expected_historical_count:
        raise ValueError(
            f"Expected {args.expected_historical_count} historical rows, "
            f"found {len(historical)}"
        )
    if len(audited) != args.expected_audited_count:
        raise ValueError(
            f"Expected {args.expected_audited_count} audited rows, found {len(audited)}"
        )
    if len(labels) != args.expected_population_count:
        raise ValueError(
            f"Expected population {args.expected_population_count}, found {len(labels)}"
        )
    if queries["query_id"].duplicated().any() or labels["query_id"].duplicated().any():
        raise ValueError("Unified population query_id values must be unique")
    if set(queries["query_id"]) != set(labels["query_id"]):
        raise ValueError("Unified queries and labels do not align")

    labels["ground_truth_siren"] = labels["ground_truth_siret"].map(
        lambda value: value[:9] if value else ""
    )
    labels["historical_ground_truth_siren"] = labels[
        "historical_ground_truth_siret"
    ].map(lambda value: value[:9] if value else "")
    exact = labels["label_kind"].eq("MATCH_EXACT")
    if labels.loc[exact, "ground_truth_siret"].isna().any():
        raise ValueError("MATCH_EXACT label lacks a SIRET")
    if labels.loc[~exact, "ground_truth_siret"].notna().any():
        raise ValueError("Open labels must not carry a SIRET")

    states = _snapshot_states(
        args.establishment_snapshot, labels.loc[exact, "ground_truth_siret"]
    )
    labels["ground_truth_state"] = labels["ground_truth_siret"].map(states).fillna("")
    labels["exact_metric_eligible"] = exact
    labels["identity_training_eligible"] = exact
    labels["operational_training_eligible"] = exact & labels[
        "ground_truth_state"
    ].eq("A")
    human_weight = labels["label_is_human_validated"].map({True: 4.0, False: 1.0})
    state_weight = labels["ground_truth_state"].map({"A": 1.0, "F": 0.5}).fillna(0.0)
    labels["ranker_weight"] = (human_weight * state_weight).where(exact, 0.0)
    labels["acceptor_weight"] = human_weight.where(
        labels["label_is_human_validated"],
        exact.map({True: 1.0, False: 0.25}),
    )

    assignments = _build_folds(labels, seed=args.seed)
    fold_map = assignments.set_index("query_id")["oof_fold"]
    queries["oof_fold"] = queries["query_id"].map(fold_map).astype("int8")
    labels["oof_fold"] = labels["query_id"].map(fold_map).astype("int8")
    queries["legacy_split_status"] = "CONSUMED_DEVELOPMENT_ONLY"

    query_columns = [
        "query_id", "crm_record_id", "crm_name", "crm_address", "crm_postcode",
        "crm_city", "crm_insee", "crm_name_norm", "crm_address_norm",
        "crm_city_norm", "reference_date", "oof_fold", "legacy_split_status",
    ]
    label_columns = [
        "query_id", "label_kind", "ground_truth_siret", "ground_truth_siren",
        "historical_ground_truth_siret", "historical_ground_truth_siren",
        "ground_truth_state", "label_source", "validator", "reliability",
        "evidence_reference", "label_is_human_validated", "exact_metric_eligible",
        "identity_training_eligible", "operational_training_eligible",
        "ranker_weight", "acceptor_weight", "legacy_split", "oof_fold",
    ]
    queries = queries[query_columns].sort_values("query_id").reset_index(drop=True)
    labels = labels[label_columns].sort_values("query_id").reset_index(drop=True)

    tracked_paths = [
        args.crm_source,
        args.audited_labels,
        args.audited_canonical,
        args.control_corrections,
        args.fresh_queries,
        args.establishment_snapshot,
    ]
    inputs.extend(
        {"path": str(path), "sha256": file_sha256(path)} for path in tracked_paths
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "seed": args.seed,
        "fold_count": FOLD_COUNT,
        "inputs": sorted(inputs, key=lambda item: item["path"]),
        "population_policy": "17054_v3_plus_43_audited_fresh",
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting build directory: {destination}")
        return destination

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        queries.to_parquet(temporary / "queries.parquet", index=False)
        labels.to_parquet(temporary / "labels.parquet", index=False)
        assignments.to_parquet(temporary / "fold_assignments.parquet", index=False)
        pd.DataFrame(external_controls).to_parquet(
            temporary / "external_regression_controls.parquet", index=False
        )
        exact_output = labels["label_kind"].eq("MATCH_EXACT")
        counts = {
            "queries": len(queries),
            "historical": len(historical),
            "audited_total": len(audited),
            "audited_historical_overrides": len(audited_ids & base_ids),
            "audited_fresh_additions": len(fresh_ids),
            "control_corrections_applied": len(controls) - len(external_controls),
            "external_regression_controls": len(external_controls),
            "labels": {str(k): int(v) for k, v in labels["label_kind"].value_counts().items()},
            "states_exact": {
                str(k): int(v)
                for k, v in labels.loc[
                    exact_output, "ground_truth_state"
                ].value_counts().items()
            },
            "folds": {
                str(k): int(v) for k, v in labels["oof_fold"].value_counts().sort_index().items()
            },
        }
        report = (
            "# Population V4.12-L apprise\n\n"
            f"- Population : **{len(labels)}** requêtes ;\n"
            f"- historique V3 : {len(historical)} ;\n"
            f"- remplacements audités : {len(audited_ids & base_ids)} ;\n"
            f"- ajouts frais audités : {len(fresh_ids)} ;\n"
            f"- labels : `{counts['labels']}` ;\n"
            f"- états des labels exacts : `{counts['states_exact']}` ;\n"
            f"- plis OOF : `{counts['folds']}`.\n\n"
            "Aucun hit, rang, candidat ou score de retrieval n'entre dans ce build. "
            "Les anciens splits train/dev/test sont consommés et ne constituent plus "
            "un test indépendant. Les deux contrôles frais contre-audités restent hors "
            "apprentissage dans `external_regression_controls.parquet`.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        output_names = [
            "queries.parquet",
            "labels.parquet",
            "fold_assignments.parquet",
            "external_regression_controls.parquet",
            "report.md",
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "counts": counts,
            "qualification": {
                "policy": "direct-evidence-v3.0",
                "human_validated": False,
                "retrieval_inputs_used": False,
            },
            "development_contract": {
                "independent_test_available": False,
                "legacy_splits_status": "CONSUMED_DEVELOPMENT_ONLY",
                "evaluation": "FIVE_FOLD_OOF_GROUPED_BY_SIREN_COMPONENT",
                "candidate_ceiling": 100,
                "positive_injection_allowed": False,
                "candidate_artifact_status": "NOT_BUILT_IN_POPULATION_MILESTONE",
                "production_candidate_policy": "PREFER_ACTIVE",
                "closed_examples_role": "IDENTITY_AUXILIARY_WEIGHT_0_5",
            },
            "outputs": {
                name: file_sha256(temporary / name) for name in output_names
            },
            "qualification_build_ids": [
                manifest["build_id"] for manifest in qualification_manifests
            ],
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crm-source", type=Path, default=Path("data/crm_ok_gt.csv"))
    parser.add_argument(
        "--qualification-dir",
        type=Path,
        action="append",
        default=None,
        help="Repeat for each V3 split; defaults to the frozen train/dev/test builds.",
    )
    parser.add_argument(
        "--audited-labels",
        type=Path,
        default=Path("reports/v412_review_local_identifiable_labels_279.csv"),
    )
    parser.add_argument(
        "--audited-canonical",
        type=Path,
        default=Path("reports/v412_review_trusted_labels_279.csv"),
    )
    parser.add_argument(
        "--control-corrections",
        type=Path,
        default=Path("reports/v412_control_label_counteraudit_4.csv"),
    )
    parser.add_argument("--fresh-queries", type=Path, default=DEFAULT_FRESH_QUERIES)
    parser.add_argument(
        "--establishment-snapshot",
        type=Path,
        default=Path("data/StockEtablissement_utf8.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
            "v4_12_learned_unified_population"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--expected-historical-count", type=int, default=17_054)
    parser.add_argument("--expected-audited-count", type=int, default=279)
    parser.add_argument("--expected-fresh-count", type=int, default=43)
    parser.add_argument("--expected-population-count", type=int, default=17_097)
    args = parser.parse_args()
    if args.qualification_dir is None:
        args.qualification_dir = list(DEFAULT_QUALIFICATION_DIRS)
    return args


if __name__ == "__main__":
    print(build(parse_args()))
