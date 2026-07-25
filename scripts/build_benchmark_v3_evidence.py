#!/usr/bin/env python3
"""Build a provisional benchmark view based on direct CRM-to-SIRET evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
from typing import Any, Mapping

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_benchmark_v2_qualification import (  # noqa: E402
    _validate_bound_file,
)
from scripts.run_v9_retrieval_experiment import (  # noqa: E402
    _binary_metric,
    _git_commit,
)
from src.xgb_matcher.blocking import (  # noqa: E402
    address_hash,
    candidate_address_hash,
)
from src.xgb_matcher.contracts import GroundTruthKind  # noqa: E402
from src.xgb_matcher.features import (  # noqa: E402
    make_features_from_preprocessed,
    preprocess_crm_row,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-benchmark-v3-evidence-1"
POLICY_VERSION = "direct-evidence-v3.0"
EVIDENCE_CLASSES = {
    "NAME_AND_ADDRESS",
    "NAME_ONLY",
    "ADDRESS_ONLY",
    "NO_DIRECT_EVIDENCE",
}


def classify_direct_evidence(
    features: Mapping[str, float],
    *,
    exact_address_hash: bool,
    crm_number_present: bool,
    candidate_number_present: bool,
) -> tuple[str, bool, bool]:
    """Classify evidence without using any retrieval result."""
    strong_name = bool(
        features.get("name_norm_exact", 0.0)
        or (
            features.get("name_jaro_max", 0.0) >= 0.85
            and features.get("name_token_overlap_max", 0.0) >= 0.50
        )
        or (
            (
                features.get("name_contains_crm_max", 0.0)
                or features.get("name_crm_contains_cand_max", 0.0)
                or features.get("acronym_match_max", 0.0)
            )
            and features.get("name_jaro_max", 0.0) >= 0.75
        )
    )
    number_compatible = bool(
        features.get("street_number_match", 0.0)
        or not crm_number_present
        or not candidate_number_present
    )
    strong_address = bool(
        exact_address_hash
        or (
            features.get("postcode_match", 0.0)
            and features.get("street_name_jaro", 0.0) >= 0.90
            and number_compatible
        )
    )
    if strong_name and strong_address:
        return "NAME_AND_ADDRESS", strong_name, strong_address
    if strong_name:
        return "NAME_ONLY", strong_name, strong_address
    if strong_address:
        return "ADDRESS_ONLY", strong_name, strong_address
    return "NO_DIRECT_EVIDENCE", strong_name, strong_address


def _clean(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    return value


def _candidate_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "siret": _clean(row.get("ground_truth_siret")),
        "siren": _clean(row.get("ground_truth_siren")),
        "denomination": _clean(row.get("denomination")),
        "enseigne1": _clean(row.get("enseigne1")),
        "enseigne2": _clean(row.get("enseigne2")),
        "enseigne3": _clean(row.get("enseigne3")),
        "is_siege": bool(_clean(row.get("is_siege")) or False),
        "numeroVoie": _clean(row.get("numeroVoie")),
        "typeVoie": _clean(row.get("typeVoie")),
        "libelleVoie": _clean(row.get("libelleVoie")),
        "complementAdresse": _clean(row.get("complementAdresse")),
        "postcode": _clean(row.get("candidate_postcode")),
        "city": _clean(row.get("candidate_city")),
        "insee": _clean(row.get("candidate_insee")),
        "cj_ul": (
            str(int(row["cj_ul"]))
            if _clean(row.get("cj_ul")) is not None
            else None
        ),
        "etat_admin": _clean(row.get("ground_truth_state")),
        "sigle_ul": _clean(row.get("sigle_ul")),
        "denomination_ul": _clean(row.get("denomination_ul")),
        "denomination_usuelle_ul": _clean(
            row.get("denomination_usuelle_ul")
        ),
        "nom_ul": _clean(row.get("nom_ul")),
        "nom_usage_ul": _clean(row.get("nom_usage_ul")),
        "prenom_usuel_ul": _clean(row.get("prenom_usuel_ul")),
        "pseudonyme_ul": _clean(row.get("pseudonyme_ul")),
    }


def audit_evidence_rows(joined: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        crm = preprocess_crm_row(
            {
                "crm_name": _clean(row.get("crm_name")) or "",
                "crm_address": _clean(row.get("crm_address")) or "",
                "crm_city": _clean(row.get("crm_city")) or "",
                "crm_city_addr": _clean(row.get("crm_city")) or "",
                "postcode": _clean(row.get("postcode")) or "",
                "insee": _clean(row.get("insee")) or "",
            }
        )
        candidate = _candidate_from_row(row)
        features = make_features_from_preprocessed(
            crm,
            candidate,
            skip_semantic=True,
        )
        crm_hash = address_hash(
            crm.get("crm_street_num"),
            crm.get("crm_street_name"),
        )
        candidate_hash = candidate_address_hash(candidate)
        exact_hash = bool(crm_hash and crm_hash == candidate_hash)
        evidence_class, strong_name, strong_address = classify_direct_evidence(
            features,
            exact_address_hash=exact_hash,
            crm_number_present=bool(crm.get("crm_street_num")),
            candidate_number_present=bool(candidate.get("numeroVoie")),
        )
        records.append(
            {
                "query_id": str(row["query_id"]),
                "split": str(row["split"]),
                "historical_ground_truth_siret": str(
                    row["ground_truth_siret"]
                ),
                "historical_ground_truth_siren": str(
                    row["ground_truth_siren"]
                ),
                "direct_evidence_class": evidence_class,
                "strong_name_evidence": strong_name,
                "strong_address_evidence": strong_address,
                "exact_address_hash": exact_hash,
                "crm_address_hash": crm_hash,
                "candidate_address_hash": candidate_hash,
                "name_jaro_max": float(features["name_jaro_max"]),
                "name_token_overlap_max": float(
                    features["name_token_overlap_max"]
                ),
                "name_norm_exact": float(features["name_norm_exact"]),
                "name_contains_crm_max": float(
                    features["name_contains_crm_max"]
                ),
                "name_crm_contains_cand_max": float(
                    features["name_crm_contains_cand_max"]
                ),
                "acronym_match_max": float(features["acronym_match_max"]),
                "name_sim_max_etab": float(features["name_sim_max_etab"]),
                "name_sim_max_ul": float(features["name_sim_max_ul"]),
                "addr_jaro": float(features["addr_jaro"]),
                "street_name_jaro": float(features["street_name_jaro"]),
                "street_number_match": float(
                    features["street_number_match"]
                ),
                "postcode_match": float(features["postcode_match"]),
            }
        )
    return pd.DataFrame(records)


def apply_evidence_policy(
    v2_benchmark: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "query_id",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
        "qualification_reason",
        "exact_metric_eligible",
        "historical_ground_truth_siret",
        "historical_ground_truth_siren",
    }
    missing = required - set(v2_benchmark.columns)
    if missing:
        raise ValueError(f"V2 benchmark is missing columns: {sorted(missing)}")
    if set(evidence["direct_evidence_class"]) - EVIDENCE_CLASSES:
        raise ValueError("Evidence contains unsupported classes")

    v2 = v2_benchmark.copy()
    audit = evidence.copy()
    v2["query_id"] = v2["query_id"].astype(str)
    audit["query_id"] = audit["query_id"].astype(str)
    if v2["query_id"].duplicated().any() or audit["query_id"].duplicated().any():
        raise ValueError("query_id must be unique")
    if set(v2["query_id"]) != set(audit["query_id"]):
        raise ValueError("V2 benchmark and evidence query IDs differ")

    audit_columns = [
        column
        for column in audit.columns
        if column
        not in {
            "split",
            "historical_ground_truth_siret",
            "historical_ground_truth_siren",
        }
    ]
    output = v2.merge(
        audit[audit_columns],
        on="query_id",
        validate="one_to_one",
    )
    output["v2_label_kind"] = output["label_kind"].astype(str)
    output["v2_qualification_reason"] = output[
        "qualification_reason"
    ].astype(str)
    no_evidence = (
        output["v2_label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
        & output["direct_evidence_class"].eq("NO_DIRECT_EVIDENCE")
    )
    output.loc[no_evidence, "label_kind"] = GroundTruthKind.UNRESOLVED.value
    output.loc[
        no_evidence,
        "qualification_reason",
    ] = "NO_DIRECT_CRM_TO_HISTORICAL_SIRET_EVIDENCE"
    output["exact_metric_eligible"] = output["label_kind"].eq(
        GroundTruthKind.MATCH_EXACT.value
    )
    open_mask = ~output["exact_metric_eligible"]
    output.loc[open_mask, "ground_truth_siret"] = None
    output.loc[open_mask, "ground_truth_siren"] = None
    output["evidence_policy_version"] = POLICY_VERSION
    output["evidence_is_human_validated"] = False
    return output.sort_values(
        "query_id",
        key=lambda values: values.astype(str),
    ).reset_index(drop=True)


def evaluate_retrieval(
    qualified: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "query_id",
        "hit_at_100",
        "baseline_hit_at_100",
        "oracle_hit",
        "candidate_count",
    }
    missing = required - set(retrieval.columns)
    if missing:
        raise ValueError(f"Retrieval artifact is missing columns: {sorted(missing)}")
    raw = retrieval.copy()
    raw["query_id"] = raw["query_id"].astype(str)
    scope = qualified[
        [
            "query_id",
            "v2_label_kind",
            "label_kind",
            "direct_evidence_class",
            "exact_metric_eligible",
        ]
    ].copy()
    scope["query_id"] = scope["query_id"].astype(str)
    if set(raw["query_id"]) != set(scope["query_id"]):
        raise ValueError("Qualified benchmark and retrieval query IDs differ")
    if raw["candidate_count"].astype(int).gt(100).any():
        raise ValueError("Retrieval artifact exceeds the 100-candidate ceiling")
    merged = raw.merge(scope, on="query_id", validate="one_to_one")
    v2_exact = merged[
        merged["v2_label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    ]
    v3_exact = merged[
        merged["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    ]

    def metric_set(frame: pd.DataFrame) -> dict[str, Any]:
        successes = int(frame["hit_at_100"].astype(bool).sum())
        required_successes = math.ceil(0.99 * len(frame))
        return {
            "admission_at_100": _binary_metric(frame["hit_at_100"]),
            "frozen_sparse_at_100": _binary_metric(
                frame["baseline_hit_at_100"]
            ),
            "internal_oracle": _binary_metric(frame["oracle_hit"]),
            "target_rate": 0.99,
            "required_successes": required_successes,
            "gap_successes": max(0, required_successes - successes),
            "losses": {
                "not_seen_by_internal_oracle": int(
                    (~frame["oracle_hit"].astype(bool)).sum()
                ),
                "seen_then_pruned": int(
                    (
                        frame["oracle_hit"].astype(bool)
                        & ~frame["hit_at_100"].astype(bool)
                    ).sum()
                ),
            },
        }

    by_evidence = {}
    for evidence_class, group in merged.groupby(
        "direct_evidence_class",
        sort=True,
    ):
        by_evidence[str(evidence_class)] = {
            "query_count": int(len(group)),
            "historical_label_hit_at_100": _binary_metric(
                group["hit_at_100"]
            ),
            "historical_label_oracle": _binary_metric(group["oracle_hit"]),
        }
    return {
        "historical_all_queries": metric_set(merged),
        "v2_exact_metric": metric_set(v2_exact),
        "v3_exact_metric": metric_set(v3_exact),
        "by_direct_evidence_class": by_evidence,
        "candidate_ceiling": {
            "max": int(merged["candidate_count"].astype(int).max()),
            "over_100": int(
                merged["candidate_count"].astype(int).gt(100).sum()
            ),
        },
        "review_routing_not_measured": True,
    }


def summarize(qualified: pd.DataFrame) -> dict[str, Any]:
    return {
        "query_count": int(len(qualified)),
        "label_counts": {
            str(key): int(value)
            for key, value in qualified["label_kind"]
            .value_counts()
            .sort_index()
            .items()
        },
        "v2_label_counts": {
            str(key): int(value)
            for key, value in qualified["v2_label_kind"]
            .value_counts()
            .sort_index()
            .items()
        },
        "direct_evidence_class_counts": {
            str(key): int(value)
            for key, value in qualified["direct_evidence_class"]
            .value_counts()
            .sort_index()
            .items()
        },
        "moved_from_v2_exact_to_unresolved": int(
            (
                qualified["v2_label_kind"].eq(
                    GroundTruthKind.MATCH_EXACT.value
                )
                & qualified["label_kind"].eq(
                    GroundTruthKind.UNRESOLVED.value
                )
            ).sum()
        ),
        "exact_metric_eligible": int(
            qualified["exact_metric_eligible"].astype(bool).sum()
        ),
        "human_validated": False,
        "automatic_relabels": 0,
    }


def _load_joined(
    *,
    benchmark_path: Path,
    establishment_snapshot: Path,
    legal_unit_snapshot: Path,
    split: str,
) -> pd.DataFrame:
    connection = duckdb.connect()
    return connection.execute(
        """
        SELECT
            b.*,
            CAST(e.numeroVoieEtablissement AS VARCHAR) AS numeroVoie,
            e.typeVoieEtablissement AS typeVoie,
            e.libelleVoieEtablissement AS libelleVoie,
            e.complementAdresseEtablissement AS complementAdresse,
            e.codePostalEtablissement AS candidate_postcode,
            e.libelleCommuneEtablissement AS candidate_city,
            e.codeCommuneEtablissement AS candidate_insee,
            e.etablissementSiege AS is_siege,
            e.enseigne1Etablissement AS enseigne1,
            e.enseigne2Etablissement AS enseigne2,
            e.enseigne3Etablissement AS enseigne3,
            e.denominationUsuelleEtablissement AS denomination,
            u.categorieJuridiqueUniteLegale AS cj_ul,
            u.sigleUniteLegale AS sigle_ul,
            u.denominationUniteLegale AS denomination_ul,
            CONCAT_WS(
                ' ',
                u.denominationUsuelle1UniteLegale,
                u.denominationUsuelle2UniteLegale,
                u.denominationUsuelle3UniteLegale
            ) AS denomination_usuelle_ul,
            u.nomUniteLegale AS nom_ul,
            u.nomUsageUniteLegale AS nom_usage_ul,
            u.prenomUsuelUniteLegale AS prenom_usuel_ul,
            u.pseudonymeUniteLegale AS pseudonyme_ul
        FROM read_parquet(?) b
        JOIN read_parquet(?) e
          ON b.ground_truth_siret = CAST(e.siret AS VARCHAR)
        LEFT JOIN read_parquet(?) u
          ON b.ground_truth_siren = CAST(u.siren AS VARCHAR)
        WHERE b.split = ?
        ORDER BY CAST(b.query_id AS BIGINT)
        """,
        [
            benchmark_path.resolve().as_posix(),
            establishment_snapshot.resolve().as_posix(),
            legal_unit_snapshot.resolve().as_posix(),
            split,
        ],
    ).df()


def _report(
    *,
    summary: dict[str, Any],
    metrics: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# Qualification V3 par preuve directe",
        "",
        f"- Build : `{manifest['build_id']}`",
        f"- Split : `{manifest['split']}`",
        f"- Requêtes : {summary['query_count']}",
        "- Statut : rétrospectif, mécanique, non certifié humainement.",
        "",
        "## Qualification",
        "",
        f"- `MATCH_EXACT` : {summary['label_counts'].get('MATCH_EXACT', 0)} ;",
        f"- `AMBIGUOUS` : {summary['label_counts'].get('AMBIGUOUS', 0)} ;",
        f"- `UNRESOLVED` : {summary['label_counts'].get('UNRESOLVED', 0)} ;",
        f"- V2 exacts passés en non résolu faute de preuve directe : "
        f"{summary['moved_from_v2_exact_to_unresolved']} ;",
        "- remplacement automatique : 0.",
        "",
    ]
    if metrics:
        for label, key in [
            ("Historique", "historical_all_queries"),
            ("V2 exact", "v2_exact_metric"),
            ("V3 exact identifiable", "v3_exact_metric"),
        ]:
            metric = metrics[key]["admission_at_100"]
            lines.append(
                f"- {label} : {metric['successes']}/{metric['total']} "
                f"= {metric['rate']:.3%}."
            )
        v3 = metrics["v3_exact_metric"]
        lines.extend(
            [
                "",
                f"Écart V3 au gate de 99 % : {v3['gap_successes']} requêtes. "
                f"Pertes : {v3['losses']['not_seen_by_internal_oracle']} non "
                f"vue(s), {v3['losses']['seen_then_pruned']} vue(s) puis "
                "éliminée(s).",
                "",
            ]
        )
    lines.extend(
        [
            "Ce résultat ne vaut pas validation aveugle : la politique a été "
            "définie après observation du dev. Le test reste fermé.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--v2-benchmark", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--establishment-snapshot", type=Path, required=True)
    parser.add_argument("--legal-unit-snapshot", type=Path, required=True)
    parser.add_argument("--policy-document", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--retrieval-raw", type=Path)
    parser.add_argument("--retrieval-manifest", type=Path)
    args = parser.parse_args()
    if bool(args.retrieval_raw) != bool(args.retrieval_manifest):
        raise ValueError(
            "--retrieval-raw and --retrieval-manifest must be provided together"
        )

    benchmark_manifest = json.loads(
        args.benchmark_manifest.read_text(encoding="utf-8")
    )
    v2_manifest = json.loads(args.v2_manifest.read_text(encoding="utf-8"))
    _validate_bound_file(
        args.benchmark,
        benchmark_manifest,
        manifest_section="output_sha256",
    )
    _validate_bound_file(
        args.v2_benchmark,
        v2_manifest,
        manifest_section="outputs",
    )
    if v2_manifest.get("benchmark_build_id") != benchmark_manifest.get(
        "build_id"
    ):
        raise ValueError("V2 and historical benchmark build IDs differ")
    if v2_manifest.get("split") != args.split:
        raise ValueError("V2 split differs from requested split")
    if file_sha256(args.establishment_snapshot) != benchmark_manifest.get(
        "establishment_snapshot_sha256"
    ):
        raise ValueError("Establishment snapshot hash mismatch")
    if file_sha256(args.legal_unit_snapshot) != benchmark_manifest.get(
        "legal_unit_snapshot_sha256"
    ):
        raise ValueError("Legal-unit snapshot hash mismatch")

    joined = _load_joined(
        benchmark_path=args.benchmark,
        establishment_snapshot=args.establishment_snapshot,
        legal_unit_snapshot=args.legal_unit_snapshot,
        split=args.split,
    )
    evidence = audit_evidence_rows(joined)
    v2_benchmark = pd.read_parquet(args.v2_benchmark)
    qualified = apply_evidence_policy(v2_benchmark, evidence)
    summary = summarize(qualified)

    metrics = None
    retrieval_manifest: dict[str, Any] | None = None
    if args.retrieval_raw and args.retrieval_manifest:
        retrieval_manifest = json.loads(
            args.retrieval_manifest.read_text(encoding="utf-8")
        )
        _validate_bound_file(
            args.retrieval_raw,
            retrieval_manifest,
            manifest_section="outputs",
        )
        if retrieval_manifest.get("benchmark_build_id") != benchmark_manifest.get(
            "build_id"
        ):
            raise ValueError("Retrieval and benchmark build IDs differ")
        if retrieval_manifest.get("split") != args.split:
            raise ValueError("Retrieval split differs from requested split")
        metrics = evaluate_retrieval(
            qualified,
            pd.read_parquet(args.retrieval_raw),
        )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "policy_document_sha256": file_sha256(args.policy_document),
        "benchmark_build_id": benchmark_manifest["build_id"],
        "historical_benchmark_sha256": file_sha256(args.benchmark),
        "v2_benchmark_sha256": file_sha256(args.v2_benchmark),
        "establishment_snapshot_sha256": file_sha256(
            args.establishment_snapshot
        ),
        "legal_unit_snapshot_sha256": file_sha256(args.legal_unit_snapshot),
        "split": args.split,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = args.output_root / build_id
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    paths = {
        "benchmark.parquet": output_dir / "benchmark.parquet",
        "exact_benchmark.parquet": output_dir / "exact_benchmark.parquet",
        "evidence.parquet": output_dir / "evidence.parquet",
        "labels.parquet": output_dir / "labels.parquet",
        "summary.json": output_dir / "summary.json",
        "report.md": output_dir / "report.md",
    }
    qualified.to_parquet(paths["benchmark.parquet"], index=False)
    qualified[qualified["exact_metric_eligible"]].to_parquet(
        paths["exact_benchmark.parquet"],
        index=False,
    )
    evidence.to_parquet(paths["evidence.parquet"], index=False)
    label_columns = [
        "query_id",
        "split",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
        "historical_ground_truth_siret",
        "historical_ground_truth_siren",
        "v2_label_kind",
        "direct_evidence_class",
        "qualification_reason",
        "exact_metric_eligible",
        "evidence_policy_version",
        "evidence_is_human_validated",
    ]
    qualified[label_columns].to_parquet(paths["labels.parquet"], index=False)
    paths["summary.json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if metrics is not None:
        paths["retrieval_metrics.json"] = (
            output_dir / "retrieval_metrics.json"
        )
        paths["retrieval_metrics.json"].write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "status": "PROVISIONAL_RETROSPECTIVE_EVIDENCE_QUALIFICATION",
        "human_validation_required_for_label_correction": True,
        "automatic_relabels": 0,
        "source_test_untouched": True,
        "inputs": {
            "benchmark_manifest_sha256": file_sha256(args.benchmark_manifest),
            "v2_manifest_sha256": file_sha256(args.v2_manifest),
            "retrieval_manifest_sha256": (
                file_sha256(args.retrieval_manifest)
                if args.retrieval_manifest
                else None
            ),
            "retrieval_raw_sha256": (
                file_sha256(args.retrieval_raw) if args.retrieval_raw else None
            ),
        },
    }
    paths["report.md"].write_text(
        _report(summary=summary, metrics=metrics, manifest=manifest),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        name: file_sha256(path) for name, path in paths.items()
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": summary,
                "retrieval_metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
