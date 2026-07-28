#!/usr/bin/env python3
"""Build the immutable V4.11 registry of already-consumed CRM rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "sireto-v4.11-consumed-population-registry-1"
NORMALIZATION_VERSION = "nfkc-upper-space-v1"
CRM_COLUMNS = (
    "SITE",
    "CODE_POSTAL",
    "CODE_INSEE",
    "SERVICE ID",
    "COMMUNE",
    "SIRET",
    "SITE_CLI_ADRESSE",
    "SITE_CLI_COMMUNE",
)
EXPECTED_HASHES = {
    "crm": "f770215cd0d0fcc654b750b90dbba835acbf4efb5c74ed269d339e046c2b049d",
    "closed": "4c533813218dced6627da238b885db47e45745d784ae9078a4aaa836680308b6",
    "fresh": "0effe19ae7f649ee6a03e73c0858d8f87710015f630971495a8cc5a2461b8279",
}
EXPECTED_COUNTS = {
    "crm": 23_609,
    "closed": 17_054,
    "fresh": 6_330,
    "consumed": 23_384,
    "unseen": 225,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.strip().upper().split())


def normalize_siret(value: Any) -> str:
    digits = "".join(character for character in canonical_text(value) if character.isdigit())
    return digits if len(digits) == 14 else ""


def row_fingerprint(row: pd.Series) -> str:
    payload = {column: canonical_text(row[column]) for column in CRM_COLUMNS}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_nonempty(values: pd.Series) -> set[str]:
    return {value for value in values.map(canonical_text).tolist() if value}


def _normalized_sirets(values: pd.Series) -> set[str]:
    return {value for value in values.map(normalize_siret).tolist() if value}


def build_registry(
    crm_path: Path,
    closed_path: Path,
    fresh_path: Path,
    *,
    verify_expected: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_hashes = {
        "crm": file_sha256(crm_path),
        "closed": file_sha256(closed_path),
        "fresh": file_sha256(fresh_path),
    }
    if verify_expected and input_hashes != EXPECTED_HASHES:
        raise RuntimeError(
            f"STOP_INPUT_DRIFT: expected {EXPECTED_HASHES}, observed {input_hashes}"
        )

    crm = pd.read_csv(
        crm_path,
        sep=";",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    if tuple(crm.columns) != CRM_COLUMNS:
        raise RuntimeError(
            f"STOP_REGISTRY_INTEGRITY: unexpected CRM columns {list(crm.columns)}"
        )
    closed = pd.read_parquet(closed_path)
    fresh = pd.read_parquet(fresh_path)

    if verify_expected:
        observed = {"crm": len(crm), "closed": len(closed), "fresh": len(fresh)}
        expected = {key: EXPECTED_COUNTS[key] for key in observed}
        if observed != expected:
            raise RuntimeError(
                f"STOP_REGISTRY_INTEGRITY: expected counts {expected}, observed {observed}"
            )

    closed_services = _normalized_nonempty(closed["crm_record_id"])
    fresh_services = _normalized_nonempty(fresh["crm_record_id"])
    closed_sirets = _normalized_sirets(closed["ground_truth_siret"])
    fresh_sirets = _normalized_sirets(fresh["historical_ground_truth_siret"])

    registry = crm.copy()
    registry.insert(0, "source_row_number", range(1, len(registry) + 1))
    registry["service_id_norm"] = registry["SERVICE ID"].map(canonical_text)
    registry["input_siret_norm"] = registry["SIRET"].map(normalize_siret)
    registry["row_fingerprint_sha256"] = registry.apply(row_fingerprint, axis=1)
    registry["source_key"] = registry.apply(
        lambda row: (
            f"SERVICE:{row['service_id_norm']}"
            if row["service_id_norm"]
            else f"ROW:{row['source_row_number']}:{row['row_fingerprint_sha256']}"
        ),
        axis=1,
    )
    registry["matched_closed_by_service"] = registry["service_id_norm"].isin(
        closed_services
    )
    registry["matched_closed_by_siret"] = registry["input_siret_norm"].isin(
        closed_sirets
    )
    registry["matched_fresh_by_service"] = registry["service_id_norm"].isin(
        fresh_services
    )
    registry["matched_fresh_by_siret"] = registry["input_siret_norm"].isin(
        fresh_sirets
    )
    match_columns = [
        "matched_closed_by_service",
        "matched_closed_by_siret",
        "matched_fresh_by_service",
        "matched_fresh_by_siret",
    ]
    registry["consumed_closed"] = registry[
        ["matched_closed_by_service", "matched_closed_by_siret"]
    ].any(axis=1)
    registry["consumed_fresh"] = registry[
        ["matched_fresh_by_service", "matched_fresh_by_siret"]
    ].any(axis=1)
    registry["population_status"] = registry[match_columns].any(axis=1).map(
        {True: "CONSUMED", False: "UNSEEN"}
    )
    registry["consumption_sources"] = registry.apply(
        lambda row: "|".join(
            source
            for source, present in (
                ("CLOSED", bool(row["consumed_closed"])),
                ("V4_FRESH", bool(row["consumed_fresh"])),
            )
            if present
        ),
        axis=1,
    )

    duplicate_source_keys = int(registry["source_key"].duplicated().sum())
    duplicate_fingerprints = int(registry["row_fingerprint_sha256"].duplicated().sum())
    closed_count = int(registry["consumed_closed"].sum())
    fresh_count = int(registry["consumed_fresh"].sum())
    overlap_count = int(
        (registry["consumed_closed"] & registry["consumed_fresh"]).sum()
    )
    consumed_count = int(registry["population_status"].eq("CONSUMED").sum())
    unseen_count = int(registry["population_status"].eq("UNSEEN").sum())
    unseen_service_missing = int(
        (
            registry["population_status"].eq("UNSEEN")
            & registry["service_id_norm"].eq("")
        ).sum()
    )

    integrity = {
        "source_key_unique": duplicate_source_keys == 0,
        "all_rows_have_fingerprint": bool(
            registry["row_fingerprint_sha256"].str.len().eq(64).all()
        ),
        "closed_source_rows": closed_count,
        "fresh_source_rows": fresh_count,
        "closed_fresh_source_overlap": overlap_count,
        "consumed_rows": consumed_count,
        "unseen_rows": unseen_count,
        "unseen_missing_service_id": unseen_service_missing,
        "duplicate_source_key_count": duplicate_source_keys,
        "duplicate_fingerprint_count": duplicate_fingerprints,
    }
    if verify_expected:
        expected_integrity = {
            "closed_source_rows": EXPECTED_COUNTS["closed"],
            "fresh_source_rows": EXPECTED_COUNTS["fresh"],
            "closed_fresh_source_overlap": 0,
            "consumed_rows": EXPECTED_COUNTS["consumed"],
            "unseen_rows": EXPECTED_COUNTS["unseen"],
            "unseen_missing_service_id": EXPECTED_COUNTS["unseen"],
        }
        failures = {
            key: {"expected": expected, "observed": integrity[key]}
            for key, expected in expected_integrity.items()
            if integrity[key] != expected
        }
        if not integrity["source_key_unique"]:
            failures["source_key_unique"] = {"expected": True, "observed": False}
        if not integrity["all_rows_have_fingerprint"]:
            failures["all_rows_have_fingerprint"] = {
                "expected": True,
                "observed": False,
            }
        if failures:
            raise RuntimeError(f"STOP_REGISTRY_INTEGRITY: {failures}")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "input_hashes": input_hashes,
        "input_counts": {
            "crm": len(crm),
            "closed": len(closed),
            "fresh": len(fresh),
        },
        "integrity": integrity,
    }
    return registry, audit


def write_build(
    registry: pd.DataFrame,
    audit: dict[str, Any],
    output_root: Path,
    script_path: Path,
) -> Path:
    script_hash = file_sha256(script_path)
    build_spec = {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "input_hashes": audit["input_hashes"],
        "script_sha256": script_hash,
    }
    build_id = hashlib.sha256(
        json.dumps(build_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    destination = output_root / build_id
    if destination.exists():
        raise FileExistsError(f"immutable build already exists: {destination}")

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=output_root))
    try:
        registry_path = staging / "source_registry.parquet"
        consumed_path = staging / "consumed.parquet"
        unseen_path = staging / "unseen.parquet"
        registry.to_parquet(registry_path, index=False)
        registry[registry["population_status"].eq("CONSUMED")].to_parquet(
            consumed_path, index=False
        )
        registry[registry["population_status"].eq("UNSEEN")].to_parquet(
            unseen_path, index=False
        )
        manifest = {
            **build_spec,
            "build_id": build_id,
            "contract": "V411_CONSUMED_POPULATION_REGISTRY",
            "verdict": "PASS_REGISTRY",
            "proof_scope": "lineage_only_not_model_validation",
            **audit,
            "outputs": {
                name: {
                    "sha256": file_sha256(path),
                    "rows": int(pd.read_parquet(path, columns=["source_key"]).shape[0]),
                }
                for name, path in (
                    ("source_registry.parquet", registry_path),
                    ("consumed.parquet", consumed_path),
                    ("unseen.parquet", unseen_path),
                )
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crm", type=Path, default=Path("data/entrainements.csv"))
    parser.add_argument(
        "--closed",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_V9/benchmarks/closed/"
            "c33b80855f560074/benchmark.parquet"
        ),
    )
    parser.add_argument(
        "--fresh",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/"
            "v4_fresh_expansion/14047b719ef90f6f/pool/benchmark.parquet"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/"
            "v4_11_consumed_population"
        ),
    )
    parser.add_argument("--skip-expected-checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry, audit = build_registry(
        args.crm,
        args.closed,
        args.fresh,
        verify_expected=not args.skip_expected_checks,
    )
    destination = write_build(
        registry, audit, args.output_root, Path(__file__).resolve()
    )
    print(destination)


if __name__ == "__main__":
    main()
