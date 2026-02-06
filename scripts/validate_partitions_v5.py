"""Validate candidate partitions v5 (schema + duplicates + filters).

Stop-the-line validator for data/candidates_v6_all.
"""

from __future__ import annotations

import argparse
import csv
import re
import random
import sys
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.dataset as ds


EXPECTED_SCHEMA: Dict[str, pa.DataType] = {
    "siret": pa.string(),
    "siren": pa.string(),
    "denomination": pa.string(),
    "enseigne1": pa.string(),
    "enseigne2": pa.string(),
    "enseigne3": pa.string(),
    "etablissementSiege": pa.bool_(),
    "is_siege": pa.bool_(),
    "numeroVoie": pa.string(),
    "typeVoie": pa.string(),
    "libelleVoie": pa.string(),
    "complementAdresse": pa.string(),
    "postcode": pa.string(),
    "city": pa.string(),
    "insee": pa.string(),
    "cj_ul": pa.string(),
    "etat_admin": pa.string(),
    "last_treatment_date": pa.timestamp("us"),
    "sigle_ul": pa.string(),
    "denomination_ul": pa.string(),
    "denomination_usuelle_ul": pa.string(),
    "nom_ul": pa.string(),
    "prenom_usuel_ul": pa.string(),
    "pm_dirigeant_names": pa.string(),
}


def _schema_issues(dataset: ds.Dataset) -> List[str]:
    issues: List[str] = []
    schema = dataset.schema
    for col, expected_type in EXPECTED_SCHEMA.items():
        if col not in schema.names:
            issues.append(f"missing column: {col}")
            continue
        actual = schema.field(col).type
        if actual != expected_type:
            issues.append(f"type mismatch for {col}: {actual} != {expected_type}")
    return issues


def _partition_dirs(root: Path, key: str) -> List[Path]:
    if not root.exists():
        return []
    prefix = f"{key}="
    return [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    if path.name.startswith(prefix):
        return path.name[len(prefix):]
    return ""


def _check_duplicates_in_partition(path: Path, key: str) -> List[str]:
    issues: List[str] = []
    dataset = ds.dataset(
        path,
        format="parquet",
        partitioning="hive",
        schema=pa.schema([(k, v) for k, v in EXPECTED_SCHEMA.items()]),
    )
    try:
        table = dataset.to_table(columns=["siret", key])
    except Exception as exc:
        return [f"failed to read partition {path}: {exc}"]

    if table.num_rows == 0:
        return issues

    df = table.to_pandas()
    if df["siret"].isna().any():
        issues.append(f"null siret in partition {path}")
    if df["siret"].duplicated().any():
        issues.append(f"duplicate siret in partition {path}")

    if key in df.columns:
        uniq = df[key].dropna().unique()
        if len(uniq) > 1:
            issues.append(f"multiple {key} values in partition {path}")
    return issues


def _check_filter(dataset: ds.Dataset, key: str, value_raw: str) -> List[str]:
    issues: List[str] = []
    try:
        table = dataset.to_table(filter=ds.field(key) == value_raw, columns=["siret", key])
    except Exception as exc:
        return [f"filter failed for {key}={value_raw}: {exc}"]
    if table.num_rows == 0:
        issues.append(f"empty filter result for {key}={value_raw}")
        return issues
    df = table.to_pandas()
    if df[key].dropna().nunique() > 1:
        issues.append(f"filter result contains multiple {key} values for {key}={value_raw}")
    return issues


def _sample(items: List[Path], count: int, seed: int) -> List[Path]:
    if count <= 0 or count >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, count)


def _check_partitions(
    root: Path,
    key: str,
    sample_partitions: int,
    seed: int,
    full_scan: bool,
    check_filters: bool,
    require_corsica: bool,
) -> List[str]:
    issues: List[str] = []
    dataset = ds.dataset(
        root,
        format="parquet",
        partitioning="hive",
        schema=pa.schema([(k, v) for k, v in EXPECTED_SCHEMA.items()]),
    )
    issues.extend(_schema_issues(dataset))

    partitions = _partition_dirs(root, key)
    if not partitions:
        issues.append(f"no partitions found for {key} under {root}")
        return issues

    if key == "insee" and require_corsica:
        has_corsica = any(
            p.name.startswith("insee=2A") or p.name.startswith("insee=2B")
            for p in partitions
        )
        if not has_corsica:
            issues.append("missing Corsica partitions (insee=2A/2B)")

    targets = partitions if full_scan else _sample(partitions, sample_partitions, seed)
    for part in targets:
        issues.extend(_check_duplicates_in_partition(part, key))

    if check_filters:
        for part in targets[: min(5, len(targets))]:
            value_raw = _partition_value(part, key)
            issues.extend(_check_filter(dataset, key, value_raw))

    return issues


def _normalize_code(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s or None


def _scope_has_corsica(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ";"
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            insee = row.get("insee") or row.get("CODE_INSEE") or row.get("crm_insee")
            insee = _normalize_code(insee) if insee is not None else None
            if insee in {"2A", "2B"}:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate candidate partitions v5")
    parser.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v6_all"))
    parser.add_argument("--sample-partitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-scan", action="store_true")
    parser.add_argument("--skip-filter-check", action="store_true")
    parser.add_argument("--require-corsica", action="store_true", default=False)
    parser.add_argument(
        "--scope-csv",
        type=Path,
        default=None,
        help="Optional scope CSV. If it contains 2A/2B INSEE codes, Corsica is required.",
    )
    args = parser.parse_args()

    base = args.partitions_dir
    insee_root = base / "insee"
    cp_root = base / "cp"

    all_issues: List[str] = []
    if not insee_root.exists() or not cp_root.exists():
        print(f"ERROR: partitions not found under {base}")
        return 2

    require_corsica = args.require_corsica
    if not require_corsica and args.scope_csv and args.scope_csv.exists():
        require_corsica = _scope_has_corsica(args.scope_csv)

    all_issues.extend(
        _check_partitions(
            insee_root,
            key="insee",
            sample_partitions=args.sample_partitions,
            seed=args.seed,
            full_scan=args.full_scan,
            check_filters=not args.skip_filter_check,
            require_corsica=require_corsica,
        )
    )
    all_issues.extend(
        _check_partitions(
            cp_root,
            key="postcode",
            sample_partitions=args.sample_partitions,
            seed=args.seed,
            full_scan=args.full_scan,
            check_filters=not args.skip_filter_check,
            require_corsica=False,
        )
    )

    if all_issues:
        print("ERROR: validation failed")
        for issue in all_issues:
            print(f"- {issue}")
        return 1

    print("OK: partitions validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
