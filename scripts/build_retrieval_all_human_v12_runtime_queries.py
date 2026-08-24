#!/usr/bin/env python3
"""Build the sealed label-blind query projection used by the v12 profiler.

The source is the exact truth-bearing input whose bytes are pinned by the v11
candidate manifest.  This builder predicate-scans folds 2/3/4 and requests a
strict allowlist of query fields from Arrow; truth columns are never converted
to Python or written.  The output directory is published atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.profile_retrieval_all_human_v12 import (  # noqa: E402
    ALLOWED_FOLDS,
    QUERY_SOURCE_MANIFEST_SCHEMA,
)
from scripts.run_retrieval_top100_cascade_v12 import (  # noqa: E402
    V11_CANDIDATE_SCHEMA,
    V11_LABEL_BLIND_RUN_KINDS,
    V11_SOURCE_MANIFEST_SCHEMA,
)


SAFE_QUERY_COLUMNS = (
    "query_id",
    "oof_fold",
    "fold",
    "crm_name",
    "crm_name_raw",
    "name",
    "nom",
    "crm_addr",
    "crm_address",
    "crm_address_raw",
    "address",
    "adresse",
    "numeroVoie",
    "number",
    "numero",
    "postcode",
    "crm_cp",
    "crm_postcode",
    "crm_postcode_raw",
    "insee",
    "crm_insee",
    "crm_insee_raw",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _fold_column(names: set[str]) -> str:
    if "oof_fold" in names:
        return "oof_fold"
    if "fold" in names:
        return "fold"
    raise ValueError("v12 runtime query source has no fold/oof_fold")


def _is_scalar_query_type(dtype: pa.DataType) -> bool:
    return not (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
        or pa.types.is_struct(dtype)
        or pa.types.is_map(dtype)
        or pa.types.is_union(dtype)
    )


def build(args: argparse.Namespace) -> Path:
    source = _require_file(args.source, "v11 source input")
    v11_manifest_path = _require_file(
        args.v11_candidate_manifest, "v11 candidate manifest"
    )
    v11 = _json(v11_manifest_path)
    if (
        v11.get("schema_version") != V11_SOURCE_MANIFEST_SCHEMA
        or v11.get("candidate_schema_version") != V11_CANDIDATE_SCHEMA
        or v11.get("run_kind") not in V11_LABEL_BLIND_RUN_KINDS
        or v11.get("positive_injection") is not False
        or v11.get("human_labels_opened_by_retrieval") is not False
        or set(v11.get("folds") or ()) != ALLOWED_FOLDS
        or v11.get("source_input_sha256") != sha256_file(source)
    ):
        raise ValueError(
            "projection source is not the exact folds-2/3/4 input sealed by v11"
        )

    dataset = ds.dataset(source, format="parquet")
    names = set(dataset.schema.names)
    fold_column = _fold_column(names)
    if "query_id" not in names:
        raise ValueError("v12 runtime query source has no query_id")
    columns = [name for name in SAFE_QUERY_COLUMNS if name in names]
    if fold_column not in columns:
        columns.insert(1, fold_column)
    for name in columns:
        if not _is_scalar_query_type(dataset.schema.field(name).type):
            raise ValueError(f"query projection refuses non-scalar column {name}")

    # Arrow reads only the allowlisted query fields.  Human SIRETs and
    # operational alternatives remain physically unmaterialised.
    table = dataset.to_table(
        columns=columns,
        filter=ds.field(fold_column).isin(sorted(ALLOWED_FOLDS)),
    )
    if table.num_rows < 1:
        raise ValueError("query projection contains no folds-2/3/4 rows")
    fold_values = {int(value) for value in table[fold_column].to_pylist()}
    if fold_values != ALLOWED_FOLDS:
        raise ValueError("query projection must contain every fold 2/3/4 and no fold 0/1")
    query_ids = [str(value or "") for value in table["query_id"].to_pylist()]
    if "" in query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("query projection query_id values must be non-empty and unique")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        query_path = stage / "queries.parquet"
        pq.write_table(table, query_path, compression="zstd")
        manifest = {
            "schema_version": QUERY_SOURCE_MANIFEST_SCHEMA,
            "run_kind": "LABEL_BLIND_QUERY_PROJECTION",
            "projection_operation": "ARROW_ALLOWLIST_AND_FOLDS_234_FILTER_ONLY",
            "source_input_sha256": sha256_file(source),
            "v11_candidate_manifest_sha256": sha256_file(v11_manifest_path),
            "query_parquet_sha256": sha256_file(query_path),
            "projection_code_sha256": sha256_file(Path(__file__).resolve()),
            "rows": int(table.num_rows),
            "projection_columns": list(table.column_names),
            "folds": [2, 3, 4],
            "fold_counts": {
                str(fold): sum(int(value) == fold for value in table[fold_column].to_pylist())
                for fold in sorted(ALLOWED_FOLDS)
            },
            "fold0_opened": False,
            "fold1_opened": False,
            "labels_present": False,
            "positive_injection": False,
            "query_id_unique": True,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v11-candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    print(build(parse_args()))


if __name__ == "__main__":
    main()
