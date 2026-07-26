#!/usr/bin/env python3
"""Compare V4.1 retrieval variants A/B/C on an explicit development set."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.features import preprocess_crm_row  # noqa: E402
from src.xgb_matcher.partitioned_store import (  # noqa: E402
    PartitionedCandidateStore,
)
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache  # noqa: E402
from src.xgb_matcher.v41_retrieval import (  # noqa: E402
    V41CandidateRetriever,
    V41GlobalCandidateStore,
    V41RetrievalConfig,
    V41RetrievalVariant,
    normalize_input_siret,
)
from src.xgb_matcher.v9_dataset import file_sha256, read_table  # noqa: E402


SCHEMA_VERSION = "sireto-v4.1-retrieval-evaluation-1"
VARIANT_ORDER = ("A", "B", "C")
FORBIDDEN_BENCHMARK_TOKENS = ("test", "holdout")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _rate(values: pd.Series) -> dict[str, Any]:
    boolean = values.fillna(False).astype(bool)
    total = int(len(boolean))
    successes = int(boolean.sum())
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else 0.0,
    }


def _p95(values: pd.Series) -> float:
    return float(np.quantile(values.astype(float), 0.95)) if len(values) else 0.0


def _normalize_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    output = str(value).strip()
    return output if output and output.lower() != "nan" else None


def assert_dev_only(
    benchmark: pd.DataFrame,
    *,
    benchmark_path: Path,
    split: str,
    benchmark_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Reject any test/final population before retrieval is instantiated."""

    if split.strip().lower() != "dev":
        raise ValueError("V4.1 retrieval comparison is restricted to split=dev")
    lowered_parts = [part.lower() for part in benchmark_path.parts]
    filename_tokens = set(
        filter(None, re.split(r"[^a-z0-9]+", benchmark_path.stem.lower()))
    )
    if (
        any(part in FORBIDDEN_BENCHMARK_TOKENS for part in lowered_parts[:-1])
        or bool(filename_tokens & set(FORBIDDEN_BENCHMARK_TOKENS))
    ):
        raise ValueError("Refusing a benchmark path marked test/holdout")
    if "split" not in benchmark:
        raise ValueError("Benchmark must carry an explicit split column")
    observed_splits = {
        str(value).strip().lower()
        for value in benchmark["split"].dropna().unique()
    }
    fresh_dev_only = (
        observed_splits == {"fresh"}
        and "fresh_role" in benchmark
        and {
            str(value).strip().lower()
            for value in benchmark["fresh_role"].dropna().unique()
        }
        == {"dev_new"}
    )
    if observed_splits != {"dev"} and not fresh_dev_only:
        raise ValueError(
            "Benchmark file must contain dev rows only; test/holdout/train "
            f"rows were not read selectively: {sorted(observed_splits)}"
        )
    for column in ("holdout_opened", "is_holdout", "is_test"):
        if column in benchmark and benchmark[column].fillna(False).astype(bool).any():
            raise ValueError(f"Benchmark is marked as forbidden by {column}")
    if benchmark_manifest is not None:
        manifest_role = " ".join(
            str(benchmark_manifest.get(key) or "")
            for key in ("split", "role", "purpose", "status")
        ).lower()
        if any(token in manifest_role for token in FORBIDDEN_BENCHMARK_TOKENS):
            raise ValueError("Benchmark manifest is marked test/holdout")


def load_dev_benchmark(
    benchmark_path: Path,
    *,
    split: str = "dev",
    benchmark_manifest: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    benchmark = read_table(benchmark_path)
    assert_dev_only(
        benchmark,
        benchmark_path=benchmark_path,
        split=split,
        benchmark_manifest=benchmark_manifest,
    )
    required = {
        "query_id",
        "crm_record_id",
        "crm_name",
        "crm_address",
        "crm_city",
        "postcode",
        "insee",
        "ground_truth_siret",
    }
    missing = sorted(required - set(benchmark.columns))
    if missing:
        raise ValueError(f"Benchmark is missing columns: {missing}")
    benchmark = benchmark.copy()
    if "label_kind" in benchmark:
        benchmark = benchmark[
            benchmark["label_kind"].astype(str).eq("MATCH_EXACT")
        ].copy()
        if benchmark.empty:
            raise ValueError("Development benchmark has no MATCH_EXACT rows")
    benchmark["query_id"] = benchmark["query_id"].map(_normalize_id)
    benchmark["crm_record_id"] = benchmark["crm_record_id"].map(_normalize_id)
    if benchmark["query_id"].isna().any() or benchmark["query_id"].duplicated().any():
        raise ValueError("Development query_id values must be present and unique")
    if (
        benchmark["crm_record_id"].isna().any()
        or benchmark["crm_record_id"].duplicated().any()
    ):
        raise ValueError("Development crm_record_id values must be present and unique")
    truth = benchmark["ground_truth_siret"].map(normalize_input_siret)
    if truth.isna().any():
        raise ValueError("Every development row requires an exact 14-digit truth SIRET")
    benchmark["ground_truth_siret"] = truth
    benchmark["ground_truth_siren"] = truth.str[:9]
    return benchmark.reset_index(drop=True)


def join_crm_input_siret(
    benchmark: pd.DataFrame,
    crm_source_path: Path,
) -> pd.DataFrame:
    source = read_table(crm_source_path)
    required = {"SERVICE ID", "SIRET"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"CRM source is missing columns: {missing}")
    source = source[["SERVICE ID", "SIRET"]].copy()
    source["crm_record_id"] = source["SERVICE ID"].map(_normalize_id)
    source = source[source["crm_record_id"].notna()].copy()
    if source["crm_record_id"].duplicated().any():
        raise ValueError("CRM SERVICE ID values must be unique for the dev join")
    source = source.rename(columns={"SIRET": "input_siret_raw"})
    source["_crm_joined"] = True
    joined = benchmark.merge(
        source[["crm_record_id", "input_siret_raw", "_crm_joined"]],
        on="crm_record_id",
        how="left",
        validate="one_to_one",
    )
    if joined["_crm_joined"].isna().any():
        missing_ids = joined.loc[
            joined["_crm_joined"].isna(), "crm_record_id"
        ].head(5)
        raise ValueError(
            "CRM input SIRET join is incomplete for IDs: "
            + ", ".join(missing_ids.astype(str))
        )
    joined = joined.drop(columns=["_crm_joined"])
    joined["input_siret"] = joined["input_siret_raw"].map(normalize_input_siret)
    joined["input_equals_truth"] = (
        joined["input_siret"].fillna("")
        == joined["ground_truth_siret"].astype(str)
    )
    return joined


def _segment_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for state in sorted(frame["input_siret_state"].astype(str).unique()):
        masks[f"input_state={state}"] = frame["input_siret_state"].eq(state)
    for value in (False, True):
        masks[f"input_equals_truth={str(value).lower()}"] = (
            frame["input_equals_truth"].astype(bool).eq(value)
        )
        masks[f"multi_site={str(value).lower()}"] = (
            frame["multi_site"].astype(bool).eq(value)
        )
    return masks


def summarize_raw(raw: pd.DataFrame) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        subset = raw[raw["variant"].eq(variant)].copy()
        if subset.empty:
            raise ValueError(f"Missing raw results for variant {variant}")
        segments = {
            name: _rate(subset.loc[mask, "hit_at_100"])
            for name, mask in _segment_masks(subset).items()
            if int(mask.sum()) > 0
        }
        variants[variant] = {
            "recall_at_100": _rate(subset["hit_at_100"]),
            "segments": segments,
            "candidate_counts": {
                "max": int(subset["candidate_count"].max()),
                "mean": float(subset["candidate_count"].mean()),
                "over_100": int(subset["candidate_count"].gt(100).sum()),
            },
            "closed_candidate_count": int(
                subset["closed_candidate_count"].sum()
            ),
            "latency_ms": {
                "p50": float(subset["latency_ms"].median()),
                "p95": _p95(subset["latency_ms"]),
            },
        }
    selection = select_variant(variants)
    return {"variants": variants, "selection": selection}


def select_variant(variants: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered V4.1 gates and simplicity tie-break."""

    baseline = variants["A"]
    baseline_p95 = float(baseline["latency_ms"]["p95"])
    baseline_segments = baseline["segments"]
    gates: dict[str, Any] = {}
    eligible: list[str] = []
    for variant in VARIANT_ORDER:
        values = variants[variant]
        segment_deltas = {
            name: float(values["segments"][name]["rate"])
            - float(baseline_segments[name]["rate"])
            for name in baseline_segments
            if name in values["segments"]
        }
        variant_p95 = float(values["latency_ms"]["p95"])
        if baseline_p95 > 0:
            latency_ratio = variant_p95 / baseline_p95
        else:
            latency_ratio = 1.0 if variant_p95 <= 0 else float("inf")
        checks = {
            "recall_at_100_at_least_99pct": (
                float(values["recall_at_100"]["rate"]) >= 0.99
            ),
            "no_segment_regression_over_2pp": all(
                delta >= -0.02 for delta in segment_deltas.values()
            ),
            "latency_p95_at_most_2x_a": latency_ratio <= 2.0,
            "zero_closed_candidates": int(values["closed_candidate_count"]) == 0,
            "candidate_ceiling_100": (
                int(values["candidate_counts"]["max"]) <= 100
                and int(values["candidate_counts"]["over_100"]) == 0
            ),
        }
        success = all(checks.values())
        gates[variant] = {
            "success": success,
            "checks": checks,
            "segment_deltas_vs_a": segment_deltas,
            "latency_ratio_vs_a": latency_ratio,
        }
        if success:
            eligible.append(variant)

    selected: str | None = None
    if eligible:
        best_recall = max(
            float(variants[variant]["recall_at_100"]["rate"])
            for variant in eligible
        )
        within_one_point = [
            variant
            for variant in VARIANT_ORDER
            if (
                variant in eligible
                and best_recall
                - float(variants[variant]["recall_at_100"]["rate"])
                < 0.01
            )
        ]
        selected = within_one_point[0]
    return {
        "selected_variant": selected,
        "verdict": "GO" if selected is not None else "PIVOT",
        "tie_break": "within <1pp of best eligible recall: A then B then C",
        "gates": gates,
    }


def _artifact_identity(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"path": str(path), "sha256": file_sha256(path)}
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "path": str(path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "declared_database_sha256": manifest.get("database_sha256"),
        }
    manifest_files = sorted((path / "manifest").glob("*.parquet"))
    if manifest_files:
        return {
            "path": str(path),
            "manifest_files": {
                str(item.relative_to(path)): file_sha256(item)
                for item in manifest_files
            },
        }
    raise ValueError(f"Cannot establish artifact identity for directory: {path}")


def evaluate(
    *,
    benchmark_path: Path,
    crm_source_path: Path,
    partitions_dir: Path,
    global_store_path: Path,
    cache_dir: Path,
    output_dir: Path,
    benchmark_manifest_path: Path | None = None,
    split: str = "dev",
    max_rows: int = 0,
    partitioned_store: Any = None,
    global_store: Any = None,
    retrievers: Mapping[str, Any] | None = None,
    clock: Any = time.perf_counter,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Immutable output directory exists: {output_dir}")
    benchmark_manifest = (
        json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
        if benchmark_manifest_path is not None
        else None
    )
    benchmark = load_dev_benchmark(
        benchmark_path,
        split=split,
        benchmark_manifest=benchmark_manifest,
    )
    observed_benchmark_hash = file_sha256(benchmark_path)
    if benchmark_manifest is not None:
        output_hashes = {
            **benchmark_manifest.get("output_sha256", {}),
            **benchmark_manifest.get("outputs", {}),
        }
        relative_hint = f"{benchmark_path.parent.name}/{benchmark_path.name}"
        expected = (
            output_hashes.get(relative_hint)
            or output_hashes.get(benchmark_path.name)
        )
        if expected is not None and expected != observed_benchmark_hash:
            raise ValueError("Benchmark hash does not match its manifest")
    if max_rows:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        benchmark = benchmark.head(max_rows).copy()
    benchmark = join_crm_input_siret(benchmark, crm_source_path)

    owns_partitioned_store = partitioned_store is None
    owns_global_store = global_store is None
    partitioned_store = partitioned_store or PartitionedCandidateStore(
        partitions_dir
    )
    global_store = global_store or V41GlobalCandidateStore(global_store_path)
    shared_in_memory_cache: OrderedDict = OrderedDict()
    config_by_variant = {
        variant: V41RetrievalConfig(
            variant=V41RetrievalVariant(variant),
            max_candidates=100,
        )
        for variant in VARIANT_ORDER
    }
    persistent_cache = TfidfPersistentCache(
        config_by_variant["A"].sparse_config().tfidf_artifact_hash(),
        cache_dir=cache_dir,
    )
    if retrievers is None:
        retrievers = {
            variant: V41CandidateRetriever(
                partitioned_store=partitioned_store,
                global_store=global_store,
                config=config,
                in_memory_tfidf_cache=shared_in_memory_cache,
            )
            for variant, config in config_by_variant.items()
        }

    truth_sirens = benchmark["ground_truth_siren"].drop_duplicates().tolist()
    active_truth_sites = global_store.get_active_siblings(
        truth_sirens,
        max_per_siren=2,
    )
    multi_site_by_siren = {
        siren: len(active_truth_sites.get(siren, [])) > 1
        for siren in truth_sirens
    }
    records: list[dict[str, Any]] = []
    try:
        for position, row in benchmark.iterrows():
            crm_row = {
                "query_id": str(row["query_id"]),
                "crm_id": str(row["query_id"]),
                "crm_name": row.get("crm_name") or "",
                "crm_address": row.get("crm_address") or "",
                "crm_city": row.get("crm_city") or "",
                "crm_city_addr": row.get("crm_city") or "",
                "postcode": row.get("postcode") or "",
                "insee": row.get("insee") or "",
            }
            crm_pre = preprocess_crm_row(crm_row)
            # Rotate the first variant per query so cache cold-start work is not
            # systematically attributed to A, B, or C.
            order = [
                VARIANT_ORDER[(position + offset) % len(VARIANT_ORDER)]
                for offset in range(len(VARIANT_ORDER))
            ]
            for variant in order:
                started = clock()
                result = retrievers[variant].build(
                    crm_row=crm_row,
                    crm_pre=crm_pre,
                    input_siret=row["input_siret_raw"],
                    # Truth is deliberately not passed to retrieval.  There is
                    # no positive injection path in this evaluator.
                    gt_siret=None,
                    persistent_cache=persistent_cache,
                )
                latency_ms = (clock() - started) * 1000.0
                candidate_sirets = [
                    str(candidate["siret"]) for candidate in result.candidates
                ]
                truth = str(row["ground_truth_siret"])
                records.append(
                    {
                        "variant": variant,
                        "query_id": str(row["query_id"]),
                        "crm_record_id": str(row["crm_record_id"]),
                        "ground_truth_siret": truth,
                        "ground_truth_siren": str(row["ground_truth_siren"]),
                        "input_siret_raw": str(row["input_siret_raw"]),
                        "input_siret": result.input_siret.normalized_siret,
                        "input_siret_state": result.input_siret.state.value,
                        "input_equals_truth": bool(row["input_equals_truth"]),
                        "multi_site": multi_site_by_siren.get(
                            str(row["ground_truth_siren"]), False
                        ),
                        "hit_at_100": truth in candidate_sirets,
                        "ground_truth_rank": (
                            candidate_sirets.index(truth) + 1
                            if truth in candidate_sirets
                            else None
                        ),
                        "candidate_count": len(candidate_sirets),
                        "closed_candidate_count": sum(
                            str(candidate.get("etat_admin") or "").upper() == "F"
                            for candidate in result.candidates
                        ),
                        "candidate_sirets_json": json.dumps(candidate_sirets),
                        "channels_json": json.dumps(
                            result.channels, sort_keys=True
                        ),
                        "latency_ms": latency_ms,
                        "positive_injected": False,
                    }
                )
            while len(shared_in_memory_cache) > 20:
                shared_in_memory_cache.popitem(last=False)
    finally:
        if owns_global_store and hasattr(global_store, "close"):
            global_store.close()
        # PartitionedCandidateStore currently has no close method.
        _ = owns_partitioned_store

    raw = pd.DataFrame(records).sort_values(
        ["variant", "query_id"]
    ).reset_index(drop=True)
    expected_rows = len(benchmark) * len(VARIANT_ORDER)
    if len(raw) != expected_rows:
        raise AssertionError("Every development query must have A/B/C results")
    if raw["positive_injected"].any():
        raise AssertionError("Positive injection is forbidden")
    summary = summarize_raw(raw)
    summary["query_count"] = int(len(benchmark))
    summary["tfidf_cache"] = persistent_cache.stats()

    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_results.parquet"
    summary_path = output_dir / "summary.json"
    raw.to_parquet(raw_path, index=False)
    _json_dump(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "split": "dev",
        "query_count": int(len(benchmark)),
        "positive_injection": False,
        "inputs": {
            "benchmark": {
                "path": str(benchmark_path),
                "sha256": observed_benchmark_hash,
            },
            "benchmark_manifest": (
                {
                    "path": str(benchmark_manifest_path),
                    "sha256": file_sha256(benchmark_manifest_path),
                }
                if benchmark_manifest_path is not None
                else None
            ),
            "crm_source": {
                "path": str(crm_source_path),
                "sha256": file_sha256(crm_source_path),
            },
            "partitions": {
                **_artifact_identity(partitions_dir),
                "declared_sha256": (
                    benchmark_manifest.get("partitions_sha256")
                    if benchmark_manifest is not None
                    else None
                ),
            },
            "global_store": _artifact_identity(global_store_path),
        },
        "cache_dir": str(cache_dir),
        "retrieval": {
            variant: {
                "config": config.to_dict(),
                "signature": config.signature().hash,
                "tfidf_artifact_hash": config.tfidf_artifact_hash(),
            }
            for variant, config in (
                (
                    name,
                    config_by_variant[name].sparse_config(),
                )
                for name in VARIANT_ORDER
            )
        },
        "selection_contract": {
            "recall_at_100_min": 0.99,
            "max_segment_regression_vs_a": 0.02,
            "max_latency_p95_ratio_vs_a": 2.0,
            "candidate_ceiling": 100,
            "closed_candidate_ceiling": 0,
            "tie_within_pp_strict": 1.0,
            "simplicity_order": list(VARIANT_ORDER),
        },
        "outputs": {
            "raw_results.parquet": file_sha256(raw_path),
            "summary.json": file_sha256(summary_path),
        },
    }
    _json_dump(output_dir / "manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path)
    parser.add_argument("--crm-source", type=Path, required=True)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument("--global-store", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    output = evaluate(
        benchmark_path=args.benchmark,
        benchmark_manifest_path=args.benchmark_manifest,
        crm_source_path=args.crm_source,
        partitions_dir=args.partitions_dir,
        global_store_path=args.global_store,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        split=args.split,
        max_rows=args.max_rows,
    )
    print(output)


if __name__ == "__main__":
    main()
