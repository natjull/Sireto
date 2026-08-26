"""Truth-side evaluation for the sealed v16 native-parent generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from .retrieval_parent_v16 import fuse_parent_signals, sha256_file


EVALUATION_SCHEMA_VERSION = "sireto-parent-v16-native-workbench-evaluation-1"
MISS_KIND = "BASE_TOP1750_MISS"


def _verify_manifest_files(root: Path, manifest: Mapping[str, Any]) -> None:
    for name, record in manifest["files"].items():
        if sha256_file(root / name) != str(record["sha256"]):
            raise ValueError(f"artifact hash drifted: {root / name}")


def _rank_by_query(
    rows_by_query: Mapping[str, list[Mapping[str, Any]]],
    *,
    channels: set[str],
    weights: Mapping[str, float],
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for query_id, rows in rows_by_query.items():
        fused = fuse_parent_signals(
            rows,
            allowed_channels=channels,
            channel_weights=weights,
            cap=512,
        )
        output[query_id] = {str(row["siren"]): int(row["parent_rank"]) for row in fused}
    return output


def _balanced_candidate_ranks(
    rows_by_query: Mapping[str, list[Mapping[str, Any]]],
    channel_order: list[str],
) -> dict[str, dict[str, int]]:
    """Build a label-blind, channel-balanced native-parent pool capped at 512."""
    output: dict[str, dict[str, int]] = {}
    for query_id, rows in rows_by_query.items():
        lookup = {
            (str(row["channel"]), int(row["channel_rank"])): str(row["siren"])
            for row in rows
        }
        ranks: dict[str, int] = {}
        for channel_rank in range(1, 513):
            for channel in channel_order:
                siren = lookup.get((channel, channel_rank))
                if siren and siren not in ranks:
                    ranks[siren] = len(ranks) + 1
                    if len(ranks) == 512:
                        break
            if len(ranks) == 512:
                break
        output[query_id] = ranks
    return output


def _report_ranks(
    ranks: Mapping[str, Mapping[str, int]],
    labels: list[Mapping[str, Any]],
    cutoffs: list[int],
) -> dict[str, Any]:
    miss_rr = 0.0
    control_rr = 0.0
    for label in labels:
        query_id = str(label["query_id"])
        truth_siren = str(label["human_ground_truth_siret"])[:9]
        rank = ranks.get(query_id, {}).get(truth_siren)
        reciprocal = 1.0 / rank if rank else 0.0
        if str(label["cohort_kind"]) == MISS_KIND:
            miss_rr += reciprocal
        else:
            control_rr += reciprocal
    report: dict[str, Any] = {
        "cutoffs": {},
        "miss_mrr": miss_rr / 522,
        "control_mrr": control_rr / 512,
    }
    for cutoff in cutoffs:
        misses_by_fold = {2: 0, 3: 0, 4: 0}
        controls_by_fold = {2: 0, 3: 0, 4: 0}
        for label in labels:
            query_id = str(label["query_id"])
            truth_siren = str(label["human_ground_truth_siret"])[:9]
            hit = int(ranks.get(query_id, {}).get(truth_siren, cutoff + 1) <= cutoff)
            fold = int(label["fold"])
            if str(label["cohort_kind"]) == MISS_KIND:
                misses_by_fold[fold] += hit
            else:
                controls_by_fold[fold] += hit
        report["cutoffs"][str(cutoff)] = {
            "miss_hits_by_fold": {str(key): value for key, value in misses_by_fold.items()},
            "miss_hits": sum(misses_by_fold.values()),
            "miss_recall": sum(misses_by_fold.values()) / 522,
            "control_hits_by_fold": {str(key): value for key, value in controls_by_fold.items()},
            "control_hits": sum(controls_by_fold.values()),
            "control_recall": sum(controls_by_fold.values()) / 512,
        }
    return report


def evaluate_parent_generation(
    *,
    generation_dir: Path,
    cohort_dir: Path,
    baseline_parents_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    config_sha256: str,
) -> Path:
    generation_dir = Path(generation_dir).resolve()
    cohort_dir = Path(cohort_dir).resolve()
    baseline_parents_path = Path(baseline_parents_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    generation_manifest = json.loads((generation_dir / "manifest.json").read_text())
    cohort_manifest = json.loads((cohort_dir / "manifest.json").read_text())
    _verify_manifest_files(generation_dir, generation_manifest)
    _verify_manifest_files(cohort_dir, cohort_manifest)
    expected_baseline_sha = str(config["authorities"]["v15_parents"]["sha256"])
    if sha256_file(baseline_parents_path) != expected_baseline_sha:
        raise ValueError("sealed v15 parent baseline drifted")

    signals_table = pq.read_table(generation_dir / "parent_signals.parquet")
    forbidden = {"human_ground_truth_siret", "cohort_kind", "label", "target"}
    if forbidden & set(signals_table.column_names):
        raise ValueError("truth-side column leaked into v16 generation")
    signals = signals_table.to_pylist()
    labels = pq.read_table(cohort_dir / "cohort_labels.parquet").to_pylist()
    queries = pq.read_table(cohort_dir / "queries.parquet").to_pylist()
    query_ids = {str(row["query_id"]) for row in queries}
    if len(query_ids) != 1034 or any(int(row["fold"]) not in {2, 3, 4} for row in queries):
        raise ValueError("cohort population/fold authority drifted")
    if {str(row["query_id"]) for row in labels} != query_ids:
        raise ValueError("truth-side labels differ from the sealed query population")
    if any(str(row["query_id"]) not in query_ids for row in signals):
        raise ValueError("generation emitted a query outside the sealed population")
    if generation_manifest.get("truth_columns_opened") is not False:
        raise ValueError("generator did not attest label blindness")
    if generation_manifest.get("native_parent_only") is not True or generation_manifest.get("siret_truncation") is not False:
        raise ValueError("generator did not attest native-parent semantics")

    rows_by_query: dict[str, list[Mapping[str, Any]]] = {query_id: [] for query_id in query_ids}
    for row in signals:
        rows_by_query[str(row["query_id"])].append(row)
    specs = {str(row["name"]): row for row in config["channels"]}
    weights = {name: float(row.get("weight", 1.0)) for name, row in specs.items()}
    cutoffs = [int(value) for value in config["evaluation"]["cutoffs"]]

    baseline_rows = pq.read_table(baseline_parents_path).to_pylist()
    baseline_ranks: dict[str, dict[str, int]] = {query_id: {} for query_id in query_ids}
    for row in baseline_rows:
        query_id = str(row["query_id"])
        if query_id in baseline_ranks:
            baseline_ranks[query_id][str(row["siren"])] = int(row["parent_rank"])
    reports: dict[str, Any] = {
        "SEALED_V15_BASELINE": _report_ranks(baseline_ranks, labels, cutoffs)
    }
    raw_union_ranks = {
        query_id: {str(row["siren"]): 1 for row in rows}
        for query_id, rows in rows_by_query.items()
    }
    reports["RAW_NATIVE_SIGNAL_UNION_ORACLE"] = _report_ranks(
        raw_union_ranks, labels, [512]
    )
    balanced_candidate_ranks = _balanced_candidate_ranks(rows_by_query, list(specs))
    reports["ALL_NATIVE_BALANCED_CANDIDATE"] = _report_ranks(
        balanced_candidate_ranks, labels, cutoffs
    )
    variant_ranks: dict[str, dict[str, dict[str, int]]] = {}
    for variant in config["ablations"]:
        name = str(variant["name"])
        channels = set(str(value) for value in variant["channels"])
        unknown = channels - set(specs)
        if unknown:
            raise ValueError(f"ablation {name} references unknown channels: {sorted(unknown)}")
        ranks = _rank_by_query(rows_by_query, channels=channels, weights=weights)
        variant_ranks[name] = ranks
        reports[name] = _report_ranks(ranks, labels, cutoffs)

    channel_reports: dict[str, Any] = {}
    for channel in specs:
        ranks = _rank_by_query(rows_by_query, channels={channel}, weights=weights)
        channel_reports[channel] = _report_ranks(ranks, labels, cutoffs)

    primary_name = str(config["evaluation"]["primary_ablation"])
    primary = reports[primary_name]["cutoffs"]["512"]
    candidate = reports["ALL_NATIVE_BALANCED_CANDIDATE"]["cutoffs"]["512"]
    baseline = reports["SEALED_V15_BASELINE"]["cutoffs"]["512"]
    gates = config["gates"]
    kill_by_fold = {str(key): int(value) for key, value in gates["hard_kill_parent_hits_at_512_by_fold"].items()}
    confidence_by_fold = {str(key): int(value) for key, value in gates["confidence_parent_hits_at_512_by_fold"].items()}
    hard_kill = any(int(candidate["miss_hits_by_fold"][fold]) < required for fold, required in kill_by_fold.items())
    pass_target = (
        not hard_kill
        and all(int(candidate["miss_hits_by_fold"][fold]) >= required for fold, required in confidence_by_fold.items())
        and all(int(primary["miss_hits_by_fold"][fold]) >= required for fold, required in confidence_by_fold.items())
        and int(primary["miss_hits"]) >= int(gates["confidence_parent_hits_at_512"])
        and int(primary["miss_hits"]) >= int(baseline["miss_hits"])
    )
    improvement = int(primary["miss_hits"]) - int(baseline["miss_hits"])
    promising = not hard_kill and improvement >= int(gates["minimum_promising_absolute_gain"])
    decision = (
        "PASS_PARENT_V16_FEASIBILITY"
        if pass_target
        else "PROMISING_PARENT_V16" if promising
        else "PIVOT_PARENT_INDEX"
    )
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "decision": decision,
        "config_sha256": config_sha256,
        "checks": {
            "fold0_opened": False,
            "fold1_opened": False,
            "truth_in_generation": False,
            "native_parent_only": True,
            "base_controls_immutably_preserved": sum(str(row["cohort_kind"]) != MISS_KIND for row in labels) == 512,
            "hard_kill_parent_oracle": hard_kill,
            "target_gate": pass_target,
        },
        "gates": gates,
        "primary_ablation": primary_name,
        "primary_gain_vs_v15": improvement,
        "ablations": reports,
        "individual_channels": channel_reports,
        "generation_manifest_sha256": sha256_file(generation_dir / "manifest.json"),
        "cohort_manifest_sha256": sha256_file(cohort_dir / "manifest.json"),
        "baseline_parents_sha256": expected_baseline_sha,
    }
    output_dir.mkdir(parents=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = ["EVALUATION_SCHEMA_VERSION", "evaluate_parent_generation"]
