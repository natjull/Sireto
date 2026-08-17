#!/usr/bin/env python3
"""Audit or publish the exact 20k counted balanced synthetic-GT corpus.

Only full-SIRENE exact rows already sealed in the cumulative production
registry are eligible.  The finalizer never generates text, reruns retrieval,
or truncates a partially trusted batch.  Arbitrary-size residual batches are
handled upstream, so the final corpus can contain exactly 20,000 independent
variants without padding to 20,001 for target groups.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import manage_synthetic_gt_balanced_registry as registry_lib
from scripts import run_synthetic_gt_agentic_loop as loop
from scripts import select_synthetic_gt_balanced_production as production
from scripts import select_synthetic_gt_fragment_pilot as fragments

exact_counts = production.exact_counts


SCHEMA_VERSION = "sireto-synthetic-gt-balanced-final-1"


def sha256(path: Path) -> str:
    return registry_lib.sha256(path)


def load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("balanced plan must be an object")
    return value


def cap_limits(plan: dict[str, Any]) -> dict[str, int]:
    target = int(plan["objective"]["promoted_variant_target"])
    caps = plan["global_caps"]
    subset = int(math.floor(target * float(caps["name_token_subset_share"])))
    return {
        "inspiration_ref": int(caps["inspiration_ref_uses"]),
        "exact_operator": int(math.ceil(
            target * 2 * float(caps["exact_operator_share"])
        )),
        "relation_pair": int(math.ceil(
            target * float(caps["relation_pair_share"])
        )),
        "official_name_alias": int(math.floor(
            target * float(caps["official_name_alias_share"])
        )),
        "name_token_subset": subset,
        "name_token_subset_signature": min(
            int(math.floor(
                target * float(caps["name_token_subset_signature_share_global"])
            )),
            int(math.floor(
                subset
                * float(caps["name_token_subset_signature_share_within_family"])
            )),
        ),
    }


def quota_audit(
    summary: dict[str, Any], plan: dict[str, Any], *, require_complete: bool,
) -> dict[str, Any]:
    target = int(plan["objective"]["promoted_variant_target"])
    promoted = int(summary["promoted_variants"])
    if promoted > target:
        raise ValueError(f"registry exceeds target: {promoted}>{target}")
    quota_targets = {
        "difficulty": exact_counts(
            target, plan["corpus_balance"]["difficulty"]
        ),
        "augmentation_strata": exact_counts(
            target, plan["corpus_balance"]["augmentation_strata"]
        ),
    }
    observed = {
        "difficulty": {
            key: int(summary["difficulty_counts"].get(key, 0))
            for key in quota_targets["difficulty"]
        },
        "augmentation_strata": {
            key: int(summary["augmentation_stratum_counts"].get(key, 0))
            for key in quota_targets["augmentation_strata"]
        },
    }
    deficits = {
        dimension: {
            key: quota_targets[dimension][key] - observed[dimension][key]
            for key in quota_targets[dimension]
        }
        for dimension in quota_targets
    }
    if any(value < 0 for values in deficits.values() for value in values.values()):
        raise ValueError(f"a final quota has already been exceeded: {deficits}")
    remaining = target - promoted
    if any(sum(values.values()) != remaining for values in deficits.values()):
        raise ValueError("quota deficits disagree with registry residual")

    limits = cap_limits(plan)
    subset_count = sum(
        count for signature, count in summary["relation_pair_counts"].items()
        if signature.startswith("name:TOKEN_SUBSET+")
    )
    official_name_alias_count = sum(
        count for signature, count in summary["relation_pair_counts"].items()
        if signature.startswith("name:OFFICIAL_NAME_ALIAS+")
    )
    usage = {
        "inspiration_ref": max(summary["inspiration_ref_counts"].values(), default=0),
        "exact_operator": max(summary["exact_operator_counts"].values(), default=0),
        "relation_pair": max(summary["relation_pair_counts"].values(), default=0),
        "official_name_alias": official_name_alias_count,
        "name_token_subset": subset_count,
        "name_token_subset_signature": max(
            summary["name_token_subset_signature_counts"].values(), default=0
        ),
    }
    if any(usage[key] > limits[key] for key in limits):
        raise ValueError(f"a cumulative cap is exceeded: usage={usage}, limits={limits}")
    maximum_targets = int(plan["objective"]["maximum_unique_targets"])
    distinct_targets = int(summary["distinct_target_sirets"])
    if distinct_targets > maximum_targets:
        raise ValueError("maximum unique target count exceeded")
    remaining_target_slots = maximum_targets - distinct_targets
    if require_complete and promoted != target:
        raise ValueError(f"registry is incomplete: {promoted}/{target}")
    if require_complete and any(
        value for values in deficits.values() for value in values.values()
    ):
        raise ValueError(f"final quotas are not exact: {deficits}")
    return {
        "target": target,
        "promoted": promoted,
        "remaining": remaining,
        "quota_targets": quota_targets,
        "observed": observed,
        "deficits": deficits,
        "cap_limits": limits,
        "cap_usage": usage,
        "cap_headroom": {key: limits[key] - usage[key] for key in limits},
        "distinct_targets": distinct_targets,
        "remaining_target_slots": remaining_target_slots,
        "minimum_required_average_variants_per_remaining_target": (
            remaining / remaining_target_slots
            if remaining_target_slots else (0.0 if not remaining else None)
        ),
    }


def final_state_audit(
    state_variant_counts: Counter[str],
    state_target_counts: Counter[str],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on final variant balance and distinct-identity envelope."""
    target = int(plan["objective"]["promoted_variant_target"])
    expected_variants = exact_counts(
        target, plan["corpus_balance"]["state_variants"]
    )
    observed_variants = {
        state: int(state_variant_counts[state]) for state in expected_variants
    }
    if observed_variants != expected_variants:
        raise ValueError(
            "final state variant quotas are not exact: "
            f"observed={observed_variants}, expected={expected_variants}"
        )
    unexpected_states = set(state_variant_counts) - set(expected_variants)
    if unexpected_states:
        raise ValueError(f"unexpected final target states: {sorted(unexpected_states)}")
    distinct_targets = sum(state_target_counts.values())
    if distinct_targets <= 0:
        raise ValueError("final corpus has no target identities")
    active_share = state_target_counts["A"] / distinct_targets
    lower, upper = (
        float(value) for value in
        plan["corpus_balance"]["target_identity_active_share_bounds"]
    )
    (lower_active, lower_closed), (upper_active, upper_closed) = (
        production.identity_share_coefficients((lower, upper))
    )
    lower_margin = (
        lower_active * state_target_counts["A"]
        + lower_closed * state_target_counts["F"]
    )
    upper_margin = (
        upper_active * state_target_counts["A"]
        + upper_closed * state_target_counts["F"]
    )
    if lower_margin < 0 or upper_margin > 0:
        raise ValueError(
            "final active target identity share is outside bounds: "
            f"{active_share:.12f} not in [{lower:.12f}, {upper:.12f}]"
        )
    return {
        "state_variant_counts": observed_variants,
        "state_variant_targets": expected_variants,
        "state_target_counts": dict(state_target_counts),
        "active_target_identity_share": active_share,
        "active_target_identity_share_bounds": [lower, upper],
        "active_target_identity_share_integer_margins": {
            "lower": lower_margin,
            "upper": upper_margin,
        },
    }


def context_capacity(
    summary: dict[str, Any], plan: dict[str, Any], remaining_easy: int,
    additional_excluded_sirens: set[str] | None = None,
) -> dict[str, Any]:
    context_path = Path(plan["sources"]["official_context"]["path"])
    if sha256(context_path) != plan["sources"]["official_context"]["sha256"]:
        raise ValueError("official-context source hash differs from the plan")
    excluded = set(summary["excluded_target_sirens"])
    excluded.update(additional_excluded_sirens or set())
    extension_manifest_path = context_path.with_suffix(
        context_path.suffix + ".manifest.json"
    )
    extension: dict[str, Any] = {}
    extension_sirets: set[str] = set()
    extension_sirens: set[str] = set()
    if extension_manifest_path.exists():
        extension = json.loads(extension_manifest_path.read_text(encoding="utf-8"))
        if extension.get("output_sha256") != sha256(context_path):
            raise ValueError("official-context extension manifest is unsealed")
        candidate_path = Path(extension.get("candidate_output", ""))
        if candidate_path.exists() and extension.get(
            "candidate_output_sha256"
        ) == sha256(candidate_path):
            for _raw, value in loop.iter_jsonl_raw(candidate_path):
                extension_sirets.add(str(value["source_siret"]))
                extension_sirens.add(str(value["source_siren"]))
    available = 0
    document_frequencies: Counter[str] = Counter()
    available_extension_contexts: list[dict[str, Any]] = []
    for _raw, value in loop.iter_jsonl_raw(context_path):
        document_frequencies.update(set(loop.normalized_words(
            fragments.baseline(value)["name"]
        )))
        if str(value["target_siren"]) not in excluded:
            available += 1
            if str(value["target_siret"]) in extension_sirets:
                available_extension_contexts.append(value)
    result: dict[str, Any] = {
        "available_unattempted_targets": available,
        "identity_variant_upper_bound": available * int(
            plan["objective"]["maximum_variants_per_target"]
        ),
    }
    if extension_sirens:
        used = len(extension_sirens.intersection(excluded))
        remaining_targets = len(extension_sirens) - used
        gross = int(extension.get("easy_capacity", {}).get("total_variants", 0))
        conservative = max(
            remaining_targets,
            gross - used * int(plan["objective"]["maximum_variants_per_target"]),
        )
        grouped = fragments.group_fragments([
            value for _raw, value in loop.iter_jsonl_raw(Path(
                plan["sources"]["field_inspiration_bank"]["path"]
            ))
        ])
        support_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        exact_remaining = 0
        capacity_counts: Counter[int] = Counter()
        for context in available_extension_contexts:
            names, locations = production.safe_capabilities(
                context, grouped, document_frequencies,
                tuple(production.SAFE_NAME_RELATIONS), support_cache,
            )
            bundles = production.candidate_bundles(
                context, names, locations, "FINAL-CAPACITY-AUDIT", limit=64,
            )
            capacity = max((
                sum(
                    production.difficulty(context, pair) == "EASY"
                    for pair in bundle
                )
                for bundle in bundles
            ), default=0)
            exact_remaining += capacity
            capacity_counts[capacity] += 1
        result["easy_extension"] = {
            "gross_safe_variant_capacity": gross,
            "used_or_attempted_targets": used,
            "remaining_targets": remaining_targets,
            "conservative_remaining_safe_variant_capacity": conservative,
            "exact_structural_remaining_safe_variant_capacity": exact_remaining,
            "remaining_targets_by_safe_easy_capacity": dict(
                sorted(capacity_counts.items())
            ),
            "remaining_easy_quota": remaining_easy,
            "proves_remaining_easy_quota": exact_remaining >= remaining_easy,
            "proof_scope": (
                "target-specific exact safe bundles before coupled future "
                "operator/ref/pair caps; current cap headroom is reported separately"
            ),
        }
    return result


def audited_registry(
    registry_path: Path, plan_path: Path, *, require_complete: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry = registry_lib.load_registry(registry_path)
    summary = registry_lib.snapshot(registry)
    if registry.get("summary") != summary:
        raise ValueError("registry summary differs from sealed batch reconstruction")
    plan = load_plan(plan_path)
    if int(registry["promoted_variant_target"]) != int(
        plan["objective"]["promoted_variant_target"]
    ):
        raise ValueError("registry and plan targets differ")
    return registry, summary, quota_audit(
        summary, plan, require_complete=require_complete,
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    _registry, summary, result = audited_registry(
        args.registry, args.plan, require_complete=False,
    )
    plan = load_plan(args.plan)
    in_flight_sirens: set[str] = set()
    for path in args.exclude_seed_input:
        for _raw, value in loop.iter_jsonl_raw(path):
            siren = str(value.get("target_siren", ""))
            if loop.valid_siren(siren):
                in_flight_sirens.add(siren)
    easy_key = "EASY"
    result["context_capacity"] = context_capacity(
        summary, plan, result["deficits"]["difficulty"].get(easy_key, 0),
        in_flight_sirens,
    )
    result["in_flight_excluded_sirens"] = len(in_flight_sirens)
    result["capacity_gates"] = {
        "identity_variants_cover_residual": (
            result["context_capacity"]["identity_variant_upper_bound"]
            >= result["remaining"]
        ),
        "target_slots_cover_residual_at_max_three": (
            result["remaining_target_slots"] * 3 >= result["remaining"]
        ),
        "easy_structural_capacity_covers_quota": result["context_capacity"].get(
            "easy_extension", {}
        ).get("proves_remaining_easy_quota", False),
    }
    if not result["remaining"]:
        result["status"] = "READY_TO_FINALIZE"
    elif all(result["capacity_gates"].values()):
        result["status"] = "CONTINUE_GO"
    else:
        result["status"] = "CAPACITY_RISK"
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return result


def atomic_publish_directory(stage: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    os.replace(stage, output)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    registry, summary, audit_result = audited_registry(
        args.registry, args.plan, require_complete=True,
    )
    plan = load_plan(args.plan)
    quarantined = registry_lib.quarantine_records(registry)
    records: list[dict[str, Any]] = []
    target_counts: Counter[str] = Counter()
    target_states: dict[str, str] = {}
    state_variant_counts: Counter[str] = Counter()
    keys: set[tuple[str, str]] = set()
    surfaces: set[tuple[str, str, str, str, str]] = set()
    source_batches: dict[str, Any] = {}
    for batch in sorted(registry["batches"], key=lambda value: value["batch_id"]):
        seed = Path(batch["seed_input"]["path"])
        promoted = Path(batch["promoted"]["path"])
        promotion_manifest = Path(batch["promotion_manifest"]["path"])
        validated = registry_lib.validate_promoted_batch(
            seed, promoted, promotion_manifest,
        )
        frozen_states = production.seed_target_states(seed)
        source_batches[batch["batch_id"]] = {
            "seed_input": {"path": str(seed), "sha256": sha256(seed)},
            "promoted": {"path": str(promoted), "sha256": sha256(promoted)},
            "promotion_manifest": {
                "path": str(promotion_manifest),
                "sha256": sha256(promotion_manifest),
            },
        }
        for value in validated["records"]:
            key = value["key"]
            if key in quarantined:
                continue
            crm = value["promoted"]["crm"]
            surface = tuple(str(crm.get(field, "")) for field in (
                "name", "address", "postcode", "city", "insee"
            ))
            if key in keys or surface in surfaces:
                raise ValueError(f"duplicate final key or surface: {key}")
            keys.add(key)
            surfaces.add(surface)
            target_counts[value["target_siret"]] += 1
            state = frozen_states[value["target_siret"]]
            previous_state = target_states.get(value["target_siret"])
            if previous_state is not None and previous_state != state:
                raise ValueError(
                    f"target state changed across batches: {value['target_siret']}"
                )
            target_states[value["target_siret"]] = state
            state_variant_counts[state] += 1
            records.append({
                **value["promoted"],
                "balanced_contract": value["contract"],
                "counting_provenance": {
                    "batch_id": batch["batch_id"],
                    "seed_input_sha256": batch["seed_input"]["sha256"],
                    "promoted_sha256": batch["promoted"]["sha256"],
                },
            })
    target = int(plan["objective"]["promoted_variant_target"])
    if len(records) != target:
        raise RuntimeError(f"final row count differs from target: {len(records)}")
    maximum_per_target = int(plan["objective"]["maximum_variants_per_target"])
    if max(target_counts.values(), default=0) > maximum_per_target:
        raise ValueError("maximum promoted variants per target exceeded")
    state_audit = final_state_audit(
        state_variant_counts, Counter(target_states.values()), plan,
    )
    records.sort(key=lambda value: (
        value["counting_provenance"]["batch_id"],
        str(value["seed_id"]), str(value["variant_id"]),
    ))

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        dir=args.output.parent, prefix=f".{args.output.name}.staging-",
    ))
    try:
        data_path = stage / "promoted_20000.jsonl"
        loop.write_jsonl_atomic(data_path, records)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "rows": len(records),
            "distinct_exact_crm_surfaces": len(surfaces),
            "distinct_keys": len(keys),
            "distinct_target_sirets": len(target_counts),
            "maximum_variants_per_target": max(target_counts.values(), default=0),
            "state_audit": state_audit,
            "audit": audit_result,
            "gates": {
                "exactly_20000_independent_variants": len(records) == 20_000,
                "target_group_padding_required": False,
                "all_rows_full_sirene_exact_promotions": True,
                "all_final_quotas_exact": True,
                "all_global_caps_respected": True,
                "duplicate_key_count": 0,
                "duplicate_surface_count": 0,
                "positive_injection": False,
                "qualification_uses_retrieval_or_model_scores": False,
                "text_generation_by_finalizer": False,
                "quarantined_v1_rows_excluded": len(quarantined),
            },
            "source_hashes": {
                "registry": sha256(args.registry),
                "plan": sha256(args.plan),
                "quarantine_report": registry.get("quarantine", {}).get("sha256"),
            },
            "source_batches": source_batches,
            "files": {"promoted_20000.jsonl": sha256(data_path)},
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in (data_path, manifest_path):
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        atomic_publish_directory(stage, args.output)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--registry", type=Path, required=True,
    )
    result.add_argument(
        "--plan", type=Path,
        default=Path("config/synthetic_gt_balanced_v1_plan.json"),
    )
    subparsers = result.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument(
        "--exclude-seed-input", type=Path, action="append", default=[],
        help="Read-only reservation of an in-flight, not-yet-registered batch.",
    )
    audit_parser.set_defaults(func=audit)
    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--output", type=Path, required=True)
    final_parser.set_defaults(func=finalize)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
