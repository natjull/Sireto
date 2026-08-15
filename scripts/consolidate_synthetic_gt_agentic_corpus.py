#!/usr/bin/env python3
"""Audit and consolidate an agentic SIRETO GT ledger.

This program is deliberately a selector and verifier, never a text generator.
Every published CRM field must already exist byte-for-byte in a stored Luna
GENERATOR response.  The program fails closed when provenance, review,
deduplication, fold isolation, or response fidelity cannot be proven.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_synthetic_gt_agentic_contracts as contracts
from scripts import run_synthetic_gt_agentic_loop as loop


SCHEMA_VERSION = "sireto-synthetic-gt-consolidated-1"
SELECTION_VERSION = "rare-preserving-hash-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = loop.load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return value


def assert_hash(path: Path, expected: Any, label: str) -> str:
    observed = sha256_path(path)
    if observed != str(expected or ""):
        raise ValueError(f"{label} hash mismatch: {observed} != {expected}")
    return observed


def read_jsonl_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for _raw, value in loop.iter_jsonl_raw(path):
        item_key = str(value.get(key, ""))
        if not item_key or item_key in result:
            raise ValueError(f"missing or duplicate {key} in {path}: {item_key!r}")
        result[item_key] = value
    return result


def read_crm_sirens(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not reader.fieldnames or "gt_siret" not in reader.fieldnames:
            raise ValueError("CRM ground truth must contain gt_siret")
        result: set[str] = set()
        for row in reader:
            digits = "".join(character for character in str(row.get("gt_siret") or "") if character.isdigit())
            if len(digits) == 14:
                result.add(digits[:9])
        return result


def deterministic_key(seed_id: str, variant_id: str, selection_seed: int) -> str:
    material = f"{SELECTION_VERSION}|{selection_seed}|{seed_id}|{variant_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def family_quotas(
    counts: dict[str, int], limit: int, preserve_below: int
) -> dict[str, int]:
    if sum(counts.values()) < limit:
        raise ValueError(f"only {sum(counts.values())} accepted variants for requested {limit}")
    rare = {family for family, count in counts.items() if count <= preserve_below}
    quotas = {family: (count if family in rare else 0) for family, count in counts.items()}
    remaining = limit - sum(quotas.values())
    if remaining < 0:
        raise ValueError("rare-family preservation exceeds requested corpus size")
    common_total = sum(count for family, count in counts.items() if family not in rare)
    if remaining > common_total:
        raise ValueError("not enough common-family rows to fill requested corpus")
    if common_total:
        exact = {
            family: remaining * count / common_total
            for family, count in counts.items()
            if family not in rare
        }
        for family, value in exact.items():
            quotas[family] = min(counts[family], int(value))
        residual = limit - sum(quotas.values())
        order = sorted(
            exact,
            key=lambda family: (-(exact[family] - int(exact[family])), family),
        )
        for family in order:
            if residual == 0:
                break
            if quotas[family] < counts[family]:
                quotas[family] += 1
                residual -= 1
    if sum(quotas.values()) != limit:
        raise ValueError("quota allocation did not reach the requested corpus size")
    return dict(sorted(quotas.items()))


def select_rows(
    rows: list[sqlite3.Row], limit: int, selection_seed: int, preserve_below: int
) -> tuple[list[sqlite3.Row], dict[str, int]]:
    buckets: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        families = json.loads(row["families_json"])
        if not isinstance(families, list) or len(families) != 1:
            raise ValueError(f"variant must declare exactly one family: {row['seed_id']}:{row['variant_id']}")
        buckets[str(families[0])].append(row)
    counts = {family: len(values) for family, values in buckets.items()}
    quotas = family_quotas(counts, limit, preserve_below)
    selected: list[sqlite3.Row] = []
    for family, values in buckets.items():
        values.sort(
            key=lambda row: deterministic_key(
                str(row["seed_id"]), str(row["variant_id"]), selection_seed
            )
        )
        selected.extend(values[: quotas[family]])
    selected.sort(
        key=lambda row: deterministic_key(
            str(row["seed_id"]), str(row["variant_id"]), selection_seed
        )
    )
    if len(selected) != limit:
        raise ValueError(f"selected {len(selected)} rows instead of {limit}")
    return selected, quotas


def response_for_hash(
    tasks_by_hash: dict[str, sqlite3.Row], response_sha256: str, role: str
) -> dict[str, Any]:
    task = tasks_by_hash.get(response_sha256)
    if task is None or task["role"] != role or task["status"] != "COMPLETED":
        raise ValueError(f"missing completed {role} response {response_sha256}")
    raw = str(task["raw_response"] or "")
    if sha256_bytes(raw.encode("utf-8")) != response_sha256:
        raise ValueError(f"stored {role} response hash mismatch: {task['task_id']}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"stored {role} response is not an object: {task['task_id']}")
    return value


def check_generator_fidelity(
    row: sqlite3.Row, tasks_by_hash: dict[str, sqlite3.Row]
) -> None:
    response = response_for_hash(
        tasks_by_hash, str(row["generator_response_sha256"]), "GENERATOR"
    )
    response_seed = response.get("seed", {})
    if (
        response_seed.get("siret") != row["target_siret"]
        or response_seed.get("siren") != row["target_siren"]
    ):
        raise ValueError(f"generator seed mismatch: {row['seed_id']}")
    variants = {
        str(value.get("variant_id")): value
        for value in response.get("variants", [])
        if isinstance(value, dict)
    }
    raw_variant = variants.get(str(row["variant_id"]))
    if raw_variant is None:
        raise ValueError(f"variant absent from raw generator response: {row['seed_id']}:{row['variant_id']}")
    if raw_variant.get("crm") != json.loads(row["crm_json"]):
        raise ValueError(f"CRM differs from raw Luna response: {row['seed_id']}:{row['variant_id']}")
    if raw_variant.get("corruption_families_observed") != json.loads(row["families_json"]):
        raise ValueError(f"family differs from raw Luna response: {row['seed_id']}:{row['variant_id']}")
    if raw_variant.get("transformation_summary") != row["transformation_summary"]:
        raise ValueError(f"summary differs from raw Luna response: {row['seed_id']}:{row['variant_id']}")


def check_review_independence(
    connection: sqlite3.Connection, selected_seed_ids: set[str]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    placeholders = ",".join("?" for _ in selected_seed_ids)
    if not placeholders:
        raise ValueError("no selected seeds")
    tasks = connection.execute(
        f"""SELECT * FROM tasks
            WHERE seed_id IN ({placeholders}) AND role IN ('CRITIC','ADJUDICATOR')
              AND status='COMPLETED'""",
        tuple(sorted(selected_seed_ids)),
    ).fetchall()
    critic_seeds: set[str] = set()
    raw_decisions: dict[tuple[str, str], set[str]] = defaultdict(set)
    roles_by_seed: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        raw = str(task["raw_response"] or "")
        if sha256_bytes(raw.encode("utf-8")) != task["response_sha256"]:
            raise ValueError(f"review response hash mismatch: {task['task_id']}")
        request = json.loads(task["task_json"])
        response = json.loads(raw)
        seed_id = str(task["seed_id"])
        roles_by_seed[seed_id].add(str(task["role"]))
        for decision in response.get("decisions", []):
            if isinstance(decision, dict):
                raw_decisions[(seed_id, str(task["role"]))].add(loop.canonical_json(decision))
        if task["role"] == "CRITIC":
            critic_seeds.add(seed_id)
            variants = request.get("input", {}).get("variants", [])
            forbidden = {"transformation_summary", "generator_rationale", "corruption_families_observed"}
            if any(forbidden & set(value) for value in variants if isinstance(value, dict)):
                raise ValueError(f"critic saw generator rationale/family: {task['task_id']}")
            if response.get("independent") is not True or response.get("generator_rationale_seen") is not False:
                raise ValueError(f"critic did not attest independence: {task['task_id']}")
        counts[f"{task['role']}_COMPLETED"] += 1
    missing = selected_seed_ids - critic_seeds
    if missing:
        raise ValueError(f"selected seeds without independent critic: {sorted(missing)[:3]}")
    placeholders = ",".join("?" for _ in selected_seed_ids)
    variants = connection.execute(
        f"""SELECT seed_id, variant_id, critic_json, critic_decision,
                   adjudicator_json, adjudicator_decision, final_decision
            FROM variants WHERE seed_id IN ({placeholders})""",
        tuple(sorted(selected_seed_ids)),
    ).fetchall()
    disagreement_seeds: set[str] = set()
    for variant in variants:
        seed_id = str(variant["seed_id"])
        critic_json = str(variant["critic_json"] or "")
        if not critic_json or critic_json not in raw_decisions[(seed_id, "CRITIC")]:
            raise ValueError(f"stored critic decision lacks matching raw response: {seed_id}:{variant['variant_id']}")
        critic = json.loads(critic_json)
        if critic.get("decision") != variant["critic_decision"]:
            raise ValueError(f"critic decision drift: {seed_id}:{variant['variant_id']}")
        if variant["critic_decision"] != "ACCEPT":
            disagreement_seeds.add(seed_id)
        if variant["adjudicator_json"] is not None:
            adjudicator_json = str(variant["adjudicator_json"])
            if adjudicator_json not in raw_decisions[(seed_id, "ADJUDICATOR")]:
                raise ValueError(f"stored adjudicator decision lacks matching raw response: {seed_id}:{variant['variant_id']}")
            adjudicator = json.loads(adjudicator_json)
            if adjudicator.get("decision") != variant["adjudicator_decision"]:
                raise ValueError(f"adjudicator decision drift: {seed_id}:{variant['variant_id']}")
        if variant["final_decision"] == "ACCEPT" and variant["critic_decision"] != "ACCEPT":
            if variant["adjudicator_decision"] != "ACCEPT":
                raise ValueError(f"accepted disagreement lacks ACCEPT adjudication: {seed_id}:{variant['variant_id']}")
    adjudicated_seeds = {
        seed_id for seed_id, roles in roles_by_seed.items() if "ADJUDICATOR" in roles
    }
    if adjudicated_seeds != disagreement_seeds:
        raise ValueError(
            "adjudication routing mismatch: "
            f"unexpected={sorted(adjudicated_seeds - disagreement_seeds)[:3]} "
            f"missing={sorted(disagreement_seeds - adjudicated_seeds)[:3]}"
        )
    return dict(sorted(counts.items()))


def audit_sources(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    seed_manifest = load_manifest(args.seed_manifest)
    candidate_manifest = load_manifest(args.official_candidates_manifest)
    intake_manifest = load_manifest(args.source_intake_manifest)
    seed_hash = assert_hash(args.seed_input, seed_manifest.get("output_sha256"), "seed input")
    candidate_hash = assert_hash(
        args.official_candidates, candidate_manifest.get("output_sha256"), "official candidates"
    )
    intake_hash = assert_hash(args.source_intake, intake_manifest.get("output_sha256"), "source intake")
    crm_hash = assert_hash(args.crm, intake_manifest.get("crm_sha256"), "CRM exclusion source")
    profile_hash = sha256_path(args.observed_profile)
    assignments_hash = sha256_path(args.assignments)
    plan_hash = sha256_path(args.plan)
    fold_hash = sha256_path(args.fold_assignments)
    expected_inputs = seed_manifest.get("inputs", {})
    if expected_inputs.get("official_candidates_sha256") != candidate_hash:
        raise ValueError("seed manifest does not bind official candidates")
    if expected_inputs.get("observed_train_profile_sha256") != profile_hash:
        raise ValueError("seed manifest does not bind observed train profile")
    if expected_inputs.get("luna_assignments_sha256") != assignments_hash:
        raise ValueError("seed manifest does not bind Luna assignments")
    if candidate_manifest.get("source_sha256") != intake_hash:
        raise ValueError("candidate manifest does not bind source intake")
    if intake_manifest.get("one_siret_per_siren") is not True:
        raise ValueError("source intake is not one-SIRET-per-SIREN")
    if intake_manifest.get("text_generation") != "none" or candidate_manifest.get("text_generation") != "none":
        raise ValueError("official source preparation claims text generation")
    profile = loop.load_json(args.observed_profile)
    plan = loop.load_json(args.plan)
    evidence = profile.get("evidence", {})
    profile_sources = evidence.get("source_sha256", {})
    if evidence.get("comparison") != "CRM_OK_GT_STRICT_TRAIN_VS_OFFICIAL_SIRENE":
        raise ValueError("observed profile is not the strict train-only comparison")
    if evidence.get("plan_sha256") != plan_hash:
        raise ValueError("observed profile does not bind the frozen corpus plan")
    if profile_sources.get("crm_ok_gt") != crm_hash:
        raise ValueError("observed profile does not bind the CRM source")
    if profile_sources.get("fold_assignments") != fold_hash:
        raise ValueError("observed profile does not bind fold assignments")
    plan_population = plan.get("population", {})
    plan_folds = plan.get("sources", {}).get("fold_assignments", {})
    if plan_folds.get("sha256") != fold_hash:
        raise ValueError("frozen corpus plan does not bind fold assignments")
    if sorted(plan_population.get("forbidden_oof_folds", [])) != [0, 1]:
        raise ValueError("frozen plan does not forbid folds 0 and 1")
    if set(plan_population.get("forbidden_legacy_splits", [])) != {"dev", "test"}:
        raise ValueError("frozen plan does not forbid dev and test")
    if plan_population.get("test_final_opened") is not False:
        raise ValueError("frozen plan reports the final test as opened")
    if profile.get("rows") != plan_population.get("allowed_rows") or evidence.get("allowed_rows") != profile.get("rows"):
        raise ValueError("observed profile row count differs from frozen train-only population")
    maps = plan.get("maps_assisted", {})
    if maps.get("enabled") is not False or maps.get("max_requests") != 0 or maps.get("max_cost_eur") != 0.0:
        raise ValueError("frozen plan does not prove Maps disabled at zero spend")
    candidates = read_jsonl_index(args.official_candidates, "source_siret")
    return {
        "seed_input_sha256": seed_hash,
        "official_candidates_sha256": candidate_hash,
        "source_intake_sha256": intake_hash,
        "crm_exclusion_sha256": crm_hash,
        "observed_train_profile_sha256": profile_hash,
        "luna_assignments_sha256": assignments_hash,
        "corpus_plan_sha256": plan_hash,
        "fold_assignments_sha256": fold_hash,
    }, candidates


def verify_seed_provenance(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
    candidates: dict[str, dict[str, Any]],
) -> tuple[set[str], dict[str, str]]:
    seed_input = read_jsonl_index(args.seed_input, "seed_id")
    db_seeds = connection.execute(
        "SELECT * FROM seeds WHERE run_id=? ORDER BY seed_id", (args.run_id,)
    ).fetchall()
    if len(db_seeds) != len(seed_input):
        raise ValueError("ledger seed count differs from immutable seed input")
    crm_sirens = read_crm_sirens(args.crm)
    source_records: dict[str, str] = {}
    target_sirens: set[str] = set()
    for seed in db_seeds:
        input_seed = seed_input.get(str(seed["seed_id"]))
        if input_seed is None:
            raise ValueError(f"ledger seed absent from seed input: {seed['seed_id']}")
        for key in ("target_siret", "target_siren", "source_kind", "oof_fold", "legacy_split"):
            if input_seed.get(key) != seed[key]:
                raise ValueError(f"ledger seed differs from seed input for {seed['seed_id']}:{key}")
        if seed["source_kind"] != "SIRENE_ONLY_TRAIN" or seed["oof_fold"] != -1 or seed["legacy_split"] != "train_synthetic":
            raise ValueError(f"non-train-only seed: {seed['seed_id']}")
        if seed["target_siret"][:9] != seed["target_siren"]:
            raise ValueError(f"SIRET/SIREN mismatch: {seed['seed_id']}")
        if seed["target_siren"] in crm_sirens:
            raise ValueError(f"fold/test leakage through CRM SIREN: {seed['target_siren']}")
        candidate = candidates.get(str(seed["target_siret"]))
        if candidate is None or candidate.get("source_siren") != seed["target_siren"]:
            raise ValueError(f"seed absent from official candidates: {seed['seed_id']}")
        expected_card = contracts.candidate_card(candidate)
        actual_card = json.loads(seed["seed_card_json"])
        for key, value in expected_card.items():
            if key == "risk_flags":
                continue
            if actual_card.get(key) != value:
                raise ValueError(f"official field drift for {seed['seed_id']}:{key}")
        if json.loads(seed["profile_json"]) != input_seed.get("observed_train_profile"):
            raise ValueError(f"profile drift for {seed['seed_id']}")
        target_sirens.add(str(seed["target_siren"]))
        source_records[str(seed["target_siret"])] = str(candidate.get("source_record_sha256", ""))
        if len(source_records[str(seed["target_siret"])]) != 64:
            raise ValueError(f"missing official source-record digest: {seed['seed_id']}")
    if len(target_sirens) != len(db_seeds):
        raise ValueError("production ledger does not preserve one seed per SIREN")
    return target_sirens, source_records


def output_row(row: sqlite3.Row, source_record_sha256: str) -> dict[str, Any]:
    crm = json.loads(row["crm_json"])
    family = json.loads(row["families_json"])[0]
    provenance = {
        "seed_id": row["seed_id"],
        "variant_id": row["variant_id"],
        "target_siret": row["target_siret"],
        "source_record_sha256": source_record_sha256,
        "generator_response_sha256": row["generator_response_sha256"],
        "crm": crm,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": f"{row['seed_id']}:{row['variant_id']}",
        "seed_query_id": row["seed_id"],
        "seed_source": "SIRENE_OFFICIAL_LUNA_DIRECT",
        "source_kind": row["source_kind"],
        "base_view": "OFFICIAL_SIRENE",
        "crm_name": crm["name"],
        "crm_address": crm["address"],
        "crm_postcode": crm["postcode"],
        "crm_city": crm["city"],
        "crm_insee": crm["insee"],
        "target_siret": row["target_siret"],
        "target_siren": row["target_siren"],
        "target_state": json.loads(row["seed_card_json"])["state"],
        "oof_fold": -1,
        "siren_component_id": row["target_siren"],
        "corruption_family": family,
        "variant_index": int(str(row["variant_id"])[1:]),
        "confidence_weight": 1.0,
        "guard_status": "ACCEPT",
        "generator_version": row["generator_prompt_version"],
        "generator_response_sha256": row["generator_response_sha256"],
        "critic_decision": row["critic_decision"],
        "adjudicator_decision": row["adjudicator_decision"],
        "source_record_sha256": source_record_sha256,
        "provenance_digest": loop.digest_json(provenance),
    }


def consolidate(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 1 or args.preserve_rare_below < 0:
        raise ValueError("limit must be positive and preserve threshold non-negative")
    source_hashes, candidates = audit_sources(args)
    connection = loop.connect(args.db)
    try:
        connection.execute("BEGIN")
        run = connection.execute("SELECT * FROM runs WHERE run_id=?", (args.run_id,)).fetchone()
        if run is None:
            raise ValueError(f"unknown run: {args.run_id}")
        target_sirens, source_records = verify_seed_provenance(connection, args, candidates)
        status = loop.status_payload(connection, args.run_id)
        if args.require_complete and status["seeds"] != {"COMPLETED": len(target_sirens)}:
            raise ValueError(f"run is not completely supervised: {status['seeds']}")
        rows = connection.execute(
            """SELECT s.*, v.*, gt.prompt_version AS generator_prompt_version
               FROM seeds s JOIN variants v USING(run_id, seed_id)
               JOIN tasks gt ON gt.run_id=v.run_id AND gt.seed_id=v.seed_id
                 AND gt.role='GENERATOR' AND gt.status='COMPLETED'
                 AND gt.response_sha256=v.generator_response_sha256
               WHERE s.run_id=? AND s.status='COMPLETED' AND v.final_decision='ACCEPT'
               ORDER BY s.seed_id, v.variant_id""",
            (args.run_id,),
        ).fetchall()
        if len({(row["seed_id"], row["variant_id"]) for row in rows}) != len(rows):
            raise ValueError("accepted variants joined more than one raw generator task")
        tasks_by_hash = {
            str(task["response_sha256"]): task
            for task in connection.execute(
                "SELECT * FROM tasks WHERE run_id=? AND status='COMPLETED' AND response_sha256 IS NOT NULL",
                (args.run_id,),
            )
        }
        exact_json: set[str] = set()
        surfaces: set[str] = set()
        for row in rows:
            preflight = json.loads(row["preflight_json"])
            if preflight.get("passed") is not True or preflight.get("errors"):
                raise ValueError(f"accepted row failed preflight: {row['seed_id']}:{row['variant_id']}")
            crm = json.loads(row["crm_json"])
            if loop.leaked_identifier(crm, row["target_siret"], row["target_siren"]):
                raise ValueError(f"identifier leak: {row['seed_id']}:{row['variant_id']}")
            exact = loop.canonical_json(crm)
            surface = loop.surface_fingerprint(crm)
            if exact in exact_json or surface in surfaces:
                raise ValueError(f"duplicate CRM surface: {row['seed_id']}:{row['variant_id']}")
            exact_json.add(exact)
            surfaces.add(surface)
            check_generator_fidelity(row, tasks_by_hash)
        selected, quotas = select_rows(
            rows, args.limit, args.selection_seed, args.preserve_rare_below
        )
        review_counts = check_review_independence(
            connection, {str(row["seed_id"]) for row in selected}
        )
        output = [
            output_row(row, source_records[str(row["target_siret"])])
            for row in selected
        ]
        if len({row["example_id"] for row in output}) != args.limit:
            raise ValueError("duplicate example_id after consolidation")
        if len({loop.canonical_json({key: row[key] for key in (
            "crm_name", "crm_address", "crm_postcode", "crm_city", "crm_insee"
        )}) for row in output}) != args.limit:
            raise ValueError("duplicate exact CRM after consolidation")
        args.output.mkdir(parents=True, exist_ok=True)
        data_name = f"accepted_{args.limit}.jsonl"
        data_path = args.output / data_name
        loop.write_jsonl_atomic(data_path, output)
        family_counts = dict(sorted(Counter(row["corruption_family"] for row in output).items()))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "audit_status": "PASS",
            "run_id": args.run_id,
            "selection": {
                "version": SELECTION_VERSION,
                "seed": args.selection_seed,
                "limit": args.limit,
                "preserve_rare_below": args.preserve_rare_below,
                "family_quotas": quotas,
            },
            "counts": {
                "published": len(output),
                "published_distinct_exact_crm": len(output),
                "published_distinct_example_id": len(output),
                "published_distinct_target_siret": len({row["target_siret"] for row in output}),
                "published_distinct_target_siren": len({row["target_siren"] for row in output}),
                "ledger_accepted_audited": len(rows),
                "ledger_seed_sirens": len(target_sirens),
            },
            "family_counts": family_counts,
            "review_tasks": review_counts,
            "gates": {
                "exactly_requested_count": len(output) == args.limit,
                "all_final_accept": True,
                "all_preflight_passed": True,
                "all_crm_fields_exactly_equal_to_raw_luna": True,
                "all_generator_raw_hashes_verified": True,
                "all_selected_seeds_independently_criticised": True,
                "all_sources_sirene_only_train": True,
                "all_target_sirens_disjoint_from_crm_ok_gt": True,
                "folds_0_1_and_test_preserved": True,
                "maps_requests": 0,
                "python_text_generation": False,
                "exact_crm_deduplication": True,
                "surface_crm_deduplication": True,
            },
            "source_hashes": source_hashes,
            "ledger_status": status,
            "files": {data_name: sha256_path(data_path)},
        }
        manifest_path = args.output / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        connection.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--db", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--seed-input", type=Path, required=True)
    result.add_argument("--seed-manifest", type=Path, required=True)
    result.add_argument("--official-candidates", type=Path, required=True)
    result.add_argument("--official-candidates-manifest", type=Path, required=True)
    result.add_argument("--source-intake", type=Path, required=True)
    result.add_argument("--source-intake-manifest", type=Path, required=True)
    result.add_argument("--observed-profile", type=Path, required=True)
    result.add_argument("--assignments", type=Path, required=True)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--fold-assignments", type=Path, required=True)
    result.add_argument("--crm", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--limit", type=int, default=20_000)
    result.add_argument("--selection-seed", type=int, default=42)
    result.add_argument("--preserve-rare-below", type=int, default=1_000)
    result.add_argument("--require-complete", action="store_true")
    result.set_defaults(func=consolidate)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    manifest = args.func(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
