#!/usr/bin/env python3
"""Final, read-only audit of the counted balanced synthetic GT corpus.

The command does not rerun generation or retrieval.  It reconstructs every
registered batch, verifies its sealed full-SIRENE audit and provenance chain,
compares the cumulative synthetic distribution with the existing real GT, and
materialises one bounded deterministic realism sample.  A completed review of
that exact sample may be supplied to obtain the downstream PASS/PAUSE verdict.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_synthetic_gt_distribution as distribution
from scripts import manage_synthetic_gt_balanced_registry as registry_lib
from scripts import run_synthetic_gt_agentic_loop as loop


DEFAULT_REGISTRY = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "synthetic_gt_corpus/balanced_v1/production_registry.json"
)
DEFAULT_CORPUS_PLAN = ROOT / "config/synthetic_gt_corpus_plan.json"
DEFAULT_BALANCED_PLAN = ROOT / "config/synthetic_gt_balanced_v1_plan.json"
SCHEMA_VERSION = "sireto-synthetic-gt-balanced-final-audit-1"
SAMPLE_SCHEMA_VERSION = "sireto-synthetic-gt-realism-sample-1"
REVIEW_SCHEMA_VERSION = "sireto-synthetic-gt-realism-review-1"
REALISM_DECISIONS = {"PASS", "BORDERLINE", "CERTAIN_FALSE_REALISM"}
HEX64 = re.compile(r"[0-9a-f]{64}")
TRAIN_FOLDS = {2, 3, 4}
CRM_FIELDS = ("name", "address", "postcode", "city", "insee")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return registry_lib.jsonl(path)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def _resolve_source(plan_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (plan_path.parent.parent / path).resolve()


def _verify_plan_sources(plan_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    plan = _json(plan_path)
    required = (
        "crm_ok_gt", "fold_assignments", "sirene_establishments", "sirene_legal_units",
    )
    paths: dict[str, Path] = {}
    for key in required:
        source = plan.get("sources", {}).get(key, {})
        path = _resolve_source(plan_path, str(source.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = registry_lib.sha256(path)
        if actual != source.get("sha256"):
            raise ValueError(f"source hash mismatch for {key}: {actual}")
        paths[key] = path
    return plan, paths


def _location_relation(contract: dict[str, Any]) -> tuple[str, str]:
    values = [
        (str(field), str(relation))
        for field, relation in contract.get("field_relations", {}).items()
        if field != "name"
    ]
    if len(values) != 1:
        raise ValueError(f"contract has no unique location relation: {values}")
    return values[0]


def _validate_fragment(fragment: dict[str, Any], field: str, relation: str) -> None:
    if fragment.get("field") != field or fragment.get("relation") != relation:
        raise ValueError("field inspiration differs from its frozen relation")
    for key in ("inspiration_ref", "provenance_digest"):
        if not HEX64.fullmatch(str(fragment.get(key, ""))):
            raise ValueError(f"invalid inspiration provenance {key}")
    if fragment.get("schema_version") != "sireto-synthetic-field-inspiration-1":
        raise ValueError("unsupported field-inspiration schema")
    fold = int(fragment.get("source_fold", -99))
    split = str(fragment.get("source_legacy_split", ""))
    if relation == "OFFICIAL_NAME_ALIAS":
        if not (
            fold == -1
            and split == "sirene_official"
            and fragment.get("evidence_source_type") == "SIRENE_OFFICIAL_NAME_OPTION"
        ):
            raise ValueError("authoritative alias lacks SIRENE provenance")
    elif fold not in TRAIN_FOLDS or split != "train":
        raise ValueError("observed inspiration is not train-only")
    if not isinstance(fragment.get("operation_parameters"), dict):
        raise ValueError("field inspiration lacks operation parameters")


def _validate_contract_provenance(contract: dict[str, Any]) -> None:
    relations = contract.get("field_relations", {})
    if "name" not in relations:
        raise ValueError("contract lacks name relation")
    location_field, _ = _location_relation(contract)
    inspirations = contract.get("field_inspirations", {})
    if set(inspirations) != {"name", location_field}:
        raise ValueError("contract inspirations do not cover exactly its target fields")
    for field, fragment in inspirations.items():
        _validate_fragment(fragment, field, str(relations[field]))
    evidence = contract.get("targeting_evidence", {})
    if evidence.get("identity_free_aggregate") is not True:
        raise ValueError("targeting evidence is not identity-free")
    if not HEX64.fullmatch(str(evidence.get("catalog_sha256", ""))):
        raise ValueError("targeting evidence lacks its frozen catalog hash")


def _batch_sidecars(batch: dict[str, Any]) -> tuple[Path, Path]:
    seed_input = Path(batch["seed_input"]["path"])
    root = seed_input.parent
    batch_id = str(batch["batch_id"])
    return root / f"{batch_id}.sqlite", root / f"{batch_id}_full_sirene_audit.json"


def _baseline(seed: dict[str, Any]) -> dict[str, Any]:
    target = seed.get("seed_card", {}).get("official_context", {}).get("target", {})
    address = target.get("address", {})
    return {
        "names": [
            str(value.get("value", ""))
            for value in target.get("names", [])
            if str(value.get("value", "")).strip()
        ],
        "address": " ".join(
            str(address.get(field, "")).strip()
            for field in ("number", "repetition_index", "street_type", "street")
            if str(address.get(field, "")).strip()
        ),
        "postcode": str(address.get("postcode", "")),
        "city": str(address.get("city", "")),
        "insee": str(address.get("insee", "")),
        "state": str(target.get("state", "")),
    }


def _validate_promoted_provenance(
    promoted: dict[str, Any], contract: dict[str, Any], target_siret: str,
    target_siren: str,
) -> None:
    if not loop.valid_siret(target_siret) or not loop.valid_siren(target_siren):
        raise ValueError("invalid promoted SIRET/SIREN")
    if target_siret[:9] != target_siren:
        raise ValueError("promoted SIRET and SIREN disagree")
    if promoted.get("variant_contract_sha256") != loop.digest_json(contract):
        raise ValueError("promoted row is not bound to its frozen variant contract")
    provenance = promoted.get("promotion_provenance", {})
    response_sha = str(promoted.get("generator_response_sha256", ""))
    if not (
        promoted.get("final_decision") == "ACCEPT"
        and promoted.get("critic_decision") == "ACCEPT"
        and provenance.get("critic_decision") == "ACCEPT"
        and str(provenance.get("accepted_task_id", "")).strip()
        and HEX64.fullmatch(response_sha)
        and provenance.get("generator_response_sha256") == response_sha
    ):
        raise ValueError("promoted row lacks accepted agentic provenance")
    crm = promoted.get("crm", {})
    if set(CRM_FIELDS) - set(crm) or not str(crm.get("name", "")).strip():
        raise ValueError("promoted CRM lacks required fields or name identity")
    if not str(crm.get("address", "")).strip():
        raise ValueError("promoted CRM lacks address identity")


def _collect_registered_rows(
    registry_path: Path,
    expected_sirene_hashes: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    registry = registry_lib.load_registry(registry_path)
    quarantined = registry_lib.quarantine_records(registry)
    reconstructed = registry_lib.snapshot(registry)
    if registry.get("summary") != reconstructed:
        raise ValueError("registry summary differs from sealed batch reconstruction")
    target = int(registry["promoted_variant_target"])
    if int(reconstructed["promoted_variants"]) < target:
        raise ValueError(
            f"final audit requires >= {target} promoted variants; "
            f"found {reconstructed['promoted_variants']}"
        )

    all_rows: list[dict[str, Any]] = []
    full_audit_files: list[dict[str, Any]] = []
    for batch in registry["batches"]:
        seed_input = Path(batch["seed_input"]["path"])
        promoted_path = Path(batch["promoted"]["path"])
        promotion_manifest_path = Path(batch["promotion_manifest"]["path"])
        validated = registry_lib.validate_promoted_batch(
            seed_input, promoted_path, promotion_manifest_path
        )
        promotion_manifest = _json(promotion_manifest_path)
        ledger_path, full_audit_path = _batch_sidecars(batch)
        if not ledger_path.is_file() or not full_audit_path.is_file():
            raise FileNotFoundError(f"missing sealed batch sidecar for {batch['batch_id']}")
        source_hashes = promotion_manifest.get("source_hashes", {})
        if source_hashes.get("db") != registry_lib.sha256(ledger_path):
            raise ValueError(f"ledger hash mismatch for {batch['batch_id']}")
        if source_hashes.get("full_audit") != registry_lib.sha256(full_audit_path):
            raise ValueError(f"full-SIRENE audit hash mismatch for {batch['batch_id']}")
        full_audit = _json(full_audit_path)
        if not (
            full_audit.get("schema_version") == "sireto-synthetic-gt-full-sirene-audit-1"
            and full_audit.get("run_id") == batch.get("run_id")
            and full_audit.get("ledger_sha256") == source_hashes.get("db")
            and full_audit.get("positive_injection") is False
            and full_audit.get("qualification_uses_retrieval_or_model_scores") is False
            and full_audit.get("source_hashes") == expected_sirene_hashes
        ):
            raise ValueError(f"invalid full-SIRENE audit contract for {batch['batch_id']}")
        audit_rows = {
            (str(value.get("seed_id", "")), str(value.get("variant_id", ""))): value
            for value in full_audit.get("rows", [])
        }
        if len(audit_rows) != len(full_audit.get("rows", [])):
            raise ValueError(f"duplicate full-SIRENE keys for {batch['batch_id']}")
        seeds = {str(value["seed_id"]): value for value in _jsonl(seed_input)}
        accepted_from_batch = 0
        for record in validated["records"]:
            promoted = record["promoted"]
            key = record["key"]
            audit_row = audit_rows.get(key)
            if audit_row is None:
                raise ValueError(f"promoted row absent from full-SIRENE audit: {key}")
            if audit_row.get("full_sirene_qualification") != promoted.get(
                "full_sirene_qualification"
            ):
                raise ValueError(f"promoted full-SIRENE qualification drift: {key}")
            qualification = audit_row.get("full_sirene_qualification", {})
            if not (
                audit_row.get("variant_promotable_exact") is True
                and qualification.get("decision") == "EXACT_IDENTIFIABLE"
                and qualification.get("exact_witness") == "G_N_A"
                and qualification.get("target_naturally_returned") is True
                and qualification.get("candidate_sirets", {}).get("G_N_A")
                == [record["target_siret"]]
            ):
                raise ValueError(f"promoted row is not full-SIRENE exact: {key}")
            contract = record["contract"]
            _validate_contract_provenance(contract)
            _validate_promoted_provenance(
                promoted, contract, record["target_siret"], record["target_siren"]
            )
            seed = seeds.get(key[0])
            if seed is None:
                raise ValueError(f"missing frozen seed: {key[0]}")
            if key in quarantined:
                continue
            location_field, location_relation = _location_relation(contract)
            accepted_from_batch += 1
            all_rows.append({
                **promoted,
                "batch_id": str(batch["batch_id"]),
                "contract": contract,
                "name_relation": str(contract["field_relations"]["name"]),
                "location_field": location_field,
                "location_relation": location_relation,
                "official_baseline": _baseline(seed),
            })
        full_audit_files.append({
            "batch_id": str(batch["batch_id"]),
            "run_id": str(batch.get("run_id", "")),
            "ledger_sha256": source_hashes["db"],
            "full_sirene_audit_sha256": source_hashes["full_audit"],
            "promoted_rows": accepted_from_batch,
        })
    if len(all_rows) != int(reconstructed["promoted_variants"]):
        raise ValueError("collected row count differs from reconstructed registry")
    return registry, all_rows, {
        "registry_sha256": registry_lib.sha256(registry_path),
        "registered_batches": len(registry["batches"]),
        "sealed_full_sirene_audits": full_audit_files,
        "validated_rows": len(all_rows),
        "valid_contract_provenance_rows": len(all_rows),
        "valid_agentic_provenance_rows": len(all_rows),
        "quarantined_v1_rows_excluded": len(quarantined),
        "quarantine_report_sha256": registry.get("quarantine", {}).get("sha256"),
    }


def _real_rows(
    crm_path: Path, fold_path: Path, corpus_plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    crm = pd.read_csv(crm_path, sep=";", dtype=str, keep_default_na=False).reset_index(
        names="query_id"
    )
    required = {
        "crm_name", "crm_adresse", "crm_cp", "crm_commune", "crm_insee",
        "gt_siret", "sirene_etat",
    }
    if missing := required - set(crm.columns):
        raise ValueError(f"real CRM lacks columns: {sorted(missing)}")
    folds = pd.read_parquet(fold_path)
    fold_columns = {"query_id", "siren_component_id", "oof_fold", "legacy_split"}
    if fold_columns - set(folds.columns):
        raise ValueError("fold assignments lack required columns")
    crm["query_id"] = crm["query_id"].astype(str)
    folds = folds.copy()
    folds["query_id"] = folds["query_id"].astype(str)
    joined = crm.merge(
        folds[["query_id", "siren_component_id", "oof_fold", "legacy_split"]],
        on="query_id", how="left", validate="one_to_one",
    )
    if joined["oof_fold"].isna().any():
        raise ValueError("real CRM rows are missing fold assignments")

    population = corpus_plan.get("population", {})
    generator = corpus_plan.get("generator", {})
    if len(joined) != int(population.get("expected_joined_rows", -1)):
        raise ValueError("real CRM/fold join differs from the frozen corpus plan")

    # Reconstruct the exact leakage-safe population used by the generator.
    # Merely selecting folds 2/3/4 is insufficient: a nominal train row must
    # also be removed when its SIREN or its connected SIREN component occurs
    # in a forbidden dev/test row.  This derivation must happen before the
    # allowed split is selected, exactly as in the frozen runtime.
    forbidden = (
        joined["oof_fold"].astype("Int64").isin(population["forbidden_oof_folds"])
        | joined["legacy_split"].isin(population["forbidden_legacy_splits"])
    )
    forbidden_components = {
        str(value).strip()
        for value in joined.loc[forbidden, "siren_component_id"]
        if str(value).strip()
    }
    forbidden_sirens = {
        str(value).strip()[:9]
        for value in joined.loc[forbidden, "gt_siret"]
        if loop.valid_siret(str(value).strip())
    }

    allowed = joined[
        joined["oof_fold"].astype("Int64").isin(generator["allowed_oof_folds"])
        & joined["legacy_split"].eq(generator["allowed_legacy_split"])
    ].copy()
    allowed_sirens = allowed["gt_siret"].astype(str).str.strip().str[:9]
    allowed = allowed[
        ~allowed["siren_component_id"].astype(str).str.strip().isin(forbidden_components)
        & ~allowed_sirens.isin(forbidden_sirens)
    ].copy()

    expected_by_fold = {
        int(key): int(value)
        for key, value in population.get("allowed_by_fold", {}).items()
    }
    actual_by_fold = {
        int(key): int(value)
        for key, value in allowed["oof_fold"].astype(int).value_counts().sort_index().items()
    }
    actual_states = {
        str(key): int(value)
        for key, value in allowed["sirene_etat"].value_counts().sort_index().items()
    }
    expected_states = {
        str(key): int(value)
        for key, value in population.get("expected_target_state_counts", {}).items()
    }
    population_checks = {
        "rows": (len(allowed), int(population.get("allowed_rows", -1))),
        "components": (
            allowed["siren_component_id"].astype(str).str.strip().nunique(),
            int(population.get("allowed_components", -1)),
        ),
        "sirens": (
            allowed["gt_siret"].astype(str).str.strip().str[:9].nunique(),
            int(population.get("allowed_sirens", -1)),
        ),
    }
    if any(actual != expected for actual, expected in population_checks.values()):
        raise ValueError(f"strict real train population differs from plan: {population_checks}")
    if actual_by_fold != expected_by_fold or actual_states != expected_states:
        raise ValueError(
            "strict real train distribution differs from plan: "
            f"folds={actual_by_fold}, states={actual_states}"
        )

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [{
            "query_id": str(row.query_id),
            "target_siret": str(row.gt_siret).zfill(14),
            "target_siren": str(row.gt_siret).zfill(14)[:9],
            "target_state": str(row.sirene_etat),
            "crm": {
                "name": str(row.crm_name), "address": str(row.crm_adresse),
                "postcode": str(row.crm_cp), "city": str(row.crm_commune),
                "insee": str(row.crm_insee),
            },
            "corruption_families_observed": [],
        } for row in frame.itertuples(index=False)]

    return records(joined), records(allowed)


def _state_counts(rows: Iterable[dict[str, Any]], *, synthetic: bool) -> dict[str, int]:
    values = (
        row.get("official_baseline", {}).get("state", "") if synthetic
        else row.get("target_state", "")
        for row in rows
    )
    return dict(sorted(Counter(map(str, values)).items()))


def _corpus_distribution(
    real_all: list[dict[str, Any]], real_train: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
) -> dict[str, Any]:
    real_sirens = {row["target_siren"] for row in real_all}
    synthetic_sirens = {str(row["target_siren"]) for row in synthetic}
    overlap = real_sirens & synthetic_sirens
    if overlap:
        raise ValueError(f"synthetic truth SIRENs overlap real GT: {sorted(overlap)[:5]}")
    real_surfaces = {distribution.fingerprint(row["crm"]) for row in real_all}
    synthetic_surfaces = {distribution.fingerprint(row["crm"]) for row in synthetic}
    surface_overlap = real_surfaces & synthetic_surfaces
    if surface_overlap:
        raise ValueError("synthetic CRM surfaces collide with the existing real dataset")
    train_union = [*real_train, *synthetic]
    synthetic_targets = Counter(str(row["target_siret"]) for row in synthetic)
    synthetic_effective_weight = sum(0.5 / synthetic_targets[str(row["target_siret"])] for row in synthetic)
    real_weight = float(len(real_train))
    return {
        "real_all": {
            **distribution.summary(real_all), "target_state_counts": _state_counts(real_all, synthetic=False),
        },
        "real_train_folds_2_3_4": {
            **distribution.summary(real_train), "target_state_counts": _state_counts(real_train, synthetic=False),
        },
        "synthetic_promoted": {
            **distribution.summary(synthetic), "target_state_counts": _state_counts(synthetic, synthetic=True),
            "difficulty_counts": dict(sorted(Counter(str(row["difficulty"]) for row in synthetic).items())),
            "augmentation_stratum_counts": dict(sorted(Counter(str(row["augmentation_stratum"]) for row in synthetic).items())),
            "name_relation_counts": dict(sorted(Counter(row["name_relation"] for row in synthetic).items())),
            "location_relation_counts": dict(sorted(Counter(row["location_relation"] for row in synthetic).items())),
        },
        "raw_available_train_union": distribution.summary(train_union),
        "composition": {
            "real_train_rows": len(real_train),
            "synthetic_available_rows": len(synthetic),
            "raw_synthetic_share": len(synthetic) / len(train_union),
            "real_scene_weight_reference": real_weight,
            "synthetic_effective_weight_if_all_rows_used": synthetic_effective_weight,
            "effective_synthetic_weight_share_if_all_rows_used": (
                synthetic_effective_weight / (real_weight + synthetic_effective_weight)
            ),
            "downstream_frozen_selection_note": (
                "The actual model mix remains a deterministic 2:1 real/synthetic "
                "selection after non-injected retrieval eligibility; this audit does "
                "not claim that all available synthetic rows are trainable."
            ),
        },
        "cross_source": {
            "truth_siren_overlap": 0,
            "crm_surface_overlap": 0,
        },
    }


def qualification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = 0
    operational_alternative = 0
    operational_only = 0
    for row in rows:
        qualification = row.get("full_sirene_qualification", {})
        is_exact = qualification.get("decision") == "EXACT_IDENTIFIABLE"
        is_operational = qualification.get("operational_equivalence") is True
        exact += is_exact
        operational_alternative += is_operational
        operational_only += is_operational and not is_exact
    return {
        "rows": len(rows),
        "exact_identifiable_rows": exact,
        "exact_identifiable_rate": exact / len(rows) if rows else 0.0,
        "rows_with_operational_equivalent_alternative": operational_alternative,
        "operational_only_rows": operational_only,
        "views_are_separate": True,
        "operational_rows_are_not_substituted_for_exact_gate": True,
    }


def _allocate(cells: dict[tuple[str, ...], list[dict[str, Any]]], total: int) -> dict[tuple[str, ...], int]:
    if total < len(cells):
        raise ValueError(f"sample size {total} cannot cover {len(cells)} populated strata")
    allocation = {key: 1 for key in cells}
    remaining = total - len(cells)
    residual = {key: len(values) - 1 for key, values in cells.items()}
    capacity = sum(residual.values())
    if remaining > capacity:
        raise ValueError("sample size exceeds corpus")
    if not remaining:
        return allocation
    ideals = {key: remaining * residual[key] / capacity for key in cells}
    for key in cells:
        allocation[key] += math.floor(ideals[key])
    left = total - sum(allocation.values())
    ranked = sorted(
        cells,
        key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key),
    )
    for key in ranked[:left]:
        allocation[key] += 1
    return allocation


def stratified_realism_sample(
    rows: list[dict[str, Any]], sample_size: int, salt: str,
    excluded_seed_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if sample_size <= 0:
        raise ValueError("realism sample size must be positive")
    rows_by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_seed_ids = excluded_seed_ids or set()
    for row in rows:
        seed_id = str(row["seed_id"])
        if seed_id not in excluded_seed_ids:
            rows_by_seed[seed_id].append(row)
    representatives = [
        min(values, key=lambda row: _stable_digest(
            salt, "SEED_REPRESENTATIVE", row["batch_id"], row["seed_id"],
            row["variant_id"],
        ))
        for values in rows_by_seed.values()
    ]
    sample_size = min(sample_size, len(representatives))
    fine_dimensions = [
        "difficulty", "augmentation_stratum", "name_relation", "location_relation",
    ]
    coarse_dimensions = fine_dimensions[:-1]

    def key(row: dict[str, Any], dimensions: list[str]) -> tuple[str, ...]:
        return tuple(str(row.get(name, "")) for name in dimensions)

    fine_count = len({key(row, fine_dimensions) for row in representatives})
    dimensions = fine_dimensions if fine_count <= sample_size else coarse_dimensions
    cells: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in representatives:
        cells[key(row, dimensions)].append(row)
    for values in cells.values():
        values.sort(key=lambda row: _stable_digest(
            salt, row["batch_id"], row["seed_id"], row["variant_id"]
        ))
    allocation = _allocate(cells, sample_size)
    selected = [row for cell in sorted(cells) for row in cells[cell][:allocation[cell]]]
    selected.sort(key=lambda row: _stable_digest(
        salt, "OUTPUT", row["batch_id"], row["seed_id"], row["variant_id"]
    ))
    sample = []
    for row in selected:
        contract = row["contract"]
        sample_id = _stable_digest(
            SAMPLE_SCHEMA_VERSION, salt, row["batch_id"], row["seed_id"], row["variant_id"]
        )
        sample.append({
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "sample_id": sample_id,
            "batch_id": row["batch_id"],
            "seed_id": row["seed_id"],
            "variant_id": row["variant_id"],
            "target_siret": row["target_siret"],
            "difficulty": row["difficulty"],
            "augmentation_stratum": row["augmentation_stratum"],
            "name_relation": row["name_relation"],
            "location_field": row["location_field"],
            "location_relation": row["location_relation"],
            "official_baseline": row["official_baseline"],
            "crm": row["crm"],
            "field_inspirations": contract.get("field_inspirations", {}),
            "transformation_summary": row.get("transformation_summary", ""),
        })
    return sample, dimensions


def excluded_realism_seeds(path: Path | None) -> tuple[set[str], dict[str, Any]]:
    if path is None:
        return set(), {
            "excluded_prior_sample_rows": 0,
            "excluded_prior_sample_seed_ids": 0,
            "excluded_prior_sample_sha256": None,
        }
    rows = _jsonl(path)
    seed_ids: set[str] = set()
    sample_ids: set[str] = set()
    for row in rows:
        if row.get("schema_version") != SAMPLE_SCHEMA_VERSION:
            raise ValueError("excluded realism sample has unsupported schema")
        sample_id = str(row.get("sample_id", ""))
        seed_id = str(row.get("seed_id", ""))
        if not HEX64.fullmatch(sample_id) or not seed_id:
            raise ValueError("excluded realism sample has an invalid row identity")
        if sample_id in sample_ids:
            raise ValueError("excluded realism sample contains duplicate sample IDs")
        sample_ids.add(sample_id)
        seed_ids.add(seed_id)
    return seed_ids, {
        "excluded_prior_sample_rows": len(rows),
        "excluded_prior_sample_seed_ids": len(seed_ids),
        "excluded_prior_sample_sha256": registry_lib.sha256(path),
    }


def realism_review_summary(
    sample: list[dict[str, Any]], reviews: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if reviews is None:
        return {
            "status": "PENDING_BOUNDED_REVIEW",
            "sample_rows": len(sample),
            "pause_threshold_certain_false_realism": 2,
        }
    expected = {row["sample_id"] for row in sample}
    by_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if review.get("schema_version") != REVIEW_SCHEMA_VERSION:
            raise ValueError("unsupported realism review schema")
        sample_id = str(review.get("sample_id", ""))
        decision = str(review.get("decision", ""))
        if sample_id in by_id or sample_id not in expected:
            raise ValueError("realism review contains duplicate or unknown sample id")
        if decision not in REALISM_DECISIONS:
            raise ValueError(f"unsupported realism decision: {decision}")
        if decision != "PASS" and not str(review.get("reason", "")).strip():
            raise ValueError("non-PASS realism decisions require a reason")
        by_id[sample_id] = review
    if set(by_id) != expected:
        raise ValueError("realism review does not cover the frozen sample exactly")
    counts = Counter(str(value["decision"]) for value in by_id.values())
    certain = counts["CERTAIN_FALSE_REALISM"]
    upper = 1 - math.pow(0.05, 1 / len(sample)) if not certain and sample else None
    return {
        "status": "COMPLETE",
        "sample_rows": len(sample),
        "decision_counts": {key: counts.get(key, 0) for key in sorted(REALISM_DECISIONS)},
        "certain_false_realism_rate": certain / len(sample) if sample else 0.0,
        "one_sided_95pct_upper_if_zero_certain": upper,
        "pause_threshold_certain_false_realism": 2,
        "verdict": "PAUSE_DOWNSTREAM" if certain >= 2 else "PASS",
        "certain_false_realism": [
            {"sample_id": key, "reason": value.get("reason", "")}
            for key, value in sorted(by_id.items())
            if value["decision"] == "CERTAIN_FALSE_REALISM"
        ],
        "borderline": [
            {"sample_id": key, "reason": value.get("reason", "")}
            for key, value in sorted(by_id.items())
            if value["decision"] == "BORDERLINE"
        ],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for value in values:
            target.write(_canonical(value).decode("utf-8") + "\n")


def run(args: argparse.Namespace) -> Path:
    balanced_plan = _json(args.balanced_plan)
    expected_target = int(
        balanced_plan.get("objective", {}).get("promoted_variant_target", -1)
    )
    registry = registry_lib.load_registry(args.registry)
    if int(registry.get("promoted_variant_target", -2)) != expected_target:
        raise ValueError("balanced plan and registry promoted targets differ")
    recorded_count = int(registry.get("summary", {}).get("promoted_variants", -1))
    if recorded_count < expected_target:
        raise ValueError(
            f"final audit is post-production only: {recorded_count}/{expected_target} "
            "promoted variants are currently registered"
        )
    corpus_plan, source_paths = _verify_plan_sources(args.corpus_plan)
    expected_sirene_hashes = {
        key: corpus_plan["sources"][key]["sha256"]
        for key in ("sirene_establishments", "sirene_legal_units")
    }
    _registry, synthetic, provenance = _collect_registered_rows(
        args.registry, expected_sirene_hashes
    )
    real_all, real_train = _real_rows(
        source_paths["crm_ok_gt"], source_paths["fold_assignments"], corpus_plan
    )
    expected_real_all = int(corpus_plan.get("sources", {}).get("crm_ok_gt", {}).get("row_count", -1))
    expected_real_train = int(corpus_plan.get("population", {}).get("allowed_rows", -1))
    if len(real_all) != expected_real_all or len(real_train) != expected_real_train:
        raise ValueError("real dataset counts differ from frozen corpus plan")
    excluded_seed_ids, exclusion_provenance = excluded_realism_seeds(
        getattr(args, "exclude_realism_sample", None)
    )
    sample, dimensions = stratified_realism_sample(
        synthetic, args.realism_sample_size, args.realism_salt, excluded_seed_ids
    )
    reviews = _jsonl(args.realism_review) if args.realism_review else None
    realism = realism_review_summary(sample, reviews)
    exact_operational = qualification_summary(synthetic)
    if exact_operational["exact_identifiable_rows"] != len(synthetic):
        raise ValueError("final corpus contains a non-exact promoted row")
    report = {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": "POST_PRODUCTION_ONLY",
        "production_runner_modified": False,
        "retrieval_or_model_scores_used_for_qualification": False,
        "positive_injection": False,
        "corpus_complete": len(synthetic) >= expected_target,
        "promoted_variant_target": expected_target,
        "promoted_variants": len(synthetic),
        "source_hashes": {
            "registry": registry_lib.sha256(args.registry),
            "corpus_plan": registry_lib.sha256(args.corpus_plan),
            "balanced_plan": registry_lib.sha256(args.balanced_plan),
            **{key: registry_lib.sha256(path) for key, path in source_paths.items()},
        },
        "deterministic_invariants": {
            **provenance,
            "all_promoted_rows_full_sirene_exact": True,
            "all_promoted_rows_have_contract_provenance": True,
            "all_promoted_rows_have_agentic_provenance": True,
            "all_synthetic_sirens_disjoint_from_real_gt": True,
            "all_synthetic_surfaces_disjoint_from_real_gt": True,
        },
        "qualification": exact_operational,
        "distribution": _corpus_distribution(real_all, real_train, synthetic),
        "realism_audit": {
            **realism,
            "sample_schema_version": SAMPLE_SCHEMA_VERSION,
            "sample_sha256": hashlib.sha256(
                b"".join(_canonical(value) + b"\n" for value in sample)
            ).hexdigest(),
            "sample_salt": args.realism_salt,
            "stratification_dimensions": dimensions,
            "one_surface_per_seed": len({row["seed_id"] for row in sample}) == len(sample),
            "disjoint_from_prior_sample": not bool(
                {row["seed_id"] for row in sample} & excluded_seed_ids
            ),
            **exclusion_provenance,
            "populated_strata_covered": len({
                tuple(str(row[name]) for name in dimensions) for row in sample
            }),
            "review_sha256": registry_lib.sha256(args.realism_review)
            if args.realism_review else None,
        },
        "final_status": (
            realism.get("verdict", "PENDING_BOUNDED_REALISM_REVIEW")
        ),
    }

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        _write_jsonl(temporary / "realism_sample.jsonl", sample)
        _write_json(temporary / "report.json", report)
        if args.realism_review:
            shutil.copyfile(args.realism_review, temporary / "realism_review.jsonl")
        output_names = sorted(path.name for path in temporary.iterdir())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "files": {
                name: {
                    "sha256": registry_lib.sha256(temporary / name),
                    "size_bytes": (temporary / name).stat().st_size,
                }
                for name in output_names
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "output": str(args.output),
        "promoted_variants": len(synthetic),
        "real_train_rows": len(real_train),
        "realism_sample_rows": len(sample),
        "final_status": report["final_status"],
    }, ensure_ascii=False, sort_keys=True))
    return args.output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--corpus-plan", type=Path, default=DEFAULT_CORPUS_PLAN)
    result.add_argument("--balanced-plan", type=Path, default=DEFAULT_BALANCED_PLAN)
    result.add_argument("--realism-review", type=Path)
    result.add_argument(
        "--exclude-realism-sample", type=Path,
        help="Prior sealed sample whose seed IDs must not be selected again.",
    )
    result.add_argument("--realism-sample-size", type=int, default=200)
    result.add_argument("--realism-salt", default="SIRETO_BALANCED_FINAL_REALISM_V1")
    result.add_argument("--output", type=Path, required=True)
    result.set_defaults(func=run)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
