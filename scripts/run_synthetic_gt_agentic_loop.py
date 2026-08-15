#!/usr/bin/env python3
"""Durable agentic orchestration for SIRETO synthetic CRM generation.

This module deliberately cannot generate or rewrite CRM text.  It leases
official seed cards to LLM workers, validates their byte-preserved JSON
responses, routes risky cases to independent review, and publishes only
supervised variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "config" / "synthetic_gt_agentic_message_schema_v2.json"
SCHEMA_VERSION = "sireto-synthetic-gt-agentic-message-2"
PROMPT_VERSIONS = {
    "GENERATOR": "sireto-gt-generator-v7",
    "CRITIC": "sireto-gt-critic-v7",
    "ADJUDICATOR": "sireto-gt-adjudicator-v4",
}
PENDING_BY_ROLE = {
    "GENERATOR": "PENDING_GENERATOR",
    "CRITIC": "PENDING_CRITIC",
    "ADJUDICATOR": "PENDING_ADJUDICATOR",
}
LEASED_BY_ROLE = {
    "GENERATOR": "LEASED_GENERATOR",
    "CRITIC": "LEASED_CRITIC",
    "ADJUDICATOR": "LEASED_ADJUDICATOR",
}
CRM_FIELDS = ("name", "address", "postcode", "city", "insee")
FINAL_DECISIONS = {"ACCEPT", "SILVER", "REJECT"}
REQUESTED_FAMILIES_BY_DIMENSION = {
    "name": {
        "ACCENT_PUNCTUATION",
        "ACRONYM_TOKENIZATION",
        "ENSEIGNE_VS_DENOMINATION",
        "LEGAL_FORM",
        "OCR_LIMITED",
        "TOKEN_ORDER",
    },
    "address": {
        "ADDRESS_ABBREVIATION",
        "ADDRESS_OCR",
        "ADDRESS_TOKEN_ORDER",
        "COMMUNE_VARIANT",
    },
    "orthographic": {"ACCENT_PUNCTUATION", "OCR_LIMITED"},
}
VARIANT_BY_DIMENSION = {"name": "v1", "address": "v2", "orthographic": "v3"}
LEGAL_FORM_TOKENS = {
    "ASS", "ASSO", "ASSOCIATION", "EARL", "EI", "EIRL", "EURL", "GAEC",
    "GIE", "SA", "SARL", "SAS", "SASU", "SC", "SCI", "SCP", "SELARL",
    "SEM", "SNC",
}
STREET_TYPE_ABBREVIATIONS = {
    "R": "RUE", "AV": "AVENUE", "BD": "BOULEVARD", "CH": "CHEMIN",
    "CHE": "CHEMIN", "CHEM": "CHEMIN", "IMP": "IMPASSE", "PL": "PLACE", "RTE": "ROUTE",
    "ALL": "ALLEE", "QU": "QUAI", "RES": "RESIDENCE",
}
PUNCTUATION = set("'’.-,()/&")
COMPOSITE_FAMILY = "OBSERVED_COMPOSITE_ANALOGY"
COMPOSITE_RELATIONS_BY_FIELD = {
    "name": {
        "TOKEN_ORDER", "TOKEN_SUBSET", "LEGAL_FORM_REMOVE", "JOIN_SPLIT",
        "PUNCTUATION_REMOVED", "DIACRITIC_REMOVED",
    },
    "address": {
        "ADDRESS_ABBREVIATE", "ADDRESS_TYPE_ORDER", "ADDRESS_TOKEN_SUBSET", "JOIN_SPLIT",
        "PUNCTUATION_REMOVED", "DIACRITIC_REMOVED",
    },
    "city": {
        "JOIN_SPLIT", "PUNCTUATION_REMOVED", "DIACRITIC_REMOVED",
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def now_ts() -> int:
    return int(time.time())


def valid_siret(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9]{14}", str(value or "")))


def valid_siren(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9]{9}", str(value or "")))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl_raw(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            raw = raw_line.rstrip("\r\n")
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL object required at {path}:{line_number}")
            yield raw, value


def iter_response_raw(path: Path, input_format: str) -> Iterable[tuple[str, dict[str, Any]]]:
    if input_format == "jsonl":
        yield from iter_jsonl_raw(path)
        return
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"empty JSON response: {path}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required at {path}")
    yield raw, value


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            policy_json TEXT NOT NULL,
            schema_sha256 TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS seeds (
            run_id TEXT NOT NULL,
            seed_id TEXT NOT NULL,
            target_siret TEXT NOT NULL,
            target_siren TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            oof_fold INTEGER NOT NULL,
            legacy_split TEXT NOT NULL,
            seed_card_json TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            risk_flags_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 0,
            lease_role TEXT,
            lease_worker TEXT,
            lease_expires_at INTEGER,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (run_id, seed_id),
            UNIQUE (run_id, target_siret),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            seed_id TEXT NOT NULL,
            role TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            task_json TEXT NOT NULL,
            status TEXT NOT NULL,
            raw_response TEXT,
            response_sha256 TEXT,
            created_at INTEGER NOT NULL,
            completed_at INTEGER,
            FOREIGN KEY (run_id, seed_id) REFERENCES seeds(run_id, seed_id)
        );

        CREATE TABLE IF NOT EXISTS variants (
            run_id TEXT NOT NULL,
            seed_id TEXT NOT NULL,
            variant_id TEXT NOT NULL,
            crm_json TEXT NOT NULL,
            crm_fingerprint TEXT NOT NULL,
            families_json TEXT NOT NULL,
            transformation_summary TEXT NOT NULL,
            generator_response_sha256 TEXT NOT NULL,
            preflight_json TEXT NOT NULL,
            critic_json TEXT,
            critic_decision TEXT,
            adjudicator_json TEXT,
            adjudicator_decision TEXT,
            final_decision TEXT,
            final_reason TEXT,
            PRIMARY KEY (run_id, seed_id, variant_id),
            FOREIGN KEY (run_id, seed_id) REFERENCES seeds(run_id, seed_id)
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            seed_id TEXT,
            event_type TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS seeds_status_idx
            ON seeds(run_id, status, seed_id);
        CREATE INDEX IF NOT EXISTS tasks_seed_role_idx
            ON tasks(run_id, seed_id, role, status);
        """
    )


def event(
    connection: sqlite3.Connection,
    run_id: str,
    seed_id: str | None,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    encoded = canonical_json(payload)
    connection.execute(
        """INSERT INTO events
           (run_id, seed_id, event_type, payload_sha256, payload_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, seed_id, event_type, hashlib.sha256(encoded.encode()).hexdigest(), encoded, now_ts()),
    )


def normalize_seed_input(seed: dict[str, Any]) -> dict[str, Any]:
    """Accept the v2 seed envelope or the audited v1 request-card envelope."""
    required = {
        "seed_id", "target_siret", "target_siren", "source_kind", "oof_fold",
        "legacy_split", "seed_card", "observed_train_profile", "risk_flags",
    }
    if required.issubset(seed):
        return seed
    if {
        "schema_version", "seed", "seed_card", "observed_train_profile"
    }.issubset(seed) and seed.get("schema_version") == "sireto-synthetic-gt-agentic-request-1":
        identity = seed["seed"]
        siret = str(identity.get("siret", ""))
        source = str(identity.get("seed_source", "CRM_OK_GT_TRAIN"))
        return {
            "seed_id": f"{source}:{siret}",
            "target_siret": siret,
            "target_siren": str(identity.get("siren", "")),
            "source_kind": source if source in {"SIRENE_ONLY_TRAIN", "MAPS_ASSISTED"} else "SIRENE_SYNTHETIC",
            "oof_fold": identity.get("oof_fold", -1 if source == "SIRENE_ONLY_TRAIN" else None),
            "legacy_split": "train" if source == "CRM_OK_GT_TRAIN" else "train_synthetic",
            "seed_card": seed["seed_card"],
            "observed_train_profile": seed["observed_train_profile"],
            "risk_flags": list(seed["seed_card"].get("risk_flags", [])),
        }
    return seed


def validate_seed(seed: dict[str, Any]) -> dict[str, Any]:
    seed = normalize_seed_input(seed)
    required = {
        "seed_id",
        "target_siret",
        "target_siren",
        "source_kind",
        "oof_fold",
        "legacy_split",
        "seed_card",
        "observed_train_profile",
        "risk_flags",
    }
    missing = sorted(required - set(seed))
    if missing:
        raise ValueError(f"seed missing fields: {missing}")
    siret = str(seed["target_siret"])
    siren = str(seed["target_siren"])
    if not valid_siret(siret) or not valid_siren(siren) or siret[:9] != siren:
        raise ValueError(f"invalid SIRET/SIREN relation for seed {seed.get('seed_id')}")
    fold = int(seed["oof_fold"])
    if seed["source_kind"] == "SIRENE_ONLY_TRAIN":
        if fold != -1:
            raise ValueError(f"SIRENE_ONLY_TRAIN requires oof_fold=-1 for seed {seed['seed_id']}")
    elif fold not in {2, 3, 4}:
        raise ValueError(f"forbidden fold for seed {seed['seed_id']}: {fold}")
    if seed["legacy_split"] not in {"train", "train_synthetic"}:
        raise ValueError(f"forbidden split for seed {seed['seed_id']}")
    if seed["source_kind"] not in {"SIRENE_SYNTHETIC", "SIRENE_ONLY_TRAIN", "MAPS_ASSISTED"}:
        raise ValueError(f"unsupported source kind for seed {seed['seed_id']}")
    if not isinstance(seed["seed_card"], dict) or not isinstance(seed["observed_train_profile"], dict):
        raise ValueError("seed_card and observed_train_profile must be objects")
    if not isinstance(seed["risk_flags"], list) or not all(
        isinstance(value, str) for value in seed["risk_flags"]
    ):
        raise ValueError("risk_flags must be a list of strings")
    validate_requested_families(seed["seed_id"], seed["seed_card"])
    validate_profile_evidence(
        seed["seed_id"], seed["seed_card"], seed["observed_train_profile"]
    )
    validate_composite_contracts(
        seed["seed_id"], seed["seed_card"], seed["observed_train_profile"]
    )
    return seed


def validate_requested_families(seed_id: str, seed_card: dict[str, Any]) -> None:
    """Reject mechanically assigned corruption families that cannot fit their field."""
    requested = seed_card.get("requested_families")
    if requested is None:
        return
    if not isinstance(requested, dict) or set(requested) != set(REQUESTED_FAMILIES_BY_DIMENSION):
        raise ValueError(
            f"requested_families must define name/address/orthographic for seed {seed_id}"
        )
    for dimension in ("name", "address", "orthographic"):
        family = requested[dimension]
        if family not in REQUESTED_FAMILIES_BY_DIMENSION[dimension]:
            raise ValueError(
                f"family {family} is incompatible with dimension {dimension} for seed {seed_id}"
            )


def validate_profile_evidence(
    seed_id: str,
    seed_card: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    requested = seed_card.get("requested_families")
    if not requested:
        return
    rows = profile.get("rows")
    phenomena = profile.get("phenomena")
    supported = profile.get("supported_families")
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError(f"observed train profile has no rows for seed {seed_id}")
    if not isinstance(phenomena, dict) or not isinstance(supported, list):
        raise ValueError(f"observed train profile lacks family evidence for seed {seed_id}")
    for family in requested.values():
        if family not in supported or not isinstance(phenomena.get(family), int) or phenomena[family] <= 0:
            raise ValueError(
                f"family {family} has no observed train evidence for seed {seed_id}"
            )

    names = [
        str(value).strip()
        for key in ("name_options", "enseigne_options")
        for value in seed_card.get(key, [])
        if str(value).strip()
    ]
    if not names:
        raise ValueError(f"name corruption requested without an official name for seed {seed_id}")
    if requested["name"] == "ENSEIGNE_VS_DENOMINATION":
        baseline = comparison_fingerprint({
            "name": str(seed_card.get("name_options", [""])[0]),
            "address": "", "postcode": "", "city": "", "insee": "",
        })
        alternatives = {
            comparison_fingerprint({
                "name": str(value), "address": "", "postcode": "", "city": "", "insee": ""
            })
            for value in seed_card.get("enseigne_options", [])
            if str(value).strip()
        }
        if not alternatives or alternatives == {baseline}:
            raise ValueError(
                f"ENSEIGNE_VS_DENOMINATION requires a distinct official enseigne for seed {seed_id}"
            )
    if not str(seed_card.get("address", "")).strip():
        raise ValueError(f"address corruption requested without an official address for seed {seed_id}")
    if requested["address"] == "ADDRESS_ABBREVIATION" and not str(
        seed_card.get("street_type", "")
    ).strip():
        raise ValueError(f"ADDRESS_ABBREVIATION requires a street type for seed {seed_id}")
    if requested["address"] == "COMMUNE_VARIANT" and not str(seed_card.get("city", "")).strip():
        raise ValueError(f"COMMUNE_VARIANT requires an official city for seed {seed_id}")
    for dimension in ("name", "address", "orthographic"):
        family = requested[dimension]
        if not source_supports_family(seed_card, dimension, family):
            raise ValueError(
                f"source does not support family {family} in dimension {dimension} "
                f"for seed {seed_id}"
            )


def validate_composite_contracts(
    seed_id: str,
    seed_card: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    contracts = seed_card.get("composite_contracts")
    if contracts is None:
        return
    if seed_card.get("generation_mode") != "OBSERVED_COMPOSITE_ANALOGY_V2":
        raise ValueError(f"unsupported composite generation mode for seed {seed_id}")
    if not isinstance(contracts, list) or len(contracts) != 3:
        raise ValueError(f"composite contracts must contain exactly three variants for seed {seed_id}")
    expected_ids = {"v1", "v2", "v3"}
    refs: list[str] = []
    signatures: list[str] = []
    for contract in contracts:
        variant_id = contract.get("variant_id")
        fields = set(contract.get("target_fields", []))
        if (
            variant_id not in expected_ids
            or "name" not in fields
            or not ({"address", "city"} & fields)
            or not fields.issubset({"name", "address", "city"})
        ):
            raise ValueError(f"invalid composite field mask for seed {seed_id}: {variant_id}")
        if contract.get("requested_family") != COMPOSITE_FAMILY:
            raise ValueError(f"invalid composite family for seed {seed_id}: {variant_id}")
        field_relations = contract.get("field_relations")
        if not isinstance(field_relations, dict) or set(field_relations) != fields:
            raise ValueError(f"composite field relations mismatch for seed {seed_id}: {variant_id}")
        field_inspirations = contract.get("field_inspirations")
        if field_inspirations is not None:
            if not isinstance(field_inspirations, dict) or set(field_inspirations) != fields:
                raise ValueError(f"composite field inspirations mismatch for seed {seed_id}: {variant_id}")
        else:
            ref = str(contract.get("inspiration_ref", ""))
            inspiration = contract.get("inspiration")
            if not re.fullmatch(r"[0-9a-f]{64}", ref) or not isinstance(inspiration, dict):
                raise ValueError(f"invalid composite inspiration for seed {seed_id}: {variant_id}")
            if inspiration.get("inspiration_ref") != ref:
                raise ValueError(f"composite inspiration ref mismatch for seed {seed_id}: {variant_id}")
            if set(inspiration.get("structural_signature", {}).get("changed_fields", [])) != fields:
                raise ValueError(f"composite inspiration mask mismatch for seed {seed_id}: {variant_id}")
            if inspiration.get("source_fold") not in {2, 3, 4}:
                raise ValueError(f"protected inspiration fold for seed {seed_id}: {variant_id}")
        for field, relation in field_relations.items():
            if relation not in COMPOSITE_RELATIONS_BY_FIELD.get(field, set()):
                raise ValueError(
                    f"unsupported composite relation {field}:{relation} for seed {seed_id}: {variant_id}"
                )
            if field_inspirations is not None:
                fragment = field_inspirations[field]
                ref = str(fragment.get("inspiration_ref", ""))
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", ref)
                    or fragment.get("field") != field
                    or fragment.get("source_fold") not in {2, 3, 4}
                    or fragment.get("relation") != relation
                ):
                    raise ValueError(
                        f"invalid field inspiration {field} for seed {seed_id}: {variant_id}"
                    )
                inspiration_source = str(fragment.get("official_value", ""))
                inspiration_target = str(fragment.get("observed_crm_value", ""))
                expected_parameters = fragment.get("operation_parameters")
                refs.append(ref)
            else:
                inspiration_source = str(inspiration.get("official", {}).get(field, ""))
                inspiration_target = str(inspiration.get("observed_crm", {}).get(field, ""))
                expected_parameters = None
            inspiration_relation = composite_relation_class(
                field, inspiration_source, inspiration_target,
            )
            if inspiration_relation != relation:
                raise ValueError(
                    f"inspiration relation mismatch {field}:{relation}!={inspiration_relation} "
                    f"for seed {seed_id}: {variant_id}"
                )
            if field_inspirations is not None:
                observed_parameters = composite_operation_parameters(
                    field, relation, inspiration_source, inspiration_target
                )
                if expected_parameters != observed_parameters:
                    raise ValueError(
                        f"field inspiration operator mismatch {field} for seed {seed_id}: {variant_id}"
                    )
        if field_inspirations is None:
            refs.append(ref)
        signatures.append(digest_json({"fields": sorted(fields), "relations": field_relations}))
    if {contract.get("variant_id") for contract in contracts} != expected_ids:
        raise ValueError(f"composite variant ids must be exactly v1/v2/v3 for seed {seed_id}")
    if len(set(refs)) != len(refs):
        raise ValueError(f"composite inspirations must be unique within seed {seed_id}")
    if len(set(signatures)) != 3:
        raise ValueError(f"composite operation signatures must differ within seed {seed_id}")
    if profile.get("rows", 0) <= 0 or COMPOSITE_FAMILY not in profile.get("supported_families", []):
        raise ValueError(f"composite profile evidence missing for seed {seed_id}")
    qualification = seed_card.get("qualification", {})
    if not qualification.get("pre_generation_exact_eligible"):
        raise ValueError(f"composite seed is not pre-generation exact eligible: {seed_id}")
    if not all(
        qualification.get(key)
        for key in ("siblings_complete", "same_address_complete", "same_name_geography_complete")
    ):
        raise ValueError(f"composite official context is incomplete: {seed_id}")


def command_init(args: argparse.Namespace) -> None:
    schema = load_json(args.schema)
    Draft202012Validator.check_schema(schema)
    schema_hash = hashlib.sha256(args.schema.read_bytes()).hexdigest()
    policy = {
        "schema_version": SCHEMA_VERSION,
        "critic_mode": args.critic_mode,
        "critic_sample_modulus": args.critic_sample_modulus,
        "critic_sample_slots": args.critic_sample_slots,
        "allow_easy_supervisor": bool(args.allow_easy_supervisor),
        "max_generator_attempts": args.max_generator_attempts,
        "variants_per_seed": 3,
    }
    if args.critic_mode != "all" and not args.allow_easy_supervisor:
        raise ValueError("targeted critic requires --allow-easy-supervisor after a passed pilot")
    if not (0 <= args.critic_sample_slots <= args.critic_sample_modulus):
        raise ValueError("invalid critic sample slots/modulus")
    rows = [validate_seed(value) for _raw, value in iter_jsonl_raw(args.seeds)]
    if not rows:
        raise ValueError("seed JSONL is empty")
    target_sirets = [row["target_siret"] for row in rows]
    seed_ids = [str(row["seed_id"]) for row in rows]
    if len(set(target_sirets)) != len(target_sirets) or len(set(seed_ids)) != len(seed_ids):
        raise ValueError("duplicate seed_id or target_siret")
    profile_hashes = {digest_json(row["observed_train_profile"]) for row in rows}
    if len(profile_hashes) != 1:
        raise ValueError("all seeds in a run must use one immutable observed train profile")
    policy["observed_train_profile_sha256"] = next(iter(profile_hashes))

    with connect(args.db) as connection:
        create_schema(connection)
        connection.execute(
            "INSERT INTO runs(run_id, policy_json, schema_sha256, created_at) VALUES (?, ?, ?, ?)",
            (args.run_id, canonical_json(policy), schema_hash, now_ts()),
        )
        for row in rows:
            connection.execute(
                """INSERT INTO seeds
                   (run_id, seed_id, target_siret, target_siren, source_kind,
                    oof_fold, legacy_split, seed_card_json, profile_json,
                    risk_flags_json, status, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_GENERATOR', ?)""",
                (
                    args.run_id,
                    str(row["seed_id"]),
                    row["target_siret"],
                    row["target_siren"],
                    row["source_kind"],
                    int(row["oof_fold"]),
                    row["legacy_split"],
                    canonical_json(row["seed_card"]),
                    canonical_json(row["observed_train_profile"]),
                    canonical_json(sorted(set(row["risk_flags"]))),
                    now_ts(),
                ),
            )
        event(connection, args.run_id, None, "RUN_INITIALIZED", {"seed_count": len(rows), "policy": policy})
    print(canonical_json({"run_id": args.run_id, "seed_count": len(rows), "status": "INITIALIZED"}))


def reap_expired(connection: sqlite3.Connection, run_id: str) -> int:
    now = now_ts()
    expired = connection.execute(
        """SELECT seed_id, lease_role FROM seeds
           WHERE run_id=? AND lease_expires_at IS NOT NULL AND lease_expires_at < ?""",
        (run_id, now),
    ).fetchall()
    for row in expired:
        role = row["lease_role"]
        if role not in PENDING_BY_ROLE:
            raise RuntimeError(f"invalid expired lease role: {role}")
        connection.execute(
            """UPDATE seeds SET status=?, lease_role=NULL, lease_worker=NULL,
               lease_expires_at=NULL, updated_at=? WHERE run_id=? AND seed_id=?""",
            (PENDING_BY_ROLE[role], now, run_id, row["seed_id"]),
        )
        connection.execute(
            """UPDATE tasks SET status='EXPIRED'
               WHERE run_id=? AND seed_id=? AND role=? AND status='LEASED'""",
            (run_id, row["seed_id"], role),
        )
        event(connection, run_id, row["seed_id"], "LEASE_EXPIRED", {"role": role})
    return len(expired)


def critic_input(connection: sqlite3.Connection, run_id: str, seed: sqlite3.Row) -> dict[str, Any]:
    seed_card = json.loads(seed["seed_card_json"])
    variants = connection.execute(
        """SELECT variant_id, crm_json FROM variants
           WHERE run_id=? AND seed_id=? ORDER BY variant_id""",
        (run_id, seed["seed_id"]),
    ).fetchall()
    return {
        "seed": {"siret": seed["target_siret"], "siren": seed["target_siren"]},
        "seed_card": llm_seed_card(seed_card),
        "baseline_crm": official_baseline(seed_card),
        "variant_contract": variant_contract(seed_card),
        "variants": [
            {"variant_id": row["variant_id"], "crm": json.loads(row["crm_json"])} for row in variants
        ],
    }


def adjudicator_input(connection: sqlite3.Connection, run_id: str, seed: sqlite3.Row) -> dict[str, Any]:
    seed_card = json.loads(seed["seed_card_json"])
    variants = connection.execute(
        """SELECT variant_id, crm_json, preflight_json, critic_json, critic_decision
           FROM variants WHERE run_id=? AND seed_id=? ORDER BY variant_id""",
        (run_id, seed["seed_id"]),
    ).fetchall()
    return {
        "seed": {"siret": seed["target_siret"], "siren": seed["target_siren"]},
        "seed_card": llm_seed_card(seed_card),
        "baseline_crm": official_baseline(seed_card),
        "variant_contract": variant_contract(seed_card),
        "variants": [
            {
                "variant_id": row["variant_id"],
                "crm": json.loads(row["crm_json"]),
                "preflight": json.loads(row["preflight_json"]),
                "critic": json.loads(row["critic_json"]) if row["critic_json"] else None,
                "critic_decision": row["critic_decision"],
            }
            for row in variants
        ],
    }


def make_task(
    connection: sqlite3.Connection,
    run_id: str,
    role: str,
    worker_id: str,
    batch_id: str,
    seed: sqlite3.Row,
) -> dict[str, Any]:
    if role == "GENERATOR":
        seed_card = json.loads(seed["seed_card_json"])
        requested = seed_card.get("requested_families")
        role_input = {
            "seed": {"siret": seed["target_siret"], "siren": seed["target_siren"]},
            "source_kind": seed["source_kind"],
            "seed_card": llm_seed_card(seed_card),
            "observed_train_profile": json.loads(seed["profile_json"]),
            "risk_flags": json.loads(seed["risk_flags_json"]),
        }
        if requested or seed_card.get("composite_contracts") is not None:
            role_input["baseline_crm"] = official_baseline(seed_card)
            role_input["variant_contract"] = variant_contract(seed_card)
        if int(seed["attempt"]) > 0:
            previous = connection.execute(
                """SELECT variant_id, crm_fingerprint, preflight_json FROM variants
                   WHERE run_id=? AND seed_id=? ORDER BY variant_id""",
                (run_id, seed["seed_id"]),
            ).fetchall()
            if previous:
                role_input["retry_context"] = {
                    "previous_fingerprints_to_avoid": [row["crm_fingerprint"] for row in previous],
                    "previous_preflight_errors": sorted({
                        error
                        for row in previous
                        for error in json.loads(row["preflight_json"])["errors"]
                    }),
                }
    elif role == "CRITIC":
        role_input = critic_input(connection, run_id, seed)
    else:
        role_input = adjudicator_input(connection, run_id, seed)
    input_sha256 = digest_json(role_input)
    task_ordinal = int(connection.execute(
        "SELECT COUNT(*) FROM tasks WHERE run_id=? AND seed_id=? AND role=?",
        (run_id, seed["seed_id"], role),
    ).fetchone()[0]) + 1
    task_id = hashlib.sha256(
        f"{run_id}|{role}|{seed['seed_id']}|{task_ordinal}|{batch_id}".encode()
    ).hexdigest()[:32]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "run_id": run_id,
        "batch_id": batch_id,
        "role": role,
        "prompt_version": PROMPT_VERSIONS[role],
        "input_sha256": input_sha256,
        "input": role_input,
    }


def command_lease(args: argparse.Namespace) -> None:
    role = args.role
    connection = connect(args.db)
    try:
        connection.execute("BEGIN IMMEDIATE")
        create_schema(connection)
        reap_expired(connection, args.run_id)
        seeds = connection.execute(
            """SELECT * FROM seeds WHERE run_id=? AND status=?
               ORDER BY seed_id LIMIT ?""",
            (args.run_id, PENDING_BY_ROLE[role], args.limit),
        ).fetchall()
        batch_id = f"{role.lower()}-{now_ts()}-{hashlib.sha256(args.worker_id.encode()).hexdigest()[:8]}"
        tasks: list[dict[str, Any]] = []
        expiry = now_ts() + args.ttl_seconds
        for seed in seeds:
            task = make_task(connection, args.run_id, role, args.worker_id, batch_id, seed)
            connection.execute(
                """INSERT INTO tasks
                   (task_id, run_id, seed_id, role, batch_id, worker_id,
                    prompt_version, input_sha256, task_json, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'LEASED', ?)""",
                (
                    task["task_id"], args.run_id, seed["seed_id"], role, batch_id,
                    args.worker_id, PROMPT_VERSIONS[role], task["input_sha256"],
                    canonical_json(task), now_ts(),
                ),
            )
            connection.execute(
                """UPDATE seeds SET status=?, lease_role=?, lease_worker=?,
                   lease_expires_at=?, updated_at=?
                   WHERE run_id=? AND seed_id=?""",
                (
                    LEASED_BY_ROLE[role], role, args.worker_id, expiry,
                    now_ts(), args.run_id, seed["seed_id"],
                ),
            )
            event(connection, args.run_id, seed["seed_id"], "TASK_LEASED", {
                "task_id": task["task_id"], "role": role, "worker_id": args.worker_id,
                "expires_at": expiry,
            })
            tasks.append(task)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    if args.output:
        write_jsonl_atomic(args.output, tasks)
    else:
        for task in tasks:
            print(canonical_json(task))


def response_validator(schema_path: Path) -> Draft202012Validator:
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def comparison_fingerprint(crm: dict[str, str]) -> str:
    joined = "|".join(crm[field] for field in CRM_FIELDS).casefold()
    return "".join(character for character in joined if character.isalnum())


def surface_fingerprint(crm: dict[str, str]) -> str:
    return "\x1f".join(normalized_surface(crm[field]) for field in CRM_FIELDS)


def normalized_words(value: Any) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", str(value or "").casefold())
    plain = "".join(character for character in decomposed if not unicodedata.combining(character))
    return re.findall(r"[a-z0-9]+", plain)


def normalized_alnum(value: Any) -> str:
    return "".join(normalized_words(value))


def normalized_surface(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def punctuation_marks(value: Any) -> set[str]:
    return {character for character in str(value or "") if character in PUNCTUATION}


def has_diacritic(value: Any) -> bool:
    return any(unicodedata.combining(character) for character in unicodedata.normalize("NFD", str(value or "")))


def diacritic_profile(value: Any) -> list[frozenset[str]]:
    profile: list[frozenset[str]] = []
    for character in str(value or ""):
        decomposed = unicodedata.normalize("NFD", character)
        bases = [item for item in decomposed if item.isalnum()]
        marks = frozenset(item for item in decomposed if unicodedata.combining(item))
        profile.extend(marks for _base in bases)
    return profile


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_character in enumerate(left, 1):
        current = [row_index]
        for column_index, right_character in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def official_baseline(seed_card: dict[str, Any]) -> dict[str, str]:
    names = [str(value).strip() for value in seed_card.get("name_options", []) if str(value).strip()]
    return {
        "name": names[0] if names else "",
        "address": str(seed_card.get("address", "")).strip(),
        "postcode": str(seed_card.get("postcode", "")).strip(),
        "city": str(seed_card.get("city", "")).strip(),
        "insee": str(seed_card.get("insee", "")).strip(),
    }


def llm_seed_card(seed_card: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded public card, excluding raw competitor identities."""
    return {
        key: value
        for key, value in seed_card.items()
        if key != "internal_context"
    }


def orthographic_target_field(seed_card: dict[str, Any], family: str) -> str:
    baseline = official_baseline(seed_card)
    if family == "ACCENT_PUNCTUATION":
        preferred = ["name", "address", "city"]
        requested = seed_card.get("requested_families", {})
        if requested.get("name") == "ACCENT_PUNCTUATION":
            preferred = ["address", "city", "name"]
        for field in preferred:
            if has_diacritic(baseline[field]) or punctuation_marks(baseline[field]):
                return field
    return "name"


def variant_contract(seed_card: dict[str, Any]) -> list[dict[str, Any]]:
    composite = seed_card.get("composite_contracts")
    if composite is not None:
        return composite
    requested = seed_card.get("requested_families")
    if not requested:
        return []
    targets = {
        "name": ["name"],
        "address": ["city"] if requested["address"] == "COMMUNE_VARIANT" else ["address"],
        "orthographic": [orthographic_target_field(seed_card, requested["orthographic"])],
    }
    return [
        {
            "variant_id": VARIANT_BY_DIMENSION[dimension],
            "target_dimension": dimension,
            "target_fields": targets[dimension],
            "requested_family": requested[dimension],
        }
        for dimension in ("name", "address", "orthographic")
    ]


def source_supports_family(seed_card: dict[str, Any], dimension: str, family: str) -> bool:
    baseline = official_baseline(seed_card)
    source = baseline[variant_contract(seed_card)[
        {"name": 0, "address": 1, "orthographic": 2}[dimension]
    ]["target_fields"][0]]
    words = normalized_words(source)
    if family == "ENSEIGNE_VS_DENOMINATION":
        names = [str(value) for value in seed_card.get("name_options", []) if str(value).strip()]
        enseignes = [str(value) for value in seed_card.get("enseigne_options", []) if str(value).strip()]
        if not names or not enseignes:
            return False
        baseline = comparison_fingerprint({
            "name": names[0], "address": "", "postcode": "", "city": "", "insee": ""
        })
        alternatives = {
            comparison_fingerprint({
                "name": value, "address": "", "postcode": "", "city": "", "insee": ""
            })
            for value in enseignes
        }
        return bool(alternatives - {baseline})
    if family == "LEGAL_FORM":
        return any(word.upper() in LEGAL_FORM_TOKENS for word in words)
    if family in {"TOKEN_ORDER", "ADDRESS_TOKEN_ORDER"}:
        return len(words) >= 2
    if family == "ACRONYM_TOKENIZATION":
        surface = str(source or "").casefold()
        return bool(
            len(words) >= 2
            and re.search(
                r"(?:\b[a-z]{1,3}[.\-][a-z0-9]+|[a-z0-9]+[.\-][a-z]{1,3}\b)",
                surface,
            )
        )
    if family == "ACCENT_PUNCTUATION":
        return has_diacritic(source) or bool(punctuation_marks(source))
    if family in {"OCR_LIMITED", "ADDRESS_OCR"}:
        evidence_key = (
            "address_ocr_substitution_pairs" if family == "ADDRESS_OCR"
            else "ocr_substitution_pairs"
        )
        allowed_sources = {
            str(value.get("source", "")).casefold()
            for value in seed_card.get(evidence_key, [])
            if isinstance(value, dict) and str(value.get("source", "")).strip()
        }
        return len(normalized_alnum(source)) >= 4 and any(
            character in allowed_sources for character in normalized_alnum(source)
        )
    if family == "ADDRESS_ABBREVIATION":
        street_type = normalized_surface(seed_card.get("street_type", "")).upper()
        aliases = {
            alias
            for alias, canonical in STREET_TYPE_ABBREVIATIONS.items()
            if canonical == street_type and alias != street_type
        }
        return bool(
            aliases
            and street_type in {word.upper() for word in words}
        )
    if family == "COMMUNE_VARIANT":
        return has_diacritic(source) or bool(punctuation_marks(source)) or len(words) >= 2
    return bool(source)


def expanded_street_words(value: Any) -> list[str]:
    return [STREET_TYPE_ABBREVIATIONS.get(word.upper(), word.upper()) for word in normalized_words(value)]


def _composite_mark_relation(source: str, target: str) -> str | None:
    source_punctuation = Counter(character for character in source if character in PUNCTUATION)
    target_punctuation = Counter(character for character in target if character in PUNCTUATION)
    punctuation_added = target_punctuation - source_punctuation
    punctuation_removed = source_punctuation - target_punctuation
    source_marks = sum(
        bool(unicodedata.combining(character))
        for character in unicodedata.normalize("NFD", source)
    )
    target_marks = sum(
        bool(unicodedata.combining(character))
        for character in unicodedata.normalize("NFD", target)
    )
    delta_kinds = sum(bool(value) for value in (punctuation_added, punctuation_removed)) + (source_marks != target_marks)
    if delta_kinds > 1:
        return None
    if punctuation_added:
        return "PUNCTUATION_ADDED"
    if punctuation_removed:
        return "PUNCTUATION_REMOVED"
    if target_marks > source_marks:
        return "DIACRITIC_ADDED"
    if target_marks < source_marks:
        return "DIACRITIC_REMOVED"
    if normalized_words(source) != normalized_words(target):
        return "JOIN_SPLIT"
    return None


def composite_relation_class(field: str, source: str, target: str) -> str | None:
    if normalized_surface(source) == normalized_surface(target):
        return None
    source_words = normalized_words(source)
    target_words = normalized_words(target)
    if (
        field == "address"
        and source_words != target_words
        and expanded_street_words(source) == expanded_street_words(target)
    ):
        return "ADDRESS_ABBREVIATE"
    if Counter(source_words) == Counter(target_words) and source_words != target_words:
        if field == "address":
            street_types = set(STREET_TYPE_ABBREVIATIONS.values())
            source_core = [
                word for word in source_words
                if not word.isdigit()
                and STREET_TYPE_ABBREVIATIONS.get(word.upper(), word.upper()) not in street_types
            ]
            target_core = [
                word for word in target_words
                if not word.isdigit()
                and STREET_TYPE_ABBREVIATIONS.get(word.upper(), word.upper()) not in street_types
            ]
            return "ADDRESS_TYPE_ORDER" if source_core == target_core else None
        return "TOKEN_ORDER" if field == "name" else None
    if normalized_alnum(source) == normalized_alnum(target):
        return _composite_mark_relation(source, target)
    removed = Counter(source_words) - Counter(target_words)
    if target_words and not (Counter(target_words) - Counter(source_words)) and removed:
        if field == "name":
            return (
                "LEGAL_FORM_REMOVE"
                if all(word.upper() in LEGAL_FORM_TOKENS for word in removed)
                else "TOKEN_SUBSET"
            )
        if field == "address":
            positions = composite_stable_positions(source_words, target_words)
            source_digits = [value for value in source_words if value.isdigit()]
            target_digits = [value for value in target_words if value.isdigit()]
            if (
                positions is not None
                and len(target_words) >= 2
                and len(target_words) * 2 >= len(source_words)
                and source_digits == target_digits
            ):
                return "ADDRESS_TOKEN_SUBSET"
    return None


def composite_stable_positions(source_words: list[str], target_words: list[str]) -> list[int] | None:
    positions: list[int] = []
    cursor = 0
    for target in target_words:
        try:
            index = source_words.index(target, cursor)
        except ValueError:
            return None
        positions.append(index)
        cursor = index + 1
    return positions


def composite_permutation(source_words: list[str], target_words: list[str]) -> list[int] | None:
    if Counter(source_words) != Counter(target_words):
        return None
    available: dict[str, list[int]] = {}
    for index, word in enumerate(source_words):
        available.setdefault(word, []).append(index)
    result: list[int] = []
    for word in target_words:
        result.append(available[word].pop(0))
    return result if result != list(range(len(source_words))) else None


def composite_punctuation_edits(source: str, target: str) -> list[dict[str, Any]] | None:
    """Locate removed punctuation by the token boundary it occupied."""
    source_marks: list[tuple[int, str]] = []
    token_index = -1
    in_token = False
    for character in source:
        if character.isalnum():
            if not in_token:
                token_index += 1
                in_token = True
        else:
            if character in PUNCTUATION:
                source_marks.append((token_index, character))
            if character.isspace() or character in PUNCTUATION:
                in_token = False
    target_counter = Counter(character for character in target if character in PUNCTUATION)
    source_counter = Counter(character for character in source if character in PUNCTUATION)
    if target_counter - source_counter:
        return None
    remaining = source_counter - target_counter
    edits: list[dict[str, Any]] = []
    for boundary, mark in source_marks:
        if remaining[mark] > 0:
            edits.append({"after_token_index": boundary, "mark": mark})
            remaining[mark] -= 1
    return edits or None


def composite_diacritic_edits(source: str, target: str) -> list[dict[str, Any]] | None:
    def projection(value: str) -> list[tuple[str, tuple[str, ...]]]:
        result: list[tuple[str, tuple[str, ...]]] = []
        for character in value:
            decomposed = unicodedata.normalize("NFD", character)
            base = "".join(
                item.casefold() for item in decomposed if not unicodedata.combining(item)
            )
            marks = tuple(sorted(
                unicodedata.name(item, "") for item in decomposed if unicodedata.combining(item)
            ))
            if base.isalnum():
                result.append((base, marks))
        return result

    left, right = projection(source), projection(target)
    if len(left) != len(right) or [value[0] for value in left] != [value[0] for value in right]:
        return None
    edits: list[dict[str, Any]] = []
    for index, ((base, source_marks), (_, target_marks)) in enumerate(zip(left, right, strict=True)):
        if set(target_marks) - set(source_marks):
            return None
        removed = sorted(set(source_marks) - set(target_marks))
        if removed:
            edits.append({"alnum_index": index, "base": base, "removed_marks": removed})
    return edits or None


def composite_operation_parameters(
    field: str, relation: str, source: str, target: str
) -> dict[str, Any] | None:
    left, right = normalized_words(source), normalized_words(target)
    if relation in {"TOKEN_SUBSET", "ADDRESS_TOKEN_SUBSET"}:
        positions = composite_stable_positions(left, right)
        if positions is None or len(right) < 2 or len(right) * 2 < len(left):
            return None
        return {"source_token_count": len(left), "retained_positions": positions}
    if relation in {"TOKEN_ORDER", "ADDRESS_TYPE_ORDER"}:
        positions = composite_permutation(left, right)
        return None if positions is None else {
            "source_token_count": len(left), "permutation": positions,
        }
    if relation == "LEGAL_FORM_REMOVE":
        positions = composite_stable_positions(left, right)
        removed = [left[index] for index in range(len(left)) if positions is not None and index not in positions]
        if positions is None or not removed or not all(value.upper() in LEGAL_FORM_TOKENS for value in removed):
            return None
        return {
            "source_token_count": len(left), "retained_positions": positions,
            "removed_legal_forms": removed,
        }
    if relation == "ADDRESS_ABBREVIATE":
        if len(left) != len(right) or expanded_street_words(source) != expanded_street_words(target):
            return None
        pairs = [
            {"source": source_word.upper(), "target": target_word.upper()}
            for source_word, target_word in zip(left, right, strict=True)
            if source_word.upper() != target_word.upper()
        ]
        return None if not pairs else {"pairs": pairs}
    if relation == "PUNCTUATION_REMOVED":
        edits = composite_punctuation_edits(source, target)
        return None if edits is None else {"edits": edits}
    if relation == "DIACRITIC_REMOVED":
        edits = composite_diacritic_edits(source, target)
        return None if edits is None else {"edits": edits}
    if relation == "JOIN_SPLIT":
        if normalized_alnum(source) != normalized_alnum(target) or len(right) >= len(left):
            return None
        groups: list[list[int]] = []
        cursor = 0
        for target_word in right:
            joined = ""
            group: list[int] = []
            while cursor < len(left) and len(joined) < len(target_word):
                joined += left[cursor]
                group.append(cursor)
                cursor += 1
            if joined != target_word:
                return None
            groups.append(group)
        if cursor != len(left) or all(len(group) == 1 for group in groups):
            return None
        return {"source_token_count": len(left), "groups": groups}
    return None


def family_change_errors(
    family: str,
    source: str,
    target: str,
    seed_card: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    source_surface, target_surface = normalized_surface(source), normalized_surface(target)
    source_words, target_words = normalized_words(source), normalized_words(target)
    source_alnum, target_alnum = normalized_alnum(source), normalized_alnum(target)
    if source_surface == target_surface:
        return ["TARGET_FIELD_UNCHANGED"]
    if family == "ENSEIGNE_VS_DENOMINATION":
        alternatives = {
            normalized_surface(value)
            for value in seed_card.get("enseigne_options", [])
            if str(value).strip()
        }
        if target_surface not in alternatives or target_surface == source_surface:
            errors.append("NOT_AN_OFFICIAL_ALTERNATE_NAME")
    elif family == "LEGAL_FORM":
        source_core = [word for word in source_words if word.upper() not in LEGAL_FORM_TOKENS]
        target_core = [word for word in target_words if word.upper() not in LEGAL_FORM_TOKENS]
        source_forms = [word for word in source_words if word.upper() in LEGAL_FORM_TOKENS]
        target_forms = [word for word in target_words if word.upper() in LEGAL_FORM_TOKENS]
        if source_core != target_core or source_forms == target_forms:
            errors.append("LEGAL_FORM_NOT_ACTUALLY_CHANGED")
    elif family in {"TOKEN_ORDER", "ADDRESS_TOKEN_ORDER"}:
        if Counter(source_words) != Counter(target_words) or source_words == target_words:
            errors.append("TOKEN_ORDER_NOT_ACTUALLY_CHANGED")
    elif family == "ACRONYM_TOKENIZATION":
        if source_alnum != target_alnum or source_words == target_words:
            errors.append("ACRONYM_TOKENIZATION_NOT_ACTUALLY_CHANGED")
    elif family in {"OCR_LIMITED", "ADDRESS_OCR"}:
        distance = edit_distance(source_alnum, target_alnum)
        maximum = max(2, int(len(source_alnum) * 0.15))
        evidence_key = (
            "address_ocr_substitution_pairs" if family == "ADDRESS_OCR"
            else "ocr_substitution_pairs"
        )
        allowed_pairs = {
            (str(value.get("source", "")).casefold(), str(value.get("target", "")).casefold())
            for value in seed_card.get(evidence_key, [])
            if isinstance(value, dict)
        }
        observed_pairs = {
            (left, right)
            for left, right in zip(source_alnum, target_alnum)
            if left != right
        }
        if (
            len(source_alnum) != len(target_alnum)
            or distance < 1
            or distance > maximum
            or not observed_pairs
            or not observed_pairs.issubset(allowed_pairs)
        ):
            errors.append("OCR_SUBSTITUTION_NOT_LIMITED_OR_ABSENT")
        if family == "ADDRESS_OCR":
            source_digits = "".join(character for character in str(source) if character.isdigit())
            target_digits = "".join(character for character in str(target) if character.isdigit())
            if source_digits != target_digits:
                errors.append("ADDRESS_OCR_CHANGED_NUMBER")
    elif family == "ACCENT_PUNCTUATION":
        if source_alnum != target_alnum:
            errors.append("ACCENT_PUNCTUATION_CHANGED_ALPHANUMERICS")
        if target_surface == source_surface or normalized_surface(source).upper() == normalized_surface(target).upper():
            errors.append("ACCENT_PUNCTUATION_CASE_ONLY")
        if not has_diacritic(source) and not punctuation_marks(source):
            errors.append("SOURCE_HAS_NO_ACCENT_OR_PUNCTUATION")
        source_punctuation = Counter(
            character for character in str(source or "") if character in PUNCTUATION
        )
        target_punctuation = Counter(
            character for character in str(target or "") if character in PUNCTUATION
        )
        source_diacritics = diacritic_profile(source)
        target_diacritics = diacritic_profile(target)
        added_diacritic = (
            len(source_diacritics) != len(target_diacritics)
            or any(
                not target_marks.issubset(source_marks)
                for source_marks, target_marks in zip(source_diacritics, target_diacritics)
            )
        )
        if any(
            target_punctuation[mark] > source_punctuation[mark]
            for mark in target_punctuation
        ) or added_diacritic:
            errors.append("ACCENT_PUNCTUATION_ADDED_GRATUITOUS_MARK")
    elif family == "ADDRESS_ABBREVIATION":
        if expanded_street_words(source) != expanded_street_words(target):
            errors.append("ADDRESS_ABBREVIATION_CHANGED_CONTENT")
        abbreviations = set(STREET_TYPE_ABBREVIATIONS)
        if not ({word.upper() for word in target_words} & abbreviations):
            errors.append("ADDRESS_TYPE_NOT_ABBREVIATED")
    elif family == "COMMUNE_VARIANT":
        if source_alnum != target_alnum:
            errors.append("COMMUNE_VARIANT_CHANGED_ALPHANUMERICS")
    return errors


def leaked_identifier(crm: dict[str, str], siret: str, siren: str) -> bool:
    joined = " ".join(crm.values())
    if siret in joined or siren in joined:
        return True
    return bool(re.search(r"(?<![0-9])(?:[0-9]{14}|[0-9]{9})(?![0-9])", joined))


def added_marks(source: str, target: str) -> bool:
    source_punctuation = Counter(character for character in source if character in PUNCTUATION)
    target_punctuation = Counter(character for character in target if character in PUNCTUATION)
    if target_punctuation - source_punctuation:
        return True
    source_combining = Counter(
        character
        for character in unicodedata.normalize("NFD", source)
        if unicodedata.combining(character)
    )
    target_combining = Counter(
        character
        for character in unicodedata.normalize("NFD", target)
        if unicodedata.combining(character)
    )
    return bool(target_combining - source_combining)


def composite_change_errors(
    baseline: dict[str, str],
    crm: dict[str, str],
    contract: dict[str, Any],
    seed_card: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    target_fields = set(contract["target_fields"])
    field_relations = contract["field_relations"]
    if "name" not in target_fields or not ({"address", "city"} & target_fields):
        errors.append("COMPOSITE_REQUIRES_NAME_AND_LOCATION")
    for field in CRM_FIELDS:
        source = baseline[field]
        target = crm[field].strip()
        if field not in target_fields:
            if target != source:
                errors.append(f"{field.upper()}_CHANGED_OUTSIDE_CONTRACT")
            continue
        if normalized_surface(source) == normalized_surface(target):
            errors.append(f"{field.upper()}_TARGET_UNCHANGED_OR_CASE_ONLY")
            continue
        if not target:
            errors.append(f"{field.upper()}_MISSING_FORBIDDEN")
            continue
        actual_relation = composite_relation_class(field, source, target)
        expected_relation = field_relations[field]
        if actual_relation != expected_relation:
            errors.append(
                f"{field.upper()}_RELATION_MISMATCH:{expected_relation}:{actual_relation or 'NONE'}"
            )
        if added_marks(source, target):
            errors.append(f"{field.upper()}_ADDED_MARK_FORBIDDEN")
        field_inspirations = contract.get("field_inspirations", {})
        fragment = field_inspirations.get(field) if isinstance(field_inspirations, dict) else None
        if fragment is not None:
            actual_parameters = composite_operation_parameters(
                field, expected_relation, source, target
            )
            if actual_parameters != fragment.get("operation_parameters"):
                errors.append(f"{field.upper()}_OPERATOR_PARAMETERS_MISMATCH")
            protected = [normalized_surface(value) for value in contract.get(
                "protected_target_tokens", {}
            ).get(field, [])]
            if protected and not set(protected).issubset(set(normalized_words(target))):
                errors.append(f"{field.upper()}_DISTINCTIVE_ANCHOR_REMOVED")
        if field == "name":
            if Counter(normalized_alnum(target)) - Counter(normalized_alnum(source)):
                errors.append("NAME_NEW_ALPHANUMERIC_MATERIAL")
            if expected_relation == "TOKEN_SUBSET" and (
                len(normalized_words(target)) < 2
                or len(normalized_words(target)) * 2 < len(normalized_words(source))
            ):
                errors.append("NAME_TOKEN_SUBSET_TOO_DESTRUCTIVE")
        elif field == "address":
            expanded_source = "".join(expanded_street_words(source)).casefold()
            expanded_target = "".join(expanded_street_words(target)).casefold()
            if Counter(expanded_target) - Counter(expanded_source):
                errors.append("ADDRESS_NEW_ALPHANUMERIC_MATERIAL")
            source_digits = re.findall(r"[0-9]+", source)
            target_digits = re.findall(r"[0-9]+", target)
            if source_digits != target_digits:
                errors.append("ADDRESS_NUMBER_CHANGED")
            house_number = normalized_alnum(seed_card.get("street_number", ""))
            if house_number and house_number not in normalized_words(target):
                errors.append("HOUSE_NUMBER_NOT_PRESERVED")
        elif field == "city":
            if normalized_alnum(source) != normalized_alnum(target):
                errors.append("CITY_ALPHANUMERICS_CHANGED")
    return errors


def candidate_address(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "").strip()
        for key in ("number", "repetition_index", "street_type", "street")
        if str(candidate.get(key) or "").strip()
    )


def known_context_ambiguities(crm: dict[str, str], seed_card: dict[str, Any]) -> list[str]:
    """Fail on a known official competitor compatible with all retained anchors.

    The context builder guarantees exhaustive siblings and exact address/name
    collisions at the official baseline.  This check applies the composite
    subset invariances to those full internal rows.  The critic separately
    judges semantic ambiguity; this function never uses retrieval/model scores.
    """
    crm_name = Counter(normalized_words(crm["name"]))
    crm_address = Counter(expanded_street_words(crm["address"]))
    result: list[str] = []
    for candidate in seed_card.get("internal_context", []):
        if crm["postcode"] and normalized_alnum(crm["postcode"]) != normalized_alnum(candidate.get("postcode")):
            continue
        if crm["insee"] and normalized_alnum(crm["insee"]) != normalized_alnum(candidate.get("insee")):
            continue
        if crm["city"] and normalized_alnum(crm["city"]) != normalized_alnum(candidate.get("city")):
            continue
        candidate_address_words = Counter(expanded_street_words(candidate_address(candidate)))
        if crm_address - candidate_address_words:
            continue
        names = list(dict.fromkeys(
            str(value).strip()
            for value in (
                list(candidate.get("name_values", []))
                + [candidate.get(key) for key in (
                    "usual_name", "enseigne1", "enseigne2", "enseigne3", "legal_name"
                )]
            )
            if str(value or "").strip()
        ))
        if not names or not any(not (crm_name - Counter(normalized_words(name))) for name in names):
            continue
        ref = str(candidate.get("record_sha256") or "UNKNOWN")[:16]
        result.append(ref)
    return sorted(set(result))


def generator_preflight(response: dict[str, Any], seed: sqlite3.Row) -> dict[str, Any]:
    errors: list[str] = []
    fingerprints: list[str] = []
    seed_card = json.loads(seed["seed_card_json"])
    baseline = official_baseline(seed_card)
    contracts = {
        value["variant_id"]: value for value in variant_contract(seed_card)
    }
    expected_ids = {"v1", "v2", "v3"}
    observed_ids = {variant["variant_id"] for variant in response["variants"]}
    if observed_ids != expected_ids:
        errors.append("VARIANT_IDS_NOT_EXACT_V1_V2_V3")
    for variant in response["variants"]:
        crm = variant["crm"]
        contract = contracts.get(variant["variant_id"])
        if not any(crm[field].strip() for field in ("name", "address")):
            errors.append(f"{variant['variant_id']}:NO_NAME_OR_ADDRESS")
        if leaked_identifier(crm, seed["target_siret"], seed["target_siren"]):
            errors.append(f"{variant['variant_id']}:IDENTIFIER_LEAK")
        comparison = comparison_fingerprint(crm)
        if not comparison:
            errors.append(f"{variant['variant_id']}:EMPTY_FINGERPRINT")
        fingerprints.append(surface_fingerprint(crm))
        if contract:
            observed = variant["corruption_families_observed"]
            if observed != [contract["requested_family"]]:
                errors.append(f"{variant['variant_id']}:DECLARED_FAMILY_MISMATCH")
            if contract["requested_family"] == COMPOSITE_FAMILY:
                for reason in composite_change_errors(baseline, crm, contract, seed_card):
                    errors.append(f"{variant['variant_id']}:{reason}")
                for competitor_ref in known_context_ambiguities(crm, seed_card):
                    errors.append(f"{variant['variant_id']}:KNOWN_CONTEXT_AMBIGUOUS:{competitor_ref}")
            else:
                target_fields = set(contract["target_fields"])
                for field in CRM_FIELDS:
                    if field not in target_fields and crm[field].strip() != baseline[field]:
                        errors.append(f"{variant['variant_id']}:{field.upper()}_CHANGED_OUTSIDE_CONTRACT")
                for field in target_fields:
                    for reason in family_change_errors(
                        contract["requested_family"], baseline[field], crm[field], seed_card
                    ):
                        errors.append(f"{variant['variant_id']}:{reason}")
    if len(set(fingerprints)) != len(fingerprints):
        errors.append("DUPLICATE_OR_COSMETIC_VARIANTS")
    return {"passed": not errors, "errors": sorted(set(errors)), "checked_fields": list(CRM_FIELDS)}


def should_critic(seed: sqlite3.Row, policy: dict[str, Any]) -> bool:
    if policy["critic_mode"] == "all":
        return True
    flags = set(json.loads(seed["risk_flags_json"]))
    if flags or seed["source_kind"] == "MAPS_ASSISTED":
        return True
    bucket = int.from_bytes(hashlib.sha256(seed["seed_id"].encode()).digest()[:8], "big")
    return bucket % int(policy["critic_sample_modulus"]) < int(policy["critic_sample_slots"])


def verify_response_envelope(response: dict[str, Any], task: sqlite3.Row, role: str) -> None:
    expected = {
        "task_id": task["task_id"],
        "run_id": task["run_id"],
        "batch_id": task["batch_id"],
        "role": role,
        "prompt_version": task["prompt_version"],
        "input_sha256": task["input_sha256"],
    }
    for key, value in expected.items():
        if response.get(key) != value:
            raise ValueError(f"response envelope mismatch for {key}")


def store_generator(
    connection: sqlite3.Connection,
    task: sqlite3.Row,
    seed: sqlite3.Row,
    raw: str,
    response: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    response_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    connection.execute(
        "DELETE FROM variants WHERE run_id=? AND seed_id=?",
        (task["run_id"], seed["seed_id"]),
    )
    preflight = generator_preflight(response, seed)
    fingerprints = {
        variant["variant_id"]: comparison_fingerprint(variant["crm"])
        for variant in response["variants"]
    }
    existing_fingerprints = {
        row[0]
        for row in connection.execute(
            "SELECT crm_fingerprint FROM variants WHERE run_id=?",
            (task["run_id"],),
        )
    }
    global_duplicates = sorted(
        variant_id for variant_id, fingerprint in fingerprints.items()
        if fingerprint in existing_fingerprints
    )
    if global_duplicates:
        preflight["errors"] = sorted(set(preflight["errors"] + [
            f"{variant_id}:GLOBAL_DUPLICATE" for variant_id in global_duplicates
        ]))
        preflight["passed"] = False
    for variant in response["variants"]:
        connection.execute(
            """INSERT INTO variants
               (run_id, seed_id, variant_id, crm_json, crm_fingerprint, families_json,
                transformation_summary, generator_response_sha256, preflight_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task["run_id"], seed["seed_id"], variant["variant_id"],
                canonical_json(variant["crm"]),
                fingerprints[variant["variant_id"]],
                canonical_json(variant["corruption_families_observed"]),
                variant["transformation_summary"], response_sha, canonical_json(preflight),
            ),
        )
    if not preflight["passed"]:
        next_status = (
            "PENDING_GENERATOR"
            if int(seed["attempt"]) + 1 < int(policy["max_generator_attempts"])
            else "READY_SUPERVISOR"
        )
        connection.execute(
            """UPDATE variants SET final_decision='REJECT', final_reason=?
               WHERE run_id=? AND seed_id=?""",
            ("PREFLIGHT:" + ",".join(preflight["errors"]), task["run_id"], seed["seed_id"]),
        )
    elif should_critic(seed, policy):
        next_status = "PENDING_CRITIC"
    elif policy["allow_easy_supervisor"]:
        next_status = "READY_SUPERVISOR"
    else:
        next_status = "PENDING_CRITIC"
    connection.execute(
        """UPDATE seeds SET status=?, attempt=attempt+1, lease_role=NULL,
           lease_worker=NULL, lease_expires_at=NULL, updated_at=?
           WHERE run_id=? AND seed_id=?""",
        (next_status, now_ts(), task["run_id"], seed["seed_id"]),
    )
    event(connection, task["run_id"], seed["seed_id"], "GENERATOR_SUBMITTED", {
        "response_sha256": response_sha, "preflight": preflight, "next_status": next_status,
    })


def decision_map(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = response["decisions"]
    result = {value["variant_id"]: value for value in values}
    if set(result) != {"v1", "v2", "v3"} or len(values) != 3:
        raise ValueError("decisions must cover v1, v2 and v3 exactly once")
    return result


def store_critic(
    connection: sqlite3.Connection,
    task: sqlite3.Row,
    seed: sqlite3.Row,
    response: dict[str, Any],
) -> None:
    decisions = decision_map(response)
    for variant_id, value in decisions.items():
        connection.execute(
            """UPDATE variants SET critic_json=?, critic_decision=?
               WHERE run_id=? AND seed_id=? AND variant_id=?""",
            (canonical_json(value), value["decision"], task["run_id"], seed["seed_id"], variant_id),
        )
    next_status = (
        "READY_SUPERVISOR"
        if all(value["decision"] == "ACCEPT" for value in decisions.values())
        else "PENDING_ADJUDICATOR"
    )
    connection.execute(
        """UPDATE seeds SET status=?, lease_role=NULL, lease_worker=NULL,
           lease_expires_at=NULL, updated_at=? WHERE run_id=? AND seed_id=?""",
        (next_status, now_ts(), task["run_id"], seed["seed_id"]),
    )
    event(connection, task["run_id"], seed["seed_id"], "CRITIC_SUBMITTED", {
        "decisions": {key: value["decision"] for key, value in decisions.items()},
        "next_status": next_status,
    })


def store_adjudicator(
    connection: sqlite3.Connection,
    task: sqlite3.Row,
    seed: sqlite3.Row,
    response: dict[str, Any],
) -> None:
    decisions = decision_map(response)
    for variant_id, value in decisions.items():
        connection.execute(
            """UPDATE variants SET adjudicator_json=?, adjudicator_decision=?
               WHERE run_id=? AND seed_id=? AND variant_id=?""",
            (canonical_json(value), value["decision"], task["run_id"], seed["seed_id"], variant_id),
        )
    connection.execute(
        """UPDATE seeds SET status='READY_SUPERVISOR', lease_role=NULL,
           lease_worker=NULL, lease_expires_at=NULL, updated_at=?
           WHERE run_id=? AND seed_id=?""",
        (now_ts(), task["run_id"], seed["seed_id"]),
    )
    event(connection, task["run_id"], seed["seed_id"], "ADJUDICATOR_SUBMITTED", {
        "decisions": {key: value["decision"] for key, value in decisions.items()},
    })


def command_submit(args: argparse.Namespace) -> None:
    validator = response_validator(args.schema)
    accepted = 0
    with connect(args.db) as connection:
        create_schema(connection)
        for raw, response in iter_response_raw(args.input, args.input_format):
            errors = sorted(validator.iter_errors(response), key=lambda value: list(value.path))
            if errors:
                path = ".".join(str(value) for value in errors[0].path)
                raise ValueError(f"response schema error at {path}: {errors[0].message}")
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?", (response["task_id"],)
            ).fetchone()
            if task is None:
                raise ValueError(f"unknown task: {response['task_id']}")
            if task["status"] != "LEASED" or task["worker_id"] != args.worker_id:
                raise ValueError(f"task is not leased by {args.worker_id}: {task['task_id']}")
            if task["role"] != args.role:
                raise ValueError("task role mismatch")
            seed = connection.execute(
                "SELECT * FROM seeds WHERE run_id=? AND seed_id=?",
                (task["run_id"], task["seed_id"]),
            ).fetchone()
            if seed["status"] != LEASED_BY_ROLE[args.role]:
                raise ValueError(f"seed is not leased for {args.role}")
            if seed["lease_worker"] != args.worker_id or seed["lease_role"] != args.role:
                raise ValueError("stale or foreign seed lease")
            verify_response_envelope(response, task, args.role)
            expected_seed = {"siret": seed["target_siret"], "siren": seed["target_siren"]}
            if response["seed"] != expected_seed:
                raise ValueError("response seed differs from leased seed")
            run = connection.execute(
                "SELECT policy_json FROM runs WHERE run_id=?", (task["run_id"],)
            ).fetchone()
            policy = json.loads(run["policy_json"])
            with connection:
                if args.role == "GENERATOR":
                    store_generator(connection, task, seed, raw, response, policy)
                elif args.role == "CRITIC":
                    store_critic(connection, task, seed, response)
                else:
                    store_adjudicator(connection, task, seed, response)
                response_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                connection.execute(
                    """UPDATE tasks SET status='COMPLETED', raw_response=?,
                       response_sha256=?, completed_at=? WHERE task_id=?""",
                    (raw, response_sha, now_ts(), task["task_id"]),
                )
            accepted += 1
    print(canonical_json({"role": args.role, "submitted": accepted, "worker_id": args.worker_id}))


def command_abandon(args: argparse.Namespace) -> None:
    with connect(args.db) as connection:
        create_schema(connection)
        task = connection.execute("SELECT * FROM tasks WHERE task_id=?", (args.task_id,)).fetchone()
        if task is None:
            raise ValueError(f"unknown task: {args.task_id}")
        if task["status"] != "LEASED" or task["role"] != args.role or task["worker_id"] != args.worker_id:
            raise ValueError("task is not leased by the requested worker and role")
        seed = connection.execute(
            "SELECT * FROM seeds WHERE run_id=? AND seed_id=?", (task["run_id"], task["seed_id"])
        ).fetchone()
        with connection:
            connection.execute(
                "UPDATE tasks SET status='ABANDONED', completed_at=? WHERE task_id=?",
                (now_ts(), args.task_id),
            )
            connection.execute(
                """UPDATE seeds SET status=?, lease_role=NULL,
                   lease_worker=NULL, lease_expires_at=NULL, updated_at=?
                   WHERE run_id=? AND seed_id=?""",
                (PENDING_BY_ROLE[args.role], now_ts(), task["run_id"], seed["seed_id"]),
            )
            event(connection, task["run_id"], seed["seed_id"], "TASK_ABANDONED_REQUEUED", {
                "task_id": args.task_id, "role": args.role, "reason": args.reason,
                "next_status": PENDING_BY_ROLE[args.role],
            })
    print(canonical_json({
        "task_id": args.task_id,
        "task_status": "ABANDONED",
        "seed_status": PENDING_BY_ROLE[args.role],
        "reason": args.reason,
    }))


def command_supervise(args: argparse.Namespace) -> None:
    completed = 0
    decisions_counter: Counter[str] = Counter()
    with connect(args.db) as connection:
        create_schema(connection)
        seeds = connection.execute(
            """SELECT * FROM seeds WHERE run_id=? AND status='READY_SUPERVISOR'
               ORDER BY seed_id LIMIT ?""",
            (args.run_id, args.limit),
        ).fetchall()
        with connection:
            for seed in seeds:
                variants = connection.execute(
                    """SELECT * FROM variants WHERE run_id=? AND seed_id=? ORDER BY variant_id""",
                    (args.run_id, seed["seed_id"]),
                ).fetchall()
                if len(variants) != 3:
                    raise RuntimeError(
                        f"supervisor requires exactly three variants for seed {seed['seed_id']}; "
                        f"found {len(variants)}"
                    )
                for variant in variants:
                    preflight = json.loads(variant["preflight_json"])
                    if not preflight["passed"]:
                        decision, reason = "REJECT", "DETERMINISTIC_PREFLIGHT_FAILED"
                    elif variant["critic_decision"] == "REJECT":
                        decision, reason = "REJECT", "CRITIC_REJECT"
                    elif variant["critic_decision"] == "SILVER" and variant["adjudicator_decision"] != "ACCEPT":
                        decision, reason = "SILVER", "UNRESOLVED_CRITIC_SILVER"
                    elif variant["adjudicator_decision"] in {"SILVER", "REJECT"}:
                        decision, reason = variant["adjudicator_decision"], "ADJUDICATOR_DOWNGRADE"
                    elif variant["critic_decision"] == "ACCEPT" or variant["adjudicator_decision"] == "ACCEPT":
                        decision, reason = "ACCEPT", "AGENTIC_REVIEW_PASS"
                    else:
                        decision, reason = "ACCEPT", "PREFLIGHT_PASS_EASY_POLICY"
                    connection.execute(
                        """UPDATE variants SET final_decision=?, final_reason=?
                           WHERE run_id=? AND seed_id=? AND variant_id=?""",
                        (decision, reason, args.run_id, seed["seed_id"], variant["variant_id"]),
                    )
                    decisions_counter[decision] += 1
                connection.execute(
                    """UPDATE seeds SET status='COMPLETED', updated_at=?
                       WHERE run_id=? AND seed_id=?""",
                    (now_ts(), args.run_id, seed["seed_id"]),
                )
                event(connection, args.run_id, seed["seed_id"], "SUPERVISED", {
                    "variant_decisions": {
                        row["variant_id"]: connection.execute(
                            """SELECT final_decision FROM variants
                               WHERE run_id=? AND seed_id=? AND variant_id=?""",
                            (args.run_id, seed["seed_id"], row["variant_id"]),
                        ).fetchone()[0]
                        for row in variants
                    }
                })
                completed += 1
    print(canonical_json({"supervised_seeds": completed, "variant_decisions": dict(decisions_counter)}))


def status_payload(connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    seed_counts = {
        row["status"]: int(row["count"])
        for row in connection.execute(
            "SELECT status, COUNT(*) AS count FROM seeds WHERE run_id=? GROUP BY status", (run_id,)
        )
    }
    variant_counts = {
        str(row["final_decision"] or "UNDECIDED"): int(row["count"])
        for row in connection.execute(
            """SELECT final_decision, COUNT(*) AS count FROM variants
               WHERE run_id=? GROUP BY final_decision""",
            (run_id,),
        )
    }
    task_counts = {
        f"{row['role']}:{row['status']}": int(row["count"])
        for row in connection.execute(
            """SELECT role, status, COUNT(*) AS count FROM tasks
               WHERE run_id=? GROUP BY role, status""",
            (run_id,),
        )
    }
    return {"run_id": run_id, "seeds": seed_counts, "variants": variant_counts, "tasks": task_counts}


def command_status(args: argparse.Namespace) -> None:
    with connect(args.db) as connection:
        create_schema(connection)
        print(json.dumps(status_payload(connection, args.run_id), ensure_ascii=False, sort_keys=True, indent=2))


def export_row(seed: sqlite3.Row, variant: sqlite3.Row) -> dict[str, Any]:
    return {
        "seed_id": seed["seed_id"],
        "source_kind": seed["source_kind"],
        "target_siret": seed["target_siret"],
        "target_siren": seed["target_siren"],
        "oof_fold": int(seed["oof_fold"]),
        "variant_id": variant["variant_id"],
        "crm": json.loads(variant["crm_json"]),
        "corruption_families_observed": json.loads(variant["families_json"]),
        "transformation_summary": variant["transformation_summary"],
        "generator_response_sha256": variant["generator_response_sha256"],
        "critic_decision": variant["critic_decision"],
        "adjudicator_decision": variant["adjudicator_decision"],
        "final_decision": variant["final_decision"],
        "final_reason": variant["final_reason"],
    }


def command_export(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    with connect(args.db) as connection:
        create_schema(connection)
        rows = connection.execute(
            """SELECT s.*, v.variant_id, v.crm_json, v.families_json,
                      v.transformation_summary, v.generator_response_sha256,
                      v.preflight_json, v.critic_decision, v.adjudicator_decision,
                      v.final_decision, v.final_reason
               FROM seeds s JOIN variants v USING(run_id, seed_id)
               WHERE s.run_id=? AND s.status='COMPLETED'
               ORDER BY s.seed_id, v.variant_id""",
            (args.run_id,),
        ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {value: [] for value in FINAL_DECISIONS}
        for row in rows:
            grouped[row["final_decision"]].append(export_row(row, row))
        for decision, values in grouped.items():
            write_jsonl_atomic(args.output / f"{decision.lower()}.jsonl", values)
        manifest = {
            "schema_version": "sireto-synthetic-gt-agentic-export-1",
            "run_id": args.run_id,
            "counts": {key: len(value) for key, value in grouped.items()},
            "files": {
                f"{key.lower()}.jsonl": hashlib.sha256(
                    (args.output / f"{key.lower()}.jsonl").read_bytes()
                ).hexdigest()
                for key in grouped
            },
            "status": status_payload(connection, args.run_id),
        }
        manifest_path = args.output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(canonical_json({"output": str(args.output), "counts": manifest["counts"]}))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--db", type=Path, required=True)
    sub = root.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--run-id", required=True)
    init.add_argument("--seeds", type=Path, required=True)
    init.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    init.add_argument("--critic-mode", choices=["all", "risk"], default="all")
    init.add_argument("--critic-sample-modulus", type=int, default=10)
    init.add_argument("--critic-sample-slots", type=int, default=1)
    init.add_argument("--allow-easy-supervisor", action="store_true")
    init.add_argument("--max-generator-attempts", type=int, default=2)
    init.set_defaults(func=command_init)

    lease = sub.add_parser("lease")
    lease.add_argument("--run-id", required=True)
    lease.add_argument("--role", choices=sorted(PENDING_BY_ROLE), required=True)
    lease.add_argument("--worker-id", required=True)
    lease.add_argument("--limit", type=int, default=8)
    lease.add_argument("--ttl-seconds", type=int, default=1800)
    lease.add_argument("--output", type=Path)
    lease.set_defaults(func=command_lease)

    submit = sub.add_parser("submit")
    submit.add_argument("--role", choices=sorted(PENDING_BY_ROLE), required=True)
    submit.add_argument("--worker-id", required=True)
    submit.add_argument("--input", type=Path, required=True)
    submit.add_argument("--input-format", choices=["jsonl", "json"], default="jsonl")
    submit.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    submit.set_defaults(func=command_submit)

    abandon = sub.add_parser("abandon")
    abandon.add_argument("--task-id", required=True)
    abandon.add_argument("--role", choices=sorted(PENDING_BY_ROLE), required=True)
    abandon.add_argument("--worker-id", required=True)
    abandon.add_argument("--reason", required=True)
    abandon.set_defaults(func=command_abandon)

    supervise = sub.add_parser("supervise")
    supervise.add_argument("--run-id", required=True)
    supervise.add_argument("--limit", type=int, default=1000)
    supervise.set_defaults(func=command_supervise)

    status = sub.add_parser("status")
    status.add_argument("--run-id", required=True)
    status.set_defaults(func=command_status)

    export = sub.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=command_export)
    return root


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
