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
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "config" / "synthetic_gt_agentic_message_schema_v2.json"
SCHEMA_VERSION = "sireto-synthetic-gt-agentic-message-2"
PROMPT_VERSIONS = {
    "GENERATOR": "sireto-gt-generator-v2",
    "CRITIC": "sireto-gt-critic-v2",
    "ADJUDICATOR": "sireto-gt-adjudicator-v2",
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
            "source_kind": "MAPS_ASSISTED" if source == "MAPS_ASSISTED" else "SIRENE_SYNTHETIC",
            "oof_fold": identity.get("oof_fold"),
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
    if fold not in {2, 3, 4}:
        raise ValueError(f"forbidden fold for seed {seed['seed_id']}: {fold}")
    if seed["legacy_split"] not in {"train", "train_synthetic"}:
        raise ValueError(f"forbidden split for seed {seed['seed_id']}")
    if seed["source_kind"] not in {"SIRENE_SYNTHETIC", "MAPS_ASSISTED"}:
        raise ValueError(f"unsupported source kind for seed {seed['seed_id']}")
    if not isinstance(seed["seed_card"], dict) or not isinstance(seed["observed_train_profile"], dict):
        raise ValueError("seed_card and observed_train_profile must be objects")
    if not isinstance(seed["risk_flags"], list) or not all(
        isinstance(value, str) for value in seed["risk_flags"]
    ):
        raise ValueError("risk_flags must be a list of strings")
    return seed


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
    variants = connection.execute(
        """SELECT variant_id, crm_json FROM variants
           WHERE run_id=? AND seed_id=? ORDER BY variant_id""",
        (run_id, seed["seed_id"]),
    ).fetchall()
    return {
        "seed": {"siret": seed["target_siret"], "siren": seed["target_siren"]},
        "seed_card": json.loads(seed["seed_card_json"]),
        "variants": [
            {"variant_id": row["variant_id"], "crm": json.loads(row["crm_json"])} for row in variants
        ],
    }


def adjudicator_input(connection: sqlite3.Connection, run_id: str, seed: sqlite3.Row) -> dict[str, Any]:
    variants = connection.execute(
        """SELECT variant_id, crm_json, preflight_json, critic_json, critic_decision
           FROM variants WHERE run_id=? AND seed_id=? ORDER BY variant_id""",
        (run_id, seed["seed_id"]),
    ).fetchall()
    return {
        "seed": {"siret": seed["target_siret"], "siren": seed["target_siren"]},
        "seed_card": json.loads(seed["seed_card_json"]),
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
        role_input = {
            "seed": {"siret": seed["target_siret"], "siren": seed["target_siren"]},
            "source_kind": seed["source_kind"],
            "seed_card": json.loads(seed["seed_card_json"]),
            "observed_train_profile": json.loads(seed["profile_json"]),
            "risk_flags": json.loads(seed["risk_flags_json"]),
        }
    elif role == "CRITIC":
        role_input = critic_input(connection, run_id, seed)
    else:
        role_input = adjudicator_input(connection, run_id, seed)
    input_sha256 = digest_json(role_input)
    attempt = int(seed["attempt"]) + (1 if role == "GENERATOR" else 0)
    task_id = hashlib.sha256(
        f"{run_id}|{role}|{seed['seed_id']}|{attempt}|{batch_id}".encode()
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
                   lease_expires_at=?, attempt=attempt+?, updated_at=?
                   WHERE run_id=? AND seed_id=?""",
                (
                    LEASED_BY_ROLE[role], role, args.worker_id, expiry,
                    1 if role == "GENERATOR" else 0, now_ts(), args.run_id, seed["seed_id"],
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


def leaked_identifier(crm: dict[str, str], siret: str, siren: str) -> bool:
    joined = " ".join(crm.values())
    if siret in joined or siren in joined:
        return True
    return bool(re.search(r"(?<![0-9])(?:[0-9]{14}|[0-9]{9})(?![0-9])", joined))


def generator_preflight(response: dict[str, Any], seed: sqlite3.Row) -> dict[str, Any]:
    errors: list[str] = []
    fingerprints: list[str] = []
    expected_ids = {"v1", "v2", "v3"}
    observed_ids = {variant["variant_id"] for variant in response["variants"]}
    if observed_ids != expected_ids:
        errors.append("VARIANT_IDS_NOT_EXACT_V1_V2_V3")
    for variant in response["variants"]:
        crm = variant["crm"]
        if not any(crm[field].strip() for field in ("name", "address")):
            errors.append(f"{variant['variant_id']}:NO_NAME_OR_ADDRESS")
        if leaked_identifier(crm, seed["target_siret"], seed["target_siren"]):
            errors.append(f"{variant['variant_id']}:IDENTIFIER_LEAK")
        fingerprint = comparison_fingerprint(crm)
        if not fingerprint:
            errors.append(f"{variant['variant_id']}:EMPTY_FINGERPRINT")
        fingerprints.append(fingerprint)
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
            if int(seed["attempt"]) < int(policy["max_generator_attempts"])
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
        """UPDATE seeds SET status=?, lease_role=NULL, lease_worker=NULL,
           lease_expires_at=NULL, updated_at=? WHERE run_id=? AND seed_id=?""",
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
        for raw, response in iter_jsonl_raw(args.input):
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
    submit.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    submit.set_defaults(func=command_submit)

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
