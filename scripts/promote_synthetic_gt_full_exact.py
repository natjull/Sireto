#!/usr/bin/env python3
"""Promote Luna GT variants proven exact by full SIRENE.

Legacy runs remain atomic 3/3.  Balanced production opts into independent
per-variant promotion so one exhausted sibling cannot discard two valid pairs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


VARIANT_IDS = {"v1", "v2", "v3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [value for _raw, value in loop.iter_jsonl_raw(path)]


def raw_generator_variant(raw_response: str, variant_id: str) -> dict[str, Any]:
    envelope = json.loads(raw_response)
    variants = [
        value for value in envelope.get("variants", [])
        if value.get("variant_id") == variant_id
    ]
    if len(variants) != 1:
        raise ValueError(f"raw generator response lacks unique {variant_id}")
    return variants[0]


def atomic_rename_exclusive(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renameatx_np = libc.renameatx_np
        renameatx_np.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            -2, os.fsencode(source), -2, os.fsencode(destination), 0x00000004
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), destination)
        return
    if destination.exists():
        raise FileExistsError(destination)
    os.rename(source, destination)


def promote(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = read_json(args.full_audit)
    if audit.get("run_id") != args.run_id:
        raise ValueError("full audit run_id mismatch")
    if audit.get("positive_injection") is not False:
        raise ValueError("full audit does not prove zero positive injection")
    if audit.get("qualification_uses_retrieval_or_model_scores") is not False:
        raise ValueError("full audit used retrieval/model qualification")
    ledger_sha = sha256(args.db)
    if audit.get("ledger_sha256") != ledger_sha:
        raise ValueError("full audit ledger hash does not match frozen DB")

    audit_by_key = {
        (value["seed_id"], value["variant_id"]): value
        for value in audit.get("rows", [])
    }
    promotable_by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in audit_by_key.values():
        qualification = value.get("full_sirene_qualification", {})
        exact_variant = (
            value.get("variant_promotable_exact") is True
            and qualification.get("decision") == "EXACT_IDENTIFIABLE"
            and qualification.get("exact_witness") == "G_N_A"
            and qualification.get("target_naturally_returned") is True
        )
        exact_atomic = (
            value.get("seed_ledger_3_of_3_accept") is True
            and value.get("seed_promotable_3_of_3_exact") is True
            and exact_variant
        )
        if exact_variant if args.promotion_mode == "per-variant" else exact_atomic:
            promotable_by_seed[value["seed_id"]].append(value)
    if args.promotion_mode == "atomic-three":
        promotable_by_seed = {
            seed_id: values for seed_id, values in promotable_by_seed.items()
            if {value["variant_id"] for value in values} == VARIANT_IDS and len(values) == 3
        }

    diagnostic_rows = read_jsonl(args.diagnostic_export / "accept.jsonl")
    diagnostic_by_key = {
        (value["seed_id"], value["variant_id"]): value for value in diagnostic_rows
    }
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    run_row = connection.execute(
        "SELECT policy_json, schema_sha256 FROM runs WHERE run_id=?", (args.run_id,)
    ).fetchone()
    if run_row is None:
        raise ValueError("unknown run")
    run_policy = json.loads(run_row["policy_json"])
    if args.promotion_mode == "per-variant" and not run_policy.get(
        "allow_partial_seed_promotion", False
    ):
        raise ValueError("per-variant promotion requires an explicitly partial run policy")
    records = connection.execute(
        """
        SELECT v.seed_id, v.variant_id, v.crm_json, v.families_json,
               v.transformation_summary, v.generator_response_sha256,
               v.preflight_json, v.critic_json, v.critic_decision,
               v.adjudicator_json, v.adjudicator_decision,
               v.final_decision, v.final_reason,
               g.status AS slot_status, g.accepted_task_id,
               t.role AS task_role, t.variant_id AS task_variant_id,
               t.status AS task_status, t.raw_response, t.response_sha256,
               s.target_siret, s.target_siren, s.source_kind, s.oof_fold
        FROM variants v
        JOIN generator_slots g
          ON g.run_id=v.run_id AND g.seed_id=v.seed_id AND g.variant_id=v.variant_id
        JOIN tasks t ON t.task_id=g.accepted_task_id
        JOIN seeds s ON s.run_id=v.run_id AND s.seed_id=v.seed_id
        WHERE v.run_id=?
        ORDER BY v.seed_id, v.variant_id
        """,
        (args.run_id,),
    ).fetchall()
    critic_seed_counts = Counter(
        row["seed_id"] for row in connection.execute(
            "SELECT seed_id FROM tasks WHERE run_id=? AND role='CRITIC' AND status='COMPLETED'",
            (args.run_id,),
        )
    )
    connection.close()

    promoted: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    seen_by_seed: Counter[str] = Counter()
    for row in records:
        seed_id, variant_id = row["seed_id"], row["variant_id"]
        if seed_id not in promotable_by_seed:
            continue
        key = (seed_id, variant_id)
        diagnostic = diagnostic_by_key.get(key)
        full_row = audit_by_key.get(key)
        if diagnostic is None or full_row is None:
            raise ValueError(f"missing diagnostic/audit row: {key}")
        if (
            row["slot_status"] != "PASSED"
            or row["task_role"] != "GENERATOR"
            or row["task_status"] != "COMPLETED"
            or row["task_variant_id"] != variant_id
            or row["final_decision"] != "ACCEPT"
            or row["critic_decision"] != "ACCEPT"
            or critic_seed_counts[seed_id] < 1
        ):
            raise ValueError(f"ledger provenance/decision gate failed: {key}")
        if not row["raw_response"]:
            raise ValueError(f"missing raw Luna response: {key}")
        raw_sha = hashlib.sha256(row["raw_response"].encode("utf-8")).hexdigest()
        if not (
            raw_sha == row["response_sha256"] == row["generator_response_sha256"]
        ):
            raise ValueError(f"generator response hashes disagree: {key}")
        crm = json.loads(row["crm_json"])
        raw_variant = raw_generator_variant(row["raw_response"], variant_id)
        if raw_variant.get("crm") != crm or diagnostic.get("crm") != crm:
            raise ValueError(f"CRM is not byte-structurally identical to Luna raw: {key}")
        preflight = json.loads(row["preflight_json"])
        if preflight.get("passed") is not True:
            raise ValueError(f"preflight not passed: {key}")
        fingerprint = loop.surface_fingerprint(crm)
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate promoted CRM surface: {key}")
        fingerprints.add(fingerprint)
        seen_by_seed[seed_id] += 1
        promoted.append({
            **diagnostic,
            "full_sirene_qualification": full_row["full_sirene_qualification"],
            "promotion_provenance": {
                "accepted_task_id": row["accepted_task_id"],
                "generator_response_sha256": raw_sha,
                "critic_decision": row["critic_decision"],
            },
        })
    if not promoted:
        raise ValueError("no exact variant is promotable")
    if args.promotion_mode == "atomic-three" and any(
        count != 3 for count in seen_by_seed.values()
    ):
        raise ValueError("promotion is not atomic 3-of-3")

    stage = args.output.parent / f".{args.output.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=False)
    try:
        promoted_path = stage / "promoted.jsonl"
        loop.write_jsonl_atomic(promoted_path, promoted)
        manifest = {
            "schema_version": "sireto-synthetic-gt-full-exact-promotion-2",
            "run_id": args.run_id,
            "promoted_seeds": len(seen_by_seed),
            "promoted_variants": len(promoted),
            "promotion_mode": args.promotion_mode,
            "atomic_three_of_three": args.promotion_mode == "atomic-three",
            "run_policy": run_policy,
            "message_schema_sha256": run_row["schema_sha256"],
            "exact_witness": "G_N_A",
            "positive_injection": False,
            "qualification_uses_retrieval_or_model_scores": False,
            "source_hashes": {
                "db": ledger_sha,
                "full_audit": sha256(args.full_audit),
                "diagnostic_accept": sha256(args.diagnostic_export / "accept.jsonl"),
                "seed_input": sha256(args.seed_input),
            },
            "promoted_sha256": sha256(promoted_path),
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in (promoted_path, manifest_path):
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
        atomic_rename_exclusive(stage, args.output)
    except Exception:
        if stage.exists():
            for path in stage.iterdir():
                path.unlink()
            stage.rmdir()
        raise
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--db", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--diagnostic-export", type=Path, required=True)
    result.add_argument("--full-audit", type=Path, required=True)
    result.add_argument("--seed-input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--promotion-mode", choices=("atomic-three", "per-variant"),
        default="atomic-three",
    )
    result.set_defaults(func=promote)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
