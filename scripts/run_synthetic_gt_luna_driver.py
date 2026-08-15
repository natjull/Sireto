#!/usr/bin/env python3
"""Drain the synthetic-GT ledger with direct, structured Luna responses.

The driver never creates, edits, or repairs CRM text.  It leases tasks from
the durable runtime, asks a fresh read-only Luna process to answer each task,
stores the model response byte-for-byte, and submits it to the existing
fail-closed validator.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


ROLE_PROMPTS = {
    "GENERATOR": """You are the SIRETO GENERATOR. Write exactly three realistic CRM variants
directly from the official task. Obey baseline_crm and variant_contract exactly: modify only
target_fields, copy every other field byte-for-byte, declare only requested_family, and make the
requested change genuinely visible. The three complete CRM fingerprints must be different.

Apply the requested family literally:
- ACRONYM_TOKENIZATION: join or split existing punctuation-delimited short tokens while preserving
  every alphanumeric character in the same order. Example: G.D. FERMETURES -> GD FERMETURES, not
  G D FERMETURES (whose normalized tokens are unchanged).
- ACCENT_PUNCTUATION: only delete or normalize an accent or punctuation mark already present.
  Never introduce a new mark and never replace one mark by a different mark.
- ADDRESS_ABBREVIATION: replace only the exact full street type with its canonical abbreviation,
  e.g. RUE->R, AVENUE->AV, BOULEVARD->BD, CHEMIN->CH, IMPASSE->IMP, PLACE->PL, ROUTE->RTE,
  ALLEE->ALL, QUAI->QU, RESIDENCE->RES. Preserve number, street words, and their order exactly.
- COMMUNE_VARIANT: only remove/normalize separators, spaces, apostrophes, hyphens, or accents in
  the official city. Preserve every alphanumeric character in the same order.
- LEGAL_FORM: remove only the explicit legal-form token; preserve all other lexical tokens.

Never output case-only OCR, gratuitous punctuation, an unofficial alternate name, an unchanged
token order, or a fake address abbreviation. Never write SIRET/SIREN inside CRM fields. On retry,
read retry_context and correct every listed preflight error. Return only the structured response.""",
    "CRITIC": """You are the independent SIRETO CRITIC. Compare every CRM variant with
baseline_crm, seed_card, and variant_contract. Reject any changed non-target field, mismatched or
non-visible family, case/space-only edit, gratuitous punctuation, unsupported OCR, unofficial
alternate name, unchanged token order, or false abbreviation. Do not repair text. Set
independent=true and generator_rationale_seen=false. Return only the structured response.""",
    "ADJUDICATOR": """You are the SIRETO ADJUDICATOR. Decide each variant from the official
baseline, contract, deterministic preflight, and independent critic. Never rewrite CRM text and
never promote a critic REJECT. ACCEPT only when the exact establishment remains identifiable and
the requested family is genuinely realized. Return only the structured response.""",
}


def structured_output_compatible(value: Any) -> Any:
    """Adapt the strict local schema to the API's supported JSON-Schema subset.

    Unsupported constraints are removed only from the transport schema.  The
    unmodified message schema is still applied locally before submission.
    """
    if isinstance(value, list):
        return [structured_output_compatible(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: structured_output_compatible(item)
        for key, item in value.items()
        if key != "uniqueItems"
    }
    if "type" not in result:
        sample: Any | None = None
        if "const" in result:
            sample = result["const"]
        elif isinstance(result.get("enum"), list) and result["enum"]:
            kinds = {type(item) for item in result["enum"]}
            if len(kinds) == 1:
                sample = result["enum"][0]
        if isinstance(sample, bool):
            result["type"] = "boolean"
        elif isinstance(sample, str):
            result["type"] = "string"
        elif isinstance(sample, int):
            result["type"] = "integer"
        elif isinstance(sample, float):
            result["type"] = "number"
    return result


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def task_output_schema(task: dict[str, Any], message_schema: dict[str, Any]) -> dict[str, Any]:
    definitions = message_schema["$defs"]
    role = task["role"]
    properties: dict[str, Any] = {
        "schema_version": {"const": task["schema_version"]},
        "task_id": {"const": task["task_id"]},
        "run_id": {"const": task["run_id"]},
        "batch_id": {"const": task["batch_id"]},
        "role": {"const": role},
        "prompt_version": {"const": task["prompt_version"]},
        "input_sha256": {"const": task["input_sha256"]},
        "seed": {
            "type": "object",
            "additionalProperties": False,
            "required": ["siret", "siren"],
            "properties": {
                "siret": {"type": "string", "const": task["input"]["seed"]["siret"]},
                "siren": {"type": "string", "const": task["input"]["seed"]["siren"]},
            },
        },
    }
    required = list(properties)
    selected_defs: dict[str, Any]
    if role == "GENERATOR":
        properties["variants"] = {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {"$ref": "#/$defs/variant"},
        }
        required.append("variants")
        selected_defs = {
            "crm": copy.deepcopy(definitions["crm"]),
            "variant": copy.deepcopy(definitions["variant"]),
        }
    else:
        properties["decisions"] = {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {"$ref": "#/$defs/decision"},
        }
        required.append("decisions")
        selected_defs = {"decision": copy.deepcopy(definitions["decision"])}
        if role == "CRITIC":
            properties["independent"] = {"const": True}
            properties["generator_rationale_seen"] = {"const": False}
            required.extend(["independent", "generator_rationale_seen"])
    return structured_output_compatible({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
        "$defs": selected_defs,
    })


def worker_prompt(task: dict[str, Any]) -> str:
    return (
        ROLE_PROMPTS[task["role"]]
        + "\n\nTask JSON (the response envelope constants must match it exactly):\n"
        + loop.canonical_json(task)
    )


def codex_command(args: argparse.Namespace, schema_path: Path, output_path: Path) -> list[str]:
    executable = shutil.which(args.codex_binary)
    if executable is None:
        raise RuntimeError(f"Codex executable not found: {args.codex_binary}")
    return [
        executable,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-m",
        args.model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
        "-s",
        "read-only",
        "-C",
        str(ROOT),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
    ]


def invoke_luna(
    task: dict[str, Any],
    args: argparse.Namespace,
    message_schema: dict[str, Any],
) -> tuple[dict[str, Any], Path, str | None]:
    role_dir = args.artifacts / task["role"].lower()
    task_dir = role_dir / task["task_id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    task_path = task_dir / "task.json"
    schema_path = task_dir / "output_schema.json"
    output_path = task_dir / "raw_response.json"
    log_path = task_dir / "codex.log"
    atomic_write(task_path, json.dumps(task, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    atomic_write(
        schema_path,
        json.dumps(task_output_schema(task, message_schema), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    last_error: str | None = None
    for transport_attempt in range(1, args.transport_retries + 1):
        started = time.monotonic()
        try:
            result = subprocess.run(
                codex_command(args, schema_path, output_path),
                input=worker_prompt(task),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.agent_timeout_seconds,
                check=False,
            )
            elapsed = time.monotonic() - started
            atomic_write(
                log_path,
                f"transport_attempt={transport_attempt}\nelapsed_seconds={elapsed:.3f}\n"
                f"exit_code={result.returncode}\n{result.stdout}",
            )
            if result.returncode != 0:
                last_error = f"codex exit code {result.returncode}"
                continue
            raw = output_path.read_text(encoding="utf-8").strip()
            response = json.loads(raw)
            validator = loop.response_validator(args.schema)
            validation_errors = list(validator.iter_errors(response))
            if validation_errors:
                last_error = f"response schema error: {validation_errors[0].message}"
                continue
            return task, output_path, None
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return task, output_path, last_error or "unknown Luna transport failure"


def lease_tasks(args: argparse.Namespace, role: str, worker_id: str) -> list[dict[str, Any]]:
    output = args.artifacts / "leases" / f"{role.lower()}-{time.time_ns()}.jsonl"
    loop.command_lease(argparse.Namespace(
        db=args.db,
        run_id=args.run_id,
        role=role,
        worker_id=worker_id,
        limit=args.concurrency,
        ttl_seconds=args.lease_ttl_seconds,
        output=output,
    ))
    if not output.exists() or not output.read_text(encoding="utf-8").strip():
        return []
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]


def submit_task(args: argparse.Namespace, role: str, worker_id: str, path: Path) -> None:
    loop.command_submit(argparse.Namespace(
        db=args.db,
        role=role,
        worker_id=worker_id,
        input=path,
        input_format="json",
        schema=args.schema,
    ))


def abandon_task(
    args: argparse.Namespace,
    role: str,
    worker_id: str,
    task: dict[str, Any],
    reason: str,
) -> None:
    loop.command_abandon(argparse.Namespace(
        db=args.db,
        task_id=task["task_id"],
        role=role,
        worker_id=worker_id,
        reason=reason[:500],
    ))


def process_wave(
    args: argparse.Namespace,
    role: str,
    message_schema: dict[str, Any],
) -> int:
    worker_id = f"luna-driver-{role.lower()}-{os.getpid()}"
    tasks = lease_tasks(args, role, worker_id)
    if not tasks:
        return 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(
            lambda task: invoke_luna(task, args, message_schema), tasks
        ))
    submitted = 0
    for task, output_path, error in results:
        if error:
            abandon_task(args, role, worker_id, task, f"LUNA_TRANSPORT:{error}")
            continue
        try:
            submit_task(args, role, worker_id, output_path)
            submitted += 1
        except Exception as exc:
            abandon_task(args, role, worker_id, task, f"SUBMIT_FAILED:{type(exc).__name__}:{exc}")
    return submitted


def current_status(args: argparse.Namespace) -> dict[str, Any]:
    with loop.connect(args.db) as connection:
        loop.create_schema(connection)
        return loop.status_payload(connection, args.run_id)


def supervise(args: argparse.Namespace) -> int:
    before = current_status(args)["seeds"].get("READY_SUPERVISOR", 0)
    if before:
        loop.command_supervise(argparse.Namespace(db=args.db, run_id=args.run_id, limit=before))
    return int(before)


def drain(args: argparse.Namespace) -> None:
    args.artifacts.mkdir(parents=True, exist_ok=True)
    message_schema = loop.load_json(args.schema)
    previous_signature: str | None = None
    stagnant_cycles = 0
    for _cycle in range(args.max_cycles):
        supervised = supervise(args)
        status = current_status(args)
        total = sum(status["seeds"].values())
        if total and status["seeds"].get("COMPLETED", 0) == total:
            if args.export:
                loop.command_export(argparse.Namespace(
                    db=args.db, run_id=args.run_id, output=args.export
                ))
            print(json.dumps(status, ensure_ascii=False, sort_keys=True, indent=2))
            return
        submitted = 0
        for role in ("GENERATOR", "CRITIC", "ADJUDICATOR"):
            submitted += process_wave(args, role, message_schema)
        new_status = current_status(args)
        signature = loop.canonical_json(new_status)
        if submitted or supervised or signature != previous_signature:
            stagnant_cycles = 0
        else:
            stagnant_cycles += 1
        if stagnant_cycles >= args.max_stagnant_cycles:
            raise RuntimeError(f"agentic driver made no progress: {signature}")
        previous_signature = signature
    raise RuntimeError(f"max cycles exceeded: {loop.canonical_json(current_status(args))}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--db", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--artifacts", type=Path, required=True)
    result.add_argument("--export", type=Path)
    result.add_argument("--schema", type=Path, default=loop.DEFAULT_SCHEMA)
    result.add_argument("--model", default="gpt-5.6-luna")
    result.add_argument("--reasoning-effort", default="low")
    result.add_argument("--codex-binary", default="codex")
    result.add_argument("--concurrency", type=int, default=2)
    result.add_argument("--lease-ttl-seconds", type=int, default=1800)
    result.add_argument("--agent-timeout-seconds", type=int, default=600)
    result.add_argument("--transport-retries", type=int, default=2)
    result.add_argument("--max-cycles", type=int, default=10000)
    result.add_argument("--max-stagnant-cycles", type=int, default=2)
    result.set_defaults(func=drain)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    for name in ("concurrency", "lease_ttl_seconds", "agent_timeout_seconds", "transport_retries"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
