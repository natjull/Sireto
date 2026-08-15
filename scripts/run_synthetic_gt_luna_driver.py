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

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


ROLE_PROMPTS = {
    "GENERATOR": """You are the SIRETO GENERATOR. Write exactly three realistic CRM variants
directly from the official task. Obey baseline_crm and variant_contract exactly: modify only
target_fields, copy every other field byte-for-byte, declare only requested_family, and make the
requested change genuinely visible. The three complete CRM fingerprints must be different.

When generation_mode is OBSERVED_COMPOSITE_ANALOGY_V2, each contract contains one existing real
train official->observed_crm pair. Study that pair as an analogy for the kind, intensity, and
interaction of human CRM degradation, then YOU must directly write a new CRM variant for the
target establishment. Do not copy business, street, city, or numeric tokens from the inspiration.
Change every target_field non-trivially (case-only is not a change), including name plus address
or city. Use only lexical and numeric material already present in the target official field; an
official street type may use its listed canonical abbreviation. You may delete, reorder, join, or
normalize target material where the inspiration supports that style. Never add punctuation or a
diacritic absent from the corresponding target official field. Preserve the house number, postal
code, INSEE code, and every non-target field byte-for-byte. The code will reject rather than repair
any violation. Declare only OBSERVED_COMPOSITE_ANALOGY.

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
- TOKEN_ORDER: move one complete name token or adjacent token block; preserve the exact token
  multiset and every character. Do not reverse letters inside a token.
- ADDRESS_TOKEN_ORDER: move only the full street-type token before/after the unchanged street-name
  sequence; preserve house number and internal street-name order.
- OCR_LIMITED or ADDRESS_OCR: perform exactly one same-length character substitution listed in the
  corresponding *_ocr_substitution_pairs evidence in seed_card. Never insert/delete/truncate; for
  an address, preserve every digit including the house number.
- ENSEIGNE_VS_DENOMINATION: replace the name exactly by one distinct enseigne_options value, with
  no additional spelling or punctuation edit.

Never output case-only OCR, gratuitous punctuation, an unofficial alternate name, an unchanged
token order, or a fake address abbreviation. Never write SIRET/SIREN inside CRM fields. On retry,
read retry_context and correct every listed preflight error. Return only the structured response.""",
    "CRITIC": """You are the independent SIRETO CRITIC. Compare every CRM variant with
baseline_crm, seed_card, and variant_contract. The requested corruption is intentionally not an
official alternate value: never reject merely because the corrupted CRM text is absent from
name_options or official fields. Judge whether it realizes the exact requested family while
preserving identity and all non-target fields.

For OBSERVED_COMPOSITE_ANALOGY, independently compare the target official->generated CRM delta
with the contract's real train inspiration. ACCEPT only if all requested fields genuinely change,
the combined name+location degradation is plausible as one human CRM record, no inspiration
entity token was copied, the target establishment remains the best exact interpretation in the
bounded official_context, and the result is not merely capitalization. Distinguish exact SIRET
identifiability from operational same-SIREN/same-site equivalence in reason_codes. A closed target
with an active same-site sibling may be OPERATIONAL_ONLY but is not exact. Never see or infer the
generator rationale and never repair text.

Apply these decision rules literally:
- ACRONYM_TOKENIZATION is valid when punctuation-delimited tokens are joined/split, the normalized
  token sequence changes, and every alphanumeric character remains in the same order (G.D. -> GD).
- ACCENT_PUNCTUATION is valid when an existing accent or punctuation mark is deleted/normalized,
  including hyphen-to-space, provided no new mark or alphanumeric character is introduced. Do not
  label such an expected deletion gratuitous, unofficial, or space-only.
- ADDRESS_ABBREVIATION is valid only for the exact canonical street-type substitution in the
  contract, with number/street/order unchanged.
- COMMUNE_VARIANT is valid when separators, spaces, apostrophes, hyphens, or accents are normalized
  while every alphanumeric character remains in the same order.
- LEGAL_FORM is valid only when the explicit form token alone is removed.
- TOKEN_ORDER is valid only when the exact name-token multiset is preserved and its sequence
  changes, without reversing characters inside tokens.
- ADDRESS_TOKEN_ORDER is valid only when the full street type moves before/after the unchanged
  street-name sequence and the house number is preserved.
- OCR_LIMITED and ADDRESS_OCR require exactly one same-length substitution present in the
  corresponding evidence list; insertion, deletion, truncation and address digit changes reject.
- ENSEIGNE_VS_DENOMINATION is valid only when the target equals a distinct official
  enseigne_options value byte-for-byte after surrounding-space normalization.

Reject any changed non-target field, mismatched/non-visible family, case-only edit, added accent or
punctuation, changed alphanumeric content, unsupported OCR, unchanged tokenization, or false
abbreviation. Do not repair text. Set independent=true and generator_rationale_seen=false. Return
only the structured response.""",
    "ADJUDICATOR": """You are the SIRETO ADJUDICATOR. Decide each variant from the official
baseline, contract, deterministic preflight, and independent critic. Never rewrite CRM text and
never promote a critic REJECT. ACCEPT only when the exact establishment remains identifiable and
the requested family is genuinely realized. Return only the structured response.""",
}

COMPOSITE_ROLE_PROMPTS = {
    "GENERATOR": """You are the SIRETO composite GENERATOR. The task has exactly three
OBSERVED_COMPOSITE_ANALOGY contracts. For each contract, study its existing real train
official->observed_crm inspiration only as a transformation analogy, then YOU directly write the
new CRM for the target establishment. Return exactly v1/v2/v3 and declare only
OBSERVED_COMPOSITE_ANALOGY for every variant.

Change every target_field non-trivially; case-only is forbidden. Copy every non-target field
byte-for-byte from baseline_crm. The result must combine a changed name with changed address
and/or city exactly as the contract mask says. Never copy entity, street, city, or number material
from the inspiration. Reproduce each contract field_relations class exactly:
- TOKEN_ORDER: same complete name-token multiset, different order;
- TOKEN_SUBSET: delete whole name tokens only, retain at least two and at least half;
- LEGAL_FORM_REMOVE: remove only the explicit legal-form token;
- ADDRESS_ABBREVIATE: change only the official street type to a canonical listed alias;
- ADDRESS_TYPE_ORDER: move only the street-type token; preserve internal street-name order;
- PUNCTUATION_REMOVED/ADDED or DIACRITIC_REMOVED/ADDED: change only that kind of mark;
- JOIN_SPLIT: change word boundaries only while preserving the exact alphanumeric sequence.

Never apply a different relation because it seems easier. In particular, never delete or reorder
city letters/tokens: city changes may only preserve the exact alphanumeric sequence. Use no new
alphanumeric material. Add punctuation or a diacritic only when the exact field relation says
ADDED; otherwise add no mark. Preserve all address digits, especially the house number, and
preserve postcode and INSEE. Make all three complete CRM fingerprints distinct. Read
retry_context and correct every listed error. The validator will reject, never repair, your text.
Return only the structured JSON response with exact envelope constants.""",
    "CRITIC": """You are the independent SIRETO composite CRITIC. You did not see the generator
rationale. For each v1/v2/v3, compare baseline_crm, the generated CRM, its exact contract, its real
train inspiration, and bounded official_context. ACCEPT only if every target field changes beyond
case, all non-target fields are byte-identical, the combined name+location degradation is a
plausible single human CRM record, no inspiration entity material was copied, and the target SIRET
remains the best exact interpretation. Verify that every generated field realizes exactly its
declared field_relations class; reject internal street-name reordering, destructive token removal,
or city token deletion/reordering. A same-SIREN same-site active sibling can establish
OPERATIONAL_ONLY but never exact success. Use distinct reason_codes for REALISTIC,
EXACT_IDENTIFIABLE, OPERATIONAL_ONLY, or AMBIGUOUS. Never repair text. Set independent=true and
generator_rationale_seen=false and return only the structured JSON response.""",
    "ADJUDICATOR": """You are the SIRETO composite ADJUDICATOR. Resolve only the reviewed
v1/v2/v3 decisions using baseline, contracts, deterministic preflight and critic. Never rewrite
CRM, never override a deterministic failure, and never promote a critic REJECT. Keep exact SIRET
and operational same-SIREN/same-site conclusions separate. Return only structured JSON.""",
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
        if task["input"].get("seed_card", {}).get("generation_mode") == "OBSERVED_COMPOSITE_ANALOGY_V2":
            family = selected_defs["variant"]["properties"]["corruption_families_observed"]
            family["minItems"] = 1
            family["maxItems"] = 1
            family["items"] = {"const": loop.COMPOSITE_FAMILY}
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
    composite = task["input"].get("seed_card", {}).get("generation_mode") == "OBSERVED_COMPOSITE_ANALOGY_V2"
    prompts = COMPOSITE_ROLE_PROMPTS if composite else ROLE_PROMPTS
    return (
        prompts[task["role"]]
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
            validator = Draft202012Validator(message_schema)
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
