from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts import run_synthetic_gt_agentic_loop as loop
from scripts import run_synthetic_gt_luna_driver as driver


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(loop.canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def seed() -> dict:
    return {
        "seed_id": "seed-1",
        "target_siret": "12345678900012",
        "target_siren": "123456789",
        "source_kind": "SIRENE_SYNTHETIC",
        "oof_fold": 2,
        "legacy_split": "train_synthetic",
        "seed_card": {
            "official_names": ["Société des Fleurs"],
            "official_address": "12 RUE DES LILAS 75001 PARIS",
            "postcode": "75001",
            "city": "PARIS",
            "insee": "75056",
        },
        "observed_train_profile": {"allowed_families": ["LEGAL_FORM"]},
        "risk_flags": [],
    }


def task(role: str = "GENERATOR") -> dict:
    return {
        "schema_version": loop.SCHEMA_VERSION,
        "task_id": "a" * 32,
        "run_id": "run-1",
        "batch_id": "generator-12345678",
        "role": role,
        "prompt_version": loop.PROMPT_VERSIONS[role],
        "input_sha256": "b" * 64,
        "input": {"seed": {"siret": "12345678900012", "siren": "123456789"}},
    }


def test_dynamic_schema_freezes_task_envelope():
    value = task()
    schema = driver.task_output_schema(value, loop.load_json(loop.DEFAULT_SCHEMA))
    properties = schema["properties"]
    for field in (
        "schema_version", "task_id", "run_id", "batch_id", "role",
        "prompt_version", "input_sha256",
    ):
        assert properties[field] == {"const": value[field], "type": "string"}
    assert properties["seed"]["additionalProperties"] is False
    assert properties["seed"]["properties"]["siret"]["const"] == value["input"]["seed"]["siret"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["variants"]["minItems"] == 3
    assert "uniqueItems" not in json.dumps(schema)


def test_per_variant_dynamic_schema_freezes_single_target_variant():
    value = task()
    value["prompt_version"] = loop.PER_VARIANT_GENERATOR_PROMPT_VERSION
    value["input"]["target_variant_id"] = "v2"
    schema = driver.task_output_schema(value, loop.load_json(loop.DEFAULT_SCHEMA))
    variants = schema["properties"]["variants"]
    assert variants["minItems"] == variants["maxItems"] == 1
    assert schema["$defs"]["variant"]["properties"]["variant_id"] == {
        "const": "v2", "type": "string"
    }
    prompt = driver.worker_prompt(value)
    assert "exactly the single variant named by target_variant_id" in prompt
    assert "Do not emit either of the other" in prompt


def test_composite_output_schema_forces_single_composite_family():
    value = task()
    value["input"]["seed_card"] = {"generation_mode": "OBSERVED_COMPOSITE_ANALOGY_V2"}
    schema = driver.task_output_schema(value, loop.load_json(loop.DEFAULT_SCHEMA))
    family = schema["$defs"]["variant"]["properties"]["corruption_families_observed"]
    assert family["minItems"] == family["maxItems"] == 1
    assert family["items"] == {"const": loop.COMPOSITE_FAMILY, "type": "string"}


def test_composite_worker_prompt_excludes_legacy_family_catalogue():
    value = task()
    value["input"]["seed_card"] = {"generation_mode": "OBSERVED_COMPOSITE_ANALOGY_V2"}
    prompt = driver.worker_prompt(value)
    assert "declare only\nOBSERVED_COMPOSITE_ANALOGY" in prompt
    assert "OCR_LIMITED" not in prompt


def test_codex_command_is_ephemeral_luna_low_and_read_only(tmp_path: Path, monkeypatch):
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(driver.shutil, "which", lambda _value: str(executable))
    args = driver.parser().parse_args([
        "--db", str(tmp_path / "db.sqlite"), "--run-id", "run-1",
        "--artifacts", str(tmp_path / "artifacts"),
    ])
    command = driver.codex_command(args, tmp_path / "schema.json", tmp_path / "raw.json")
    assert "--ephemeral" in command
    assert command[command.index("-m") + 1] == "gpt-5.6-luna"
    assert 'model_reasoning_effort="low"' in command
    assert command[command.index("-s") + 1] == "read-only"
    assert "--output-schema" in command
    assert "--output-last-message" in command


def test_fake_luna_drains_cycle_and_preserves_multiline_raw_response(tmp_path: Path):
    db = tmp_path / "ledger.sqlite"
    seeds = tmp_path / "seeds.jsonl"
    write_jsonl(seeds, [seed()])
    loop.main(["--db", str(db), "init", "--run-id", "run-1", "--seeds", str(seeds)])

    fake = tmp_path / "fake-codex"
    fake.write_text(
        """#!/usr/bin/env python3
import json,sys
args=sys.argv[1:]
output=args[args.index('--output-last-message')+1]
prompt=sys.stdin.read()
task=json.loads(prompt.split('Task JSON (the response envelope constants must match it exactly):\\n',1)[1])
base={k:task[k] for k in ('schema_version','task_id','run_id','batch_id','role','prompt_version','input_sha256')}
base['seed']=task['input']['seed']
if task['role']=='GENERATOR':
    base['variants']=[
      {'variant_id':'v1','crm':{'name':'Soc des Fleurs','address':'12 rue des Lilas','postcode':'75001','city':'Paris','insee':'75056'},'corruption_families_observed':['LEGAL_FORM'],'transformation_summary':'Forme omise.'},
      {'variant_id':'v2','crm':{'name':'SOCIETE DES FLEURS','address':'12 R DES LILAS','postcode':'75001','city':'PARIS','insee':'75056'},'corruption_families_observed':['ADDRESS_ABBREVIATION'],'transformation_summary':'Voie abrégée.'},
      {'variant_id':'v3','crm':{'name':'Societe des Fleurs','address':'12 RUE DES LILAS','postcode':'75001','city':'Paris','insee':'75056'},'corruption_families_observed':['ACCENT_PUNCTUATION'],'transformation_summary':'Accent omis.'}]
else:
    base['decisions']=[{'variant_id':v,'decision':'ACCEPT','confidence':0.99,'reason_codes':['PASS'],'reason':'Identité préservée.'} for v in ('v1','v2','v3')]
    if task['role']=='CRITIC':
        base['independent']=True
        base['generator_rationale_seen']=False
with open(output,'w',encoding='utf-8') as f:
    json.dump(base,f,ensure_ascii=False,indent=2)
    f.write('\\n')
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    artifacts = tmp_path / "driver"
    export = tmp_path / "export"
    driver.main([
        "--db", str(db),
        "--run-id", "run-1",
        "--artifacts", str(artifacts),
        "--export", str(export),
        "--codex-binary", str(fake),
        "--concurrency", "1",
        "--transport-retries", "1",
        "--agent-timeout-seconds", "10",
    ])
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"ACCEPT": 3, "REJECT": 0, "SILVER": 0}
    with sqlite3.connect(db) as connection:
        rows = connection.execute(
            "SELECT raw_response FROM tasks WHERE role='GENERATOR' AND status='COMPLETED'"
        ).fetchall()
        attempt = connection.execute("SELECT attempt FROM seeds").fetchone()[0]
    assert len(rows) == 1
    raw_path = next((artifacts / "generator").glob("*/raw_response.json"))
    assert rows[0][0] == raw_path.read_text(encoding="utf-8")
    assert "\n" in rows[0][0]
    assert attempt == 1


def test_driver_source_contains_no_mechanical_crm_generator():
    source = Path(driver.__file__).read_text(encoding="utf-8")
    forbidden = ["import random", "from random", "faker", "transform_variant", "deterministic_rng"]
    assert all(token not in source for token in forbidden)
