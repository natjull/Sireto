from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts import run_synthetic_gt_agentic_loop as loop


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(loop.canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def seed(seed_id: str = "seed-1", siret: str = "12345678900012", risk_flags=None) -> dict:
    return {
        "seed_id": seed_id,
        "target_siret": siret,
        "target_siren": siret[:9],
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
        "observed_train_profile": {
            "allowed_families": ["LEGAL_FORM", "ADDRESS_ABBREVIATION", "ACCENT_PUNCTUATION"]
        },
        "risk_flags": list(risk_flags or []),
    }


def init_run(tmp_path: Path, seeds: list[dict], run_id: str = "run-1", extra=None) -> tuple[Path, str]:
    db = tmp_path / "agentic.sqlite"
    seed_path = tmp_path / f"{run_id}-seeds.jsonl"
    write_jsonl(seed_path, seeds)
    argv = ["--db", str(db), "init", "--run-id", run_id, "--seeds", str(seed_path)]
    loop.main(argv + list(extra or []))
    return db, run_id


def lease(tmp_path: Path, db: Path, run_id: str, role: str, worker: str, limit: int = 8) -> list[dict]:
    output = tmp_path / f"{role}-{worker}.jsonl"
    loop.main([
        "--db", str(db), "lease", "--run-id", run_id, "--role", role,
        "--worker-id", worker, "--limit", str(limit), "--output", str(output),
    ])
    if not output.read_text(encoding="utf-8"):
        return []
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


def generator_response(task: dict, leaked_name: str | None = None) -> dict:
    values = [
        {
            "variant_id": "v1",
            "crm": {"name": leaked_name or "Soc des Fleurs", "address": "12 rue des Lilas", "postcode": "75001", "city": "Paris", "insee": "75056"},
            "corruption_families_observed": ["LEGAL_FORM"],
            "transformation_summary": "Forme juridique omise selon le profil observé.",
        },
        {
            "variant_id": "v2",
            "crm": {"name": "SOCIETE DES FLEURS", "address": "12 R DES LILAS", "postcode": "75001", "city": "PARIS", "insee": "75056"},
            "corruption_families_observed": ["ADDRESS_ABBREVIATION"],
            "transformation_summary": "Type de voie abrégé comme dans le CRM train.",
        },
        {
            "variant_id": "v3",
            "crm": {"name": "Societe des Fleurs", "address": "12 RUE DES LILAS", "postcode": "75001", "city": "Paris", "insee": "75056"},
            "corruption_families_observed": ["ACCENT_PUNCTUATION"],
            "transformation_summary": "Accent absent dans la dénomination.",
        },
    ]
    return {
        "schema_version": loop.SCHEMA_VERSION,
        "task_id": task["task_id"],
        "run_id": task["run_id"],
        "batch_id": task["batch_id"],
        "role": "GENERATOR",
        "prompt_version": loop.PROMPT_VERSIONS["GENERATOR"],
        "input_sha256": task["input_sha256"],
        "seed": task["input"]["seed"],
        "variants": values,
    }


def review_response(task: dict, role: str, decisions=None) -> dict:
    values = decisions or [
        {"variant_id": value, "decision": "ACCEPT", "confidence": 0.99, "reason_codes": ["IDENTITY_PRESERVED"], "reason": "La variante reste identifiable."}
        for value in ("v1", "v2", "v3")
    ]
    result = {
        "schema_version": loop.SCHEMA_VERSION,
        "task_id": task["task_id"],
        "run_id": task["run_id"],
        "batch_id": task["batch_id"],
        "role": role,
        "prompt_version": loop.PROMPT_VERSIONS[role],
        "input_sha256": task["input_sha256"],
        "seed": task["input"]["seed"],
        "decisions": values,
    }
    if role == "CRITIC":
        result["independent"] = True
        result["generator_rationale_seen"] = False
    return result


def submit(tmp_path: Path, db: Path, role: str, worker: str, responses: list[dict]) -> None:
    path = tmp_path / f"submit-{role}-{worker}.jsonl"
    write_jsonl(path, responses)
    loop.main(["--db", str(db), "submit", "--role", role, "--worker-id", worker, "--input", str(path)])


def test_full_generator_critic_supervisor_cycle_preserves_agent_text(tmp_path: Path):
    db, run_id = init_run(tmp_path, [seed()])
    generator_task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    response = generator_response(generator_task)
    submit(tmp_path, db, "GENERATOR", "luna-g1", [response])

    critic_task = lease(tmp_path, db, run_id, "CRITIC", "luna-c1")[0]
    encoded_task = loop.canonical_json(critic_task)
    assert "transformation_summary" not in encoded_task
    assert "corruption_families_observed" not in encoded_task
    submit(tmp_path, db, "CRITIC", "luna-c1", [review_response(critic_task, "CRITIC")])

    loop.main(["--db", str(db), "supervise", "--run-id", run_id])
    output = tmp_path / "export"
    loop.main(["--db", str(db), "export", "--run-id", run_id, "--output", str(output)])
    accepted = [json.loads(line) for line in (output / "accept.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(accepted) == 3
    assert accepted[0]["crm"] == response["variants"][0]["crm"]
    assert accepted[0]["generator_response_sha256"] == loop.hashlib.sha256(
        loop.canonical_json(response).encode("utf-8")
    ).hexdigest()


def test_identifier_leak_is_rejected_without_rewriting_and_released_for_retry(tmp_path: Path):
    value = seed()
    db, run_id = init_run(tmp_path, [value])
    task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    response = generator_response(task, leaked_name=value["target_siret"])
    submit(tmp_path, db, "GENERATOR", "luna-g1", [response])
    with sqlite3.connect(db) as connection:
        status = connection.execute("SELECT status FROM seeds WHERE run_id=?", (run_id,)).fetchone()[0]
        stored = json.loads(connection.execute("SELECT crm_json FROM variants WHERE variant_id='v1'").fetchone()[0])
    assert status == "PENDING_GENERATOR"
    assert stored["name"] == value["target_siret"]
    assert lease(tmp_path, db, run_id, "GENERATOR", "luna-g2")


def test_leases_are_disjoint_and_expired_lease_is_recovered(tmp_path: Path):
    db, run_id = init_run(tmp_path, [seed("seed-1"), seed("seed-2", "98765432100019")])
    first = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1", limit=1)
    second = lease(tmp_path, db, run_id, "GENERATOR", "luna-g2", limit=2)
    assert len(first) == len(second) == 1
    assert first[0]["input"]["seed"] != second[0]["input"]["seed"]

    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE seeds SET lease_expires_at=0 WHERE seed_id='seed-1'")
        connection.commit()
    recovered = lease(tmp_path, db, run_id, "GENERATOR", "luna-g3", limit=1)
    assert len(recovered) == 1
    assert recovered[0]["input"]["seed"] == first[0]["input"]["seed"]
    stale_response = generator_response(first[0])
    with pytest.raises(ValueError, match="not leased"):
        submit(tmp_path, db, "GENERATOR", "luna-g1", [stale_response])


def test_abandon_requeues_seed_in_same_role(tmp_path: Path):
    db, run_id = init_run(tmp_path, [seed()])
    task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    loop.main([
        "--db", str(db), "abandon", "--task-id", task["task_id"],
        "--role", "GENERATOR", "--worker-id", "luna-g1", "--reason", "worker_failed",
    ])
    with sqlite3.connect(db) as connection:
        seed_status = connection.execute("SELECT status FROM seeds").fetchone()[0]
        task_status = connection.execute("SELECT status FROM tasks").fetchone()[0]
    assert seed_status == "PENDING_GENERATOR"
    assert task_status == "ABANDONED"
    assert lease(tmp_path, db, run_id, "GENERATOR", "luna-g2")


def test_supervisor_refuses_seed_without_three_variants(tmp_path: Path):
    db, run_id = init_run(tmp_path, [seed()])
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE seeds SET status='READY_SUPERVISOR'")
        connection.commit()
    with pytest.raises(RuntimeError, match="exactly three variants"):
        loop.main(["--db", str(db), "supervise", "--run-id", run_id])
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT status FROM seeds").fetchone()[0] == "READY_SUPERVISOR"


def test_requested_family_dimension_mismatch_is_rejected_before_lease(tmp_path: Path):
    value = seed()
    value["seed_card"]["requested_families"] = {
        "name": "ADDRESS_TOKEN_ORDER",
        "address": "LEGAL_FORM",
        "orthographic": "ENSEIGNE_VS_DENOMINATION",
    }
    with pytest.raises(ValueError, match="incompatible with dimension name"):
        init_run(tmp_path, [value])


def test_generator_task_carries_variant_contract_and_retry_errors(tmp_path: Path):
    value = seed()
    value["seed_card"].update({
        "name_options": ["Société des Fleurs"],
        "address": "12 RUE DES LILAS 75001 PARIS",
        "street_type": "RUE",
        "requested_families": {
            "name": "TOKEN_ORDER",
            "address": "ADDRESS_ABBREVIATION",
            "orthographic": "ACCENT_PUNCTUATION",
        },
    })
    db, run_id = init_run(tmp_path, [value])
    first = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    assert first["input"]["variant_contract"] == [
        {"variant_id": "v1", "target_dimension": "name", "requested_family": "TOKEN_ORDER"},
        {"variant_id": "v2", "target_dimension": "address", "requested_family": "ADDRESS_ABBREVIATION"},
        {"variant_id": "v3", "target_dimension": "orthographic", "requested_family": "ACCENT_PUNCTUATION"},
    ]
    response = generator_response(first)
    response["variants"][2]["crm"] = dict(response["variants"][0]["crm"])
    submit(tmp_path, db, "GENERATOR", "luna-g1", [response])
    retry = lease(tmp_path, db, run_id, "GENERATOR", "luna-g2")[0]
    assert retry["input"]["retry_context"]["previous_preflight_errors"] == [
        "DUPLICATE_OR_COSMETIC_VARIANTS"
    ]
    assert len(retry["input"]["retry_context"]["previous_fingerprints_to_avoid"]) == 3


def test_forbidden_fold_and_unsafe_targeted_critic_are_fail_closed(tmp_path: Path):
    invalid = seed()
    invalid["oof_fold"] = 1
    with pytest.raises(ValueError, match="forbidden fold"):
        init_run(tmp_path, [invalid])
    with pytest.raises(ValueError, match="targeted critic"):
        init_run(tmp_path, [seed()], run_id="run-risk", extra=["--critic-mode", "risk"])


def test_v1_audited_request_cards_can_initialize_the_v2_loop(tmp_path: Path):
    value = seed()
    legacy = {
        "schema_version": "sireto-synthetic-gt-agentic-request-1",
        "seed": {
            "siret": value["target_siret"],
            "siren": value["target_siren"],
            "oof_fold": 2,
            "seed_source": "CRM_OK_GT_TRAIN",
        },
        "seed_card": value["seed_card"],
        "observed_train_profile": value["observed_train_profile"],
    }
    db, run_id = init_run(tmp_path, [legacy], run_id="legacy-run")
    task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    assert task["input"]["seed"] == {"siret": value["target_siret"], "siren": value["target_siren"]}


def test_global_duplicate_agent_text_is_retried_not_silently_duplicated(tmp_path: Path):
    db, run_id = init_run(tmp_path, [seed("seed-1"), seed("seed-2", "98765432100019")])
    tasks = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1", limit=2)
    submit(tmp_path, db, "GENERATOR", "luna-g1", [generator_response(task) for task in tasks])
    with sqlite3.connect(db) as connection:
        statuses = [row[0] for row in connection.execute("SELECT status FROM seeds ORDER BY seed_id")]
    assert statuses == ["PENDING_CRITIC", "PENDING_GENERATOR"]


def test_adjudicator_can_resolve_silver_but_never_promote_critic_reject(tmp_path: Path):
    db, run_id = init_run(tmp_path, [seed()])
    generator_task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    submit(tmp_path, db, "GENERATOR", "luna-g1", [generator_response(generator_task)])
    critic_task = lease(tmp_path, db, run_id, "CRITIC", "luna-c1")[0]
    critic_decisions = [
        {"variant_id": "v1", "decision": "SILVER", "confidence": 0.7, "reason_codes": ["REVIEW"], "reason": "Doute résiduel."},
        {"variant_id": "v2", "decision": "ACCEPT", "confidence": 0.99, "reason_codes": ["PASS"], "reason": "Identité conservée."},
        {"variant_id": "v3", "decision": "REJECT", "confidence": 0.99, "reason_codes": ["SIBLING"], "reason": "Meilleur sibling possible."},
    ]
    submit(tmp_path, db, "CRITIC", "luna-c1", [review_response(critic_task, "CRITIC", critic_decisions)])
    adjudicator_task = lease(tmp_path, db, run_id, "ADJUDICATOR", "luna-a1")[0]
    submit(tmp_path, db, "ADJUDICATOR", "luna-a1", [review_response(adjudicator_task, "ADJUDICATOR")])
    loop.main(["--db", str(db), "supervise", "--run-id", run_id])
    with sqlite3.connect(db) as connection:
        values = dict(connection.execute("SELECT variant_id, final_decision FROM variants"))
    assert values == {"v1": "ACCEPT", "v2": "ACCEPT", "v3": "REJECT"}


def test_orchestrator_contains_no_mechanical_text_generator():
    source = Path(loop.__file__).read_text(encoding="utf-8")
    forbidden = ["import random", "from random", "faker", "transform_variant", "deterministic_rng"]
    assert all(token not in source for token in forbidden)
