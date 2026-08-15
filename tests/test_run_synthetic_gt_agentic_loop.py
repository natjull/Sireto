from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts import run_synthetic_gt_agentic_loop as loop


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(loop.canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def test_punctuation_edits_distinguish_identical_marks_by_position() -> None:
    source = "'NATUREL ET GOURMANDISE'"
    assert loop.composite_operation_parameters(
        "name", "PUNCTUATION_REMOVED", source, "'NATUREL ET GOURMANDISE"
    ) == {"edits": [{"after_token_index": 2, "mark": "'"}]}
    assert loop.composite_operation_parameters(
        "name", "PUNCTUATION_REMOVED", source, "NATUREL ET GOURMANDISE'"
    ) == {"edits": [{"after_token_index": -1, "mark": "'"}]}
    assert loop.composite_operation_parameters(
        "name", "PUNCTUATION_REMOVED", source, "NATUR'EL ET GOURMANDISE"
    ) is None


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


def composite_seed() -> dict:
    value = seed()
    baseline = {
        "name": "MAISON RETRAITE DES FLEURS",
        "address": "12 RUE DES LILAS",
        "postcode": "75001",
        "city": "SAINT-DENIS",
        "insee": "93066",
    }
    inspirations = []
    masks = {
        "v1": ["name", "address"],
        "v2": ["name", "city"],
        "v3": ["name", "address", "city"],
    }
    relations = {
        "v1": {"name": "TOKEN_SUBSET", "address": "ADDRESS_ABBREVIATE"},
        "v2": {"name": "TOKEN_ORDER", "city": "PUNCTUATION_REMOVED"},
        "v3": {"name": "TOKEN_SUBSET", "address": "ADDRESS_ABBREVIATE", "city": "PUNCTUATION_REMOVED"},
    }
    for index, (variant_id, fields) in enumerate(masks.items(), 1):
        ref = f"{index:064x}"
        observed = dict(baseline)
        if relations[variant_id]["name"] == "TOKEN_ORDER":
            observed["name"] = "FLEURS MAISON RETRAITE DES"
        else:
            observed["name"] = "MAISON DES FLEURS"
        if "address" in fields:
            observed["address"] = "12 R DES LILAS"
        if "city" in fields:
            observed["city"] = "SAINT DENIS"
        inspiration = {
            "inspiration_ref": ref,
            "source_fold": 2,
            "structural_signature": {"changed_fields": fields, "missing_fields": []},
            "official": baseline,
            "observed_crm": observed,
        }
        inspirations.append({
            "variant_id": variant_id,
            "requested_family": loop.COMPOSITE_FAMILY,
            "target_fields": fields,
            "field_relations": relations[variant_id],
            "inspiration_ref": ref,
            "inspiration": inspiration,
        })
    value.update({"source_kind": "SIRENE_ONLY_TRAIN", "oof_fold": -1})
    value["seed_card"] = {
        "generation_mode": "OBSERVED_COMPOSITE_ANALOGY_V2",
        "name_options": [baseline["name"]],
        "enseigne_options": [],
        "address": baseline["address"],
        "postcode": baseline["postcode"],
        "city": baseline["city"],
        "insee": baseline["insee"],
        "street_number": "12",
        "street_type": "RUE",
        "composite_contracts": inspirations,
        "official_context": {"target": {}, "official_context": []},
        "internal_context": [{
            "siret": "99999999900019", "siren": "999999999", "record_sha256": "a" * 64,
            "usual_name": "AUTRE ENTREPRISE", "enseigne1": "", "enseigne2": "", "enseigne3": "",
            "number": "99", "repetition_index": "", "street_type": "RUE", "street": "AILLEURS",
            "postcode": "75001", "city": "SAINT-DENIS", "insee": "93066",
        }],
        "qualification": {
            "pre_generation_exact_eligible": True,
            "siblings_complete": True,
            "same_address_complete": True,
            "same_name_geography_complete": True,
        },
    }
    value["observed_train_profile"] = {
        "rows": 299,
        "supported_families": [loop.COMPOSITE_FAMILY],
        "source_sha256": "b" * 64,
    }
    return value


def composite_response(task: dict) -> dict:
    variants = [
        {
            "variant_id": "v1",
            "crm": {"name": "MAISON DES FLEURS", "address": "12 R DES LILAS", "postcode": "75001", "city": "SAINT-DENIS", "insee": "93066"},
            "corruption_families_observed": [loop.COMPOSITE_FAMILY],
            "transformation_summary": "Nom raccourci et type de voie abrégé par analogie train.",
        },
        {
            "variant_id": "v2",
            "crm": {"name": "RETRAITE DES FLEURS MAISON", "address": "12 RUE DES LILAS", "postcode": "75001", "city": "SAINT DENIS", "insee": "93066"},
            "corruption_families_observed": [loop.COMPOSITE_FAMILY],
            "transformation_summary": "Ordre du nom et séparateur communal dégradés.",
        },
        {
            "variant_id": "v3",
            "crm": {"name": "FLEURS MAISON RETRAITE", "address": "12 R DES LILAS", "postcode": "75001", "city": "SAINT DENIS", "insee": "93066"},
            "corruption_families_observed": [loop.COMPOSITE_FAMILY],
            "transformation_summary": "Dégradation composite des trois champs.",
        },
    ]
    return {
        "schema_version": loop.SCHEMA_VERSION,
        "task_id": task["task_id"], "run_id": task["run_id"], "batch_id": task["batch_id"],
        "role": "GENERATOR", "prompt_version": task["prompt_version"],
        "input_sha256": task["input_sha256"], "seed": task["input"]["seed"], "variants": variants,
    }


def composite_variant_response(task: dict) -> dict:
    response = composite_response(task)
    target_variant_id = task["input"]["target_variant_id"]
    response["variants"] = [
        value for value in response["variants"] if value["variant_id"] == target_variant_id
    ]
    return response


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
    assert critic_task["input"]["baseline_crm"]["name"] == ""
    assert critic_task["input"]["variant_contract"] == []
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


def test_composite_task_hides_raw_competitor_ids_and_passes_fail_closed_preflight(tmp_path: Path):
    db, run_id = init_run(tmp_path, [composite_seed()], run_id="composite")
    task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    encoded = loop.canonical_json(task)
    assert "99999999900019" not in encoded
    assert "internal_context" not in task["input"]["seed_card"]
    assert len(task["input"]["variant_contract"]) == 3
    submit(tmp_path, db, "GENERATOR", "luna-g1", [composite_response(task)])
    with sqlite3.connect(db) as connection:
        status = connection.execute("SELECT status FROM seeds").fetchone()[0]
        preflights = [json.loads(row[0]) for row in connection.execute("SELECT preflight_json FROM variants")]
    assert status == "PENDING_CRITIC"
    assert all(value["passed"] for value in preflights)


def test_deterministic_proof_uses_canonical_hyphen_tokenization() -> None:
    card = {
        "name_options": ["GOPA-HAIDER CONFECTION"],
        "address": "11 IMP LISE DE HARME",
        "postcode": "95350",
        "city": "SAINT-BRICE-SOUS-FORET",
        "insee": "95539",
        "composite_contracts": [{
            "variant_id": "v1",
            "requested_family": loop.COMPOSITE_FAMILY,
            "target_fields": ["name"],
            "field_relations": {"name": "TOKEN_ORDER"},
            "field_inspirations": {"name": {
                "operation_parameters": {
                    "source_token_count": 3,
                    "permutation": [1, 2, 0],
                }
            }},
        }],
    }
    crm = {
        "name": "HAIDER CONFECTION GOPA",
        "address": "11 IMP LISE DE HARME",
        "postcode": "95350",
        "city": "SAINT-BRICE-SOUS-FORET",
        "insee": "95539",
    }
    proof = loop.deterministic_variant_proof(
        card, "v1", crm, {"passed": True, "errors": []}
    )
    assert proof["fields"]["name"]["source_tokens"] == [
        "gopa", "haider", "confection",
    ]
    assert proof["fields"]["name"]["operator_match"] is True


def test_duplicate_fingerprint_preserves_meaningful_surface_differences() -> None:
    first = {
        "name": "GLOUE LOCATION RAVINE DES CABRIS",
        "address": "14 CHE RANGAMA RAV DES CABRIS",
        "postcode": "97410", "city": "SAINT-PIERRE", "insee": "97416",
    }
    second = {
        "name": "G'LOUE LOCATION RAVINE DES CABRIS",
        "address": "14 CHE RANGAMA-RAV DES CABRIS",
        "postcode": "97410", "city": "SAINT PIERRE", "insee": "97416",
    }
    assert loop.comparison_fingerprint(first) == loop.comparison_fingerprint(second)
    assert loop.surface_fingerprint(first) != loop.surface_fingerprint(second)


def test_per_variant_generator_retries_only_failed_slot_then_assembles_critic_input(tmp_path: Path):
    db, run_id = init_run(
        tmp_path,
        [composite_seed()],
        run_id="composite-per-variant",
        extra=["--generator-task-mode", "per-variant", "--max-generator-attempts", "2"],
    )

    v1_task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    assert v1_task["input"]["target_variant_id"] == "v1"
    assert [value["variant_id"] for value in v1_task["input"]["variant_contract"]] == ["v1"]
    assert len(v1_task["input"]["seed_card"]["composite_contracts"]) == 1
    v1_response = composite_variant_response(v1_task)
    submit(tmp_path, db, "GENERATOR", "luna-g1", [v1_response])

    v2_bad_task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g2")[0]
    assert v2_bad_task["input"]["target_variant_id"] == "v2"
    v2_bad_response = composite_variant_response(v2_bad_task)
    v2_bad_response["variants"][0]["crm"]["name"] += " BIDON"
    submit(tmp_path, db, "GENERATOR", "luna-g2", [v2_bad_response])

    v2_retry_task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g3")[0]
    assert v2_retry_task["input"]["target_variant_id"] == "v2"
    assert any(
        "NAME_NEW_ALPHANUMERIC_MATERIAL" in value
        for value in v2_retry_task["input"]["retry_context"]["previous_preflight_errors"]
    )
    submit(tmp_path, db, "GENERATOR", "luna-g3", [composite_variant_response(v2_retry_task)])

    v3_task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g4")[0]
    assert v3_task["input"]["target_variant_id"] == "v3"
    submit(tmp_path, db, "GENERATOR", "luna-g4", [composite_variant_response(v3_task)])

    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute("SELECT status FROM seeds").fetchone()[0] == "PENDING_CRITIC"
        slots = connection.execute(
            "SELECT variant_id, status, attempt, accepted_task_id FROM generator_slots ORDER BY variant_id"
        ).fetchall()
        assert [(row["variant_id"], row["status"], row["attempt"]) for row in slots] == [
            ("v1", "PASSED", 1), ("v2", "PASSED", 2), ("v3", "PASSED", 1)
        ]
        assert all(row["accepted_task_id"] for row in slots)
        generator_tasks = connection.execute(
            "SELECT variant_id, raw_response, response_sha256 FROM tasks WHERE role='GENERATOR' ORDER BY created_at, task_id"
        ).fetchall()
        assert len(generator_tasks) == 4
        assert all(
            row["response_sha256"] == loop.hashlib.sha256(row["raw_response"].encode("utf-8")).hexdigest()
            for row in generator_tasks
        )

    critic_task = lease(tmp_path, db, run_id, "CRITIC", "luna-c1")[0]
    assert [value["variant_id"] for value in critic_task["input"]["variants"]] == ["v1", "v2", "v3"]
    for value in critic_task["input"]["variants"]:
        proof = value["deterministic_proof"]
        assert proof["preflight_passed"] is True
        assert proof["canonical_tokenizer"] == "normalized_words_v1_hyphen_is_separator"
        assert proof["proof_sha256"]
        assert proof["fields"]
        assert all(field["operator_match"] for field in proof["fields"].values())


def test_per_variant_mode_rejects_seed_without_exact_variant_contracts(tmp_path: Path):
    with pytest.raises(ValueError, match="requires exact v1/v2/v3 contracts"):
        init_run(
            tmp_path,
            [seed()],
            run_id="per-variant-without-contract",
            extra=["--generator-task-mode", "per-variant"],
        )


def test_per_variant_exhaustion_never_sends_partial_seed_to_critic(tmp_path: Path):
    db, run_id = init_run(
        tmp_path,
        [composite_seed()],
        run_id="composite-exhausted",
        extra=["--generator-task-mode", "per-variant", "--max-generator-attempts", "1"],
    )
    for expected in ("v1", "v2", "v3"):
        task = lease(tmp_path, db, run_id, "GENERATOR", f"luna-{expected}")[0]
        assert task["input"]["target_variant_id"] == expected
        response = composite_variant_response(task)
        if expected == "v1":
            response["variants"][0]["crm"]["name"] += " BIDON"
        submit(tmp_path, db, "GENERATOR", f"luna-{expected}", [response])

    assert lease(tmp_path, db, run_id, "CRITIC", "luna-c1") == []
    loop.main(["--db", str(db), "supervise", "--run-id", run_id])
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT status FROM seeds").fetchone()[0] == "COMPLETED"
        decisions = connection.execute(
            "SELECT variant_id, final_decision, final_reason FROM variants ORDER BY variant_id"
        ).fetchall()
    assert decisions == [
        ("v1", "REJECT", "SEED_GENERATION_INCOMPLETE"),
        ("v2", "REJECT", "SEED_GENERATION_INCOMPLETE"),
        ("v3", "REJECT", "SEED_GENERATION_INCOMPLETE"),
    ]


def test_composite_preflight_rejects_case_only_added_tokens_and_marks(tmp_path: Path):
    db, run_id = init_run(tmp_path, [composite_seed()], run_id="composite-bad")
    task = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    response = composite_response(task)
    response["variants"][0]["crm"]["name"] = "maison retraite des fleurs"
    response["variants"][1]["crm"]["name"] = "MAISON RETRAITE DES FLEURS BIDON"
    response["variants"][2]["crm"]["city"] = "SAINT.DENIS"
    submit(tmp_path, db, "GENERATOR", "luna-g1", [response])
    with sqlite3.connect(db) as connection:
        status = connection.execute("SELECT status FROM seeds").fetchone()[0]
        errors = json.loads(connection.execute("SELECT preflight_json FROM variants LIMIT 1").fetchone()[0])["errors"]
    assert status == "PENDING_GENERATOR"
    assert any("TARGET_UNCHANGED_OR_CASE_ONLY" in error for error in errors)
    assert any("NAME_NEW_ALPHANUMERIC_MATERIAL" in error for error in errors)
    assert any("CITY_ADDED_MARK_FORBIDDEN" in error for error in errors)


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


def test_max_generator_attempts_is_total_attempts_not_retries(tmp_path: Path):
    value = seed()
    db, run_id = init_run(
        tmp_path, [value], extra=["--max-generator-attempts", "2"]
    )
    first = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    submit(tmp_path, db, "GENERATOR", "luna-g1", [
        generator_response(first, leaked_name=value["target_siret"])
    ])
    second = lease(tmp_path, db, run_id, "GENERATOR", "luna-g2")[0]
    submit(tmp_path, db, "GENERATOR", "luna-g2", [
        generator_response(second, leaked_name=value["target_siret"])
    ])
    with sqlite3.connect(db) as connection:
        status, attempt = connection.execute(
            "SELECT status, attempt FROM seeds WHERE run_id=?", (run_id,)
        ).fetchone()
    assert (status, attempt) == ("READY_SUPERVISOR", 2)
    assert lease(tmp_path, db, run_id, "GENERATOR", "luna-g3") == []


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
        seed_status, attempt = connection.execute(
            "SELECT status, attempt FROM seeds"
        ).fetchone()
        task_status = connection.execute("SELECT status FROM tasks").fetchone()[0]
    assert seed_status == "PENDING_GENERATOR"
    assert attempt == 0
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
    value["observed_train_profile"] = {
        "rows": 100,
        "phenomena": {
            "TOKEN_ORDER": 10,
            "ADDRESS_ABBREVIATION": 20,
            "ACCENT_PUNCTUATION": 30,
        },
        "supported_families": [
            "TOKEN_ORDER", "ADDRESS_ABBREVIATION", "ACCENT_PUNCTUATION"
        ],
    }
    db, run_id = init_run(tmp_path, [value])
    first = lease(tmp_path, db, run_id, "GENERATOR", "luna-g1")[0]
    assert first["input"]["variant_contract"] == [
        {"variant_id": "v1", "target_dimension": "name", "target_fields": ["name"], "requested_family": "TOKEN_ORDER"},
        {"variant_id": "v2", "target_dimension": "address", "target_fields": ["address"], "requested_family": "ADDRESS_ABBREVIATION"},
        {"variant_id": "v3", "target_dimension": "orthographic", "target_fields": ["name"], "requested_family": "ACCENT_PUNCTUATION"},
    ]
    assert first["input"]["baseline_crm"] == {
        "name": "Société des Fleurs",
        "address": "12 RUE DES LILAS 75001 PARIS",
        "postcode": "75001",
        "city": "PARIS",
        "insee": "75056",
    }
    response = generator_response(first)
    for variant, family in zip(response["variants"], (
        "TOKEN_ORDER", "ADDRESS_ABBREVIATION", "ACCENT_PUNCTUATION"
    )):
        variant["corruption_families_observed"] = [family]
    response["variants"][2]["crm"] = dict(response["variants"][0]["crm"])
    submit(tmp_path, db, "GENERATOR", "luna-g1", [response])
    retry = lease(tmp_path, db, run_id, "GENERATOR", "luna-g2")[0]
    assert "DUPLICATE_OR_COSMETIC_VARIANTS" in retry["input"]["retry_context"][
        "previous_preflight_errors"
    ]
    assert len(retry["input"]["retry_context"]["previous_fingerprints_to_avoid"]) == 3


def test_family_semantic_checks_reject_false_claims_and_accept_real_changes():
    card = {
        "name_options": ["SOCIETE DES FLEURS"],
        "enseigne_options": ["FLEURS DE PARIS"],
        "ocr_substitution_pairs": [{"source": "o", "target": "0", "count": 1}],
    }
    assert loop.family_change_errors("OCR_LIMITED", "SOCIETE", "societe", card) == [
        "TARGET_FIELD_UNCHANGED"
    ]
    assert "ACCENT_PUNCTUATION_ADDED_GRATUITOUS_MARK" in loop.family_change_errors(
        "ACCENT_PUNCTUATION", "L'ÉTOILE", "L'ÉTOILE.", card
    )
    assert loop.family_change_errors(
        "ENSEIGNE_VS_DENOMINATION", "SOCIETE DES FLEURS", "SOCIETE-DES-FLEURS", card
    ) == ["NOT_AN_OFFICIAL_ALTERNATE_NAME"]
    assert loop.family_change_errors("OCR_LIMITED", "SOCIETE", "S0CIETE", card) == []
    assert loop.family_change_errors("ACCENT_PUNCTUATION", "L'ÉTOILE", "L ETOILE", card) == []
    assert "ACCENT_PUNCTUATION_ADDED_GRATUITOUS_MARK" in loop.family_change_errors(
        "ACCENT_PUNCTUATION", "ETOILE-BLEUE", "ÉTOILE BLEUE", card
    )
    assert loop.family_change_errors(
        "ADDRESS_ABBREVIATION", "12 RUE DES LILAS", "12 R DES LILAS", card
    ) == []
    assert loop.family_change_errors(
        "ENSEIGNE_VS_DENOMINATION", "SOCIETE DES FLEURS", "FLEURS DE PARIS", card
    ) == []
    dotted = {
        "name": "G.D. FERMETURES", "address": "1 RUE A", "postcode": "57000",
        "city": "METZ", "insee": "57463",
    }
    compact = dict(dotted, name="GD FERMETURES")
    assert loop.comparison_fingerprint(dotted) == loop.comparison_fingerprint(compact)
    assert loop.surface_fingerprint(dotted) != loop.surface_fingerprint(compact)


def test_source_family_feasibility_rejects_fake_abbreviation_and_generic_hyphen():
    base = {
        "name_options": ["NEPTUNE-SERVICES"],
        "address": "12 LOT DES TILLEULS",
        "street_type": "LOT",
        "city": "PARIS",
        "postcode": "75001",
        "insee": "75056",
        "requested_families": {
            "name": "ACRONYM_TOKENIZATION",
            "address": "ADDRESS_ABBREVIATION",
            "orthographic": "ACCENT_PUNCTUATION",
        },
    }
    assert not loop.source_supports_family(base, "name", "ACRONYM_TOKENIZATION")
    assert not loop.source_supports_family(base, "address", "ADDRESS_ABBREVIATION")
    base["name_options"] = ["J.O.B. ELEC"]
    base["address"] = "12 RUE DES TILLEULS"
    base["street_type"] = "RUE"
    assert loop.source_supports_family(base, "name", "ACRONYM_TOKENIZATION")
    assert loop.source_supports_family(base, "address", "ADDRESS_ABBREVIATION")


def test_ocr_requires_observed_substitution_and_preserves_address_number():
    card = {
        "ocr_substitution_pairs": [{"source": "o", "target": "0", "count": 1}],
        "address_ocr_substitution_pairs": [
            {"source": "i", "target": "l", "count": 1}
        ],
    }
    assert loop.family_change_errors("OCR_LIMITED", "SOCIETE", "S0CIETE", card) == []
    assert "OCR_SUBSTITUTION_NOT_LIMITED_OR_ABSENT" in loop.family_change_errors(
        "OCR_LIMITED", "SOCIETE", "SACIETE", card
    )
    assert "OCR_SUBSTITUTION_NOT_LIMITED_OR_ABSENT" in loop.family_change_errors(
        "OCR_LIMITED", "SOCIETE", "SOCETE", card
    )
    assert "ADDRESS_OCR_CHANGED_NUMBER" in loop.family_change_errors(
        "ADDRESS_OCR", "12 RUE LILAS", "13 RUE LLLAS", card
    )


def test_enseigne_feasibility_uses_comparison_fingerprint():
    card = {
        "name_options": ["L'ETOILE"],
        "enseigne_options": ["L ETOILE"],
        "address": "1 RUE A", "street_type": "RUE", "city": "PARIS",
        "postcode": "75001", "insee": "75056",
        "requested_families": {
            "name": "ENSEIGNE_VS_DENOMINATION",
            "address": "ADDRESS_ABBREVIATION",
            "orthographic": "ACCENT_PUNCTUATION",
        },
    }
    assert not loop.source_supports_family(
        card, "name", "ENSEIGNE_VS_DENOMINATION"
    )


def test_requested_family_requires_nonempty_observed_train_evidence(tmp_path: Path):
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
    value["observed_train_profile"] = {
        "rows": 0,
        "phenomena": {},
        "supported_families": [
            "TOKEN_ORDER", "ADDRESS_ABBREVIATION", "ACCENT_PUNCTUATION"
        ],
    }
    with pytest.raises(ValueError, match="observed train profile has no rows"):
        init_run(tmp_path, [value])


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
