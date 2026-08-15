from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import prepare_synthetic_gt_agentic_contracts as prepare
from scripts import run_synthetic_gt_agentic_loop as loop


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def candidate() -> dict:
    return {
        "source_siret": "12345678900012",
        "source_siren": "123456789",
        "official_fields": {
            "siret": "12345678900012",
            "siren": "123456789",
            "state": "A",
            "name": "SOCIÉTÉ L'ÉTOILE",
            "enseigne": "L'ÉTOILE FLEURIE",
            "street_number": "12",
            "street_type": "RUE",
            "street": "DE L'ÉTOILE",
            "postcode": "75001",
            "city": "PARIS",
            "insee": "75056",
        },
        "selection_eligibility": {},
    }


def profile() -> dict:
    return {
        "rows": 100,
        "fields": {},
        "phenomena": {
            "ACCENT_PUNCTUATION": 20,
            "ADDRESS_ABBREVIATION": 30,
        },
        "supported_families": ["ACCENT_PUNCTUATION", "ADDRESS_ABBREVIATION"],
    }


def test_preparer_copies_official_fields_and_luna_contract_only(tmp_path: Path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(loop.canonical_json(candidate()) + "\n", encoding="utf-8")
    assignments = tmp_path / "assignments.json"
    write_json(assignments, {
        "selections": [{
            "siret": "12345678900012",
            "requested_families": {
                "name": "ACCENT_PUNCTUATION",
                "address": "ADDRESS_ABBREVIATION",
                "orthographic": "ACCENT_PUNCTUATION",
            },
        }]
    })
    observed = tmp_path / "profile.json"
    write_json(observed, profile())
    output, manifest = tmp_path / "seeds.jsonl", tmp_path / "manifest.json"
    prepare.main([
        "--candidates", str(candidates), "--assignments", str(assignments),
        "--profile", str(observed), "--output", str(output),
        "--manifest", str(manifest), "--count", "1", "--seed-prefix", "v17",
    ])
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["seed_card"]["name_options"] == ["SOCIÉTÉ L'ÉTOILE"]
    assert row["seed_card"]["enseigne_options"] == ["L'ÉTOILE FLEURIE"]
    assert row["seed_card"]["address"] == "12 RUE DE L'ÉTOILE"
    assert row["seed_card"]["requested_families"] == json.loads(
        assignments.read_text(encoding="utf-8")
    )["selections"][0]["requested_families"]
    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["seed_count"] == report["distinct_siret"] == report["distinct_siren"] == 1


def test_preparer_rejects_nonofficial_luna_selection(tmp_path: Path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(loop.canonical_json(candidate()) + "\n", encoding="utf-8")
    assignments = tmp_path / "assignments.json"
    write_json(assignments, {
        "selections": [{
            "siret": "98765432100019",
            "requested_families": {
                "name": "ACCENT_PUNCTUATION",
                "address": "ADDRESS_ABBREVIATION",
                "orthographic": "ACCENT_PUNCTUATION",
            },
        }]
    })
    observed = tmp_path / "profile.json"
    write_json(observed, profile())
    with pytest.raises(ValueError, match="non-official"):
        prepare.main([
            "--candidates", str(candidates), "--assignments", str(assignments),
            "--profile", str(observed), "--output", str(tmp_path / "out.jsonl"),
            "--manifest", str(tmp_path / "manifest.json"), "--count", "1",
        ])


def test_preparer_contains_no_mechanical_crm_generator():
    source = Path(prepare.__file__).read_text(encoding="utf-8")
    forbidden = ["import random", "from random", "faker", "transform_variant", "deterministic_rng"]
    assert all(token not in source for token in forbidden)
