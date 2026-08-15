from __future__ import annotations

import json
from pathlib import Path

from scripts import run_synthetic_gt_agentic_loop as loop
from scripts import select_synthetic_gt_composite_pilot as selector


def context(index: int, state: str, *, multi: bool = False, multi_active: bool = False) -> dict:
    siret = f"{100000000 + index:09d}00001"
    relatives = []
    if multi:
        relatives.append({
            "site_ref": f"CTX-{index}", "state": "A" if multi_active else "F",
            "relation_tags": ["SAME_SIREN"],
        })
    target = {
        "site_ref": "TARGET", "state": state,
        "names": [{"kind": "OFFICIAL_NAME", "value": f"MAISON DES FLEURS {index}"}],
        "address": {"number": "12", "repetition_index": "", "street_type": "RUE", "street": "DES LILAS", "postcode": "75001", "city": "SAINT-DENIS", "insee": "93066"},
    }
    return {
        "target_siret": siret, "target_siren": siret[:9], "target": target,
        "qualification": {"pre_generation_exact_eligible": True},
        "internal_context": relatives,
        "llm_view": {"target": target, "official_context": [], "context_summary": {}},
        "context_sha256": f"sha-{index}",
    }


def inspiration(index: int, fields: list[str]) -> dict:
    official = {
        "name": "MAISON DES FLEURS", "address": "12 RUE DES LILAS",
        "postcode": "75001", "city": "SAINT-DENIS", "insee": "93066",
    }
    observed = dict(official)
    observed.update({
        "name": "FLEURS MAISON", "address": "12 R DES LILAS", "city": "SAINT DENIS",
    })
    return {
        "inspiration_ref": f"ref-{'+'.join(fields)}-{index}", "source_fold": 2,
        "official": official, "observed_crm": observed,
        "structural_signature": {"changed_fields": fields, "missing_fields": []},
        "analogy_safety": {"lexical_tokens_subset_of_official": True, "numeric_tokens_subset_of_official": True, "added_marks": {}},
    }


def test_build_selects_balanced_stratified_pilot_without_generating_text(tmp_path: Path):
    contexts = [
        context(index, "A" if index < 15 else "F", multi=index < 12, multi_active=index < 4)
        for index in range(30)
    ]
    inspirations = [
        inspiration(index, fields)
        for fields in selector.CONTRACT_MASKS.values()
        for index in range(30)
    ]
    official = tmp_path / "contexts.jsonl"
    bank = tmp_path / "bank.jsonl"
    loop.write_jsonl_atomic(official, contexts)
    loop.write_jsonl_atomic(bank, inspirations)
    plan = {
        "sources": {
            "official_context": {"path": str(official), "sha256": selector.sha256(official)},
            "train_inspiration_bank": {"path": str(bank), "sha256": selector.sha256(bank)},
        },
        "pilot_strata": {"active_targets": 15, "closed_targets": 15, "minimum_multi_site_siren": 12, "minimum_multi_active_siren": 4},
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "pilot.jsonl"
    manifest = selector.build(type("Args", (), {"plan": plan_path, "output": output, "selection_seed": "test"})())
    selected = [value for _raw, value in loop.iter_jsonl_raw(output)]
    assert manifest["rows"] == 30
    assert manifest["planned_pairs"] == 90
    assert manifest["state_counts"] == {"A": 15, "F": 15}
    assert manifest["multi_site_targets"] >= 12
    assert manifest["multi_active_targets"] >= 4
    assert manifest["distinct_inspiration_refs"] == 90
    assert all(row["source_kind"] == "SIRENE_ONLY_TRAIN" for row in selected)
    assert all(row["seed_card"]["generation_mode"] == "OBSERVED_COMPOSITE_ANALOGY_V2" for row in selected)
    assert [c["target_fields"] for c in selected[0]["seed_card"]["composite_contracts"]] == list(selector.CONTRACT_MASKS.values())


def test_inspiration_groups_rejects_protected_fold_and_missing_fields():
    good = inspiration(1, ["name", "address"])
    bad_fold = inspiration(2, ["name", "address"])
    bad_fold["source_fold"] = 1
    bad_missing = inspiration(3, ["name", "address"])
    bad_missing["structural_signature"]["missing_fields"] = ["city"]
    grouped = selector.inspiration_groups([good, bad_fold, bad_missing])
    assert len(grouped[("name", "address")]) == 1
    assert grouped[("name", "address")][0]["inspiration_ref"] == good["inspiration_ref"]
    assert grouped[("name", "address")][0]["transfer_relations"] == {
        "name": "TOKEN_SUBSET", "address": "ADDRESS_ABBREVIATE",
    }
