import json

import pytest

from scripts import extend_synthetic_gt_official_context as extension
from scripts.extend_synthetic_gt_official_context import (
    excluded_seed_ids,
    context_is_simple_and_exact,
    bundle_capacity_profile,
    merge_jsonl,
    safe_bundle_capacity,
    select_snapshot_rows,
)


def test_excluded_seed_ids_accepts_production_and_source_identity(tmp_path) -> None:
    path = tmp_path / "seeds.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "target_siret": "12345678900012",
                "target_siren": "123456789",
            }),
            json.dumps({
                "source_siret": "98765432100019",
                "source_siren": "987654321",
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    sirets, sirens = excluded_seed_ids([path])
    assert sirets == {"12345678900012", "98765432100019"}
    assert sirens == {"123456789", "987654321"}


def test_merge_jsonl_preserves_base_bytes_and_adds_missing_newline(tmp_path) -> None:
    base = tmp_path / "base.jsonl"
    output = tmp_path / "extended.jsonl"
    base.write_bytes(b'{"base":1}')
    merge_jsonl(base, [{"extension": 2}], output)
    value = output.read_bytes()
    assert value.startswith(base.read_bytes() + b"\n")
    assert value == b'{"base":1}\n{"extension":2}\n'


def test_snapshot_selection_rejects_unknown_target_state(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported target state"):
        select_snapshot_rows(
            tmp_path / "establishments.parquet",
            tmp_path / "crm.csv",
            tmp_path / "context.jsonl",
            tmp_path / "candidates.jsonl",
            set(), 10, 10, "seed", tmp_path / "duckdb", "X",
        )


def test_closed_capacity_uses_largest_materializable_bundle(monkeypatch) -> None:
    monkeypatch.setattr(
        extension.production, "safe_capabilities",
        lambda *_args, **_kwargs: ({"name": [{}]}, {("address", "op"): [{}]}),
    )
    monkeypatch.setattr(
        extension.production, "candidate_bundles",
        lambda *_args, **_kwargs: [(1,), (1, 2, 3), (1, 2)],
    )
    capacity, _names, _locations = safe_bundle_capacity(
        {"target_siret": "12345678900012"}, {}, {}, "seed", {},
    )
    assert capacity == 3


def test_closed_context_allows_only_its_intrinsic_state_flag(monkeypatch) -> None:
    context = {
        "target": {"state": "F"},
        "qualification": {"pre_generation_exact_eligible": True},
    }
    monkeypatch.setattr(
        extension.production, "context_flags", lambda _value: {"CLOSED_TARGET"},
    )
    assert context_is_simple_and_exact(context, "F")
    monkeypatch.setattr(
        extension.production, "context_flags",
        lambda _value: {"CLOSED_TARGET", "SAME_ADDRESS_COMPETITION"},
    )
    assert not context_is_simple_and_exact(context, "F")


def test_closed_profile_exposes_hard_nonalias_and_pair_support(monkeypatch) -> None:
    hard = ("TOKEN_ORDER", ("address", "ADDRESS_TOKEN_SUBSET"))
    alias = ("OFFICIAL_NAME_ALIAS", ("address", "ADDRESS_ABBREVIATE"))
    nonalias = ("LEGAL_FORM_REMOVE", ("address", "ADDRESS_ABBREVIATE"))
    monkeypatch.setattr(
        extension.production, "safe_capabilities",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        extension.production, "candidate_bundles",
        lambda *_args, **_kwargs: [(hard, hard), (alias, nonalias, nonalias)],
    )
    profile = bundle_capacity_profile(
        {"target_siret": "12345678900012", "target": {"state": "F"},
         "internal_context": []},
        {}, {}, "seed", {},
    )
    assert profile["safe"] == 3
    assert profile["hard"] == 2
    assert profile["nonalias"] == 2
    assert profile["pair_support"] == sorted({
        extension.production.pair_signature(hard),
        extension.production.pair_signature(alias),
        extension.production.pair_signature(nonalias),
    })
