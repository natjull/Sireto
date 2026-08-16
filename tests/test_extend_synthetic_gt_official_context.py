import json

from scripts.extend_synthetic_gt_official_context import (
    excluded_seed_ids,
    merge_jsonl,
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
