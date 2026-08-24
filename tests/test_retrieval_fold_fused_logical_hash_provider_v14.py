from __future__ import annotations

from dataclasses import replace

import pyarrow as pa
import pytest

from xgb_matcher.retrieval_fold_fused_fragment_dataset_v13 import (
    logical_table_sha256_v13,
)
from xgb_matcher import retrieval_fold_fused_logical_hash_provider_v14 as provider
from xgb_matcher.retrieval_fold_fused_logical_hash_v14 import (
    LOGICAL_TABLE_HASH_ALGORITHM_ID_V14,
    LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14,
    logical_table_sha256_v14,
)


def _table() -> pa.Table:
    return pa.table(
        {
            "name": pa.array(["alpha", None, "omega"], type=pa.string()),
            "rank": pa.array([7, None, -3], type=pa.int16()),
            "active": pa.array([True, None, False], type=pa.bool_()),
        }
    )


def test_canonical_providers_select_exact_historical_and_v14_hashes() -> None:
    table = _table()
    assert provider.V13_LOGICAL_HASH_PROVIDER.table_sha256(table) == (
        logical_table_sha256_v13(table)
    )
    assert provider.V14_LOGICAL_HASH_PROVIDER.table_sha256(table) == (
        logical_table_sha256_v14(table)
    )
    assert provider.require_logical_hash_provider_v14(
        provider.V14_LOGICAL_HASH_PROVIDER
    ) is provider.V14_LOGICAL_HASH_PROVIDER


def test_v14_provider_lineage_names_algorithm_schema_callable_and_self_test() -> None:
    record = provider.logical_hash_lineage_record(
        provider.V14_LOGICAL_HASH_PROVIDER
    )
    assert set(record) == {
        "revision",
        "algorithm_id",
        "algorithm_sha256",
        "record_schema_id",
        "callable_fingerprint_sha256",
        "self_test_sha256",
    }
    assert record["revision"] == 14
    assert record["algorithm_id"] == LOGICAL_TABLE_HASH_ALGORITHM_ID_V14
    assert record["record_schema_id"] == LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14
    assert len(str(record["algorithm_sha256"])) == 64
    assert len(str(record["callable_fingerprint_sha256"])) == 64
    assert len(str(record["self_test_sha256"])) == 64


@pytest.mark.parametrize(
    "mutation",
    (
        {"revision": 13},
        {"algorithm_id": "forged-v14"},
        {"algorithm_sha256": "0" * 64},
        {"_hasher": logical_table_sha256_v14},
        {"_capability": object()},
    ),
)
def test_dataclass_lookalike_provider_is_rejected(mutation: dict[str, object]) -> None:
    clone = replace(provider.V14_LOGICAL_HASH_PROVIDER, **mutation)
    with pytest.raises(provider.LogicalHashProviderInvariantError):
        provider.require_logical_hash_provider_v14(clone)
    with pytest.raises(provider.LogicalHashProviderInvariantError):
        clone.table_sha256(_table())


def test_exact_dataclass_clone_is_rejected() -> None:
    clone = replace(provider.V14_LOGICAL_HASH_PROVIDER)
    assert clone is not provider.V14_LOGICAL_HASH_PROVIDER
    with pytest.raises(provider.LogicalHashProviderInvariantError):
        provider.require_logical_hash_provider_v14(clone)


def test_rebinding_public_alias_cannot_replace_private_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = provider.V14_LOGICAL_HASH_PROVIDER
    clone = replace(canonical)
    monkeypatch.setattr(provider, "V14_LOGICAL_HASH_PROVIDER", clone)
    with pytest.raises(provider.LogicalHashProviderInvariantError):
        provider.require_logical_hash_provider_v14(
            provider.V14_LOGICAL_HASH_PROVIDER
        )
    assert provider.require_logical_hash_provider_v14(canonical) is canonical


def test_v13_provider_cannot_enter_v14_protocol() -> None:
    with pytest.raises(provider.LogicalHashProviderInvariantError):
        provider.require_logical_hash_provider_v14(
            provider.V13_LOGICAL_HASH_PROVIDER
        )


def test_runtime_callable_code_mutation_is_detected_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forged(_table: pa.Table) -> str:
        return "0" * 64

    monkeypatch.setattr(logical_table_sha256_v14, "__code__", forged.__code__)
    with pytest.raises(provider.LogicalHashProviderInvariantError):
        provider.require_logical_hash_provider_v14(
            provider.V14_LOGICAL_HASH_PROVIDER
        )
