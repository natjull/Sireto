"""Explicit logical-table hash providers for the v13 engine and v14 runs.

The fragment engine predates the null-canonical v14 hashing rule.  This module
keeps the historical v13 behaviour as the default while giving v14 workers a
non-ambient, capability-bound provider.  Callers pass the provider through the
writer/sealer/loader stack; no global switch or monkeypatch can change a run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import marshal
from typing import Callable

import pyarrow as pa

from .retrieval_fold_fused_logical_hash_v14 import (
    LOGICAL_TABLE_HASH_ALGORITHM_ID_V14,
    LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14,
    logical_table_sha256_v14,
)


LOGICAL_TABLE_HASH_ALGORITHM_ID_V13 = (
    "SIRETO_ARROW_COMBINE_CHUNKS_IPC_SHA256_V13_1"
)


class LogicalHashProviderInvariantError(ValueError):
    """Raised when a fragment surface receives an untrusted hash provider."""


@dataclass(frozen=True)
class LogicalTableHashProvider:
    """A sealed in-process choice of logical table hash algorithm."""

    revision: int
    algorithm_id: str
    algorithm_sha256: str
    _hasher: Callable[[pa.Table], str]
    _capability: object

    def table_sha256(self, table: pa.Table) -> str:
        _require_provider(self)
        return self._hasher(table)


_V13_CAPABILITY = object()
_V14_CAPABILITY = object()


def _algorithm_sha256(algorithm_id: str) -> str:
    return hashlib.sha256(algorithm_id.encode("utf-8")).hexdigest()


def _historical_v13_table_sha256(table: pa.Table) -> str:
    """Historical v13 Arrow-IPC digest, kept byte-for-byte compatible."""

    canonical = pa.Table.from_arrays(
        [pa.concat_arrays(list(column.chunks)) for column in table.columns],
        schema=table.schema.remove_metadata(),
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, canonical.schema) as writer:
        writer.write_table(canonical, max_chunksize=max(1, canonical.num_rows))
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


_CANONICAL_V13_PROVIDER = LogicalTableHashProvider(
    revision=13,
    algorithm_id=LOGICAL_TABLE_HASH_ALGORITHM_ID_V13,
    algorithm_sha256=_algorithm_sha256(LOGICAL_TABLE_HASH_ALGORITHM_ID_V13),
    _hasher=_historical_v13_table_sha256,
    _capability=_V13_CAPABILITY,
)

_CANONICAL_V14_PROVIDER = LogicalTableHashProvider(
    revision=14,
    algorithm_id=LOGICAL_TABLE_HASH_ALGORITHM_ID_V14,
    algorithm_sha256=_algorithm_sha256(LOGICAL_TABLE_HASH_ALGORITHM_ID_V14),
    _hasher=logical_table_sha256_v14,
    _capability=_V14_CAPABILITY,
)

# Export aliases are deliberately not the authority used by ``_require``.
# Rebinding either public name cannot manufacture a capability.
V13_LOGICAL_HASH_PROVIDER = _CANONICAL_V13_PROVIDER
V14_LOGICAL_HASH_PROVIDER = _CANONICAL_V14_PROVIDER


def _callable_fingerprint(function: Callable[[pa.Table], str]) -> str:
    code = getattr(function, "__code__", None)
    if code is None:
        raise LogicalHashProviderInvariantError(
            "logical hash implementation has no inspectable code"
        )
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


_CANONICAL_CALLABLES = (
    (13, _historical_v13_table_sha256),
    (14, logical_table_sha256_v14),
)
_CANONICAL_CALLABLE_FINGERPRINTS = tuple(
    (revision, _callable_fingerprint(function))
    for revision, function in _CANONICAL_CALLABLES
)
_CANONICAL_PROVIDERS = (
    (13, _CANONICAL_V13_PROVIDER),
    (14, _CANONICAL_V14_PROVIDER),
)
_V14_SELF_TEST_SHA256 = (
    "6573db9eb89da9f9f986f61201305a4d9f22a988c1f10d174ed1180527e2c1f1"
)


def _verify_v14_self_test() -> None:
    vector = pa.table(
        {
            "s": pa.array(["alpha", None, "omega"], type=pa.string()),
            "n": pa.array([7, None, -3], type=pa.int16()),
            "b": pa.array([True, None, False], type=pa.bool_()),
        }
    )
    callable_v14 = dict(_CANONICAL_CALLABLES)[14]
    if callable_v14(vector) != _V14_SELF_TEST_SHA256:
        raise LogicalHashProviderInvariantError(
            "v14 logical hash implementation failed its sealed self-test"
        )


def _require_provider(
    provider: LogicalTableHashProvider,
    *,
    revision: int | None = None,
    _canonical_providers: tuple[tuple[int, LogicalTableHashProvider], ...] = (
        _CANONICAL_PROVIDERS
    ),
    _canonical_callables: tuple[
        tuple[int, Callable[[pa.Table], str]], ...
    ] = (
        _CANONICAL_CALLABLES
    ),
    _canonical_fingerprints: tuple[tuple[int, str], ...] = (
        _CANONICAL_CALLABLE_FINGERPRINTS
    ),
) -> LogicalTableHashProvider:
    if not isinstance(provider, LogicalTableHashProvider):
        raise LogicalHashProviderInvariantError("logical hash provider type drifted")
    expected = {
        13: (
            _V13_CAPABILITY,
            LOGICAL_TABLE_HASH_ALGORITHM_ID_V13,
            _historical_v13_table_sha256,
        ),
        14: (
            _V14_CAPABILITY,
            LOGICAL_TABLE_HASH_ALGORITHM_ID_V14,
            logical_table_sha256_v14,
        ),
    }.get(provider.revision)
    canonical = dict(_canonical_providers).get(provider.revision)
    canonical_callable = dict(_canonical_callables).get(provider.revision)
    if (
        expected is None
        or provider is not canonical
        or provider._capability is not expected[0]
        or provider.algorithm_id != expected[1]
        or provider._hasher is not expected[2]
        or provider._hasher is not canonical_callable
        or _callable_fingerprint(provider._hasher)
        != dict(_canonical_fingerprints).get(provider.revision)
        or provider.algorithm_sha256 != _algorithm_sha256(provider.algorithm_id)
        or (revision is not None and provider.revision != revision)
    ):
        raise LogicalHashProviderInvariantError("logical hash provider capability drifted")
    return provider


def require_logical_hash_provider_v14(
    provider: LogicalTableHashProvider,
) -> LogicalTableHashProvider:
    """Return only the canonical v14 provider; reject lookalike providers."""

    value = _require_provider(provider, revision=14)
    _verify_v14_self_test()
    return value


def logical_hash_lineage_record(
    provider: LogicalTableHashProvider,
) -> dict[str, object]:
    value = _require_provider(provider)
    return {
        "revision": value.revision,
        "algorithm_id": value.algorithm_id,
        "algorithm_sha256": value.algorithm_sha256,
        "record_schema_id": (
            LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14
            if value.revision == 14
            else "sireto-retrieval-logical-table-hash-record-v13-implicit"
        ),
        "callable_fingerprint_sha256": _callable_fingerprint(value._hasher),
        "self_test_sha256": (
            _V14_SELF_TEST_SHA256 if value.revision == 14 else None
        ),
    }


__all__ = [
    "LOGICAL_TABLE_HASH_ALGORITHM_ID_V13",
    "LogicalHashProviderInvariantError",
    "LogicalTableHashProvider",
    "V13_LOGICAL_HASH_PROVIDER",
    "V14_LOGICAL_HASH_PROVIDER",
    "logical_hash_lineage_record",
    "require_logical_hash_provider_v14",
]
