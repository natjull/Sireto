from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

from src.xgb_matcher import blocking
from src.xgb_matcher import features
from src.xgb_matcher import v412_unit_retrieval as unit


def candidate(
    siret: str,
    *,
    name: str = "ACME",
    state: str = "A",
    number: str = "12",
    street_type: str = "RUE",
    street: str = "DES FLEURS",
    **extra: object,
) -> dict[str, object]:
    return {
        "siret": siret,
        "siren": siret[:9],
        "denomination": name,
        "enseigne1": None,
        "enseigne2": None,
        "enseigne3": None,
        "numeroVoie": number,
        "typeVoie": street_type,
        "libelleVoie": street,
        "complementAdresse": None,
        "postcode": "75001",
        "city": "PARIS",
        "insee": "75056",
        "cj_ul": "5710",
        "etat_admin": state,
        "sigle_ul": None,
        "denomination_ul": None,
        "denomination_usuelle_ul": None,
        "nom_ul": None,
        "prenom_usuel_ul": None,
        "pm_dirigeant_names": None,
        **extra,
    }


class FakePartitionStore:
    def __init__(self, rows: list[dict[str, object]], key: str = "75056_"):
        self.rows = rows
        self.partition_keys = (key,)

    def load(self, key: str) -> list[dict[str, object]]:
        assert key in self.partition_keys
        return [dict(row) for row in self.rows]


class FixedVectorizer:
    def __init__(self, values: list[float]):
        self.values = np.asarray(values, dtype=float)
        self.queries: list[str] = []

    def transform(self, queries: list[str]) -> sparse.csr_matrix:
        self.queries.extend(queries)
        return sparse.csr_matrix(self.values.reshape(1, -1))


class FakeCache:
    def __init__(
        self,
        *,
        word: list[float],
        char: list[float],
        address: list[float],
    ):
        count = len(word)
        self.word_vectorizer = FixedVectorizer(word)
        self.char_vectorizer = FixedVectorizer(char)
        self.address_vectorizer = FixedVectorizer(address)
        # Identity matrices make q @ matrix.T preserve the supplied scores.
        matrix = sparse.identity(count, format="csr")
        self.artifacts = (
            self.word_vectorizer,
            matrix,
            [""] * count,
            self.char_vectorizer,
            matrix,
            self.address_vectorizer,
            matrix,
        )
        self.calls: list[tuple[str, int]] = []

    def get(self, key: str, aligned_pool: list[dict[str, object]]) -> tuple:
        self.calls.append((key, len(aligned_pool)))
        return self.artifacts


class FakeLookup:
    def __init__(
        self,
        states: dict[str, str] | None = None,
        *,
        missing: set[str] | None = None,
        extra: str | None = None,
    ):
        self.states = states or {}
        self.missing = missing or set()
        self.extra = extra
        self.requests: list[list[str]] = []

    def get_candidate_scene_details(
        self, sirets: list[str]
    ) -> dict[str, dict[str, str]]:
        self.requests.append(list(sirets))
        result = {
            siret: {"candidate_state": self.states.get(siret, "A")}
            for siret in sirets
            if siret not in self.missing
        }
        if self.extra:
            result[self.extra] = {"candidate_state": "A"}
        return result


def query(**updates: str) -> dict[str, str]:
    value = {
        "query_id": "q1",
        "crm_name": "ACME",
        "crm_address": "12 RUE DES FLEURS",
        "crm_postcode": "75001",
        "crm_city": "PARIS",
        "crm_insee": "75056",
    }
    value.update(updates)
    return value


def test_route_query_prefers_insee_then_cp_and_normalizes_float_codes() -> None:
    assert unit.route_query(
        query(crm_insee="75056.0"),
        {"75056_", "_75001"},
    ) == "75056_"
    assert unit.route_query(
        query(crm_insee="99999", crm_postcode="75001.0"),
        {"75056_", "_75001"},
    ) == "_75001"
    with pytest.raises(unit.UnitRetrievalError, match=unit.STOP):
        unit.route_query(query(crm_insee="", crm_postcode=""), {"75056_"})
    with pytest.raises(unit.UnitRetrievalError, match="invalid partition"):
        unit.route_query(query(), {"bad"})


def test_query_normalization_matches_frozen_acronym_and_plural_rules() -> None:
    assert unit._normalize_text_for_tfidf("Écoles J.N.C chevaux prix") == (
        "ECOLES ECOLE J N C CHEVAUX CHEVAL PRIX JNC"
    )
    assert unit._normalize_name("SAS Société-des fleurs") == "SOCIETE DES FLEURS"
    assert unit._extract_street_number("12 BIS rue A") == "12"
    assert unit._extract_street_name("12 BIS rue A") == "RUE A"


@pytest.mark.parametrize(
    ("name", "address", "insee", "postcode"),
    [
        ("Écoles J.N.C chevaux prix", "12 BIS rue des Fleurs", "75056.0", "75001"),
        ("APAJH 69 – antenne", "7-ter, avenue de l'Europe", None, "69003.0"),
        ("Mairie de L'Haÿ-les-Roses", "PLACE DU 8 MAI 1945", "94038", ""),
        ("S.A.S. AUX CHEVAUX", "42 QUAI ST-PIERRE", "nan", "13002"),
    ],
)
def test_crm_preprocessing_is_differentially_frozen_against_historical_helpers(
    name: str,
    address: str,
    insee: str | None,
    postcode: str,
) -> None:
    """The local worker copy must remain byte-for-byte behavior compatible."""

    assert unit._normalize_code(insee) == blocking.normalize_code(insee)
    assert unit._normalize_code(postcode) == blocking.normalize_code(postcode)
    assert (
        unit._normalize_text_for_tfidf(name)
        == blocking.normalize_text_for_tfidf(name)
    )
    assert (
        unit._normalize_text_for_tfidf(unit._normalize_text(address))
        == blocking.normalize_text_for_tfidf(features.normalize_text(address))
    )
    assert unit._extract_street_number(address) == features.extract_street_number(
        address
    )
    assert unit._extract_street_name(
        address
    ) == features.extract_street_name_from_address(address)

    historical = features.preprocess_crm_row(
        {
            "crm_name": name,
            "crm_address": address,
            "crm_city": "PARIS",
            "crm_city_addr": "PARIS",
            "postcode": postcode,
            "insee": insee,
        }
    )
    assert historical["crm_addr"] == unit._normalize_text(address)
    assert historical["crm_street_num"] == unit._extract_street_number(address)
    assert historical["crm_street_name"] == unit._extract_street_name(address)
    # The retrieval identity deliberately consumes the raw name, not the
    # broader feature pipeline's location-stripped name.
    assert unit._normalize_text_for_tfidf(name) == blocking.normalize_text_for_tfidf(
        name
    )


def test_sparse_and_rescue_helpers_are_differentially_frozen() -> None:
    row = sparse.csr_matrix(
        ([0.25, 0.25, 0.75, 0.5], ([0, 0, 0, 0], [9, 2, 7, 4])),
        shape=(1, 12),
    )
    assert unit._rank_sparse_scores(row, 3) == blocking._rank_sparse_scores(row, 3)

    row_candidate = candidate(
        "00000000000001",
        name="APAJH 69",
        number="12",
        street_type="AVENUE",
        street="DES FLEURS",
    )
    assert unit._candidate_address_hash(
        row_candidate
    ) == blocking.candidate_address_hash(dict(row_candidate))
    assert unit._numeric_tokens(
        unit._primary_name(row_candidate)
    ) == blocking.candidate_numeric_tokens(dict(row_candidate))


def test_sparse_address_equal_scores_follow_reversed_sparse_buffer() -> None:
    row = sparse.csr_matrix(([1.0, 1.0], ([0, 0], [2, 7])), shape=(1, 10))
    assert unit._rank_sparse_scores(row, 10) == [(7, 1.0), (2, 1.0)]


def test_sparse_argpartition_is_applied_before_stable_reverse_sort() -> None:
    values = np.arange(1.0, 507.0)
    row = sparse.csr_matrix(
        (values, (np.zeros(506, dtype=int), np.arange(506))),
        shape=(1, 506),
    )
    hits = unit._rank_sparse_scores(row, 500)
    assert len(hits) == 500
    assert hits[0] == (505, 506.0)
    assert {index for index, _ in hits}.isdisjoint(range(6))


def test_rrf_uses_lexical_string_index_as_last_tie_break() -> None:
    hits = unit._rrf({"a": [2, 10], "b": [10, 2]}, budget=10)
    assert hits[0][0] == 10  # Equal score and best rank: "10" < "2".
    assert hits[0][1] == hits[1][1]


def test_retrieval_fuses_word_char_address_rescue_and_final_siret_tie() -> None:
    rows = [
        candidate("00000000000003", name="OTHER", number="99"),
        candidate("00000000000002", name="ACME 69", number="42"),
        candidate("00000000000001", name="THIRD", number="12"),
    ]
    cache = FakeCache(
        word=[0.0, 4.0, 0.0],
        char=[0.0, 2.0, 3.0],
        address=[0.0, 0.0, 5.0],
    )
    result = unit.retrieve_unit_query(
        query=query(crm_name="ACME 69", crm_address="42 RUE DES FLEURS"),
        partition_store=FakePartitionStore(rows),
        tfidf_cache=cache,
        lookup=FakeLookup(),
    )
    assert result.partition_key == "75056_"
    assert result.raw_pool_count == 3
    assert result.aligned_pool_count == 3
    assert result.lookup_missing_count == 0
    # Candidate 1 wins name+numeric rescue, candidate 2 address+address rescue.
    assert result.candidate_sirets[:2] == (
        "00000000000002",
        "00000000000001",
    )
    assert cache.word_vectorizer.queries == ["ACME 69"]
    assert cache.address_vectorizer.queries == ["42 RUE DES FLEURS FLEUR"]


def test_no_channel_hits_pad_in_physical_order_then_final_sort_by_siret() -> None:
    rows = [
        candidate("00000000000003", number="1"),
        candidate("00000000000001", number="2"),
        candidate("00000000000002", number="3"),
    ]
    cache = FakeCache(word=[0, 0, 0], char=[0, 0, 0], address=[0, 0, 0])
    result = unit.retrieve_unit_query(
        query=query(crm_name="", crm_address=""),
        partition_store=FakePartitionStore(rows),
        tfidf_cache=cache,
        lookup=FakeLookup(),
    )
    assert result.candidate_sirets == (
        "00000000000001",
        "00000000000002",
        "00000000000003",
    )
    assert cache.word_vectorizer.queries == []
    assert cache.char_vectorizer.queries == []
    assert cache.address_vectorizer.queries == []


def test_zero_or_one_pool_skips_tfidf_and_filters_lookup_state() -> None:
    class ExplodingCache:
        def get(self, *_: object) -> tuple:
            raise AssertionError("cache must not be called")

    only = candidate("00000000000001")
    result = unit.retrieve_unit_query(
        query=query(),
        partition_store=FakePartitionStore([only]),
        tfidf_cache=ExplodingCache(),
        lookup=FakeLookup(states={"00000000000001": "F"}),
    )
    assert result.candidate_sirets == ()
    empty = unit.retrieve_unit_query(
        query=query(),
        partition_store=FakePartitionStore([]),
        tfidf_cache=ExplodingCache(),
        lookup=FakeLookup(),
    )
    assert empty.candidate_sirets == ()


def test_pool_alignment_is_last_value_wins_first_position_and_closed_removed() -> None:
    rows = [
        candidate("00000000000001", name="FIRST"),
        candidate("00000000000002", name="CLOSED", state="F"),
        candidate("00000000000001", name="LAST"),
        candidate("00000000000003", name=""),
    ]
    result = unit.retrieve_unit_query(
        query=query(),
        partition_store=FakePartitionStore(rows),
        tfidf_cache=object(),
        lookup=FakeLookup(),
    )
    assert result.raw_pool_count == 4
    assert result.aligned_pool_count == 1
    assert result.candidate_sirets == ("00000000000001",)


def test_lookup_missing_is_omitted_and_counted_and_extra_fails() -> None:
    rows = [
        candidate("00000000000001"),
        candidate("00000000000002"),
    ]
    cache = FakeCache(word=[1, 0], char=[0, 0], address=[0, 0])
    result = unit.retrieve_unit_query(
        query=query(crm_address=""),
        partition_store=FakePartitionStore(rows),
        tfidf_cache=cache,
        lookup=FakeLookup(missing={"00000000000002"}),
    )
    assert result.candidate_sirets == ("00000000000001",)
    assert result.lookup_missing_count == 1
    with pytest.raises(unit.UnitRetrievalError, match="unrequested"):
        unit.retrieve_unit_query(
            query=query(crm_address=""),
            partition_store=FakePartitionStore(rows),
            tfidf_cache=cache,
            lookup=FakeLookup(extra="99999999999999"),
        )


def test_nonfinite_query_or_product_fails_closed() -> None:
    rows = [candidate("00000000000001"), candidate("00000000000002")]
    cache = FakeCache(word=[np.nan, 0], char=[0, 0], address=[0, 0])
    with pytest.raises(unit.UnitRetrievalError, match="non-finite"):
        unit.retrieve_unit_query(
            query=query(),
            partition_store=FakePartitionStore(rows),
            tfidf_cache=cache,
            lookup=FakeLookup(),
        )


def test_candidate_ceiling_is_absolute_and_padding_is_deterministic() -> None:
    rows = [
        candidate(f"{index:014d}", number=str(index + 1))
        for index in range(120)
    ]
    zeros = [0.0] * len(rows)
    lookup = FakeLookup()
    result = unit.retrieve_unit_query(
        query=query(crm_name="", crm_address=""),
        partition_store=FakePartitionStore(rows),
        tfidf_cache=FakeCache(word=zeros, char=zeros, address=zeros),
        lookup=lookup,
    )
    assert len(result.candidate_sirets) == 100
    assert lookup.requests == [[f"{index:014d}" for index in range(100)]]
    assert result.candidate_sirets == tuple(f"{index:014d}" for index in range(100))


def test_worker_identity_excludes_fields_not_present_in_sanitized_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = {"python": "test"}
    monkeypatch.setattr(unit, "_runtime_values", lambda: runtime)
    gate = {
        "schema_version": "gate",
        "safe_input_build_id": "a" * 64,
        "query_count": 1,
        "routing_payload_sha256": "b" * 64,
        "max_rss_bytes": 123,
    }
    monkeypatch.setattr(unit.strict_stores, "_validate_run_spec", lambda value: value)
    spec = {
        "schema_version": unit.WORKER_RUN_SPEC_SCHEMA,
        "safe_input_build_id": "a" * 64,
        "safe_runtime_manifest_sha256": "c" * 64,
        "safe_queries_dev_sha256": "d" * 64,
        "query_count": 1,
        "query_id_payload_sha256": "e" * 64,
        "routing_payload_sha256": "b" * 64,
        "worker_policy_sha256": "f" * 64,
        "worker_lock_projection_sha256": "1" * 64,
        "parent_runner_sha256": "2" * 64,
        "worker_source_hashes": {"worker.py": "3" * 64},
        "strict_stores_build_id": "4" * 64,
        "strict_stores_manifest_sha256": "5" * 64,
        "retrieval": unit._RETRIEVAL_POLICY,
        "tfidf_cache": unit._TFIDF_POLICY,
        "runtime": runtime,
        "max_rss_bytes": 123,
        "gate_a_run_spec": gate,
        "declarations": unit._WORKER_DECLARATIONS,
    }
    validated = unit._validate_worker_run_spec(spec)
    identity = unit._worker_identity(validated)
    assert set(identity) == {
        "schema_version",
        "worker_policy_sha256",
        "worker_lock_projection_sha256",
        "parent_runner_sha256",
        "worker_source_hashes",
        "safe_input_build_id",
        "safe_runtime_manifest_sha256",
        "safe_queries_dev_sha256",
        "strict_stores_build_id",
        "strict_stores_manifest_sha256",
        "retrieval",
        "tfidf_cache",
        "runtime",
    }
    assert "gate_a_run_spec" not in identity


def test_fd_reader_requires_dev_fd_and_reads_inherited_descriptor(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_bytes(b'{"a":1}')
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert unit._read_inherited_fd(f"/dev/fd/{descriptor}", "test") == b'{"a":1}'
    finally:
        os.close(descriptor)
    with pytest.raises(unit.UnitRetrievalError, match="/dev/fd"):
        unit._read_inherited_fd(str(source), "test")


def test_query_parquet_requires_exact_nonnullable_schema_and_order(
    tmp_path: Path,
) -> None:
    ids = ["b", "a"]
    ids.sort(
        key=lambda value: (
            hashlib.sha256(("v412-unit-engine:" + value).encode()).hexdigest(),
            value,
        )
    )
    rows = [
        {
            "query_id": value,
            "crm_name": "N",
            "crm_address": "A",
            "crm_postcode": "75001",
            "crm_city": "PARIS",
            "crm_insee": "75056",
        }
        for value in ids
    ]
    schema = pa.schema(
        [pa.field(name, pa.string(), nullable=False) for name in unit.QUERY_COLUMNS]
    )
    path = tmp_path / "queries.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    data = path.read_bytes()
    payload = "".join(f"{value}\n" for value in ids).encode()
    spec = {
        "safe_queries_dev_sha256": hashlib.sha256(data).hexdigest(),
        "query_count": len(ids),
        "query_id_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert unit._query_table(data, spec).num_rows == 2
    with pytest.raises(unit.UnitRetrievalError, match="hash mismatch"):
        unit._query_table(data + b"x", spec)


def test_cli_exposes_exact_orchestrator_flags() -> None:
    parser = unit._parser()
    values = parser.parse_args(
        [
            "--run-spec",
            "/dev/fd/3",
            "--lookup-descriptor",
            "/dev/fd/4",
            "--queries",
            "/dev/fd/5",
            "--output-dir",
            "output",
            "--worker-build-id",
            "a" * 64,
            "--forbidden-oracle",
            "/oracle",
            "--forbidden-oracle-audit",
            "/oracle-audit",
            "--forbidden-historical",
            "/historical",
            "--forbidden-model",
            "/model",
        ]
    )
    assert values.run_spec == "/dev/fd/3"
    assert values.forbidden_oracle_audit == "/oracle-audit"


def test_child_worker_writes_only_the_three_contract_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    output = run_root / "output"
    output.mkdir(parents=True)
    query_schema = pa.schema(
        [pa.field(name, pa.string(), nullable=False) for name in unit.QUERY_COLUMNS]
    )
    query_row = query()
    query_table = pa.Table.from_pylist([query_row], schema=query_schema)
    query_path = tmp_path / "queries.parquet"
    pq.write_table(query_table, query_path)
    query_bytes = query_path.read_bytes()
    query_id_payload = b"q1\n"
    routing_payload = b"q1\0" + b"75056_\n"
    descriptor_bytes = b"{}\n"
    worker_build_id = "a" * 64
    run_spec = {
        "safe_queries_dev_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "query_count": 1,
        "query_id_payload_sha256": hashlib.sha256(query_id_payload).hexdigest(),
        "routing_payload_sha256": hashlib.sha256(routing_payload).hexdigest(),
        "max_rss_bytes": 1024,
        "gate_a_run_spec": {
            "lookup_descriptor_sha256": hashlib.sha256(
                descriptor_bytes
            ).hexdigest(),
            "allowed_read_files": [],
            "partition_records": [],
            "cache_records": [],
        },
    }
    run_spec_path = tmp_path / "run_spec.json"
    descriptor_path = tmp_path / "lookup.json"
    run_spec_path.write_bytes(json.dumps(run_spec).encode())
    descriptor_path.write_bytes(descriptor_bytes)

    class OneCache:
        partition_keys = ("75056_",)

    class LookupContext(FakeLookup):
        def __enter__(self) -> "LookupContext":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(unit, "_validate_worker_run_spec", lambda value: value)
    monkeypatch.setattr(
        unit, "compute_worker_build_id", lambda value: worker_build_id
    )
    monkeypatch.setattr(
        unit.strict_stores,
        "StrictPartitionStore",
        lambda *_args, **_kwargs: FakePartitionStore(
            [candidate("00000000000001")]
        ),
    )
    monkeypatch.setattr(
        unit.strict_stores,
        "StrictVerifiedTfidfCache",
        lambda *_args, **_kwargs: OneCache(),
    )
    monkeypatch.setattr(
        unit.strict_stores,
        "StrictSnapshotLookup",
        lambda *_args, **_kwargs: LookupContext(),
    )
    monkeypatch.setattr(unit, "_require_denied_open", lambda *_args: True)
    monkeypatch.setattr(unit, "_require_network_denied", lambda: True)
    monkeypatch.setattr(unit, "_require_write_denied", lambda _path: True)
    monkeypatch.setattr(unit, "_peak_rss_bytes", lambda: 1)
    monkeypatch.chdir(run_root)

    descriptors = [
        os.open(run_spec_path, os.O_RDONLY),
        os.open(descriptor_path, os.O_RDONLY),
        os.open(query_path, os.O_RDONLY),
    ]
    try:
        integrity = unit.run_child_worker(
            run_spec_path=f"/dev/fd/{descriptors[0]}",
            lookup_descriptor_path=f"/dev/fd/{descriptors[1]}",
            queries_path=f"/dev/fd/{descriptors[2]}",
            output_dir="output",
            worker_build_id=worker_build_id,
            forbidden_oracle="/oracle",
            forbidden_oracle_audit="/oracle-audit",
            forbidden_historical="/historical",
            forbidden_model="/model",
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)

    assert {path.name for path in output.iterdir()} == {
        "query_status.parquet",
        "candidates_top100.parquet",
        "integrity.json",
    }
    assert pq.read_table(output / "query_status.parquet").to_pylist() == [
        {"query_id": "q1", "candidate_count": 1}
    ]
    assert pq.read_table(output / "candidates_top100.parquet").to_pylist() == [
        {
            "query_id": "q1",
            "candidate_rank": 1,
            "candidate_siret": "00000000000001",
        }
    ]
    assert integrity["candidate_count"] == 1
    assert integrity["sandbox_checks"] == {
        "allowed_read": True,
        "oracle_denied": True,
        "oracle_audit_denied": True,
        "historical_denied": True,
        "model_denied": True,
        "network_denied": True,
        "write_denied": True,
    }
