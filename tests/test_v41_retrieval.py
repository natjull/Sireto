from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval import CandidatePoolResult
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.v41_retrieval import (
    InputSiretState,
    V41CandidateRetriever,
    V41GlobalCandidateStore,
    V41RetrievalConfig,
    V41RetrievalVariant,
    normalize_input_siret,
)


def _candidate(
    siret: str,
    *,
    state: str,
    name: str,
    number: str = "1",
    street: str = "RUE TEST",
    postcode: str = "01000",
    insee: str = "01001",
    is_siege: bool = False,
) -> dict:
    return {
        "siret": siret,
        "siren": siret[:9],
        "denomination": name,
        "enseigne1": None,
        "enseigne2": None,
        "enseigne3": None,
        "numeroVoie": number,
        "typeVoie": None,
        "libelleVoie": street,
        "complementAdresse": None,
        "postcode": postcode,
        "city": "TEST",
        "insee": insee,
        "etat_admin": state,
        "is_siege": is_siege,
    }


@pytest.fixture()
def global_database(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE candidates (
            siret VARCHAR,
            siren VARCHAR,
            denomination VARCHAR,
            enseigne1 VARCHAR,
            enseigne2 VARCHAR,
            enseigne3 VARCHAR,
            numeroVoie VARCHAR,
            typeVoie VARCHAR,
            libelleVoie VARCHAR,
            complementAdresse VARCHAR,
            postcode VARCHAR,
            city VARCHAR,
            insee VARCHAR,
            etat_admin VARCHAR,
            is_siege BOOLEAN
        )
        """
    )
    rows = [
        _candidate(
            "11111111100001",
            state="F",
            name="ANCIENNE ENSEIGNE",
            number="8",
            street="RUE DU TEST",
        ),
        _candidate(
            "11111111100002",
            state="F",
            name="ANCIENNE ENSEIGNE",
            number="8",
            street="RUE DU TEST",
        ),
        _candidate(
            "11111111100003",
            state="A",
            name="ANCIENNE ENSEIGNE ACTIVE",
            number="8",
            street="RUE DU TEST",
            is_siege=True,
        ),
        _candidate(
            "22222222200001",
            state="A",
            name="AUTRE ENTREPRISE",
        ),
    ]
    connection.register("candidate_rows", pa.Table.from_pylist(rows))
    connection.execute("INSERT INTO candidates SELECT * FROM candidate_rows")
    connection.close()
    return path


def test_v41_config_is_active_only_isolated_and_capped() -> None:
    config = V41RetrievalConfig(max_candidates=100).sparse_config()

    assert config.include_closed is False
    assert config.retrieval_budget == 100
    assert config.prefilter_trigger_size == 1
    assert config.version == "v4.1-active-only-1"
    assert config.signature().hash != RetrievalConfigV1(
        include_closed=False,
        retrieval_budget=100,
        fusion_mode="rrf",
    ).signature().hash

    with pytest.raises(ValueError, match="between 1 and 100"):
        V41RetrievalConfig(max_candidates=101)


def test_normalize_input_siret_does_not_repair_bad_identifiers() -> None:
    assert normalize_input_siret(" 11111111100001 ") == "11111111100001"
    assert normalize_input_siret("1111111110000X") is None
    assert normalize_input_siret("1111111110000") is None
    assert normalize_input_siret(None) is None


def test_batch_qualification_distinguishes_all_four_states(
    global_database: Path,
) -> None:
    with V41GlobalCandidateStore(global_database) as store:
        results = store.qualify_input_sirets(
            [
                "11111111100003",
                "11111111100001",
                "99999999900009",
                "not-a-siret",
            ]
        )

    assert [result.state for result in results] == [
        InputSiretState.ACTIVE,
        InputSiretState.CLOSED,
        InputSiretState.NOT_FOUND,
        InputSiretState.INVALID,
    ]
    assert results[0].candidate["denomination"] == "ANCIENNE ENSEIGNE ACTIVE"
    assert results[2].siren == "999999999"
    assert results[3].normalized_siret is None


def test_active_siblings_are_filtered_before_limit(global_database: Path) -> None:
    with V41GlobalCandidateStore(global_database) as store:
        siblings = store.get_active_siblings(
            ["111111111"],
            max_per_siren=1,
        )

    assert [row["siret"] for row in siblings["111111111"]] == [
        "11111111100003"
    ]
    assert siblings["111111111"][0]["etat_admin"] == "A"


def test_variant_c_uses_closed_alias_but_outputs_only_unique_active_candidates(
    global_database: Path,
) -> None:
    sparse_candidates = [
        _candidate(
            "11111111100001",
            state="F",
            name="ANCIENNE ENSEIGNE",
        ),
        _candidate(
            "22222222200001",
            state="A",
            name="LOCAL STALE NAME",
        ),
    ]

    def sparse_builder(*_args, **_kwargs) -> CandidatePoolResult:
        return CandidatePoolResult(candidates=sparse_candidates)

    with V41GlobalCandidateStore(global_database) as store:
        retriever = V41CandidateRetriever(
            partitioned_store=object(),
            global_store=store,
            config=V41RetrievalConfig(
                variant=V41RetrievalVariant.C_CLOSED_ALIAS,
                max_candidates=2,
            ),
            sparse_pool_builder=sparse_builder,
        )
        result = retriever.build(
            crm_row={"insee": "01001", "postcode": "01000"},
            crm_pre={"crm_name": "ANCIENNE ENSEIGNE", "crm_addr": "8 RUE DU TEST"},
            input_siret="11111111100001",
        )

    assert result.input_siret.state == InputSiretState.CLOSED
    assert result.channels["closed_alias_name"] == ["11111111100003"]
    assert result.channels["closed_alias_address"][0] == "11111111100003"
    assert {candidate["siret"] for candidate in result.candidates} == {
        "11111111100003",
        "22222222200001",
    }
    assert all(candidate["etat_admin"] == "A" for candidate in result.candidates)
    assert len(result.candidates) <= 2
    hydrated = next(
        candidate
        for candidate in result.candidates
        if candidate["siret"] == "22222222200001"
    )
    assert hydrated["denomination"] == "AUTRE ENTREPRISE"


def test_variant_b_active_input_has_direct_and_siren_channels(
    global_database: Path,
) -> None:
    def sparse_builder(*_args, **_kwargs) -> CandidatePoolResult:
        return CandidatePoolResult(candidates=[])

    with V41GlobalCandidateStore(global_database) as store:
        retriever = V41CandidateRetriever(
            partitioned_store=object(),
            global_store=store,
            config=V41RetrievalConfig(
                variant=V41RetrievalVariant.B_INPUT_EVIDENCE,
            ),
            sparse_pool_builder=sparse_builder,
        )
        result = retriever.build(
            crm_row={},
            crm_pre={},
            input_siret="11111111100003",
        )

    assert result.channels["input_siret_active"] == ["11111111100003"]
    assert result.channels["input_siren_active_sites"] == ["11111111100003"]
    assert [candidate["siret"] for candidate in result.candidates] == [
        "11111111100003"
    ]
    assert {
        "input_siret_active",
        "input_siren_active_sites",
    } <= set(result.candidates[0]["v41_channel_ranks"])


def test_variant_c_reruns_sparse_retrieval_with_closed_alias(
    global_database: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    def sparse_builder(_store, crm_row, _crm_pre, *_args, **_kwargs):
        name = str(crm_row.get("crm_name") or "")
        address = str(crm_row.get("crm_address") or "")
        calls.append((name, address))
        if "ANCIENNE ENSEIGNE" in name or "RUE DU TEST" in address:
            return CandidatePoolResult(
                candidates=[
                    _candidate(
                        "22222222200001",
                        state="A",
                        name="ANCIENNE ENSEIGNE REPRISE",
                        number="8",
                        street="RUE DU TEST",
                    )
                ]
            )
        return CandidatePoolResult(candidates=[])

    with V41GlobalCandidateStore(global_database) as store:
        retriever = V41CandidateRetriever(
            partitioned_store=object(),
            global_store=store,
            config=V41RetrievalConfig(
                variant=V41RetrievalVariant.C_CLOSED_ALIAS,
            ),
            sparse_pool_builder=sparse_builder,
        )
        result = retriever.build(
            crm_row={
                "crm_name": "NOM CRM SALE",
                "crm_address": "ADRESSE CRM SALE",
                "insee": "01001",
                "postcode": "01000",
            },
            crm_pre={},
            input_siret="11111111100001",
        )

    assert len(calls) == 3
    assert any("ANCIENNE ENSEIGNE" in name for name, _address in calls[1:])
    assert any("RUE DU TEST" in address for _name, address in calls[1:])
    assert "22222222200001" in result.channels["closed_alias_name"]
    assert "22222222200001" in result.channels["closed_alias_address"]
