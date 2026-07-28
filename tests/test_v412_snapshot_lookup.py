from pathlib import Path

import duckdb
import pytest

from src.xgb_matcher.v412_snapshot_lookup import (
    INDEX_NAME,
    LOOKUP_COLUMNS,
    TABLE_NAME,
    V412SnapshotLookup,
    inspect_lookup_schema,
    validate_requested_sirets,
)


def _database(path: Path) -> Path:
    connection = duckdb.connect(str(path))
    connection.execute(
        f"""
        CREATE TABLE {TABLE_NAME} (
            siret VARCHAR,
            candidate_state VARCHAR,
            enseigne1 VARCHAR,
            enseigne2 VARCHAR,
            enseigne3 VARCHAR,
            denomination_usuelle VARCHAR,
            activity_code VARCHAR
        )
        """
    )
    connection.executemany(
        f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("00000000000001", "A", "Zulu", None, None, "Usuelle", "62.01Z"),
            ("00000000000002", "F", None, "Deux", None, None, None),
        ],
    )
    connection.execute(
        f"CREATE UNIQUE INDEX {INDEX_NAME} ON {TABLE_NAME}(siret)"
    )
    connection.execute("CHECKPOINT")
    connection.close()
    return path


@pytest.mark.parametrize(
    "invalid",
    [
        "00000000000001",
        b"00000000000001",
        bytearray(b"00000000000001"),
        None,
        123,
        {"00000000000001"},
    ],
)
def test_request_rejects_non_sequence_or_scalar(invalid):
    with pytest.raises((TypeError, ValueError), match="STOP_V412_LOOKUP_INPUT"):
        validate_requested_sirets(invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        12345678901234,
        b"00000000000001",
        " 00000000000001",
        "00000000000001 ",
        "0000000000001",
        "000000000000001",
        "0000000000000A",
        float("nan"),
    ],
)
def test_request_rejects_every_noncanonical_element(invalid):
    with pytest.raises(ValueError, match="STOP_V412_LOOKUP_INPUT"):
        validate_requested_sirets([invalid])


def test_request_limit_applies_before_deduplication():
    with pytest.raises(ValueError, match="more than 100"):
        validate_requested_sirets(["00000000000001"] * 101)


def test_request_deduplicates_in_first_seen_order():
    assert validate_requested_sirets(
        ["00000000000002", "00000000000001", "00000000000002"]
    ) == ["00000000000002", "00000000000001"]


def test_lookup_is_sorted_omits_absent_and_preserves_nulls(tmp_path):
    path = _database(tmp_path / "lookup.duckdb")
    with V412SnapshotLookup(path) as store:
        observed = store.get_candidate_scene_details(
            ["99999999999999", "00000000000002", "00000000000001"]
        )
    assert list(observed) == ["00000000000001", "00000000000002"]
    assert observed["00000000000001"] == {
        "candidate_state": "A",
        "enseigne1": "Zulu",
        "enseigne2": None,
        "enseigne3": None,
        "denomination_usuelle": "Usuelle",
        "activity_code": "62.01Z",
    }
    assert observed["00000000000002"]["candidate_state"] == "F"


def test_lookup_empty_and_closed_behavior(tmp_path):
    store = V412SnapshotLookup(_database(tmp_path / "lookup.duckdb"))
    assert store.get_candidate_scene_details([]) == {}
    store.close()
    store.close()
    with pytest.raises(RuntimeError, match="lookup is closed"):
        store.get_candidate_scene_details([])


def test_lookup_connection_is_read_only(tmp_path):
    with V412SnapshotLookup(_database(tmp_path / "lookup.duckdb")) as store:
        with pytest.raises(duckdb.InvalidInputException):
            store._connection.execute(
                f"INSERT INTO {TABLE_NAME} VALUES "
                "('00000000000003', NULL, NULL, NULL, NULL, NULL, NULL)"
            )


def test_schema_refuses_extra_table_wrong_order_and_missing_unique_index(tmp_path):
    for drift in ("extra_table", "wrong_order", "missing_index"):
        path = tmp_path / f"{drift}.duckdb"
        connection = duckdb.connect(str(path))
        columns = LOOKUP_COLUMNS
        if drift == "wrong_order":
            columns = [LOOKUP_COLUMNS[1], LOOKUP_COLUMNS[0], *LOOKUP_COLUMNS[2:]]
        connection.execute(
            f"CREATE TABLE {TABLE_NAME} "
            f"({', '.join(f'{column} VARCHAR' for column in columns)})"
        )
        if drift != "missing_index":
            connection.execute(
                f"CREATE UNIQUE INDEX {INDEX_NAME} ON {TABLE_NAME}(siret)"
            )
        if drift == "extra_table":
            connection.execute("CREATE TABLE extra(value VARCHAR)")
        with pytest.raises(ValueError, match="STOP_V412_LOOKUP_ARTIFACT"):
            inspect_lookup_schema(connection)
        connection.close()


def test_constructor_refuses_symlink_and_wal(tmp_path):
    database = _database(tmp_path / "lookup.duckdb")
    link = tmp_path / "link.duckdb"
    link.symlink_to(database)
    with pytest.raises(ValueError, match="invalid database file"):
        V412SnapshotLookup(link)
    wal = database.with_suffix(".duckdb.wal")
    wal.write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="invalid database file"):
        V412SnapshotLookup(database)
