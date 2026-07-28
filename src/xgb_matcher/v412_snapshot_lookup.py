"""Strict read-only access to the frozen V4.12 SIRENE snapshot lookup."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
from typing import Any

import duckdb


TABLE_NAME = "candidate_details"
INDEX_NAME = "candidate_details_siret_uidx"
LOOKUP_COLUMNS = [
    "siret",
    "candidate_state",
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "denomination_usuelle",
    "activity_code",
]
DETAIL_COLUMNS = LOOKUP_COLUMNS[1:]
MAX_SIRETS_PER_CALL = 100
_SIRET = re.compile(r"^[0-9]{14}$")


def validate_requested_sirets(sirets: Sequence[str]) -> list[str]:
    """Validate without coercion, then deduplicate in first-seen order."""

    if isinstance(sirets, (str, bytes, bytearray)) or not isinstance(
        sirets, Sequence
    ):
        raise TypeError("STOP_V412_LOOKUP_INPUT: expected a sequence of strings")
    if len(sirets) > MAX_SIRETS_PER_CALL:
        raise ValueError("STOP_V412_LOOKUP_INPUT: more than 100 SIRET requested")
    unique: list[str] = []
    seen: set[str] = set()
    for value in sirets:
        if type(value) is not str or _SIRET.fullmatch(value) is None:
            raise ValueError("STOP_V412_LOOKUP_INPUT: invalid canonical SIRET")
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def inspect_lookup_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Fail closed if the frozen table or unique index differs."""

    tables = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()
    ]
    if tables != [TABLE_NAME]:
        raise ValueError("STOP_V412_LOOKUP_ARTIFACT: table set changed")
    columns = connection.execute(
        """
        SELECT column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [TABLE_NAME],
    ).fetchall()
    if [str(row[0]) for row in columns] != LOOKUP_COLUMNS:
        raise ValueError("STOP_V412_LOOKUP_ARTIFACT: column order changed")
    if any(str(row[1]).upper() != "VARCHAR" for row in columns):
        raise ValueError("STOP_V412_LOOKUP_ARTIFACT: column type changed")
    indexes = connection.execute(
        """
        SELECT index_name, is_unique, expressions
        FROM duckdb_indexes()
        WHERE schema_name = 'main' AND table_name = ?
        ORDER BY index_name
        """,
        [TABLE_NAME],
    ).fetchall()
    if len(indexes) != 1:
        raise ValueError("STOP_V412_LOOKUP_ARTIFACT: index set changed")
    index_name, is_unique, expressions = indexes[0]
    if (
        str(index_name) != INDEX_NAME
        or is_unique is not True
        or "[siret]" not in str(expressions).replace('"', "")
    ):
        raise ValueError("STOP_V412_LOOKUP_ARTIFACT: unique SIRET index changed")


class V412SnapshotLookup:
    """DuckDB-backed store exposing only the frozen, bounded lookup API."""

    def __init__(self, database_path: Path):
        supplied_path = Path(database_path)
        if supplied_path.is_symlink():
            raise ValueError("STOP_V412_LOOKUP_ARTIFACT: invalid database file")
        self.database_path = supplied_path.resolve()
        if (
            not self.database_path.is_file()
            or self.database_path.with_suffix(
                self.database_path.suffix + ".wal"
            ).exists()
        ):
            raise ValueError("STOP_V412_LOOKUP_ARTIFACT: invalid database file")
        self._connection = duckdb.connect(
            str(self.database_path), read_only=True
        )
        try:
            inspect_lookup_schema(self._connection)
        except BaseException:
            self._connection.close()
            raise
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "V412SnapshotLookup":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def get_candidate_scene_details(
        self, sirets: Sequence[str]
    ) -> dict[str, dict[str, str | None]]:
        if self._closed:
            raise RuntimeError("STOP_V412_LOOKUP_ARTIFACT: lookup is closed")
        requested = validate_requested_sirets(sirets)
        if not requested:
            return {}
        rows = self._connection.execute(
            f"""
            SELECT {", ".join(LOOKUP_COLUMNS)}
            FROM {TABLE_NAME}
            WHERE siret IN (SELECT unnest(?))
            ORDER BY siret
            """,
            [requested],
        ).fetchall()
        requested_set = set(requested)
        result: dict[str, dict[str, str | None]] = {}
        for row in rows:
            siret = row[0]
            if type(siret) is not str or siret not in requested_set:
                raise ValueError(
                    "STOP_V412_LOOKUP_ARTIFACT: lookup returned extra SIRET"
                )
            if siret in result:
                raise ValueError(
                    "STOP_V412_LOOKUP_ARTIFACT: duplicate SIRET returned"
                )
            values = row[1:]
            if any(value is not None and type(value) is not str for value in values):
                raise ValueError(
                    "STOP_V412_LOOKUP_ARTIFACT: lookup value type changed"
                )
            result[siret] = dict(zip(DETAIL_COLUMNS, values, strict=True))
        return result
