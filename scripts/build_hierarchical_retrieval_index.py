#!/usr/bin/env python3
"""Build the content-addressed Tantivy index for hierarchical SIRET retrieval.

The 42M establishment and 29M legal-unit sources are joined and streamed by
DuckDB.  No CRM labels or ground-truth columns are accepted by this builder.
Official history and succession inputs are optional, but their absence is made
explicit through ``temporal_complete=false`` in the sealed manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.hierarchical_retrieval import (  # noqa: E402
    character_ngrams,
    normalize_code,
    normalize_insee,
    normalize_text,
)


SCHEMA_VERSION = "sireto-hierarchical-index-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _relation(path: Path) -> str:
    escaped = _sql_path(path)
    return (
        f"read_parquet('{escaped}')"
        if path.suffix.lower() in {".parquet", ".pq"}
        else f"read_csv_auto('{escaped}', header=true, all_varchar=true)"
    )


def _columns(connection: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM {_relation(path)}").fetchall()
    return {str(row[0]) for row in rows}


def _first(columns: set[str], names: Iterable[str]) -> str | None:
    return next((name for name in names if name in columns), None)


def _string_agg_expression(columns: set[str], names: Iterable[str]) -> str:
    available = [name for name in names if name in columns]
    if not available:
        return "''"
    values = ", ".join(f"CAST({name} AS VARCHAR)" for name in available)
    return f"concat_ws(' ', {values})"


def _prepare_optional_views(
    connection: duckdb.DuckDBPyConnection,
    *,
    establishment_history: Path | None,
    legal_unit_history: Path | None,
    successions: Path | None,
) -> tuple[str, str, str]:
    establishment_join = ""
    legal_join = ""
    succession_join = ""
    if establishment_history:
        columns = _columns(connection, establishment_history)
        siret = _first(columns, ["siret", "siretEtablissement"])
        if not siret:
            raise ValueError("establishment history has no SIRET column")
        name = _string_agg_expression(
            columns,
            [
                "enseigne1Etablissement",
                "enseigne2Etablissement",
                "enseigne3Etablissement",
                "denominationUsuelleEtablissement",
            ],
        )
        address = _string_agg_expression(
            columns,
            [
                "numeroVoieEtablissement",
                "indiceRepetitionEtablissement",
                "typeVoieEtablissement",
                "libelleVoieEtablissement",
                "complementAdresseEtablissement",
            ],
        )
        connection.execute(
            f"""
            CREATE TEMP VIEW establishment_history_agg AS
            SELECT CAST({siret} AS VARCHAR) AS siret,
                   string_agg(DISTINCT nullif(trim({name}), ''), ' | ') AS historical_names,
                   string_agg(DISTINCT nullif(trim({address}), ''), ' | ') AS historical_addresses
            FROM {_relation(establishment_history)}
            GROUP BY 1
            """
        )
        establishment_join = "LEFT JOIN establishment_history_agg eh USING (siret)"
    if legal_unit_history:
        columns = _columns(connection, legal_unit_history)
        siren = _first(columns, ["siren", "sirenUniteLegale"])
        if not siren:
            raise ValueError("legal-unit history has no SIREN column")
        name = _string_agg_expression(
            columns,
            [
                "denominationUniteLegale",
                "denominationUsuelle1UniteLegale",
                "denominationUsuelle2UniteLegale",
                "denominationUsuelle3UniteLegale",
                "sigleUniteLegale",
                "nomUniteLegale",
                "nomUsageUniteLegale",
                "prenomUsuelUniteLegale",
            ],
        )
        connection.execute(
            f"""
            CREATE TEMP VIEW legal_unit_history_agg AS
            SELECT CAST({siren} AS VARCHAR) AS siren,
                   string_agg(DISTINCT nullif(trim({name}), ''), ' | ') AS historical_legal_names
            FROM {_relation(legal_unit_history)}
            GROUP BY 1
            """
        )
        legal_join = "LEFT JOIN legal_unit_history_agg lh USING (siren)"
    if successions:
        columns = _columns(connection, successions)
        predecessor = _first(
            columns,
            ["siretEtablissementPredecesseur", "siret_predecesseur", "predecessor_siret"],
        )
        successor = _first(
            columns,
            ["siretEtablissementSuccesseur", "siret_successeur", "successor_siret"],
        )
        if not predecessor or not successor:
            raise ValueError("successions input lacks predecessor/successor SIRET columns")
        # Links are indexed in both directions. Runtime follows exactly one hop.
        connection.execute(
            f"""
            CREATE TEMP VIEW succession_links AS
            SELECT siret, string_agg(DISTINCT linked_siret, ' ') AS linked_sirets
            FROM (
                SELECT CAST({predecessor} AS VARCHAR) siret,
                       CAST({successor} AS VARCHAR) linked_siret
                FROM {_relation(successions)}
                UNION ALL
                SELECT CAST({successor} AS VARCHAR) siret,
                       CAST({predecessor} AS VARCHAR) linked_siret
                FROM {_relation(successions)}
            )
            WHERE length(siret) = 14 AND length(linked_siret) = 14
            GROUP BY 1
            """
        )
        succession_join = "LEFT JOIN succession_links sl USING (siret)"
    return establishment_join, legal_join, succession_join


def _create_current_view(
    connection: duckdb.DuckDBPyConnection,
    establishments: Path,
    legal_units: Path,
    optional_joins: tuple[str, str, str],
) -> None:
    establishment_join, legal_join, succession_join = optional_joins
    connection.execute(
        f"""
        CREATE TEMP VIEW index_rows AS
        SELECT
            CAST(e.siret AS VARCHAR) AS siret,
            CAST(e.siren AS VARCHAR) AS siren,
            CAST(e.codeCommuneEtablissement AS VARCHAR) AS insee,
            CAST(e.codePostalEtablissement AS VARCHAR) AS postcode,
            CAST(e.numeroVoieEtablissement AS VARCHAR) AS number,
            CAST(e.indiceRepetitionEtablissement AS VARCHAR) AS number_suffix,
            CAST(e.etatAdministratifEtablissement AS VARCHAR) AS state,
            CAST(e.etablissementSiege AS BOOLEAN) AS is_siege,
            concat_ws(' | ',
                e.enseigne1Etablissement,
                e.enseigne2Etablissement,
                e.enseigne3Etablissement,
                e.denominationUsuelleEtablissement,
                u.denominationUniteLegale,
                u.denominationUsuelle1UniteLegale,
                u.denominationUsuelle2UniteLegale,
                u.denominationUsuelle3UniteLegale,
                u.sigleUniteLegale,
                concat_ws(' ', u.prenomUsuelUniteLegale, u.nomUniteLegale),
                u.nomUsageUniteLegale
                {', eh.historical_names' if establishment_join else ''}
                {', lh.historical_legal_names' if legal_join else ''}
            ) AS names,
            concat_ws(' | ',
                concat_ws(' ', e.numeroVoieEtablissement,
                    e.indiceRepetitionEtablissement, e.typeVoieEtablissement,
                    e.libelleVoieEtablissement, e.complementAdresseEtablissement)
                {', eh.historical_addresses' if establishment_join else ''}
            ) AS addresses,
            {"coalesce(sl.linked_sirets, '')" if succession_join else "''"} AS linked_sirets,
            to_json(struct_pack(
                siret := CAST(e.siret AS VARCHAR),
                siren := CAST(e.siren AS VARCHAR),
                denomination := coalesce(e.denominationUsuelleEtablissement,
                                         e.enseigne1Etablissement,
                                         u.denominationUniteLegale),
                enseigne1 := e.enseigne1Etablissement,
                enseigne2 := e.enseigne2Etablissement,
                enseigne3 := e.enseigne3Etablissement,
                numeroVoie := e.numeroVoieEtablissement,
                indiceRepetition := e.indiceRepetitionEtablissement,
                typeVoie := e.typeVoieEtablissement,
                libelleVoie := e.libelleVoieEtablissement,
                complementAdresse := e.complementAdresseEtablissement,
                postcode := e.codePostalEtablissement,
                city := e.libelleCommuneEtablissement,
                insee := e.codeCommuneEtablissement,
                etat_admin := e.etatAdministratifEtablissement,
                is_siege := e.etablissementSiege,
                sigle_ul := u.sigleUniteLegale,
                denomination_ul := u.denominationUniteLegale,
                denomination_usuelle_ul := concat_ws(' ',
                    u.denominationUsuelle1UniteLegale,
                    u.denominationUsuelle2UniteLegale,
                    u.denominationUsuelle3UniteLegale),
                nom_ul := u.nomUniteLegale,
                prenom_usuel_ul := u.prenomUsuelUniteLegale
            )) AS payload
        FROM {_relation(establishments)} e
        LEFT JOIN {_relation(legal_units)} u USING (siren)
        {establishment_join}
        {legal_join}
        {succession_join}
        WHERE e.siret IS NOT NULL AND length(CAST(e.siret AS VARCHAR)) = 14
        """
    )


def _schema(tantivy: Any) -> Any:
    builder = tantivy.SchemaBuilder()
    for field_name in ["document_type", "siret", "siren", "insee", "postcode"]:
        builder.add_text_field(
            field_name, stored=True, tokenizer_name="raw", index_option="basic"
        )
    for field_name in ["names", "addresses", "name_ngrams", "address_ngrams"]:
        builder.add_text_field(field_name, stored=field_name in {"names", "addresses"})
    for field_name in ["names_exact", "addresses_exact"]:
        builder.add_text_field(
            field_name, stored=False, tokenizer_name="raw", index_option="basic"
        )
    for field_name in [
        "number",
        "state",
        "is_siege",
        "linked_sirets",
        "payload",
    ]:
        builder.add_text_field(field_name, stored=True, tokenizer_name="raw", index_option="basic")
    return builder.build()


def _split_values(value: Any) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            normalized
            for part in str(value or "").split("|")
            if (normalized := normalize_text(part))
        )
    )


def _add_document(writer: Any, tantivy: Any, row: tuple[Any, ...]) -> None:
    (
        siret,
        siren,
        insee,
        postcode,
        number,
        number_suffix,
        state,
        is_siege,
        names_raw,
        addresses_raw,
        linked_sirets,
        payload_raw,
    ) = row
    names = _split_values(names_raw)
    addresses = _split_values(addresses_raw)
    document = tantivy.Document()
    scalar = {
        "document_type": "siret",
        "siret": normalize_code(siret, 14),
        "siren": normalize_code(siren, 9),
        "insee": normalize_insee(insee),
        "postcode": normalize_code(postcode, 5),
        "number": normalize_text(number),
        "state": str(state or "A").upper(),
        "is_siege": "1" if bool(is_siege) else "0",
        "linked_sirets": str(linked_sirets or ""),
    }
    try:
        source_payload = json.loads(payload_raw or "{}")
    except (TypeError, json.JSONDecodeError):
        source_payload = {}
    scalar["payload"] = json.dumps(
        {
            **source_payload,
            **scalar,
            "number_suffix": normalize_text(number_suffix),
            "names": list(names),
            "addresses": list(addresses),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    for key, value in scalar.items():
        document.add_text(key, value)
    for name in names:
        document.add_text("names", name)
        document.add_text("names_exact", name)
        document.add_text("name_ngrams", " ".join(character_ngrams(name)))
    for address in addresses:
        document.add_text("addresses", address)
        document.add_text("addresses_exact", address)
        document.add_text("address_ngrams", " ".join(character_ngrams(address)))
    writer.add_document(document)


def _add_siren_document(
    writer: Any, tantivy: Any, row: tuple[Any, ...]
) -> None:
    siren, insee, postcode, names_raw = row
    names = _split_values(names_raw)
    document = tantivy.Document()
    for key, value in {
        "document_type": "siren",
        "siret": "",
        "siren": normalize_code(siren, 9),
        "insee": normalize_insee(insee),
        "postcode": normalize_code(postcode, 5),
        "number": "",
        "state": "A",
        "is_siege": "0",
        "linked_sirets": "",
        "payload": "{}",
    }.items():
        document.add_text(key, value)
    for name in names:
        document.add_text("names", name)
        document.add_text("name_ngrams", " ".join(character_ngrams(name)))
    writer.add_document(document)


def _stream_query(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    batch_size: int,
) -> Iterable[list[tuple[Any, ...]]]:
    cursor = connection.execute(sql)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def build_index(args: argparse.Namespace) -> Path:
    try:
        import tantivy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Production index build requires tantivy==0.25.1. Install requirements.txt "
            "in the selected Python environment before running this script."
        ) from exc

    sources = {
        "establishments_current": args.establishments,
        "legal_units_current": args.legal_units,
        "establishments_history": args.establishments_history,
        "legal_units_history": args.legal_units_history,
        "successions": args.successions,
    }
    for role in ["establishments_current", "legal_units_current"]:
        if not sources[role] or not sources[role].is_file():
            raise FileNotFoundError(f"missing required source {role}: {sources[role]}")
    for role, path in sources.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"missing source {role}: {path}")

    source_meta = {
        role: (
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            if path
            else None
        )
        for role, path in sources.items()
    }
    if not args.retrieval_config.is_file():
        raise FileNotFoundError(f"missing retrieval config: {args.retrieval_config}")
    retrieval_config_sha256 = sha256_file(args.retrieval_config)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "source_content": {
            role: (
                {"sha256": metadata["sha256"], "size_bytes": metadata["size_bytes"]}
                if metadata
                else None
            )
            for role, metadata in source_meta.items()
        },
        "retrieval_config_sha256": retrieval_config_sha256,
        "smoke_limit": args.smoke_limit,
    }
    build_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / build_hash[:16]
    if final.exists():
        manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("build_hash") != build_hash:
            raise RuntimeError(f"content-address collision at {final}")
        return final

    stage = Path(tempfile.mkdtemp(prefix="hierarchical-index-", dir=output_root))
    try:
        tantivy_path = stage / "tantivy"
        tantivy_path.mkdir()
        schema = _schema(tantivy)
        index = tantivy.Index(schema, path=str(tantivy_path))
        writer = index.writer(heap_size=args.writer_heap_bytes, num_threads=args.writer_threads)
        connection = duckdb.connect()
        connection.execute(f"SET memory_limit='{args.duckdb_memory_limit}'")
        connection.execute(f"SET threads={args.duckdb_threads}")
        connection.execute(f"SET temp_directory='{_sql_path(args.temp_directory)}'")
        optional_joins = _prepare_optional_views(
            connection,
            establishment_history=args.establishments_history,
            legal_unit_history=args.legal_units_history,
            successions=args.successions,
        )
        _create_current_view(
            connection, args.establishments, args.legal_units, optional_joins
        )
        selected_scope = (
            f"SELECT * FROM index_rows ORDER BY siret LIMIT {args.smoke_limit}"
            if args.smoke_limit
            else "SELECT * FROM index_rows"
        )
        establishment_sql = f"""
            SELECT siret, siren, insee, postcode, number, number_suffix, state,
                   is_siege, names, addresses, linked_sirets, payload
            FROM ({selected_scope}) selected_rows
            ORDER BY siret
        """
        establishment_count = 0
        since_commit = 0
        for rows in _stream_query(connection, establishment_sql, batch_size=args.batch_size):
            for row in rows:
                _add_document(writer, tantivy, row)
            establishment_count += len(rows)
            since_commit += len(rows)
            if since_commit >= args.commit_every:
                writer.commit()
                since_commit = 0

        # True aggregate documents avoid a top-SIRET bias in SIREN retrieval.
        siren_sql = f"""
            SELECT siren, insee, postcode,
                   string_agg(DISTINCT names, ' | ') AS names
            FROM ({selected_scope}) selected_rows
            GROUP BY siren, insee, postcode
            ORDER BY siren, insee, postcode
        """
        siren_count = 0
        for rows in _stream_query(connection, siren_sql, batch_size=args.batch_size):
            for row in rows:
                _add_siren_document(writer, tantivy, row)
            siren_count += len(rows)
        writer.commit()
        index.reload()
        connection.close()

        missing_roles = [
            role
            for role in [
                "establishments_history",
                "legal_units_history",
                "successions",
            ]
            if sources[role] is None
        ]
        manifest = {
            **identity,
            "sources": source_meta,
            "retrieval_config_path": str(args.retrieval_config.resolve()),
            "build_hash": build_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": "tantivy",
            "tantivy_version": getattr(tantivy, "__version__", "unknown"),
            "build_scope": "smoke" if args.smoke_limit else "full",
            "num_establishment_documents": establishment_count,
            "num_siren_documents": siren_count,
            "num_documents": establishment_count + siren_count,
            "temporal_complete": not missing_roles,
            "missing_optional_roles": missing_roles,
            "contains_crm_labels": False,
            "limits": {
                "batch_size": args.batch_size,
                "commit_every": args.commit_every,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.rename(final)
        return final
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--establishments", type=Path, required=True)
    parser.add_argument("--legal-units", type=Path, required=True)
    parser.add_argument("--establishments-history", type=Path)
    parser.add_argument("--legal-units-history", type=Path)
    parser.add_argument("--successions", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=ROOT / "config" / "retrieval_hierarchical_v1.json",
    )
    parser.add_argument("--temp-directory", type=Path, default=Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp"))
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--commit-every", type=int, default=250_000)
    parser.add_argument("--writer-heap-bytes", type=int, default=512_000_000)
    parser.add_argument("--writer-threads", type=int, default=4)
    parser.add_argument("--duckdb-memory-limit", default="12GB")
    parser.add_argument("--duckdb-threads", type=int, default=8)
    parser.add_argument("--smoke-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.temp_directory.mkdir(parents=True, exist_ok=True)
    output = build_index(args)
    print(output)


if __name__ == "__main__":
    main()
