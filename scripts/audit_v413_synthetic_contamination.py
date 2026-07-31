#!/usr/bin/env python3
"""Synthetic proof of the three applicable V4.13 anti-overlap projections."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

try:
    from scripts import build_v412_consumed_compatibility_registry as historical
except ModuleNotFoundError:
    import build_v412_consumed_compatibility_registry as historical


APPLICABLE = ("service_id", "siret_masked", "fuzzy_historical")


class ContaminationStop(RuntimeError):
    pass


def _stop(message: str) -> None:
    raise ContaminationStop(f"STOP_V413_SYNTHETIC_CONTAMINATION: {message}")


def _historical_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "source_record_id",
        "source_record_id_equivalence_attested",
        "crm_name_raw",
        "crm_address_raw",
        "crm_postcode_raw",
        "crm_city_raw",
        "crm_insee_raw",
    }
    if not required <= set(row):
        _stop("CRM projection fields missing")
    if row["source_record_id_equivalence_attested"] is not True:
        _stop("source_record_id equivalence must be attested")
    source_id = row["source_record_id"]
    if not isinstance(source_id, str) or not source_id:
        _stop("source_record_id must be nonempty")
    return {
        "SITE": row["crm_name_raw"],
        "CODE_POSTAL": row["crm_postcode_raw"],
        "CODE_INSEE": row["crm_insee_raw"],
        "SERVICE ID": source_id,
        "COMMUNE": row["crm_city_raw"],
        "SIRET": "",
        "SITE_CLI_ADRESSE": row["crm_address_raw"],
        "SITE_CLI_COMMUNE": row["crm_city_raw"],
    }


def audit_synthetic_contamination(
    *,
    crm_rows: Iterable[Mapping[str, Any]],
    split_rows: Iterable[Mapping[str, Any]],
    keysets: Mapping[str, set[str]],
    hmac_key: bytes,
    synthetic_only: bool,
) -> dict[str, Any]:
    if synthetic_only is not True:
        _stop("real registry access forbidden")
    if set(keysets) != {*APPLICABLE, "consumed_sirens"}:
        _stop("keysets must be exactly the three applicable sets plus SIREN")
    if not isinstance(hmac_key, bytes) or len(hmac_key) < 32:
        _stop("synthetic HMAC key must contain at least 256 bits")
    if any(
        not isinstance(values, set)
        or any(not isinstance(value, str) for value in values)
        for values in keysets.values()
    ):
        _stop("keysets must be string sets")

    rows = list(crm_rows)
    private = list(split_rows)
    if not rows or len(rows) != len(private):
        _stop("CRM and private split rows must be nonempty and equal length")

    hits: dict[str, set[int]] = {name: set() for name in APPLICABLE}
    observed: dict[str, int] = {name: 0 for name in APPLICABLE}
    for ordinal, source in enumerate(rows, 1):
        projection = _historical_projection(source)
        service_norm = historical.canonical_text(projection["SERVICE ID"])
        service_digest = historical.lineage_hmac(
            hmac_key,
            historical.SERVICE_DOMAIN,
            service_norm,
        )
        masked_digest = historical.siret_masked_fingerprint(projection)
        fuzzy_digests = {
            digest
            for _, _, digest in historical.fuzzy_singletons(projection)
        }
        observed["service_id"] += 1
        observed["siret_masked"] += 1
        observed["fuzzy_historical"] += len(fuzzy_digests)
        if service_digest in keysets["service_id"]:
            hits["service_id"].add(ordinal)
        if masked_digest in keysets["siret_masked"]:
            hits["siret_masked"].add(ordinal)
        if fuzzy_digests & keysets["fuzzy_historical"]:
            hits["fuzzy_historical"].add(ordinal)

    known_sirens: set[str] = set()
    for row in private:
        sirens = row.get("authoritative_sirens")
        if (
            not isinstance(sirens, list)
            or sirens != sorted(set(sirens))
            or any(
                not isinstance(siren, str)
                or len(siren) != 9
                or not siren.isascii()
                or not siren.isdigit()
                for siren in sirens
            )
        ):
            _stop("private authoritative_sirens invalid")
        known_sirens.update(sirens)
    siren_hits = known_sirens & keysets["consumed_sirens"]
    union_rows = set().union(*hits.values())
    report = {
        "schema_version": "sireto-v4.13-synthetic-contamination-audit-1",
        "synthetic_only": True,
        "source_row_count": len(rows),
        "applicable_keysets": list(APPLICABLE),
        "excluded_keyset": "input_siret_lineage",
        "comparable_observation_counts": observed,
        "hit_row_counts": {
            name: len(values) for name, values in hits.items()
        },
        "keyset_union_hit_row_count": len(union_rows),
        "authoritatively_known_siren_count": len(known_sirens),
        "consumed_siren_hit_count": len(siren_hits),
        "verdict": "ZERO_FORBIDDEN_OVERLAP",
    }
    if union_rows or siren_hits:
        _stop("forbidden historical overlap")
    return report
