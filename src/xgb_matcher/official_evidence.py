"""Canonical, source-provenanced official evidence for SIRET retrieval.

The types in this module form the boundary between raw public snapshots and
retrieval.  They deliberately have no field for beneficial owners, directors,
or free-form announcement text.  BODACC/RNE material can therefore reach the
retrieval index only through allow-listed identity, name, address, status and
structured-identifier fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .hierarchical_retrieval import normalize_code, normalize_insee, normalize_text


OFFICIAL_EVIDENCE_SCHEMA_VERSION = "sireto-official-evidence-v1"
OFFICIAL_RELATION_SCHEMA_VERSION = "sireto-official-relation-v1"
OFFICIAL_QUARANTINE_SCHEMA_VERSION = "sireto-official-quarantine-v1"

_SIREN_RE = re.compile(r"^[0-9]{9}$")
_SIRET_RE = re.compile(r"^[0-9]{14}$")


class OfficialSource(str, Enum):
    SIRENE_CURRENT = "SIRENE_CURRENT"
    SIRENE_HISTORY = "SIRENE_HISTORY"
    SIRENE_SUCCESSION = "SIRENE_SUCCESSION"
    RNE = "RNE"
    BODACC = "BODACC"


class OfficialSubjectKind(str, Enum):
    SIRET = "SIRET"
    SIREN = "SIREN"


class OfficialNameKind(str, Enum):
    LEGAL = "LEGAL"
    USUAL = "USUAL"
    TRADE = "TRADE"
    SIGN = "SIGN"
    HISTORICAL = "HISTORICAL"


class OfficialRelationType(str, Enum):
    """Relations accepted from explicit identifier fields only."""

    ESTABLISHMENT_SUCCESSION = "ESTABLISHMENT_SUCCESSION"
    LEGAL_UNIT_SUCCESSION = "LEGAL_UNIT_SUCCESSION"
    ASSET_TRANSFER = "ASSET_TRANSFER"
    ESTABLISHMENT_OF_LEGAL_UNIT = "ESTABLISHMENT_OF_LEGAL_UNIT"


class QuarantineReason(str, Enum):
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    IDENTIFIER_MISMATCH = "IDENTIFIER_MISMATCH"
    MISSING_SUBJECT_IDENTIFIER = "MISSING_SUBJECT_IDENTIFIER"
    EMPTY_ALLOWLISTED_EVIDENCE = "EMPTY_ALLOWLISTED_EVIDENCE"
    OFFICIAL_REUSE_OPPOSITION = "OFFICIAL_REUSE_OPPOSITION"
    UNSUPPORTED_SNAPSHOT = "UNSUPPORTED_SNAPSHOT"
    MALFORMED_RECORD = "MALFORMED_RECORD"
    UNSTRUCTURED_RELATION = "UNSTRUCTURED_RELATION"
    RELATION_SELF_LOOP = "RELATION_SELF_LOOP"
    RELATION_KIND_MISMATCH = "RELATION_KIND_MISMATCH"
    DUPLICATE_LOWER_PRECEDENCE = "DUPLICATE_LOWER_PRECEDENCE"
    LOWER_PRECEDENCE_CURRENT_GEO_CONFLICT = (
        "LOWER_PRECEDENCE_CURRENT_GEO_CONFLICT"
    )
    AMBIGUOUS_TOP_PRECEDENCE_CURRENT_GEO = (
        "AMBIGUOUS_TOP_PRECEDENCE_CURRENT_GEO"
    )


# Higher values win only for canonical hard fields.  Alternative official
# names remain additive after provenance-preserving de-duplication.
SOURCE_PRECEDENCE: Mapping[OfficialSource, int] = {
    OfficialSource.SIRENE_CURRENT: 500,
    OfficialSource.SIRENE_SUCCESSION: 500,
    OfficialSource.SIRENE_HISTORY: 400,
    OfficialSource.RNE: 300,
    OfficialSource.BODACC: 200,
}


def source_precedence(source: OfficialSource | str) -> int:
    return SOURCE_PRECEDENCE[OfficialSource(source)]


def _iso_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    # The source may expose an ISO timestamp where the canonical contract only
    # needs day resolution.  Invalid dates are retained as absent, not guessed.
    candidate = text[:10]
    try:
        return date.fromisoformat(candidate).isoformat()
    except ValueError:
        return None


def normalize_siren(value: Any) -> str:
    text = normalize_code(value, 9)
    return text if _SIREN_RE.fullmatch(text) else ""


def normalize_siret(value: Any) -> str:
    text = normalize_code(value, 14)
    return text if _SIRET_RE.fullmatch(text) else ""


@dataclass(frozen=True, order=True)
class OfficialName:
    raw_value: str
    kind: OfficialNameKind = OfficialNameKind.USUAL
    normalized_value: str = ""

    def __post_init__(self) -> None:
        raw = str(self.raw_value or "").strip()
        normalized = normalize_text(self.normalized_value or raw)
        if not normalized:
            raise ValueError("official name cannot be empty")
        object.__setattr__(self, "raw_value", raw)
        object.__setattr__(self, "normalized_value", normalized)
        object.__setattr__(self, "kind", OfficialNameKind(self.kind))

    @property
    def value(self) -> str:
        """Backward-compatible normalized retrieval value."""
        return self.normalized_value

    def to_dict(self) -> dict[str, str]:
        return {
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OfficialName":
        return cls(
            raw_value=str(value.get("raw_value") or value.get("value") or ""),
            kind=value.get("kind", "USUAL"),
            normalized_value=str(value.get("normalized_value") or ""),
        )


@dataclass(frozen=True, order=True)
class OfficialAddress:
    raw_value: str
    postcode: str = ""
    insee: str = ""
    number: str = ""
    number_suffix: str = ""
    normalized_value: str = ""

    def __post_init__(self) -> None:
        raw = str(self.raw_value or "").strip()
        normalized_value = normalize_text(self.normalized_value or raw)
        postcode = normalize_code(self.postcode, 5) if self.postcode else ""
        if postcode and not re.fullmatch(r"[0-9]{5}", postcode):
            postcode = ""
        object.__setattr__(self, "raw_value", raw)
        object.__setattr__(self, "normalized_value", normalized_value)
        object.__setattr__(self, "postcode", postcode)
        object.__setattr__(self, "insee", normalize_insee(self.insee))
        object.__setattr__(self, "number", normalize_text(self.number))
        object.__setattr__(self, "number_suffix", normalize_text(self.number_suffix))
        if not any((normalized_value, postcode, self.insee)):
            raise ValueError("official address cannot be empty")

    @property
    def value(self) -> str:
        """Backward-compatible normalized retrieval value."""
        return self.normalized_value

    def to_dict(self) -> dict[str, str]:
        return {
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "postcode": self.postcode,
            "insee": self.insee,
            "number": self.number,
            "number_suffix": self.number_suffix,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OfficialAddress":
        return cls(
            raw_value=str(value.get("raw_value") or value.get("value") or ""),
            postcode=str(value.get("postcode") or ""),
            insee=str(value.get("insee") or ""),
            number=str(value.get("number") or ""),
            number_suffix=str(value.get("number_suffix") or ""),
            normalized_value=str(value.get("normalized_value") or ""),
        )


def _deduplicate_names(values: Iterable[OfficialName]) -> tuple[OfficialName, ...]:
    # Different official spellings remain auditable even when they normalize to
    # the same retrieval form; semantic de-duplication occurs at projection.
    return tuple(
        sorted(
            set(values),
            key=lambda item: (item.normalized_value, item.kind.value, item.raw_value),
        )
    )


def _deduplicate_addresses(
    values: Iterable[OfficialAddress],
) -> tuple[OfficialAddress, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda item: (
                item.insee,
                item.postcode,
                item.normalized_value,
                item.number,
                item.number_suffix,
                item.raw_value,
            ),
        )
    )


@dataclass(frozen=True)
class OfficialEvidence:
    source: OfficialSource
    source_record_id: str
    subject_kind: OfficialSubjectKind
    siren: str
    siret: str = ""
    names: tuple[OfficialName, ...] = ()
    addresses: tuple[OfficialAddress, ...] = ()
    administrative_state: str = ""
    is_headquarters: bool | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    observed_at: str | None = None
    is_current: bool = True
    evidence_id: str = field(default="", compare=True)
    schema_version: str = field(
        default=OFFICIAL_EVIDENCE_SCHEMA_VERSION, compare=False
    )

    def __post_init__(self) -> None:
        source = OfficialSource(self.source)
        subject_kind = OfficialSubjectKind(self.subject_kind)
        siren = normalize_siren(self.siren)
        siret = normalize_siret(self.siret) if self.siret else ""
        if not siren:
            raise ValueError("invalid SIREN")
        if subject_kind is OfficialSubjectKind.SIRET:
            if not siret:
                raise ValueError("SIRET evidence requires a valid SIRET")
            if not siret.startswith(siren):
                raise ValueError("SIRET/SIREN mismatch")
        elif siret:
            raise ValueError("SIREN evidence cannot carry a SIRET subject")
        names = _deduplicate_names(self.names)
        addresses = _deduplicate_addresses(self.addresses)
        state = normalize_text(self.administrative_state)
        if not any((names, addresses, state, self.is_headquarters is not None)):
            raise ValueError("empty allow-listed official evidence")
        source_record_id = str(self.source_record_id or "").strip()
        if not source_record_id:
            raise ValueError("source_record_id is required")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "subject_kind", subject_kind)
        object.__setattr__(self, "siren", siren)
        object.__setattr__(self, "siret", siret)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "addresses", addresses)
        object.__setattr__(self, "administrative_state", state)
        object.__setattr__(self, "valid_from", _iso_date(self.valid_from))
        object.__setattr__(self, "valid_to", _iso_date(self.valid_to))
        object.__setattr__(self, "observed_at", _iso_date(self.observed_at))
        object.__setattr__(self, "source_record_id", source_record_id)
        object.__setattr__(self, "schema_version", OFFICIAL_EVIDENCE_SCHEMA_VERSION)
        if not self.evidence_id:
            object.__setattr__(self, "evidence_id", self._stable_id())

    @property
    def subject_id(self) -> str:
        return self.siret if self.subject_kind is OfficialSubjectKind.SIRET else self.siren

    @property
    def priority(self) -> int:
        return source_precedence(self.source)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "source_record_id": self.source_record_id,
            "subject_kind": self.subject_kind.value,
            "siren": self.siren,
            "siret": self.siret,
            "names": [item.to_dict() for item in self.names],
            "addresses": [item.to_dict() for item in self.addresses],
            "administrative_state": self.administrative_state,
            "is_headquarters": self.is_headquarters,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "observed_at": self.observed_at,
            "is_current": self.is_current,
        }

    def _stable_id(self) -> str:
        payload = json.dumps(
            self._identity_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def semantic_key(self) -> str:
        payload = {
            key: value
            for key, value in self._identity_payload().items()
            if key not in {"source", "source_record_id", "observed_at"}
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def without_addresses(self) -> "OfficialEvidence":
        if not self.addresses:
            return self
        # A lower-precedence conflicting location is removed while its official
        # name/status observation remains usable under the authoritative site.
        return replace(self, addresses=(), evidence_id="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            **self._identity_payload(),
            "subject_id": self.subject_id,
            "source_priority": self.priority,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OfficialEvidence":
        names = value.get("names") or []
        addresses = value.get("addresses") or []
        if isinstance(names, str):
            names = json.loads(names)
        if isinstance(addresses, str):
            addresses = json.loads(addresses)
        return cls(
            source=value["source"],
            source_record_id=str(value["source_record_id"]),
            subject_kind=value["subject_kind"],
            siren=str(value["siren"]),
            siret=str(value.get("siret") or ""),
            names=tuple(OfficialName.from_dict(item) for item in names),
            addresses=tuple(OfficialAddress.from_dict(item) for item in addresses),
            administrative_state=str(value.get("administrative_state") or ""),
            is_headquarters=value.get("is_headquarters"),
            valid_from=value.get("valid_from"),
            valid_to=value.get("valid_to"),
            observed_at=value.get("observed_at"),
            is_current=bool(value.get("is_current", True)),
            evidence_id=str(value.get("evidence_id") or ""),
        )


@dataclass(frozen=True)
class OfficialRelation:
    source: OfficialSource
    source_record_id: str
    relation_type: OfficialRelationType
    from_kind: OfficialSubjectKind
    from_identifier: str
    to_kind: OfficialSubjectKind
    to_identifier: str
    effective_date: str | None = None
    observed_at: str | None = None
    relation_id: str = ""
    schema_version: str = OFFICIAL_RELATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        source = OfficialSource(self.source)
        relation_type = OfficialRelationType(self.relation_type)
        from_kind = OfficialSubjectKind(self.from_kind)
        to_kind = OfficialSubjectKind(self.to_kind)
        from_identifier = _normalize_identifier(from_kind, self.from_identifier)
        to_identifier = _normalize_identifier(to_kind, self.to_identifier)
        if not from_identifier or not to_identifier:
            raise ValueError("invalid relation identifier")
        if from_kind is to_kind and from_identifier == to_identifier:
            raise ValueError("relation self-loop")
        _validate_relation_kinds(relation_type, from_kind, to_kind)
        source_record_id = str(self.source_record_id or "").strip()
        if not source_record_id:
            raise ValueError("source_record_id is required")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "from_kind", from_kind)
        object.__setattr__(self, "to_kind", to_kind)
        object.__setattr__(self, "from_identifier", from_identifier)
        object.__setattr__(self, "to_identifier", to_identifier)
        object.__setattr__(self, "effective_date", _iso_date(self.effective_date))
        object.__setattr__(self, "observed_at", _iso_date(self.observed_at))
        object.__setattr__(self, "source_record_id", source_record_id)
        object.__setattr__(self, "schema_version", OFFICIAL_RELATION_SCHEMA_VERSION)
        if not self.relation_id:
            payload = json.dumps(
                self._identity_payload(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            object.__setattr__(self, "relation_id", hashlib.sha256(payload).hexdigest())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "source_record_id": self.source_record_id,
            "relation_type": self.relation_type.value,
            "from_kind": self.from_kind.value,
            "from_identifier": self.from_identifier,
            "to_kind": self.to_kind.value,
            "to_identifier": self.to_identifier,
            "effective_date": self.effective_date,
            "observed_at": self.observed_at,
        }

    def semantic_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.relation_type.value,
            self.from_kind.value,
            self.from_identifier,
            self.to_kind.value,
            self.to_identifier,
            self.effective_date or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relation_id": self.relation_id,
            **self._identity_payload(),
            "source_priority": source_precedence(self.source),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OfficialRelation":
        return cls(
            source=value["source"],
            source_record_id=str(value["source_record_id"]),
            relation_type=value["relation_type"],
            from_kind=value["from_kind"],
            from_identifier=str(value["from_identifier"]),
            to_kind=value["to_kind"],
            to_identifier=str(value["to_identifier"]),
            effective_date=value.get("effective_date"),
            observed_at=value.get("observed_at"),
            relation_id=str(value.get("relation_id") or ""),
        )


def _normalize_identifier(kind: OfficialSubjectKind, value: Any) -> str:
    return normalize_siret(value) if kind is OfficialSubjectKind.SIRET else normalize_siren(value)


def _validate_relation_kinds(
    relation_type: OfficialRelationType,
    from_kind: OfficialSubjectKind,
    to_kind: OfficialSubjectKind,
) -> None:
    expected = {
        OfficialRelationType.ESTABLISHMENT_SUCCESSION: (
            OfficialSubjectKind.SIRET,
            OfficialSubjectKind.SIRET,
        ),
        OfficialRelationType.LEGAL_UNIT_SUCCESSION: (
            OfficialSubjectKind.SIREN,
            OfficialSubjectKind.SIREN,
        ),
        OfficialRelationType.ASSET_TRANSFER: (
            OfficialSubjectKind.SIREN,
            OfficialSubjectKind.SIREN,
        ),
        OfficialRelationType.ESTABLISHMENT_OF_LEGAL_UNIT: (
            OfficialSubjectKind.SIRET,
            OfficialSubjectKind.SIREN,
        ),
    }[relation_type]
    if (from_kind, to_kind) != expected:
        raise ValueError("relation endpoint kind mismatch")


@dataclass(frozen=True)
class QuarantinedOfficialRecord:
    source: OfficialSource
    snapshot_role: str
    source_record_id: str
    reason: QuarantineReason
    detail: str
    record_fingerprint: str
    schema_version: str = OFFICIAL_QUARANTINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", OfficialSource(self.source))
        object.__setattr__(self, "reason", QuarantineReason(self.reason))
        object.__setattr__(self, "source_record_id", str(self.source_record_id or ""))
        # Quarantine stores neither the source record nor free-form source text.
        object.__setattr__(self, "detail", normalize_text(self.detail)[:240])
        if not re.fullmatch(r"[0-9a-f]{64}", self.record_fingerprint):
            raise ValueError("record_fingerprint must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.value,
            "snapshot_role": self.snapshot_role,
            "source_record_id": self.source_record_id,
            "reason": self.reason.value,
            "detail": self.detail,
            "record_fingerprint": self.record_fingerprint,
        }


def record_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash a raw record without retaining any raw/sensitive field."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def official_evidence_arrow_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("evidence_id", pa.string()),
            ("source", pa.string()),
            ("source_record_id", pa.string()),
            ("subject_kind", pa.string()),
            ("subject_id", pa.string()),
            ("siren", pa.string()),
            ("siret", pa.string()),
            (
                "names",
                pa.list_(
                    pa.struct(
                        [
                            ("raw_value", pa.string()),
                            ("normalized_value", pa.string()),
                            ("kind", pa.string()),
                        ]
                    )
                ),
            ),
            (
                "addresses",
                pa.list_(
                    pa.struct(
                        [
                            ("raw_value", pa.string()),
                            ("normalized_value", pa.string()),
                            ("postcode", pa.string()),
                            ("insee", pa.string()),
                            ("number", pa.string()),
                            ("number_suffix", pa.string()),
                        ]
                    )
                ),
            ),
            ("administrative_state", pa.string()),
            ("is_headquarters", pa.bool_()),
            ("valid_from", pa.string()),
            ("valid_to", pa.string()),
            ("observed_at", pa.string()),
            ("is_current", pa.bool_()),
            ("source_priority", pa.int16()),
        ]
    )


def official_relation_arrow_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("relation_id", pa.string()),
            ("source", pa.string()),
            ("source_record_id", pa.string()),
            ("relation_type", pa.string()),
            ("from_kind", pa.string()),
            ("from_identifier", pa.string()),
            ("to_kind", pa.string()),
            ("to_identifier", pa.string()),
            ("effective_date", pa.string()),
            ("observed_at", pa.string()),
            ("source_priority", pa.int16()),
        ]
    )


def official_quarantine_arrow_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("source", pa.string()),
            ("snapshot_role", pa.string()),
            ("source_record_id", pa.string()),
            ("reason", pa.string()),
            ("detail", pa.string()),
            ("record_fingerprint", pa.string()),
        ]
    )
