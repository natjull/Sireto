"""Open-set hierarchical sparse retrieval for SIRET candidates.

This module is intentionally isolated from the historical TF-IDF retriever.  It
does not learn entity aliases from CRM labels.  Current and historical strings
must come from official SIRENE/RNE inputs recorded by the index manifest.

The production backend is Tantivy.  ``InMemoryBackend`` exists for small smoke
tests and deterministic unit tests only; production construction refuses it
unless the caller opts into the test backend explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Protocol, Sequence


_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_LEADING_STREET_NUMBER_RE = re.compile(r"^(\d{1,5})(?:\s|$)")


def normalize_text(value: Any) -> str:
    """Return a deterministic, locale-independent retrieval representation."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = _NON_ALNUM_RE.sub(" ", text.upper())
    return _SPACE_RE.sub(" ", text).strip()


def normalize_code(value: Any, width: int) -> str:
    text = re.sub(r"\D", "", str(value or ""))
    return text.zfill(width) if text and len(text) <= width else text


def normalize_insee(value: Any) -> str:
    """Normalize INSEE without destroying Corsican 2A/2B commune codes."""
    return re.sub(r"[^0-9A-Z]", "", str(value or "").upper())


def _leading_street_number(address: str) -> str:
    match = _LEADING_STREET_NUMBER_RE.match(address)
    return match.group(1) if match else ""


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().upper()
    if normalized in {"1", "TRUE", "T", "YES", "Y", "OUI", "A"}:
        return True
    if normalized in {"0", "FALSE", "F", "NO", "N", "NON"}:
        return False
    return default


def character_ngrams(value: str, minimum: int = 3, maximum: int = 5) -> tuple[str, ...]:
    compact = f" {normalize_text(value).replace(' ', '_')} "
    if not compact.strip(" _"):
        return ()
    return tuple(
        sorted(
            {
                f"G{size}_{compact[index : index + size].encode('utf-8').hex().upper()}"
                for size in range(minimum, maximum + 1)
                for index in range(max(0, len(compact) - size + 1))
            }
        )
    )


@dataclass(frozen=True)
class HierarchicalRetrievalConfig:
    schema_version: str = "sireto-hierarchical-retrieval-v1"
    enabled: bool = False
    backend: str = "tantivy"
    require_tantivy_in_production: bool = True
    max_candidates: int = 100
    union_cap: int = 1000
    direct_top_k: int = 300
    name_char_top_k: int = 150
    address_char_top_k: int = 150
    siren_top_k: int = 5
    sites_per_siren: int = 32
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    include_closed: bool = True
    include_official_history: bool = True
    include_official_successions: bool = True
    rrf_k: int = 60
    direct_slots: int = 60
    hierarchical_slots: int = 30
    character_rescue_slots: int = 10
    typed_name_portfolio: bool = False
    historical_name_top_k: int = 100
    supporting_name_top_k: int = 50

    def __post_init__(self) -> None:
        if self.max_candidates < 1 or self.max_candidates > 100:
            raise ValueError("max_candidates must be between 1 and 100")
        if self.union_cap < self.max_candidates or self.union_cap > 2000:
            raise ValueError("union_cap must be between max_candidates and 2000")
        if self.siren_top_k not in {3, 5}:
            raise ValueError("siren_top_k must be 3 or 5")
        if self.sites_per_siren < 1 or self.sites_per_siren > 32:
            raise ValueError("sites_per_siren must be between 1 and 32")
        if self.char_ngram_min < 1 or self.char_ngram_max < self.char_ngram_min:
            raise ValueError("invalid character n-gram range")
        slots = self.direct_slots + self.hierarchical_slots + self.character_rescue_slots
        if slots != self.max_candidates:
            raise ValueError("fusion allocation must sum to max_candidates")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HierarchicalRetrievalConfig":
        limits = raw.get("limits") or {}
        analysis = raw.get("analysis") or {}
        fusion = raw.get("fusion") or {}
        allocation = fusion.get("allocation") or {}
        return cls(
            schema_version=str(raw.get("schema_version", cls.schema_version)),
            enabled=bool(raw.get("enabled", False)),
            backend=str(raw.get("backend", "tantivy")),
            require_tantivy_in_production=bool(
                raw.get("require_tantivy_in_production", True)
            ),
            max_candidates=int(limits.get("max_candidates", 100)),
            union_cap=int(limits.get("union_cap", 1000)),
            direct_top_k=int(limits.get("direct_top_k", 300)),
            name_char_top_k=int(limits.get("name_char_top_k", 150)),
            address_char_top_k=int(limits.get("address_char_top_k", 150)),
            siren_top_k=int(limits.get("siren_top_k", 5)),
            sites_per_siren=int(limits.get("sites_per_siren", 32)),
            char_ngram_min=int(analysis.get("char_ngram_min", 3)),
            char_ngram_max=int(analysis.get("char_ngram_max", 5)),
            include_closed=bool(analysis.get("include_closed", True)),
            include_official_history=bool(
                analysis.get("include_official_history", True)
            ),
            include_official_successions=bool(
                analysis.get("include_official_successions", True)
            ),
            rrf_k=int(fusion.get("rrf_k", 60)),
            direct_slots=int(allocation.get("direct", 60)),
            hierarchical_slots=int(allocation.get("hierarchical", 30)),
            character_rescue_slots=int(allocation.get("character_rescue", 10)),
            typed_name_portfolio=bool(analysis.get("typed_name_portfolio", False)),
            historical_name_top_k=int(limits.get("historical_name_top_k", 100)),
            supporting_name_top_k=int(limits.get("supporting_name_top_k", 50)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "HierarchicalRetrievalConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RetrievalQuery:
    name: str
    address: str = ""
    number: str = ""
    postcode: str = ""
    insee: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "RetrievalQuery":
        address = normalize_text(
            row.get("crm_addr")
            or row.get("crm_address")
            or row.get("crm_address_raw")
            or row.get("address")
            or row.get("adresse")
            or ""
        )
        return cls(
            name=normalize_text(
                row.get("crm_name")
                or row.get("crm_name_raw")
                or row.get("name")
                or row.get("nom")
                or ""
            ),
            address=address,
            number=normalize_text(
                row.get("numeroVoie")
                or row.get("number")
                or row.get("numero")
                or _leading_street_number(address)
            ),
            postcode=normalize_code(
                row.get("postcode")
                or row.get("crm_cp")
                or row.get("crm_postcode")
                or row.get("crm_postcode_raw"),
                5,
            ),
            insee=normalize_insee(
                row.get("insee") or row.get("crm_insee") or row.get("crm_insee_raw")
            ),
        )


@dataclass(frozen=True)
class IndexedEstablishment:
    siret: str
    siren: str
    insee: str
    postcode: str
    names: tuple[str, ...]
    addresses: tuple[str, ...]
    legal_names: tuple[str, ...] = ()
    trade_names: tuple[str, ...] = ()
    site_names: tuple[str, ...] = ()
    historical_names: tuple[str, ...] = ()
    supporting_names: tuple[str, ...] = ()
    historical_addresses: tuple[str, ...] = ()
    supporting_addresses: tuple[str, ...] = ()
    number: str = ""
    active: bool = True
    is_siege: bool = False
    linked_sirets: tuple[str, ...] = ()
    linked_sirens: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "IndexedEstablishment":
        def values(key: str) -> tuple[str, ...]:
            raw_values = raw.get(key) or []
            if isinstance(raw_values, str):
                raw_values = raw_values.split("|")
            return tuple(dict.fromkeys(filter(None, map(normalize_text, raw_values))))

        def identifiers(value: Any, width: int) -> tuple[str, ...]:
            if isinstance(value, str):
                value = value.replace("|", " ").split()
            return tuple(sorted({normalize_code(item, width) for item in (value or []) if item}))

        legal_names = values("legal_names") or values("legal_current_names")
        trade_names = values("trade_names") or values("trade_current_names")
        site_names = values("site_names") or values("site_current_names")
        historical_names = values("historical_names")
        supporting_names = values("supporting_names")
        names = values("names") or tuple(
            dict.fromkeys(legal_names + trade_names + site_names + historical_names + supporting_names)
        )
        addresses = values("addresses") or values("current_address_text")
        historical_addresses = values("historical_addresses") or values("historical_address_text")
        supporting_addresses = values("supporting_addresses") or values("supporting_address_text")
        siret = normalize_code(raw.get("siret"), 14)
        siren = normalize_code(raw.get("siren") or siret[:9], 9)
        return cls(
            siret=siret,
            siren=siren,
            insee=normalize_insee(raw.get("insee")),
            postcode=normalize_code(raw.get("postcode"), 5),
            names=names,
            addresses=addresses,
            legal_names=legal_names,
            trade_names=trade_names,
            site_names=site_names,
            historical_names=historical_names,
            supporting_names=supporting_names,
            historical_addresses=historical_addresses,
            supporting_addresses=supporting_addresses,
            number=normalize_text(raw.get("number") or raw.get("numeroVoie")),
            active=(
                str(raw.get("state") or raw.get("etat_admin") or "A").upper() != "F"
                if raw.get("active") is None
                else parse_bool(raw.get("active"), True)
            ),
            is_siege=parse_bool(
                raw.get("is_siege")
                if raw.get("is_siege") is not None
                else raw.get("etablissementSiege"),
                False,
            ),
            linked_sirets=identifiers(
                raw.get("linked_sirets") or raw.get("successor_sirets"), 14
            ),
            linked_sirens=identifiers(raw.get("linked_sirens"), 9),
            payload=dict(raw),
        )


@dataclass(frozen=True)
class BackendHit:
    record: IndexedEstablishment
    score: float


class RetrievalBackend(Protocol):
    def search(
        self, query: RetrievalQuery, channel: str, limit: int
    ) -> list[BackendHit]: ...

    def search_sirens(
        self, query: RetrievalQuery, limit: int
    ) -> list[tuple[str, float]]: ...

    def sites_for_siren(
        self, siren: str, query: RetrievalQuery, limit: int
    ) -> list[IndexedEstablishment]: ...

    def by_siret(self, siret: str) -> IndexedEstablishment | None: ...


def _tokens(value: str) -> set[str]:
    return set(normalize_text(value).split())


def _overlap(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    return intersection / math.sqrt(len(left_tokens) * len(right_tokens))


def _char_overlap(left: str, right: str, minimum: int, maximum: int) -> float:
    left_grams = set(character_ngrams(left, minimum, maximum))
    right_grams = set(character_ngrams(right, minimum, maximum))
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / math.sqrt(len(left_grams) * len(right_grams))


class InMemoryBackend:
    """Deterministic reference backend for tests and bounded smoke checks."""

    def __init__(
        self,
        records: Iterable[IndexedEstablishment | Mapping[str, Any]],
        *,
        char_ngram_min: int = 3,
        char_ngram_max: int = 5,
    ) -> None:
        self.records = tuple(
            record
            if isinstance(record, IndexedEstablishment)
            else IndexedEstablishment.from_mapping(record)
            for record in records
        )
        self.minimum = char_ngram_min
        self.maximum = char_ngram_max
        self._by_siret = {record.siret: record for record in self.records}
        self._by_siren: dict[str, list[IndexedEstablishment]] = defaultdict(list)
        for record in self.records:
            self._by_siren[record.siren].append(record)

    def _geographic_records(self, query: RetrievalQuery) -> tuple[IndexedEstablishment, ...]:
        insee_rows = tuple(record for record in self.records if query.insee and record.insee == query.insee)
        if insee_rows:
            return insee_rows
        return tuple(
            record
            for record in self.records
            if query.postcode and record.postcode == query.postcode
        )

    def _score(self, record: IndexedEstablishment, query: RetrievalQuery, channel: str) -> float:
        names = record.names or ("",)
        addresses = record.addresses or ("",)
        typed_names = {
            "legal_name_word": record.legal_names,
            "trade_name_word": record.trade_names,
            "site_name_word": record.site_names,
            "historical_name_word": record.historical_names,
            "supporting_name_word": record.supporting_names,
        }
        typed_exact = {
            "legal_name_exact": record.legal_names,
            "trade_name_exact": record.trade_names,
            "site_name_exact": record.site_names,
        }
        if channel in typed_names:
            return max((_overlap(query.name, name) for name in typed_names[channel]), default=0.0)
        if channel in typed_exact:
            return 1.0 if query.name and query.name in typed_exact[channel] else 0.0
        if channel == "fielded_name_bm25":
            return max(
                max((_overlap(query.name, name) for name in record.legal_names), default=0.0) * 1.25,
                max((_overlap(query.name, name) for name in record.trade_names), default=0.0) * 1.50,
                max((_overlap(query.name, name) for name in record.site_names), default=0.0) * 1.75,
            )
        if channel == "number_exact":
            return 1.0 if query.number and query.number == record.number else 0.0
        if channel == "current_name_char":
            current = record.legal_names + record.trade_names + record.site_names
            return max(
                (_char_overlap(query.name, name, self.minimum, self.maximum) for name in current),
                default=0.0,
            )
        if channel == "historical_address_word":
            return max((_overlap(query.address, value) for value in record.historical_addresses), default=0.0)
        if channel == "supporting_address_word":
            return max((_overlap(query.address, value) for value in record.supporting_addresses), default=0.0)
        if channel == "name_word":
            return max(_overlap(query.name, name) for name in names)
        if channel == "name_char":
            return max(_char_overlap(query.name, name, self.minimum, self.maximum) for name in names)
        if channel == "address_word":
            return max(_overlap(query.address, address) for address in addresses)
        if channel == "address_char":
            return max(_char_overlap(query.address, address, self.minimum, self.maximum) for address in addresses)
        if channel == "name_exact":
            return 1.0 if query.name and query.name in names else 0.0
        if channel == "address_exact":
            return 1.0 if query.address and query.address in addresses else 0.0
        raise ValueError(f"unknown retrieval channel: {channel}")

    def search(self, query: RetrievalQuery, channel: str, limit: int) -> list[BackendHit]:
        hits = [
            BackendHit(record, score)
            for record in self._geographic_records(query)
            if (score := self._score(record, query, channel)) > 0.0
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.record.siret))
        return hits[:limit]

    def search_sirens(self, query: RetrievalQuery, limit: int) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for record in self._geographic_records(query):
            name_score = max((_overlap(query.name, name) for name in record.names), default=0.0)
            char_score = max(
                (_char_overlap(query.name, name, self.minimum, self.maximum) for name in record.names),
                default=0.0,
            )
            score = max(name_score, 0.85 * char_score)
            scores[record.siren] = max(scores.get(record.siren, 0.0), score)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def sites_for_siren(
        self, siren: str, query: RetrievalQuery, limit: int
    ) -> list[IndexedEstablishment]:
        rows = [
            record
            for record in self._by_siren.get(siren, [])
            if (query.insee and record.insee == query.insee)
            or (not query.insee and query.postcode and record.postcode == query.postcode)
        ]
        rows.sort(key=lambda record: _site_sort_key(record, query))
        return rows[:limit]

    def linked_sirens(self, siren: str, limit: int = 256) -> tuple[str, ...]:
        normalized = normalize_code(siren, 9)
        linked = {
            value
            for record in self._by_siren.get(normalized, [])
            for value in record.linked_sirens
        }
        return tuple(sorted(linked))[:limit]

    def by_siret(self, siret: str) -> IndexedEstablishment | None:
        return self._by_siret.get(normalize_code(siret, 14))


class TantivyBackend:
    """Thin adapter over a content-addressed Tantivy index."""

    def __init__(
        self,
        index_path: Path | str,
        *,
        fielded_name_boosts: Mapping[str, float] | None = None,
    ) -> None:
        try:
            import tantivy  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The hierarchical production retriever requires Python package "
                "'tantivy'. Install requirements.txt in the runtime environment; "
                "the in-memory backend is test-only."
            ) from exc
        self._tantivy = tantivy
        self.index_path = Path(index_path)
        if not (self.index_path / "manifest.json").exists():
            raise FileNotFoundError(f"missing hierarchical index manifest: {self.index_path}")
        self.index = tantivy.Index.open(str(self.index_path / "tantivy"))
        self.manifest = json.loads(
            (self.index_path / "manifest.json").read_text(encoding="utf-8")
        )
        self.index.reload()
        self.searcher = self.index.searcher()
        self.fielded_name_boosts = dict(
            fielded_name_boosts
            or {"legal_names": 1.25, "trade_names": 1.50, "site_names": 1.75}
        )
        self._insee_exists: dict[str, bool] = {}

    @staticmethod
    def _first(document: Mapping[str, Any], key: str, default: Any = "") -> Any:
        value = document.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value

    def _record(self, document: Mapping[str, Any]) -> IndexedEstablishment:
        payload_raw = self._first(document, "payload", "{}")
        try:
            payload = json.loads(payload_raw)
        except (TypeError, json.JSONDecodeError):
            payload = {}
        def all_values(key: str) -> list[str]:
            value = document.get(key, [])
            if isinstance(value, list):
                return [str(item) for item in value if item]
            return [str(value)] if value else []

        return IndexedEstablishment.from_mapping(
            {
                **payload,
                "siret": self._first(document, "siret"),
                "siren": self._first(document, "siren"),
                "insee": self._first(document, "insee"),
                "postcode": self._first(document, "postcode"),
                "names": all_values("names"),
                "addresses": all_values("addresses"),
                "legal_names": all_values("legal_names"),
                "trade_names": all_values("trade_names"),
                "site_names": all_values("site_names"),
                "historical_names": all_values("historical_names"),
                "supporting_names": all_values("supporting_names"),
                "historical_addresses": all_values("historical_addresses"),
                "supporting_addresses": all_values("supporting_addresses"),
                "number": self._first(document, "number"),
                "state": self._first(document, "state", "A"),
                "is_siege": str(self._first(document, "is_siege", "0")) == "1",
                "linked_sirets": str(
                    self._first(document, "linked_sirets", "")
                ).split(),
                "linked_sirens": str(
                    self._first(document, "linked_sirens", "")
                ).split(),
            }
        )

    def _combined_query(
        self,
        query: RetrievalQuery,
        text_query: Any,
        *,
        document_type: str,
        siren: str | None = None,
    ) -> Any:
        use_insee = bool(query.insee and self._has_insee(query.insee))
        geo_field, geo_value = (
            ("insee", query.insee) if use_insee else ("postcode", query.postcode)
        )
        if not geo_value:
            return self._tantivy.Query.empty_query()
        subqueries = [
            (self._tantivy.Occur.Must, text_query),
            (
                self._tantivy.Occur.Must,
                self._tantivy.Query.term_query(
                    self.index.schema, "document_type", document_type, "basic"
                ),
            ),
            (
                self._tantivy.Occur.Must,
                self._tantivy.Query.term_query(
                    self.index.schema, geo_field, geo_value, "basic"
                ),
            ),
        ]
        if siren:
            subqueries.append(
                (
                    self._tantivy.Occur.Must,
                    self._tantivy.Query.term_query(
                        self.index.schema, "siren", normalize_code(siren, 9), "basic"
                    ),
                )
            )
        return self._tantivy.Query.boolean_query(subqueries)

    def _has_insee(self, insee: str) -> bool:
        normalized = normalize_insee(insee)
        if normalized not in self._insee_exists:
            geo_query = self._tantivy.Query.boolean_query(
                [
                    (
                        self._tantivy.Occur.Must,
                        self._tantivy.Query.term_query(
                            self.index.schema, "document_type", "siret", "basic"
                        ),
                    ),
                    (
                        self._tantivy.Occur.Must,
                        self._tantivy.Query.term_query(
                            self.index.schema, "insee", normalized, "basic"
                        ),
                    ),
                ]
            )
            self._insee_exists[normalized] = bool(
                self.searcher.search(geo_query, limit=1).hits
            )
        return self._insee_exists[normalized]

    def _query(
        self,
        query: RetrievalQuery,
        fields: Sequence[str],
        text: str,
        limit: int,
        *,
        exact: bool = False,
        document_type: str = "siret",
    ) -> list[tuple[float, Mapping[str, Any]]]:
        if not text:
            return []
        if exact:
            text_query = self._tantivy.Query.term_query(
                self.index.schema, fields[0], text, "basic"
            )
        else:
            # CRM values are data, not Tantivy query syntax.  In particular,
            # perfectly legitimate names can end in reserved words such as
            # "OR" (for example "HOTEL DE LA COTE D OR").  The strict parser
            # raises on those inputs and can abort an entire evaluation or
            # production batch.  Lenient parsing preserves the usable lexical
            # terms while treating malformed syntax as data-quality noise.
            text_query, _parse_errors = self.index.parse_query_lenient(
                text,
                default_field_names=list(fields),
                conjunction_by_default=False,
            )
        parsed = self._combined_query(query, text_query, document_type=document_type)
        result = self.searcher.search(parsed, limit=limit)
        return [
            (float(score), self.searcher.doc(address).to_dict())
            for score, address in result.hits
        ]

    def _character_query(self, field: str, text: str) -> Any:
        """Build the n-gram query without the analyzer's shared G3/G4/G5 tokens.

        The v1 index stores encoded values such as ``G3_484F54`` with the
        default analyzer.  Tantivy consequently indexes both ``g3`` and the
        hexadecimal payload.  Feeding the encoded strings back through the
        query parser adds the ultra-common ``g3``/``g4``/``g5`` terms and turns
        every character lookup into a near-global scan.  Querying only the
        hexadecimal term preserves the actual n-gram signal and avoids that
        pathological fan-out without rebuilding the frozen national index.
        """
        payload_terms = sorted(
            {
                gram.split("_", 1)[1].lower()
                for gram in character_ngrams(text)
                if "_" in gram
            }
        )
        if not payload_terms:
            return self._tantivy.Query.empty_query()
        return self._tantivy.Query.boolean_query(
            [
                (
                    self._tantivy.Occur.Should,
                    self._tantivy.Query.term_query(
                        self.index.schema, field, term, "basic"
                    ),
                )
                for term in payload_terms
            ]
        )

    def _fielded_name_query(self, text: str) -> Any:
        """Field-aware BM25 disjunction with fixed, manifest-pinned boosts.

        Tantivy scores each field with BM25 before applying the boost.  Keeping
        the three component scores as a dedicated retrieval channel avoids the
        length bias of the historical concatenated name bag; LambdaMART still
        receives every individual field channel separately.
        """
        if not text:
            return self._tantivy.Query.empty_query()
        clauses = []
        for field, boost in sorted(self.fielded_name_boosts.items()):
            parsed, _errors = self.index.parse_query_lenient(
                text, default_field_names=[field], conjunction_by_default=False
            )
            clauses.append(
                (
                    self._tantivy.Occur.Should,
                    self._tantivy.Query.boost_query(parsed, float(boost)),
                )
            )
        return self._tantivy.Query.boolean_query(clauses)

    def search(self, query: RetrievalQuery, channel: str, limit: int) -> list[BackendHit]:
        fields_and_text = {
            "name_word": (("names",), query.name, False),
            "name_char": (("name_ngrams",), " ".join(character_ngrams(query.name)), False),
            "address_word": (("addresses",), query.address, False),
            "address_char": (("address_ngrams",), " ".join(character_ngrams(query.address)), False),
            "name_exact": (("names_exact",), query.name, True),
            "address_exact": (("addresses_exact",), query.address, True),
            "legal_name_word": (("legal_names",), query.name, False),
            "trade_name_word": (("trade_names",), query.name, False),
            "site_name_word": (("site_names",), query.name, False),
            "historical_name_word": (("historical_names",), query.name, False),
            "supporting_name_word": (("supporting_names",), query.name, False),
            "legal_name_exact": (("legal_names_exact",), query.name, True),
            "trade_name_exact": (("trade_names_exact",), query.name, True),
            "site_name_exact": (("site_names_exact",), query.name, True),
            "number_exact": (("number",), query.number, True),
            "historical_address_word": (("historical_addresses",), query.address, False),
            "supporting_address_word": (("supporting_addresses",), query.address, False),
            "current_name_char": (("current_name_ngrams",), " ".join(character_ngrams(query.name)), False),
        }
        if channel == "fielded_name_bm25":
            parsed = self._combined_query(
                query, self._fielded_name_query(query.name), document_type="siret"
            )
            result = self.searcher.search(parsed, limit=limit)
            rows = [
                (float(score), self.searcher.doc(address).to_dict())
                for score, address in result.hits
            ]
        elif channel not in fields_and_text:
            raise ValueError(f"unknown retrieval channel: {channel}")
        else:
            fields, text, exact = fields_and_text[channel]
        if channel == "fielded_name_bm25":
            pass
        elif channel in {"name_char", "address_char", "current_name_char"}:
            parsed = self._combined_query(
                query,
                self._character_query(
                    fields[0],
                    query.name if channel in {"name_char", "current_name_char"} else query.address,
                ),
                document_type="siret",
            )
            result = self.searcher.search(parsed, limit=limit)
            rows = [
                (float(score), self.searcher.doc(address).to_dict())
                for score, address in result.hits
            ]
        else:
            rows = self._query(
                query, fields, text, limit, exact=exact, document_type="siret"
            )
        hits = [BackendHit(self._record(document), score) for score, document in rows]
        return sorted(hits, key=lambda hit: (-hit.score, hit.record.siret))

    def search_sirens(self, query: RetrievalQuery, limit: int) -> list[tuple[str, float]]:
        siren_fields = (
            ["legal_names", "trade_names", "site_names"]
            if self.manifest.get("typed_name_portfolio")
            else ["names"]
        )
        lexical, _lexical_errors = self.index.parse_query_lenient(
            query.name, default_field_names=siren_fields
        )
        # SIREN expansion is a small hierarchical rescue channel.  Its old
        # character disjunction dominated latency on the 39M geo/SIREN docs;
        # direct SIRET character rescue remains available below, while SIREN
        # heads use the much cheaper lexical evidence.
        parsed = self._combined_query(query, lexical, document_type="siren")
        result = self.searcher.search(parsed, limit=limit)
        hits = [
            (float(score), self.searcher.doc(address).to_dict())
            for score, address in result.hits
        ]
        scores = [
            (str(self._first(document, "siren")), score)
            for score, document in hits
        ]
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]

    def sites_for_siren(
        self, siren: str, query: RetrievalQuery, limit: int
    ) -> list[IndexedEstablishment]:
        siren_query = self._tantivy.Query.term_query(
            self.index.schema, "siren", normalize_code(siren, 9), "basic"
        )
        parsed = self._combined_query(
            query, siren_query, document_type="siret", siren=siren
        )
        result = self.searcher.search(parsed, limit=limit)
        rows = [
            self._record(self.searcher.doc(address).to_dict())
            for _score, address in result.hits
        ]
        if query.address:
            address_query, _address_errors = self.index.parse_query_lenient(
                query.address, default_field_names=["addresses"]
            )
            address_parsed = self._combined_query(
                query, address_query, document_type="siret", siren=siren
            )
            address_result = self.searcher.search(address_parsed, limit=limit)
            rows.extend(
                self._record(self.searcher.doc(address).to_dict())
                for _score, address in address_result.hits
            )
        rows = list({row.siret: row for row in rows}.values())
        return sorted(rows, key=lambda record: _site_sort_key(record, query))[:limit]

    def linked_sirens(self, siren: str, limit: int = 256) -> tuple[str, ...]:
        """Return only structured one-hop links stored on SIREN documents."""
        normalized = normalize_code(siren, 9)
        parsed = self._tantivy.Query.boolean_query(
            [
                (
                    self._tantivy.Occur.Must,
                    self._tantivy.Query.term_query(
                        self.index.schema, "document_type", "siren", "basic"
                    ),
                ),
                (
                    self._tantivy.Occur.Must,
                    self._tantivy.Query.term_query(
                        self.index.schema, "siren", normalized, "basic"
                    ),
                ),
            ]
        )
        linked: set[str] = set()
        for _score, address in self.searcher.search(parsed, limit=64).hits:
            document = self.searcher.doc(address).to_dict()
            linked.update(
                normalize_code(value, 9)
                for value in str(self._first(document, "linked_sirens", "")).split()
                if value
            )
        return tuple(sorted(linked))[:limit]

    def by_siret(self, siret: str) -> IndexedEstablishment | None:
        parsed = self._tantivy.Query.term_query(
            self.index.schema, "siret", normalize_code(siret, 14), "basic"
        )
        result = self.searcher.search(parsed, limit=1)
        return (
            self._record(self.searcher.doc(result.hits[0][1]).to_dict())
            if result.hits
            else None
        )


def _site_sort_key(record: IndexedEstablishment, query: RetrievalQuery) -> tuple[Any, ...]:
    address_score = max((_overlap(query.address, value) for value in record.addresses), default=0.0)
    number_match = bool(query.number and record.number == query.number)
    # Address and number deliberately precede headquarters status.
    return (-int(number_match), -address_score, int(record.is_siege), -int(record.active), record.siret)


@dataclass(frozen=True)
class RetrievalCandidate:
    siret: str
    siren: str
    rank: int
    score: float
    sources: tuple[str, ...]
    record: IndexedEstablishment

    def to_dict(self) -> dict[str, Any]:
        return {
            **dict(self.record.payload),
            "siret": self.siret,
            "siren": self.siren,
            "retrieval_rank": self.rank,
            "retrieval_score": self.score,
            "retrieval_source": "+".join(self.sources),
            "retrieval_channel_count": len(self.sources),
        }


class HierarchicalSiretRetriever:
    def __init__(self, backend: RetrievalBackend, config: HierarchicalRetrievalConfig) -> None:
        if not config.enabled:
            raise RuntimeError("hierarchical retrieval is disabled; explicit opt-in is required")
        self.backend = backend
        self.config = config
        self._search_pool = ThreadPoolExecutor(max_workers=12)

    def retrieve(self, row: RetrievalQuery | Mapping[str, Any]) -> list[RetrievalCandidate]:
        query = row if isinstance(row, RetrievalQuery) else RetrievalQuery.from_mapping(row)
        if not query.insee and not query.postcode:
            return []

        primary_channels = (
            [
                "legal_name_exact", "trade_name_exact", "site_name_exact", "address_exact",
                "legal_name_word", "trade_name_word", "site_name_word", "address_word",
            ]
            if self.config.typed_name_portfolio
            else ["name_exact", "address_exact", "name_word", "address_word"]
        )
        primary_futures = {
            channel: self._search_pool.submit(
                self.backend.search, query, channel, self.config.direct_top_k
            )
            for channel in primary_channels
        }
        siren_future = self._search_pool.submit(
            self.backend.search_sirens, query, self.config.siren_top_k
        )
        channels: dict[str, list[BackendHit]] = {
            channel: future.result() for channel, future in primary_futures.items()
        }
        # Character retrieval is a rescue path.  Exact official evidence makes
        # it redundant on the easy majority and skipping it materially lowers
        # latency without hiding hard cases from the lexical/hierarchical path.
        exact_channels = (
            ["legal_name_exact", "trade_name_exact", "site_name_exact", "address_exact"]
            if self.config.typed_name_portfolio else ["name_exact", "address_exact"]
        )
        has_exact_current = any(channels.get(channel) for channel in exact_channels)
        if has_exact_current:
            for channel in (
                ["current_name_char", "address_char", "historical_name_word",
                 "supporting_name_word", "historical_address_word", "supporting_address_word"]
                if self.config.typed_name_portfolio else ["name_char", "address_char"]
            ):
                channels[channel] = []
        else:
            rescue_limits = (
                {
                    "current_name_char": self.config.name_char_top_k,
                    "address_char": self.config.address_char_top_k,
                    "historical_name_word": self.config.historical_name_top_k,
                    "supporting_name_word": self.config.supporting_name_top_k,
                    "historical_address_word": self.config.historical_name_top_k,
                    "supporting_address_word": self.config.supporting_name_top_k,
                }
                if self.config.typed_name_portfolio
                else {"name_char": self.config.name_char_top_k,
                      "address_char": self.config.address_char_top_k}
            )
            char_futures = {
                channel: self._search_pool.submit(self.backend.search, query, channel, limit)
                for channel, limit in rescue_limits.items()
            }
            channels.update(
                {channel: future.result() for channel, future in char_futures.items()}
            )
        hierarchical: list[BackendHit] = []
        for siren_rank, (siren, siren_score) in enumerate(
            siren_future.result(), start=1
        ):
            for site_rank, record in enumerate(
                self.backend.sites_for_siren(siren, query, self.config.sites_per_siren), start=1
            ):
                hierarchical.append(
                    BackendHit(record, siren_score + 1.0 / (10 + siren_rank) + 1.0 / (100 + site_rank))
                )

        # One official succession hop is a rescue channel; never recurse.
        succession: list[BackendHit] = []
        for hit in [item for values in channels.values() for item in values] + hierarchical:
            for linked_siret in hit.record.linked_sirets:
                record = self.backend.by_siret(linked_siret)
                if record and (
                    (query.insee and record.insee == query.insee)
                    or (not query.insee and query.postcode and record.postcode == query.postcode)
                ):
                    succession.append(BackendHit(record, hit.score * 0.9))
            for linked_siren in hit.record.linked_sirens:
                for record in self.backend.sites_for_siren(
                    linked_siren, query, min(8, self.config.sites_per_siren)
                ):
                    succession.append(BackendHit(record, hit.score * 0.85))
        channels["official_successor"] = succession
        channels["hierarchical"] = hierarchical

        rrf_scores: dict[str, float] = defaultdict(float)
        sources: dict[str, set[str]] = defaultdict(set)
        records: dict[str, IndexedEstablishment] = {}
        for source, hits in channels.items():
            for rank, hit in enumerate(hits, start=1):
                if not self.config.include_closed and not hit.record.active:
                    continue
                records.setdefault(hit.record.siret, hit.record)
                sources[hit.record.siret].add(source)
                if source in set(exact_channels):
                    weight = 2.0
                elif source in {
                    "historical_name_word", "supporting_name_word",
                    "historical_address_word", "supporting_address_word",
                }:
                    weight = 0.6
                else:
                    weight = 1.0
                rrf_scores[hit.record.siret] += weight / (self.config.rrf_k + rank)
        if len(records) > self.config.union_cap:
            def ranked_for(source_names: set[str]) -> list[str]:
                return sorted(
                    (
                        siret
                        for siret in records
                        if sources[siret] & source_names
                    ),
                    key=lambda siret: (-rrf_scores[siret], siret),
                )

            reserved: list[str] = []
            reserved_set: set[str] = set()
            for source_names, count in [
                (
                    set(primary_channels) | {"official_successor"},
                    self.config.direct_slots,
                ),
                ({"hierarchical"}, self.config.hierarchical_slots),
                ({"name_char", "current_name_char", "address_char", "historical_name_word",
                  "supporting_name_word", "historical_address_word", "supporting_address_word"},
                 self.config.character_rescue_slots),
            ]:
                added = 0
                for siret in ranked_for(source_names):
                    if siret in reserved_set:
                        continue
                    reserved.append(siret)
                    reserved_set.add(siret)
                    added += 1
                    if added >= count:
                        break
            retained = set(reserved)
            for siret in sorted(records, key=lambda value: (-rrf_scores[value], value)):
                if len(retained) >= self.config.union_cap:
                    break
                retained.add(siret)
            records = {siret: record for siret, record in records.items() if siret in retained}
            rrf_scores = defaultdict(
                float,
                {siret: score for siret, score in rrf_scores.items() if siret in retained},
            )
            sources = defaultdict(
                set,
                {siret: value for siret, value in sources.items() if siret in retained},
            )

        direct_sources = set(primary_channels) | {"official_successor"}
        char_sources = {"name_char", "current_name_char", "address_char",
                        "historical_name_word", "supporting_name_word",
                        "historical_address_word", "supporting_address_word"}
        buckets = {
            "direct": [siret for siret in records if sources[siret] & direct_sources],
            "hierarchical": [siret for siret in records if "hierarchical" in sources[siret]],
            "character": [siret for siret in records if sources[siret] & char_sources],
        }
        for values in buckets.values():
            values.sort(key=lambda siret: (-rrf_scores[siret], siret))

        selected: list[str] = []
        selected_set: set[str] = set()

        def take(bucket: str, count: int) -> None:
            added = 0
            for siret in buckets[bucket]:
                if siret in selected_set:
                    continue
                selected.append(siret)
                selected_set.add(siret)
                added += 1
                if added >= count:
                    break

        take("direct", self.config.direct_slots)
        take("hierarchical", self.config.hierarchical_slots)
        take("character", self.config.character_rescue_slots)
        remainder = sorted(records, key=lambda siret: (-rrf_scores[siret], siret))
        for siret in remainder:
            if len(selected) >= self.config.max_candidates:
                break
            if siret not in selected_set:
                selected.append(siret)
                selected_set.add(siret)

        return [
            RetrievalCandidate(
                siret=siret,
                siren=records[siret].siren,
                rank=rank,
                score=rrf_scores[siret],
                sources=tuple(sorted(sources[siret])),
                record=records[siret],
            )
            for rank, siret in enumerate(selected[: self.config.max_candidates], start=1)
        ]


def load_production_retriever(
    *, config_path: Path | str, index_path: Path | str
) -> HierarchicalSiretRetriever:
    """Explicit opt-in runtime factory; legacy retrieval remains the default."""
    config = HierarchicalRetrievalConfig.load(config_path)
    if config.backend != "tantivy" or not config.require_tantivy_in_production:
        raise RuntimeError("production hierarchical retrieval must use the Tantivy backend")
    backend = TantivyBackend(index_path)
    if (
        config.include_official_history or config.include_official_successions
    ) and not backend.manifest.get("temporal_complete", False):
        raise RuntimeError(
            "production hierarchical index is temporally incomplete; provide official "
            "establishment/legal-unit history and succession sources"
        )
    return HierarchicalSiretRetriever(backend, config)


def exact_and_operational_hits(
    candidates: Sequence[RetrievalCandidate],
    *,
    exact_siret: str,
    operational_sirets: Iterable[str],
) -> dict[str, bool]:
    """Publish exact and operational views separately; never merge the labels."""
    retrieved = {candidate.siret for candidate in candidates}
    exact = normalize_code(exact_siret, 14)
    operational = {normalize_code(value, 14) for value in operational_sirets}
    return {
        "exact_hit": exact in retrieved,
        "operational_hit": bool(retrieved & operational),
    }


def _site_components(record: Mapping[str, Any]) -> tuple[str, str, str]:
    postcode = normalize_code(
        record.get("postcode")
        or record.get("codePostalEtablissement")
        or record.get("crm_postcode_raw"),
        5,
    )
    number = normalize_text(
        record.get("number")
        or record.get("numeroVoie")
        or record.get("numeroVoieEtablissement")
    )
    suffix = normalize_text(
        record.get("indiceRepetition")
        or record.get("indiceRepetitionEtablissement")
        or record.get("street_number_suffix")
    )
    road = normalize_text(
        " ".join(
            filter(
                None,
                [
                    str(
                        record.get("typeVoie")
                        or record.get("typeVoieEtablissement")
                        or ""
                    ),
                    str(
                        record.get("road")
                        or record.get("libelleVoie")
                        or record.get("libelleVoieEtablissement")
                        or ""
                    ),
                ],
            )
        )
    )
    return postcode, f"{number} {suffix}".strip(), road


def same_physical_site(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Strict, label-blind evidence for the operational same-site policy."""
    left_components = _site_components(left)
    right_components = _site_components(right)
    return all(left_components) and left_components == right_components


def operational_acceptable_sirets(
    exact_gt: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]]
) -> tuple[str, ...]:
    """Keep exact GT and same-SIREN/same-site alternatives as a separate view."""
    exact_siret = normalize_code(exact_gt.get("siret"), 14)
    exact_siren = normalize_code(exact_gt.get("siren") or exact_siret[:9], 9)
    acceptable = {exact_siret} if exact_siret else set()
    for candidate in candidates:
        candidate_siret = normalize_code(candidate.get("siret"), 14)
        candidate_siren = normalize_code(
            candidate.get("siren") or candidate_siret[:9], 9
        )
        if (
            candidate_siret
            and candidate_siren == exact_siren
            and same_physical_site(exact_gt, candidate)
        ):
            acceptable.add(candidate_siret)
    return tuple(sorted(acceptable))


__all__ = [
    "BackendHit",
    "HierarchicalRetrievalConfig",
    "HierarchicalSiretRetriever",
    "InMemoryBackend",
    "IndexedEstablishment",
    "RetrievalCandidate",
    "RetrievalQuery",
    "TantivyBackend",
    "character_ngrams",
    "exact_and_operational_hits",
    "load_production_retriever",
    "normalize_text",
    "normalize_insee",
    "operational_acceptable_sirets",
    "same_physical_site",
]
