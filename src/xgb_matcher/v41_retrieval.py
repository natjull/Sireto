"""V4.1 active-only retrieval with optional CRM SIRET evidence.

This module is deliberately separate from the historical retrieval code.  It
wraps the canonical sparse retriever, but gives V4.1 its own configuration
signature and enforces the V4.1 invariants at the final boundary:

* closed SIRETs may be used as aliases, never as final candidates;
* candidate SIRETs are active, unique and capped at ``max_candidates``;
* the suspect CRM SIRET is evidence, not a label;
* candidate details can be hydrated from the global DuckDB store.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .features import build_address, preprocess_crm_row
from .fusion import reciprocal_rank_fusion
from .naming import build_candidate_names, normalize_text
from .retrieval import CandidatePoolResult, build_candidate_pool
from .retrieval_config import RetrievalConfigV1


class InputSiretState(str, Enum):
    """State of the SIRET supplied by the CRM in the current SIRENE snapshot."""

    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    NOT_FOUND = "NOT_FOUND"
    INVALID = "INVALID"


class V41RetrievalVariant(str, Enum):
    """Incremental variants used by the V4.1 constant-budget ablation."""

    A_SPARSE_ACTIVE = "A"
    B_INPUT_EVIDENCE = "B"
    C_CLOSED_ALIAS = "C"


@dataclass(frozen=True)
class InputSiretQualification:
    raw_value: str | None
    normalized_siret: str | None
    siren: str | None
    state: InputSiretState
    candidate: dict[str, Any] | None = None


@dataclass(frozen=True)
class V41RetrievalConfig:
    """Configuration isolated from V7/V9 and safe for shadow inference."""

    variant: V41RetrievalVariant = V41RetrievalVariant.A_SPARSE_ACTIVE
    max_candidates: int = 100
    sparse_per_channel_k: int = 500
    input_siren_site_limit: int = 100
    rrf_k: int = 60

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", V41RetrievalVariant(self.variant))
        if not 1 <= self.max_candidates <= 100:
            raise ValueError("V4.1 max_candidates must be between 1 and 100")
        if self.sparse_per_channel_k <= 0:
            raise ValueError("sparse_per_channel_k must be positive")
        if self.input_siren_site_limit <= 0:
            raise ValueError("input_siren_site_limit must be positive")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")

    def sparse_config(self) -> RetrievalConfigV1:
        """Return the isolated active-only configuration for legacy sparse code."""

        return RetrievalConfigV1(
            version="v4.1-active-only-1",
            include_closed=False,
            sparse_retrieval_enabled=True,
            dense_retrieval_enabled=False,
            fusion_mode="rrf",
            sparse_channel_fusion_mode="separate_rrf",
            retrieval_budget=self.max_candidates,
            prefilter_k=self.sparse_per_channel_k,
            prefilter_union_cap=None,
            min_candidates=min(50, self.max_candidates),
            # Force the constant-budget code path for every non-trivial pool.
            prefilter_trigger_size=1,
            mega_insee_policy="full_insee",
            rrf_k=self.rrf_k,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "v4.1-retrieval-config-1",
            "variant": self.variant.value,
            "max_candidates": self.max_candidates,
            "sparse_per_channel_k": self.sparse_per_channel_k,
            "input_siren_site_limit": self.input_siren_site_limit,
            "rrf_k": self.rrf_k,
            "sparse_config": self.sparse_config().to_dict(),
        }

    def signature(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class V41CandidatePoolResult:
    candidates: list[dict[str, Any]]
    input_siret: InputSiretQualification
    channels: dict[str, list[str]]
    sparse_result: CandidatePoolResult


_SIRET_PATTERN = re.compile(r"^\d{14}$")
_SIREN_PATTERN = re.compile(r"^\d{9}$")


def normalize_input_siret(value: object) -> str | None:
    """Normalize a CRM SIRET without inventing or repairing digits.

    Whitespace is harmless transport formatting and is removed.  Every other
    unexpected character, a short value, or a long value makes the identifier
    invalid.  A checksum is intentionally not required: SIRENE lookup remains
    the authoritative existence check and this also supports documented public
    identifier exceptions.
    """

    if value is None:
        return None
    text = re.sub(r"\s+", "", str(value).strip())
    return text if _SIRET_PATTERN.fullmatch(text) else None


def _normalize_siren(value: object) -> str | None:
    text = re.sub(r"\s+", "", str(value or "").strip())
    return text if _SIREN_PATTERN.fullmatch(text) else None


def _is_active(candidate: Mapping[str, Any] | None) -> bool:
    return (
        candidate is not None
        and str(candidate.get("etat_admin") or "").strip().upper() == "A"
    )


def _is_closed(candidate: Mapping[str, Any] | None) -> bool:
    return (
        candidate is not None
        and str(candidate.get("etat_admin") or "").strip().upper() == "F"
    )


class V41GlobalCandidateStore:
    """Batch read-only access to the complete SIRET candidate store."""

    def __init__(
        self,
        path: Path,
        *,
        read_only: bool = True,
        table_name: str = "candidates",
    ) -> None:
        import duckdb

        if table_name != "candidates":
            raise ValueError("Only the canonical 'candidates' table is supported")
        path = Path(path)
        database_path = path / "siren_candidates.duckdb" if path.is_dir() else path
        if not database_path.exists():
            raise FileNotFoundError(database_path)
        self.database_path = database_path
        self._connection = duckdb.connect(str(database_path), read_only=read_only)
        columns = {
            str(row[0])
            for row in self._connection.execute(
                "DESCRIBE candidates"
            ).fetchall()
        }
        required = {"siret", "siren", "etat_admin"}
        missing = required - columns
        if missing:
            self._connection.close()
            raise ValueError(
                f"Global candidate store is missing required columns: {sorted(missing)}"
            )

    @staticmethod
    def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
        output = dict(row)
        output["siret"] = str(output.get("siret") or "").zfill(14)
        output["siren"] = str(output.get("siren") or "").zfill(9)
        return output

    def get_candidate_details(
        self,
        sirets: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        normalized = list(
            dict.fromkeys(
                siret
                for value in sirets
                if (siret := normalize_input_siret(value)) is not None
            )
        )
        if not normalized:
            return {}
        table = self._connection.execute(
            """
            SELECT *
            FROM candidates
            WHERE siret IN (SELECT unnest(?))
            ORDER BY siret
            """,
            [normalized],
        ).fetch_arrow_table()
        return {
            row["siret"]: row
            for raw in table.to_pylist()
            if (row := self._coerce_row(raw)).get("siret")
        }

    def get_candidate_states(
        self,
        sirets: Sequence[str],
    ) -> dict[str, str]:
        """Return current administrative states with a narrow batch query."""

        normalized = list(
            dict.fromkeys(
                siret
                for value in sirets
                if (siret := normalize_input_siret(value)) is not None
            )
        )
        if not normalized:
            return {}
        rows = self._connection.execute(
            """
            SELECT siret, etat_admin
            FROM candidates
            WHERE siret IN (SELECT unnest(?))
            ORDER BY siret
            """,
            [normalized],
        ).fetchall()
        return {
            str(siret).zfill(14): str(state or "").strip().upper()
            for siret, state in rows
        }

    def qualify_input_sirets(
        self,
        values: Sequence[object],
    ) -> list[InputSiretQualification]:
        normalized = [normalize_input_siret(value) for value in values]
        details = self.get_candidate_details(
            [siret for siret in normalized if siret is not None]
        )
        output: list[InputSiretQualification] = []
        for raw_value, siret in zip(values, normalized):
            raw_text = None if raw_value is None else str(raw_value)
            if siret is None:
                output.append(
                    InputSiretQualification(
                        raw_value=raw_text,
                        normalized_siret=None,
                        siren=None,
                        state=InputSiretState.INVALID,
                    )
                )
                continue
            candidate = details.get(siret)
            if candidate is None:
                state = InputSiretState.NOT_FOUND
            elif _is_active(candidate):
                state = InputSiretState.ACTIVE
            elif _is_closed(candidate):
                state = InputSiretState.CLOSED
            else:
                # An unknown administrative state is not safe active evidence.
                state = InputSiretState.NOT_FOUND
            output.append(
                InputSiretQualification(
                    raw_value=raw_text,
                    normalized_siret=siret,
                    siren=(
                        str(candidate.get("siren") or "").zfill(9)
                        if candidate is not None
                        else siret[:9]
                    ),
                    state=state,
                    candidate=candidate,
                )
            )
        return output

    def get_active_siblings(
        self,
        sirens: Sequence[str],
        *,
        max_per_siren: int,
        crm_insee: str = "",
        crm_postcode: str = "",
    ) -> dict[str, list[dict[str, Any]]]:
        """Return active sites, filtering state before the per-SIREN limit."""

        if max_per_siren <= 0:
            raise ValueError("max_per_siren must be positive")
        normalized = list(
            dict.fromkeys(
                siren
                for value in sirens
                if (siren := _normalize_siren(value)) is not None
            )
        )
        if not normalized:
            return {}
        # The ACTIVE predicate is deliberately inside the windowed subquery.
        # Therefore closed rows never consume candidate_rank positions.
        table = self._connection.execute(
            """
            SELECT * EXCLUDE (candidate_rank)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY siren
                        ORDER BY
                            (insee = ?) DESC,
                            (postcode = ?) DESC,
                            coalesce(is_siege, false) DESC,
                            siret
                    ) AS candidate_rank
                FROM candidates
                WHERE siren IN (SELECT unnest(?))
                  AND upper(trim(coalesce(etat_admin, ''))) = 'A'
            )
            WHERE candidate_rank <= ?
            ORDER BY siren, candidate_rank
            """,
            [
                str(crm_insee or ""),
                str(crm_postcode or ""),
                normalized,
                int(max_per_siren),
            ],
        ).fetch_arrow_table()
        output: dict[str, list[dict[str, Any]]] = {}
        for raw in table.to_pylist():
            row = self._coerce_row(raw)
            output.setdefault(row["siren"], []).append(row)
        return output

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "V41GlobalCandidateStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class V41CurrentStateStore:
    """Read current SIRET states directly from the authoritative snapshot.

    The enriched global candidate store is intentionally smaller than the
    complete SIRENE establishment snapshot.  It remains useful for candidate
    details and SIREN expansion, but it must not be the final authority for
    administrative state.
    """

    def __init__(
        self,
        snapshot_path: Path,
        *,
        siret_column: str = "siret",
        state_column: str = "etatAdministratifEtablissement",
    ) -> None:
        import duckdb

        snapshot_path = Path(snapshot_path)
        if not snapshot_path.is_file():
            raise FileNotFoundError(snapshot_path)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", siret_column):
            raise ValueError("Invalid SIRET column name")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", state_column):
            raise ValueError("Invalid state column name")
        self.snapshot_path = snapshot_path
        self.siret_column = siret_column
        self.state_column = state_column
        self._connection = duckdb.connect(":memory:")
        columns = {
            str(row[0])
            for row in self._connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(snapshot_path)],
            ).fetchall()
        }
        missing = {siret_column, state_column} - columns
        if missing:
            self._connection.close()
            raise ValueError(
                "SIRENE state snapshot is missing required columns: "
                f"{sorted(missing)}"
            )

    def get_candidate_states(
        self,
        sirets: Sequence[str],
    ) -> dict[str, str]:
        """Return authoritative current states for normalized SIRETs."""

        normalized = list(
            dict.fromkeys(
                siret
                for value in sirets
                if (siret := normalize_input_siret(value)) is not None
            )
        )
        if not normalized:
            return {}
        rows = self._connection.execute(
            f"""
            SELECT
                CAST({self.siret_column} AS VARCHAR) AS siret,
                upper(trim(CAST({self.state_column} AS VARCHAR))) AS state
            FROM read_parquet(?)
            WHERE CAST({self.siret_column} AS VARCHAR)
                  IN (SELECT unnest(?))
            ORDER BY siret
            """,
            [str(self.snapshot_path), normalized],
        ).fetchall()
        output: dict[str, str] = {}
        for raw_siret, raw_state in rows:
            siret = str(raw_siret or "").zfill(14)
            state = str(raw_state or "").strip().upper()
            previous = output.get(siret)
            if previous is not None and previous != state:
                raise ValueError(
                    f"Conflicting current states for SIRET {siret}: "
                    f"{previous!r} and {state!r}"
                )
            output[siret] = state
        return output

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "V41CurrentStateStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _candidate_name_texts(candidate: Mapping[str, Any]) -> list[str]:
    return [name.text for name in build_candidate_names(dict(candidate))]


def _rank_closed_alias_name(
    alias: Mapping[str, Any],
    active_candidates: Iterable[Mapping[str, Any]],
) -> list[str]:
    alias_names = _candidate_name_texts(alias)
    scored: list[tuple[float, str]] = []
    for candidate in active_candidates:
        candidate_names = _candidate_name_texts(candidate)
        score = max(
            (
                _token_similarity(alias_name, candidate_name)
                for alias_name in alias_names
                for candidate_name in candidate_names
            ),
            default=0.0,
        )
        siret = str(candidate.get("siret") or "")
        if score > 0.0 and siret:
            scored.append((score, siret))
    return [siret for _score, siret in sorted(scored, key=lambda x: (-x[0], x[1]))]


def _rank_closed_alias_address(
    alias: Mapping[str, Any],
    active_candidates: Iterable[Mapping[str, Any]],
) -> list[str]:
    alias_address = build_address(dict(alias))
    scored: list[tuple[float, str]] = []
    for candidate in active_candidates:
        score = _token_similarity(alias_address, build_address(dict(candidate)))
        siret = str(candidate.get("siret") or "")
        if score > 0.0 and siret:
            scored.append((score, siret))
    return [siret for _score, siret in sorted(scored, key=lambda x: (-x[0], x[1]))]


SparsePoolBuilder = Callable[..., CandidatePoolResult]


class V41CandidateRetriever:
    """Compose active sparse retrieval and suspect input-SIRET evidence."""

    def __init__(
        self,
        *,
        partitioned_store: Any,
        global_store: V41GlobalCandidateStore,
        current_state_store: Any | None = None,
        config: V41RetrievalConfig,
        sparse_pool_builder: SparsePoolBuilder = build_candidate_pool,
        in_memory_tfidf_cache: dict[tuple[str, str], tuple] | None = None,
        max_in_memory_tfidf_partitions: int = 20,
    ) -> None:
        self.partitioned_store = partitioned_store
        self.global_store = global_store
        self.current_state_store = current_state_store or global_store
        self.config = config
        self._sparse_pool_builder = sparse_pool_builder
        self._tfidf_cache = (
            in_memory_tfidf_cache
            if in_memory_tfidf_cache is not None
            else {}
        )
        if max_in_memory_tfidf_partitions <= 0:
            raise ValueError("max_in_memory_tfidf_partitions must be positive")
        self._max_in_memory_tfidf_partitions = max_in_memory_tfidf_partitions

    def _trim_tfidf_cache(self) -> None:
        while len(self._tfidf_cache) > self._max_in_memory_tfidf_partitions:
            oldest = next(iter(self._tfidf_cache))
            self._tfidf_cache.pop(oldest, None)

    def _apply_authoritative_input_state(
        self,
        qualification: InputSiretQualification,
    ) -> InputSiretQualification:
        siret = qualification.normalized_siret
        if siret is None or self.current_state_store is self.global_store:
            return qualification
        state = self.current_state_store.get_candidate_states([siret]).get(siret)
        if state == "A":
            input_state = InputSiretState.ACTIVE
        elif state == "F":
            input_state = InputSiretState.CLOSED
        else:
            input_state = InputSiretState.NOT_FOUND
        candidate = (
            None
            if qualification.candidate is None
            else {**qualification.candidate, "etat_admin": state}
        )
        return InputSiretQualification(
            raw_value=qualification.raw_value,
            normalized_siret=siret,
            siren=qualification.siren or siret[:9],
            state=input_state,
            candidate=candidate,
        )

    def build(
        self,
        *,
        crm_row: dict[str, Any],
        crm_pre: dict[str, Any],
        input_siret: object,
        input_qualification: InputSiretQualification | None = None,
        gt_siret: str | None = None,
        persistent_cache: Any = None,
        timer: Any = None,
    ) -> V41CandidatePoolResult:
        sparse_result = self._sparse_pool_builder(
            self.partitioned_store,
            crm_row,
            crm_pre,
            self.config.sparse_config(),
            self._tfidf_cache,
            gt_siret,
            persistent_cache=persistent_cache,
            timer=timer,
        )
        qualification = input_qualification or self.global_store.qualify_input_sirets(
            [input_siret]
        )[0]
        qualification = self._apply_authoritative_input_state(qualification)

        candidate_by_siret: dict[str, dict[str, Any]] = {
            str(candidate.get("siret") or ""): dict(candidate)
            for candidate in sparse_result.candidates
            if candidate.get("siret")
        }
        channels: dict[str, list[str]] = {
            "sparse_active": [
                str(candidate.get("siret") or "")
                for candidate in sparse_result.candidates
                if candidate.get("siret")
            ],
            "input_siret_active": [],
            "input_siren_active_sites": [],
            "closed_alias_name": [],
            "closed_alias_address": [],
        }

        variant = self.config.variant
        siblings: list[dict[str, Any]] = []
        if (
            variant in {
                V41RetrievalVariant.B_INPUT_EVIDENCE,
                V41RetrievalVariant.C_CLOSED_ALIAS,
            }
            and qualification.siren is not None
        ):
            sibling_map = self.global_store.get_active_siblings(
                [qualification.siren],
                max_per_siren=self.config.input_siren_site_limit,
                crm_insee=str(
                    crm_row.get("insee") or crm_row.get("crm_insee") or ""
                ),
                crm_postcode=str(
                    crm_row.get("postcode") or crm_row.get("crm_cp") or ""
                ),
            )
            siblings = sibling_map.get(qualification.siren, [])
            candidate_by_siret.update(
                {str(candidate["siret"]): dict(candidate) for candidate in siblings}
            )
            channels["input_siren_active_sites"] = [
                str(candidate["siret"]) for candidate in siblings
            ]
            channels["input_siret_active"] = (
                [qualification.normalized_siret]
                if (
                    qualification.state == InputSiretState.ACTIVE
                    and qualification.normalized_siret is not None
                )
                else []
            )
            if qualification.candidate is not None and _is_active(
                qualification.candidate
            ):
                candidate_by_siret[qualification.normalized_siret or ""] = dict(
                    qualification.candidate
                )

        if variant == V41RetrievalVariant.C_CLOSED_ALIAS:
            if (
                qualification.state == InputSiretState.CLOSED
                and qualification.candidate is not None
            ):
                alias_names = [
                    name.text
                    for name in build_candidate_names(qualification.candidate)[:3]
                    if name.text
                ]
                alias_address = build_address(qualification.candidate)
                alias_name_row = {
                    **crm_row,
                    "crm_name": " ".join(alias_names),
                    "crm_address": "",
                }
                alias_address_row = {
                    **crm_row,
                    "crm_name": "",
                    "crm_address": alias_address,
                }
                alias_name_result = self._sparse_pool_builder(
                    self.partitioned_store,
                    alias_name_row,
                    preprocess_crm_row(alias_name_row),
                    self.config.sparse_config(),
                    self._tfidf_cache,
                    None,
                    persistent_cache=persistent_cache,
                    timer=timer,
                )
                alias_address_result = self._sparse_pool_builder(
                    self.partitioned_store,
                    alias_address_row,
                    preprocess_crm_row(alias_address_row),
                    self.config.sparse_config(),
                    self._tfidf_cache,
                    None,
                    persistent_cache=persistent_cache,
                    timer=timer,
                )
                alias_name_candidates = [
                    *siblings,
                    *alias_name_result.candidates,
                ]
                alias_address_candidates = [
                    *siblings,
                    *alias_address_result.candidates,
                ]
                candidate_by_siret.update(
                    {
                        str(candidate["siret"]): dict(candidate)
                        for candidate in [
                            *alias_name_result.candidates,
                            *alias_address_result.candidates,
                        ]
                        if candidate.get("siret")
                    }
                )
                channels["closed_alias_name"] = _rank_closed_alias_name(
                    qualification.candidate,
                    alias_name_candidates,
                )
                channels["closed_alias_address"] = _rank_closed_alias_address(
                    qualification.candidate,
                    alias_address_candidates,
                )

        # Strict final safety boundary: every channel is reduced to candidates
        # proven active in the current global/local snapshot.
        current_states = self.current_state_store.get_candidate_states(
            list(candidate_by_siret)
        )
        for siret, state in current_states.items():
            if siret in candidate_by_siret:
                candidate_by_siret[siret]["etat_admin"] = state
        active_sirets = {
            siret
            for siret, state in current_states.items()
            if state == "A" and siret in candidate_by_siret
        }
        filtered_channels = {
            channel: [
                siret
                for siret in dict.fromkeys(ordered_sirets)
                if siret in active_sirets
            ]
            for channel, ordered_sirets in channels.items()
        }
        fused = reciprocal_rank_fusion(
            filtered_channels,
            budget=self.config.max_candidates,
            rrf_k=self.config.rrf_k,
        )
        candidates: list[dict[str, Any]] = []
        for hit in fused:
            siret = str(hit.key)
            candidate = dict(candidate_by_siret[siret])
            candidate["rrf_score"] = float(hit.rrf_score)
            candidate["retrieval_rank"] = int(hit.rank)
            candidate["retrieval_source"] = hit.source
            candidate["retrieval_channel_count"] = len(hit.channel_ranks)
            candidate["v41_channel_ranks"] = dict(hit.channel_ranks)
            for channel_name in (
                "sparse_active",
                "input_siret_active",
                "input_siren_active_sites",
                "closed_alias_name",
                "closed_alias_address",
            ):
                candidate[f"{channel_name}_rank"] = hit.channel_ranks.get(
                    channel_name
                )
            candidates.append(candidate)

        if len(candidates) > 100:
            raise AssertionError("V4.1 candidate ceiling violated")
        if len({candidate["siret"] for candidate in candidates}) != len(candidates):
            raise AssertionError("V4.1 candidates must be unique")
        if any(not _is_active(candidate) for candidate in candidates):
            raise AssertionError("V4.1 final candidates must all be active")
        self._trim_tfidf_cache()

        return V41CandidatePoolResult(
            candidates=candidates,
            input_siret=qualification,
            channels=filtered_channels,
            sparse_result=sparse_result,
        )


__all__ = [
    "InputSiretQualification",
    "InputSiretState",
    "V41CandidatePoolResult",
    "V41CandidateRetriever",
    "V41CurrentStateStore",
    "V41GlobalCandidateStore",
    "V41RetrievalConfig",
    "V41RetrievalVariant",
    "normalize_input_siret",
]
