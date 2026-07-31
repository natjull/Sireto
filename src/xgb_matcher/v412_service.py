"""Label-blind, query-level V4.11 + V4.12-G downstream service core.

The caller owns retrieval and candidate-feature construction.  This module
accepts one already bounded candidate pool, scores it with the frozen Ranker
C, builds the shared V4.11 scene, calls the frozen acceptor, then applies the
V4.12-G veto.  It performs no file, network, model-training or label I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import hmac
import json
import math
import os
import re
import struct
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v411_scene import (
    V411_ACCEPTOR_FEATURE_NAMES,
    build_v411_compact_scene,
)
from .v412_direct_evidence import apply_guard


CANDIDATE_CEILING = 100
FIXED_THRESHOLD = 0.8720916706888049
RANKER_C_FEATURE_ORDER = (
    "has_any_name",
    "name_count",
    "name_jaro_max",
    "name_jaro_second",
    "name_jaro_gap",
    "name_levenshtein_max",
    "name_token_overlap_max",
    "idf_name",
    "numeric_token_match",
    "name_first_word_match_max",
    "name_contains_crm_max",
    "name_crm_contains_cand_max",
    "acronym_match_max",
    "name_sim_max_etab",
    "name_sim_max_ul",
    "name_sim_max_sigle",
    "name_sim_max_pm_dirigeant",
    "type_of_max_name",
    "is_ul_name_max",
    "is_sigle_max",
    "name_length_max",
    "has_person_name",
    "person_name_jaro_max",
    "name_city_overlap_max",
    "name_is_city_like_max",
    "addr_jaro",
    "addr_levenshtein",
    "postcode_match",
    "city_match",
    "street_number_diff",
    "addr_token_overlap",
    "address_density",
    "street_name_jaro",
    "name_addr_consistency",
    "legal_form_category",
    "is_siege",
    "is_association",
    "alias_match",
    "token_overlap_ul",
    "ul_vs_pm_indicator",
    "is_crm_school",
    "geo_exact_match",
    "name_norm_exact",
    "street_number_match",
    "retrieval_rank_recip",
)
RANKER_C_FEATURE_ORDER_SHA256 = hashlib.sha256(
    json.dumps(
        list(RANKER_C_FEATURE_ORDER),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
if (
    RANKER_C_FEATURE_ORDER_SHA256
    != "760db4db1397c85ad34440819e868533a3a11684999285c30a7047eccdba4746"
):
    raise RuntimeError(
        "STOP_V412_SERVICE_INTEGRITY: frozen Ranker C feature hash changed"
    )
FORBIDDEN_FIELDS = frozenset(
    {
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
        "is_ground_truth",
        "acceptor_target",
        "correct_exact_siret",
    }
)
_SIRET = re.compile(r"^[0-9]{14}$")
_SIREN = re.compile(r"^[0-9]{9}$")

SceneBuilder = Callable[
    [Mapping[str, Any], pd.DataFrame, Any],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class ServiceTimings:
    ranker_ns: int
    scene_acceptor_ns: int
    guard_ns: int

    @property
    def downstream_ns(self) -> int:
        return self.ranker_ns + self.scene_acceptor_ns + self.guard_ns


@dataclass
class ServiceTrace:
    query_id: str
    predicted_siret: str | None
    predicted_siren: str | None
    acceptor_score: float
    threshold: float
    decision_v411: str
    review_reason_v411: str | None
    decision_v412: str
    review_reason_v412: str | None
    scored_candidates: pd.DataFrame
    scene: dict[str, Any]
    timings: ServiceTimings


@dataclass(frozen=True)
class V411Trace:
    query_id: str
    predicted_siret: str | None
    predicted_siren: str | None
    acceptor_score: float
    threshold: float
    decision_v411: str
    review_reason_v411: str | None
    scored_candidates: pd.DataFrame
    scene: dict[str, Any]
    ranker_ns: int
    scene_acceptor_ns: int
    _origin_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _integrity_tag: bytes = field(
        default=b"",
        repr=False,
        compare=False,
    )


def _require_label_blind(fields: Sequence[str], *, label: str) -> None:
    leaked = FORBIDDEN_FIELDS & set(fields)
    if leaked:
        raise ValueError(
            f"STOP_V412_SERVICE_INTEGRITY: {label} contains {sorted(leaked)}"
        )


def _validate_query(query: Mapping[str, Any]) -> str:
    _require_label_blind(list(query), label="query")
    query_id = str(query.get("query_id") or "")
    if not query_id:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: empty query identity"
        )
    return query_id


def _validate_direct_evidence(
    query_id: str,
    direct_evidence: Mapping[str, Any],
) -> None:
    _require_label_blind(list(direct_evidence), label="direct evidence")
    if str(direct_evidence.get("query_id") or "") != query_id:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: query/evidence identity mismatch"
        )
    count = direct_evidence.get("direct_candidate_count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid direct candidate count"
        )
    sole = direct_evidence.get("sole_direct_siret")
    if count == 1:
        if not isinstance(sole, str) or _SIRET.fullmatch(sole) is None:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: sole direct SIRET missing"
            )
    elif sole is not None and not pd.isna(sole):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: unexpected sole direct SIRET"
        )


def _validate_candidates(
    query_id: str,
    candidates: pd.DataFrame,
    feature_order: Sequence[str],
) -> None:
    _require_label_blind(list(candidates.columns), label="candidate pool")
    required = {
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "candidate_state",
        "retrieval_rank",
        *feature_order,
    }
    if not required.issubset(candidates.columns):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: candidate schema incomplete"
        )
    if len(candidates) > CANDIDATE_CEILING:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: candidate ceiling exceeded"
        )
    if candidates["query_id"].astype(str).ne(query_id).any():
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: candidate query mismatch"
        )
    sirets = candidates["candidate_siret"].astype(str)
    sirens = candidates["candidate_siren"].astype(str)
    if (
        sirets.duplicated().any()
        or not sirets.map(_SIRET.fullmatch).all()
        or not sirens.map(_SIREN.fullmatch).all()
        or not sirets.str[:9].eq(sirens).all()
        or not candidates["candidate_state"].astype(str).eq("A").all()
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid candidate identity/state"
        )
    ranks = sorted(candidates["retrieval_rank"].astype(int).tolist())
    if ranks != list(range(1, len(candidates) + 1)):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: retrieval ranks not contiguous"
        )
    matrix = candidates[list(feature_order)].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: non-finite ranker feature"
        )


class V412DownstreamService:
    """Persistent frozen ranker, acceptor and V4.12-G veto."""

    def __init__(
        self,
        *,
        ranker: Any,
        acceptor: Any,
        taxonomy: Any,
        ranker_feature_order: Sequence[str],
        threshold: float = FIXED_THRESHOLD,
        scene_builder: SceneBuilder = build_v411_compact_scene,
    ) -> None:
        if tuple(ranker_feature_order) != RANKER_C_FEATURE_ORDER:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: Ranker C feature order changed"
            )
        if threshold != FIXED_THRESHOLD:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: acceptor threshold changed"
            )
        self.ranker = ranker
        self.acceptor = acceptor
        self.taxonomy = taxonomy
        self.ranker_feature_order = tuple(ranker_feature_order)
        self.threshold = threshold
        self.scene_builder = scene_builder
        self._trace_origin_token = object()
        self._trace_secret = os.urandom(32)

    def rank_and_accept_one(
        self,
        *,
        query: Mapping[str, Any],
        candidates: pd.DataFrame,
    ) -> V411Trace:
        query_id = _validate_query(query)
        _validate_candidates(
            query_id,
            candidates,
            self.ranker_feature_order,
        )

        ranker_started = time.perf_counter_ns()
        scored = candidates.copy()
        scores = (
            np.asarray(
                self.ranker.predict(
                    scored[list(self.ranker_feature_order)].to_numpy(
                        dtype=np.float32
                    )
                ),
                dtype=np.float32,
            )
            if len(scored)
            else np.asarray([], dtype=np.float32)
        )
        if scores.shape != (len(scored),) or not np.isfinite(scores).all():
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid ranker scores"
            )
        scored["ranker_score"] = scores
        scored = scored.sort_values(
            [
                "ranker_score",
                "retrieval_rank",
                "candidate_siret",
            ],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        scored["ranker_rank"] = np.arange(
            1, len(scored) + 1, dtype=np.int16
        )
        ranker_ns = time.perf_counter_ns() - ranker_started

        scene_started = time.perf_counter_ns()
        scene = dict(self.scene_builder(query, scored, self.taxonomy))
        _require_label_blind(list(scene), label="scene")
        missing_scene = set(V411_ACCEPTOR_FEATURE_NAMES) - set(scene)
        if missing_scene:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: scene feature missing"
            )
        scene_matrix = np.asarray(
            [[float(scene[name]) for name in V411_ACCEPTOR_FEATURE_NAMES]],
            dtype=np.float64,
        )
        if not np.isfinite(scene_matrix).all():
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: non-finite scene feature"
            )
        probabilities = np.asarray(
            self.acceptor.predict_proba(scene_matrix),
            dtype=np.float64,
        )
        if (
            probabilities.shape != (1, 2)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
            or abs(float(probabilities[0].sum()) - 1.0) > 1e-12
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid acceptor score"
            )
        acceptor_score = float(probabilities[0, 1])
        predicted_siret_raw = scene.get("predicted_siret")
        predicted_siren_raw = scene.get("predicted_siren")
        predicted_siret = (
            str(predicted_siret_raw)
            if isinstance(predicted_siret_raw, str)
            and _SIRET.fullmatch(predicted_siret_raw)
            else None
        )
        predicted_siren = (
            str(predicted_siren_raw)
            if isinstance(predicted_siren_raw, str)
            and _SIREN.fullmatch(predicted_siren_raw)
            else None
        )
        if predicted_siret is not None and predicted_siret[:9] != predicted_siren:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: predicted SIRET/SIREN mismatch"
            )
        decision_v411 = (
            "AUTO_MATCH"
            if predicted_siret is not None
            and acceptor_score >= self.threshold
            else "REVIEW"
        )
        review_reason_v411 = (
            None
            if decision_v411 == "AUTO_MATCH"
            else (
                "NO_CANDIDATE"
                if len(scored) == 0
                else "LOW_CONFIDENCE"
            )
        )
        scene_acceptor_ns = time.perf_counter_ns() - scene_started
        if not math.isfinite(acceptor_score):
            raise AssertionError("non-finite score escaped validation")
        trace = V411Trace(
            query_id=query_id,
            predicted_siret=predicted_siret,
            predicted_siren=predicted_siren,
            acceptor_score=acceptor_score,
            threshold=self.threshold,
            decision_v411=decision_v411,
            review_reason_v411=review_reason_v411,
            scored_candidates=scored,
            scene=scene,
            ranker_ns=ranker_ns,
            scene_acceptor_ns=scene_acceptor_ns,
            _origin_token=self._trace_origin_token,
        )
        return replace(
            trace,
            _integrity_tag=self._sign_v411_trace(trace),
        )

    @staticmethod
    def _trace_text(value: str | None) -> bytes:
        payload = (value or "").encode("utf-8")
        return len(payload).to_bytes(4, "big") + payload

    def _sign_v411_trace(self, trace: V411Trace) -> bytes:
        digest = hmac.new(self._trace_secret, digestmod=hashlib.sha256)
        for value in (
            trace.query_id,
            trace.predicted_siret,
            trace.predicted_siren,
            trace.decision_v411,
            trace.review_reason_v411,
        ):
            digest.update(self._trace_text(value))
        digest.update(
            struct.pack(
                ">ddqq",
                trace.acceptor_score,
                trace.threshold,
                trace.ranker_ns,
                trace.scene_acceptor_ns,
            )
        )
        scored = trace.scored_candidates
        for column in (
            "candidate_siret",
            "candidate_siren",
        ):
            if column not in scored.columns:
                digest.update(b"MISSING")
            else:
                for value in scored[column].astype(str):
                    digest.update(self._trace_text(value))
        for column in ("retrieval_rank", "ranker_rank"):
            if column not in scored.columns:
                digest.update(b"MISSING")
            else:
                digest.update(
                    scored[column].to_numpy(dtype=np.int64).tobytes()
                )
        if "ranker_score" not in scored.columns:
            digest.update(b"MISSING")
        else:
            digest.update(
                scored["ranker_score"].to_numpy(dtype=np.float32).tobytes()
            )
        if set(RANKER_C_FEATURE_ORDER).issubset(scored.columns):
            digest.update(
                scored[list(RANKER_C_FEATURE_ORDER)]
                .to_numpy(dtype=np.float32)
                .tobytes()
            )
        else:
            digest.update(b"MISSING_FEATURES")
        for name in V411_ACCEPTOR_FEATURE_NAMES:
            value = trace.scene.get(name)
            try:
                digest.update(struct.pack(">d", float(value)))
            except (TypeError, ValueError):
                digest.update(b"INVALID")
        digest.update(
            self._trace_text(trace.scene.get("predicted_siret"))
        )
        digest.update(
            self._trace_text(trace.scene.get("predicted_siren"))
        )
        return digest.digest()

    def _validate_v411_trace(self, trace: V411Trace) -> None:
        if (
            not isinstance(trace, V411Trace)
            or trace._origin_token is not self._trace_origin_token
            or not hmac.compare_digest(
                trace._integrity_tag,
                self._sign_v411_trace(trace),
            )
            or trace.threshold != self.threshold
            or not math.isfinite(trace.acceptor_score)
            or not 0.0 <= trace.acceptor_score <= 1.0
            or not trace.query_id
            or type(trace.ranker_ns) is not int
            or trace.ranker_ns < 0
            or type(trace.scene_acceptor_ns) is not int
            or trace.scene_acceptor_ns < 0
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid V4.11 trace provenance"
            )
        predicted_siret = trace.predicted_siret
        predicted_siren = trace.predicted_siren
        if (
            (predicted_siret is not None and _SIRET.fullmatch(predicted_siret) is None)
            or (predicted_siren is not None and _SIREN.fullmatch(predicted_siren) is None)
            or (
                predicted_siret is not None
                and predicted_siret[:9] != predicted_siren
            )
            or trace.scene.get("predicted_siret") != predicted_siret
            or trace.scene.get("predicted_siren") != predicted_siren
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid V4.11 trace identity"
            )
        expected_decision = (
            "AUTO_MATCH"
            if predicted_siret is not None
            and trace.acceptor_score >= self.threshold
            else "REVIEW"
        )
        expected_reason = (
            None
            if expected_decision == "AUTO_MATCH"
            else (
                "NO_CANDIDATE"
                if len(trace.scored_candidates) == 0
                else "LOW_CONFIDENCE"
            )
        )
        if (
            trace.decision_v411 != expected_decision
            or trace.review_reason_v411 != expected_reason
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid V4.11 trace decision"
            )
        required = {"ranker_score", "ranker_rank"}
        if not required.issubset(trace.scored_candidates.columns):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid scored trace schema"
            )
        scores = trace.scored_candidates["ranker_score"].to_numpy(
            dtype=np.float32
        )
        ranks = trace.scored_candidates["ranker_rank"].astype(int).tolist()
        if (
            not np.isfinite(scores).all()
            or ranks != list(range(1, len(ranks) + 1))
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid scored trace values"
            )

    def apply_guard_to_trace(
        self,
        *,
        trace: V411Trace,
        direct_evidence: Mapping[str, Any],
    ) -> ServiceTrace:
        self._validate_v411_trace(trace)
        _validate_direct_evidence(trace.query_id, direct_evidence)
        guard_started = time.perf_counter_ns()
        sole = direct_evidence.get("sole_direct_siret")
        decision_v412, review_reason_v412 = apply_guard(
            decision_v411=trace.decision_v411,
            review_reason_v411=trace.review_reason_v411,
            predicted_siret=trace.predicted_siret,
            direct_candidate_count=int(
                direct_evidence["direct_candidate_count"]
            ),
            sole_direct_siret=(
                str(sole)
                if isinstance(sole, str) and not pd.isna(sole)
                else None
            ),
        )
        guard_ns = time.perf_counter_ns() - guard_started
        if trace.decision_v411 == "REVIEW" and decision_v412 != "REVIEW":
            raise AssertionError("V4.12-G created an AUTO decision")
        return ServiceTrace(
            query_id=trace.query_id,
            predicted_siret=trace.predicted_siret,
            predicted_siren=trace.predicted_siren,
            acceptor_score=trace.acceptor_score,
            threshold=trace.threshold,
            decision_v411=trace.decision_v411,
            review_reason_v411=trace.review_reason_v411,
            decision_v412=decision_v412,
            review_reason_v412=review_reason_v412,
            scored_candidates=trace.scored_candidates,
            scene=trace.scene,
            timings=ServiceTimings(
                ranker_ns=trace.ranker_ns,
                scene_acceptor_ns=trace.scene_acceptor_ns,
                guard_ns=guard_ns,
            ),
        )

    def infer_one(
        self,
        *,
        query: Mapping[str, Any],
        candidates: pd.DataFrame,
        direct_evidence: Mapping[str, Any],
    ) -> ServiceTrace:
        trace = self.rank_and_accept_one(
            query=query,
            candidates=candidates,
        )
        return self.apply_guard_to_trace(
            trace=trace,
            direct_evidence=direct_evidence,
        )


__all__ = [
    "CANDIDATE_CEILING",
    "FIXED_THRESHOLD",
    "FORBIDDEN_FIELDS",
    "RANKER_C_FEATURE_ORDER",
    "RANKER_C_FEATURE_ORDER_SHA256",
    "ServiceTimings",
    "ServiceTrace",
    "V411Trace",
    "V412DownstreamService",
]
