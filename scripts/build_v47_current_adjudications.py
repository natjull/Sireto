#!/usr/bin/env python3
"""Archive public evidence and build the canonical V4.7 current-top1 labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.7-current-adjudications-1"
SPEC_SCHEMA_VERSION = "sireto-v4.7-public-evidence-spec-1"
EXPECTED_DOCKET_SHA256 = (
    "7ee6bf58ed60e3d7d9b94577ae8da62eca2d076fb61459d7c1625522d92a7104"
)
EXPECTED_SCENES_SHA256 = (
    "72540dcdba6f33da0eb1875ef4bcdc8c44a2cd10083589b5e1683098cd954a08"
)
EXPECTED_DRIFT_COUNT = 37
EXPECTED_SCENE_COUNT = 172
EXPECTED_RANDOM_SCENE_COUNT = 57
SIRENE_GROUP = "SIRENE_REGISTRY"
RELATIONSHIP_TO_LABEL = {
    "SUPPORTS_CURRENT_TOP1": "TOP1_CORRECT",
    "CONTRADICTS_CURRENT_TOP1": "TOP1_WRONG",
    "AMBIGUOUS_CURRENT_TOP1": "AMBIGUOUS",
}
LABELS = {"TOP1_CORRECT", "TOP1_WRONG", "AMBIGUOUS", "UNRESOLVED"}
Fetcher = Callable[[str, float], tuple[int, str, bytes, str]]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _siret(value: Any) -> str:
    digits = "".join(character for character in _text(value) if character.isdigit())
    if len(digits) != 14:
        raise ValueError(f"Invalid SIRET: {value!r}")
    return digits


def normalize_text(value: Any) -> str:
    """Normalize evidence text without performing fuzzy or model-based matching."""

    decomposed = unicodedata.normalize("NFKD", _text(value))
    ascii_text = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).split())


def fetch_source(url: str, timeout_seconds: float) -> tuple[int, str, bytes, str]:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; SIRETO-V4.7 evidence archive; "
                "+local-research)"
            ),
            "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return (
                int(response.status),
                str(response.headers.get_content_type() or ""),
                response.read(),
                str(response.geturl()),
            )
    except HTTPError as error:
        return (
            int(error.code),
            str(error.headers.get_content_type() if error.headers else ""),
            error.read(),
            str(error.geturl() or url),
        )
    except URLError as error:
        return (0, "", str(error).encode("utf-8"), url)


def extract_evidence_text(payload: bytes, content_type: str) -> str:
    """Extract searchable text while retaining the original bytes separately."""

    lowered = content_type.lower()
    if "pdf" in lowered or payload.startswith(b"%PDF"):
        completed = subprocess.run(
            ["pdftotext", "-layout", "-", "-"],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            return ""
        return completed.stdout.decode("utf-8", errors="replace")
    soup = BeautifulSoup(payload, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return soup.get_text(" ", strip=True)


def load_spec(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError("Unsupported V4.7 public evidence specification")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("V4.7 evidence specification requires sources")
    required = {
        "audit_case_id",
        "siret_to_adjudicate",
        "source_url",
        "producer",
        "source_family",
        "independence_group",
        "relationship",
        "required_terms",
        "fact_summary",
    }
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_sources):
        missing = required - set(raw)
        if missing:
            raise ValueError(f"Evidence source {index} missing: {sorted(missing)}")
        relationship = _text(raw["relationship"]).upper()
        if relationship not in RELATIONSHIP_TO_LABEL:
            raise ValueError(f"Unsupported evidence relationship: {relationship}")
        terms = raw["required_terms"]
        if not isinstance(terms, list) or not terms or any(not _text(x) for x in terms):
            raise ValueError("Every public source needs non-empty required_terms")
        group = _text(raw["independence_group"]).upper()
        if not group or group == SIRENE_GROUP:
            raise ValueError("Public evidence must be independent from SIRENE")
        rows.append(
            {
                **raw,
                "audit_case_id": _text(raw["audit_case_id"]),
                "siret_to_adjudicate": _siret(raw["siret_to_adjudicate"]),
                "producer": _text(raw["producer"]).upper(),
                "source_family": _text(raw["source_family"]).upper(),
                "independence_group": group,
                "relationship": relationship,
                "required_terms": [_text(term) for term in terms],
                "fact_summary": _text(raw["fact_summary"]),
                "spec_source_index": index,
            }
        )
    sources = pd.DataFrame(rows)
    if sources["source_url"].astype(str).duplicated().any():
        raise ValueError("V4.7 source URLs must be unique")
    for case_id, group in sources.groupby("audit_case_id"):
        if group["relationship"].nunique() != 1:
            raise ValueError(f"{case_id}: public sources disagree on relationship")
        if group["siret_to_adjudicate"].nunique() != 1:
            raise ValueError(f"{case_id}: public sources disagree on current SIRET")
    return payload, sources


def validate_inputs(
    *,
    docket: pd.DataFrame,
    official: pd.DataFrame,
    scenes: pd.DataFrame,
    sources: pd.DataFrame,
    enforce_canonical: bool,
) -> None:
    docket_required = {
        "audit_case_id",
        "service_id",
        "siret_to_adjudicate",
        "sampling_stratum",
        "scene_status",
    }
    official_required = {
        "audit_case_id",
        "siret_to_adjudicate",
        "query_kind",
        "http_status",
        "result_count",
        "payload_json",
        "payload_sha256",
        "collected_at",
        "source_url",
        "source_family",
        "independence_group",
    }
    scene_required = {
        "audit_case_id",
        "service_id",
        "sampling_stratum",
        "replayed_top1_siret",
        "scene_status",
        "scene_adjudication_label",
        "scene_acceptor_target",
        "scene_training_eligible",
    }
    for name, frame, required in (
        ("docket", docket, docket_required),
        ("official", official, official_required),
        ("scenes", scenes, scene_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if docket["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("V4.7 docket case IDs must be unique")
    drift = scenes.loc[scenes["scene_status"].astype(str).eq("SCENE_DRIFT")]
    if set(drift["audit_case_id"].astype(str)) != set(
        docket["audit_case_id"].astype(str)
    ):
        raise ValueError("V4.7 docket and scene drift populations differ")
    docket_sirets = docket.set_index("audit_case_id")["siret_to_adjudicate"].map(_siret)
    source_sirets = sources.groupby("audit_case_id")["siret_to_adjudicate"].first()
    unknown = set(source_sirets.index) - set(docket_sirets.index)
    if unknown:
        raise ValueError(f"Evidence spec contains unknown cases: {sorted(unknown)}")
    mismatched = [
        case_id
        for case_id, siret in source_sirets.items()
        if siret != docket_sirets.loc[case_id]
    ]
    if mismatched:
        raise ValueError(f"Evidence spec pins wrong current SIRET: {mismatched}")
    if enforce_canonical:
        if len(docket) != EXPECTED_DRIFT_COUNT or len(drift) != EXPECTED_DRIFT_COUNT:
            raise ValueError("Canonical V4.7 requires exactly 37 drift cases")
        if len(scenes) != EXPECTED_SCENE_COUNT:
            raise ValueError("Canonical V4.7 requires exactly 172 scenes")
        random_count = int(
            scenes["sampling_stratum"].astype(str).eq("RANDOM_POPULATION").sum()
        )
        if random_count != EXPECTED_RANDOM_SCENE_COUNT:
            raise ValueError("Canonical V4.7 requires exactly 57 random scenes")


def archive_public_sources(
    *,
    sources: pd.DataFrame,
    raw_dir: Path,
    timeout_seconds: float,
    fetcher: Fetcher,
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in sources.sort_values(
        ["audit_case_id", "spec_source_index"]
    ).to_dict("records"):
        collected_at = datetime.now(timezone.utc).isoformat()
        status, content_type, payload, final_url = fetcher(
            row["source_url"], timeout_seconds
        )
        raw_hash = hashlib.sha256(payload).hexdigest()
        suffix = ".pdf" if (
            "pdf" in content_type.lower() or payload.startswith(b"%PDF")
        ) else ".html"
        filename = (
            f"{row['audit_case_id']}-{int(row['spec_source_index']):02d}-"
            f"{raw_hash[:12]}{suffix}"
        )
        raw_path = raw_dir / filename
        raw_path.write_bytes(payload)
        extracted = extract_evidence_text(payload, content_type)
        normalized = normalize_text(extracted)
        term_results = {
            term: normalize_text(term) in normalized for term in row["required_terms"]
        }
        terms_validated = bool(term_results) and all(term_results.values())
        usable = bool(
            200 <= int(status) < 300
            and len(payload) > 0
            and len(normalized) > 0
            and terms_validated
        )
        records.append(
            {
                **row,
                "http_status": int(status),
                "content_type": content_type,
                "final_url": final_url,
                "collected_at": collected_at,
                "raw_archive_relative_path": f"public_raw/{filename}",
                "raw_sha256": raw_hash,
                "raw_size_bytes": len(payload),
                "extracted_text_sha256": hashlib.sha256(
                    extracted.encode("utf-8")
                ).hexdigest(),
                "extracted_text_length": len(extracted),
                "required_term_results_json": json.dumps(
                    term_results, ensure_ascii=False, sort_keys=True
                ),
                "terms_validated": terms_validated,
                "usable": usable,
            }
        )
    return pd.DataFrame(records)


def build_registry_evidence(
    docket: pd.DataFrame, official: pd.DataFrame
) -> pd.DataFrame:
    top1 = official.loc[official["query_kind"].astype(str).eq("TOP1_SIRET")].copy()
    if top1["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("Official evidence has duplicate TOP1_SIRET rows")
    official_columns = [
        "audit_case_id",
        "siret_to_adjudicate",
        "http_status",
        "result_count",
        "payload_json",
        "payload_sha256",
        "collected_at",
        "source_url",
        "source_family",
        "independence_group",
    ]
    merged = docket[
        ["audit_case_id", "service_id", "siret_to_adjudicate"]
    ].merge(
        top1[official_columns],
        on=["audit_case_id", "siret_to_adjudicate"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    records: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        payload = _text(row.get("payload_json"))
        siret = _siret(row["siret_to_adjudicate"])
        usable = bool(
            row["_merge"] == "both"
            and int(row.get("http_status") or 0) == 200
            and int(row.get("result_count") or 0) > 0
            and siret in "".join(character for character in payload if character.isdigit())
        )
        records.append(
            {
                "proof_id": f"registry:{row['audit_case_id']}",
                "audit_case_id": str(row["audit_case_id"]),
                "service_id": str(row["service_id"]),
                "siret_to_adjudicate": siret,
                "producer": "API_RECHERCHE_ENTREPRISES",
                "source_family": str(row.get("source_family") or "").upper(),
                "independence_group": SIRENE_GROUP,
                "source_url": str(row.get("source_url") or ""),
                "collected_at": row.get("collected_at"),
                "raw_archive_relative_path": "official_evidence.parquet:TOP1_SIRET",
                "raw_sha256": str(row.get("payload_sha256") or ""),
                "raw_size_bytes": len(payload.encode("utf-8")),
                "extracted_text_sha256": hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest(),
                "extracted_text_length": len(payload),
                "fact_summary": (
                    f"Vue registre du SIRET courant {siret}, archivée avant "
                    "l'adjudication publique."
                ),
                "relationship": "REGISTRY_DESCRIBES_CURRENT_TOP1",
                "terms_validated": usable,
                "usable": usable,
                "required_term_results_json": json.dumps(
                    {siret: usable}, sort_keys=True
                ),
            }
        )
    return pd.DataFrame(records)


def build_public_evidence_rows(public: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in public.to_dict("records"):
        records.append(
            {
                "proof_id": (
                    f"public:{row['audit_case_id']}:{int(row['spec_source_index']):02d}"
                ),
                "audit_case_id": row["audit_case_id"],
                "service_id": None,
                "siret_to_adjudicate": row["siret_to_adjudicate"],
                "producer": row["producer"],
                "source_family": row["source_family"],
                "independence_group": row["independence_group"],
                "source_url": row["source_url"],
                "collected_at": row["collected_at"],
                "raw_archive_relative_path": row["raw_archive_relative_path"],
                "raw_sha256": row["raw_sha256"],
                "raw_size_bytes": int(row["raw_size_bytes"]),
                "extracted_text_sha256": row["extracted_text_sha256"],
                "extracted_text_length": int(row["extracted_text_length"]),
                "fact_summary": row["fact_summary"],
                "relationship": row["relationship"],
                "terms_validated": bool(row["terms_validated"]),
                "usable": bool(row["usable"]),
                "required_term_results_json": row[
                    "required_term_results_json"
                ],
            }
        )
    return pd.DataFrame(records)


def build_adjudications(
    docket: pd.DataFrame, evidence: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for docket_row in docket.sort_values("audit_case_id").to_dict("records"):
        case_id = str(docket_row["audit_case_id"])
        case_evidence = evidence.loc[evidence["audit_case_id"].eq(case_id)].copy()
        registry = case_evidence.loc[
            case_evidence["independence_group"].eq(SIRENE_GROUP)
            & case_evidence["usable"].astype(bool)
        ]
        public = case_evidence.loc[
            ~case_evidence["independence_group"].eq(SIRENE_GROUP)
            & case_evidence["usable"].astype(bool)
        ]
        relationships = sorted(set(public["relationship"].astype(str)))
        groups = sorted(set(pd.concat([registry, public])["independence_group"]))
        reliable = bool(
            len(registry) == 1
            and len(public) >= 1
            and len(relationships) == 1
            and relationships[0] in RELATIONSHIP_TO_LABEL
            and len(groups) >= 2
        )
        label = (
            RELATIONSHIP_TO_LABEL[relationships[0]] if reliable else "UNRESOLVED"
        )
        cited = pd.concat([registry, public]) if reliable else registry.iloc[0:0]
        proof_ids = cited["proof_id"].astype(str).tolist()
        fact_summaries = public["fact_summary"].astype(str).tolist()
        if reliable:
            reason = " ".join(fact_summaries)
        elif len(registry) != 1:
            reason = "Vue registre courante absente ou invalide."
        elif not len(public):
            reason = "Aucune preuve publique indépendante archivée et validée."
        else:
            reason = "Les preuves publiques archivées ne produisent pas une relation unique."
        current_siret = _siret(docket_row["siret_to_adjudicate"])
        acceptor_target: int | None
        if label == "TOP1_CORRECT":
            acceptor_target = 1
        elif label in {"TOP1_WRONG", "AMBIGUOUS"}:
            acceptor_target = 0
        else:
            acceptor_target = None
        records.append(
            {
                "audit_case_id": case_id,
                "query_id": str(docket_row.get("query_id") or ""),
                "service_id": str(docket_row["service_id"]),
                "sampling_stratum": str(docket_row["sampling_stratum"]),
                "evidence_partition": str(docket_row.get("evidence_partition") or ""),
                "siret_to_adjudicate": current_siret,
                "siren_to_adjudicate": current_siret[:9],
                "adjudication_label": label,
                "validated_correct_siret": (
                    current_siret if label == "TOP1_CORRECT" else None
                ),
                "evidence_validated": reliable,
                "training_eligible": reliable,
                "acceptor_target": acceptor_target,
                "acceptor_eligible": reliable,
                "cited_proof_count": int(len(cited)),
                "independent_evidence_group_count": int(len(groups) if reliable else 0),
                "evidence_ref_ids_json": json.dumps(
                    proof_ids, ensure_ascii=False, separators=(",", ":")
                ),
                "evidence_source_groups_json": json.dumps(
                    groups if reliable else [], ensure_ascii=False, separators=(",", ":")
                ),
                "adjudication_reason": reason,
                "adjudication_rule_version": "v4.7-public-relation-derived-1",
                "old_label_transported": False,
                "adjudicated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    output = pd.DataFrame(records)
    if len(output) != len(docket) or output["audit_case_id"].duplicated().any():
        raise ValueError("V4.7 adjudications do not cover the docket exactly once")
    validated = output["evidence_validated"].astype(bool)
    if (
        output.loc[validated, "independent_evidence_group_count"].lt(2).any()
        or output["old_label_transported"].astype(bool).any()
    ):
        raise ValueError("V4.7 adjudication evidence invariant failed")
    if not set(output["adjudication_label"]).issubset(LABELS):
        raise ValueError("V4.7 produced an unsupported label")
    return output


def build_current_labels(
    scenes: pd.DataFrame, adjudications: pd.DataFrame
) -> pd.DataFrame:
    adjudication_index = adjudications.set_index("audit_case_id")
    output = scenes.copy()
    labels: list[str] = []
    validated_values: list[bool] = []
    targets: list[int | None] = []
    origins: list[str] = []
    references: list[str] = []
    groups: list[int] = []
    for row in output.to_dict("records"):
        case_id = str(row["audit_case_id"])
        if str(row["scene_status"]) == "SCENE_DRIFT":
            current = adjudication_index.loc[case_id]
            if _siret(row["replayed_top1_siret"]) != _siret(
                current["siret_to_adjudicate"]
            ):
                raise ValueError(f"{case_id}: drift SIRET binding mismatch")
            label = str(current["adjudication_label"])
            reliable = bool(current["evidence_validated"])
            target = current["acceptor_target"] if reliable else None
            origin = "V4.7_CURRENT_TOP1"
            reference = str(current["evidence_ref_ids_json"])
            group_count = int(current["independent_evidence_group_count"])
        else:
            reliable = bool(row.get("scene_training_eligible"))
            transported = _text(row.get("scene_adjudication_label"))
            label = transported if reliable else "UNRESOLVED"
            target = row.get("scene_acceptor_target") if reliable else None
            origin = "V4.4_TRANSPORT_EXACT_TOP1"
            reference = "[]"
            group_count = 2 if reliable else 0
        labels.append(label)
        validated_values.append(reliable)
        targets.append(target)
        origins.append(origin)
        references.append(reference)
        groups.append(group_count)
    output["current_top1_siret"] = output["replayed_top1_siret"].map(_siret)
    output["current_adjudication_label"] = labels
    output["current_evidence_validated"] = validated_values
    output["current_training_eligible"] = validated_values
    output["current_acceptor_target"] = targets
    output["current_label_origin"] = origins
    output["current_evidence_ref_ids_json"] = references
    output["current_independent_evidence_group_count"] = groups
    drift = output["scene_status"].astype(str).eq("SCENE_DRIFT")
    if output.loc[drift, "current_label_origin"].ne("V4.7_CURRENT_TOP1").any():
        raise ValueError("An old label was transported to a drifted top1")
    return output


def evaluate_gate(
    adjudications: pd.DataFrame, current_labels: pd.DataFrame
) -> dict[str, Any]:
    reliable = current_labels["current_evidence_validated"].astype(bool)
    random = current_labels["sampling_stratum"].astype(str).eq("RANDOM_POPULATION")
    negative = current_labels["current_adjudication_label"].isin(
        ["TOP1_WRONG", "AMBIGUOUS"]
    )
    drift = current_labels["scene_status"].astype(str).eq("SCENE_DRIFT")
    targeted = ~random
    metrics = {
        "current_scene_count": int(len(current_labels)),
        "v47_docket_count": int(len(adjudications)),
        "current_reliable_label_count": int(reliable.sum()),
        "current_random_scene_count": int(random.sum()),
        "current_random_reliable_count": int((random & reliable).sum()),
        "current_targeted_reliable_negative_count": int(
            (targeted & reliable & negative).sum()
        ),
        "current_random_reliable_negative_count": int(
            (random & reliable & negative).sum()
        ),
        "v47_reliable_count": int(adjudications["evidence_validated"].sum()),
        "v47_unresolved_count": int(
            adjudications["adjudication_label"].eq("UNRESOLVED").sum()
        ),
        "v47_label_counts": {
            str(key): int(value)
            for key, value in adjudications["adjudication_label"]
            .value_counts()
            .items()
        },
        "drift_old_label_transport_count": int(
            (
                drift
                & current_labels["current_label_origin"].ne(
                    "V4.7_CURRENT_TOP1"
                )
            ).sum()
        ),
        "reliable_under_two_groups_count": int(
            (
                reliable
                & current_labels["current_independent_evidence_group_count"].lt(2)
            ).sum()
        ),
    }
    quality_checks = {
        "docket_exactly_37": metrics["v47_docket_count"] == 37,
        "current_scene_exactly_172": metrics["current_scene_count"] == 172,
        "current_reliable_at_least_150": (
            metrics["current_reliable_label_count"] >= 150
        ),
        "random_reliable_at_least_50": (
            metrics["current_random_reliable_count"] >= 50
        ),
        "all_reliable_have_two_groups": (
            metrics["reliable_under_two_groups_count"] == 0
        ),
        "no_old_label_transport_on_drift": (
            metrics["drift_old_label_transport_count"] == 0
        ),
    }
    quality_passed = all(quality_checks.values())
    if not quality_passed:
        verdict = "STOP_CURRENT_LABEL_QUALITY"
    elif (
        metrics["current_targeted_reliable_negative_count"] >= 20
        and metrics["current_random_reliable_negative_count"] >= 3
    ):
        verdict = "GO_ACCEPTOR_FEASIBILITY"
    elif metrics["current_targeted_reliable_negative_count"] < 20:
        verdict = "KEEP_CURRENT_STACK_SHADOW"
    else:
        verdict = "PIVOT_INDEPENDENT_EVALUATION"
    return {
        "schema_version": "sireto-v4.7-current-adjudication-gate-1",
        "verdict": verdict,
        "quality_gate_passed": quality_passed,
        "quality_checks": quality_checks,
        "thresholds": {
            "current_reliable_label_min": 150,
            "random_reliable_min": 50,
            "targeted_reliable_negative_min": 20,
            "random_reliable_negative_min": 3,
        },
        "metrics": metrics,
        "model_training_performed": False,
        "test_opened": False,
    }


def render_report(gate: dict[str, Any], public: pd.DataFrame) -> str:
    metrics = gate["metrics"]
    failed_sources = public.loc[~public["usable"].astype(bool)]
    failures = "\n".join(
        f"- `{row.audit_case_id}` — HTTP {row.http_status}, "
        f"termes_validés={bool(row.terms_validated)} — {row.source_url}"
        for row in failed_sources.itertuples(index=False)
    ) or "- Aucun."
    return (
        "# V4.7 — adjudication des top-1 courants\n\n"
        f"Verdict : **{gate['verdict']}**\n\n"
        "Aucun modèle n'a été entraîné et le test final est resté fermé.\n\n"
        "## Gate du corpus\n\n"
        f"- Labels courants fiables : {metrics['current_reliable_label_count']} / "
        f"{metrics['current_scene_count']} (minimum 150)\n"
        f"- Aléatoires fiables : {metrics['current_random_reliable_count']} / "
        f"{metrics['current_random_scene_count']} (minimum 50)\n"
        f"- Négatifs ciblés fiables : "
        f"{metrics['current_targeted_reliable_negative_count']} (minimum 20)\n"
        f"- Négatifs aléatoires fiables : "
        f"{metrics['current_random_reliable_negative_count']} (minimum 3)\n"
        f"- V4.7 résolus : {metrics['v47_reliable_count']} / "
        f"{metrics['v47_docket_count']}\n"
        f"- V4.7 non résolus : {metrics['v47_unresolved_count']}\n"
        f"- Transport d'ancien label vers un top-1 différent : "
        f"{metrics['drift_old_label_transport_count']}\n\n"
        "## Sources publiques non retenues automatiquement\n\n"
        f"{failures}\n"
    )


def build_artifact(
    *,
    docket_path: Path,
    official_dir: Path,
    scenes_path: Path,
    spec_path: Path,
    contract_path: Path,
    output_root: Path,
    timeout_seconds: float = 30.0,
    enforce_canonical: bool = True,
    fetcher: Fetcher = fetch_source,
) -> Path:
    docket_path = Path(docket_path).resolve()
    official_dir = Path(official_dir).resolve()
    official_path = official_dir / "official_evidence.parquet"
    official_manifest_path = official_dir / "manifest.json"
    scenes_path = Path(scenes_path).resolve()
    spec_path = Path(spec_path).resolve()
    contract_path = Path(contract_path).resolve()
    input_paths = {
        "docket": docket_path,
        "official_evidence": official_path,
        "official_manifest": official_manifest_path,
        "scenes": scenes_path,
        "spec": spec_path,
        "contract": contract_path,
    }
    input_hashes = {name: file_sha256(path) for name, path in input_paths.items()}
    if enforce_canonical:
        if input_hashes["docket"] != EXPECTED_DOCKET_SHA256:
            raise ValueError("Canonical V4.7 docket hash mismatch")
        if input_hashes["scenes"] != EXPECTED_SCENES_SHA256:
            raise ValueError("Canonical V4.7 scenes hash mismatch")
    official_manifest = json.loads(
        official_manifest_path.read_text(encoding="utf-8")
    )
    expected_official_hash = official_manifest.get("outputs", {}).get(
        "official_evidence.parquet"
    )
    if expected_official_hash != input_hashes["official_evidence"]:
        raise ValueError("V4.7 official evidence manifest mismatch")
    if official_manifest.get("docket_sha256") != input_hashes["docket"]:
        raise ValueError("V4.7 official evidence was built from another docket")

    spec_payload, sources = load_spec(spec_path)
    docket = pd.read_parquet(docket_path).copy()
    official = pd.read_parquet(official_path).copy()
    scenes = pd.read_parquet(scenes_path).copy()
    validate_inputs(
        docket=docket,
        official=official,
        scenes=scenes,
        sources=sources,
        enforce_canonical=enforce_canonical,
    )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": spec_payload["policy_version"],
        "input_hashes": input_hashes,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root).resolve() / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.7 adjudications exist: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent))
    try:
        public = archive_public_sources(
            sources=sources,
            raw_dir=staging / "public_raw",
            timeout_seconds=timeout_seconds,
            fetcher=fetcher,
        )
        registry_rows = build_registry_evidence(docket, official)
        public_rows = build_public_evidence_rows(public)
        evidence = pd.concat([registry_rows, public_rows], ignore_index=True)
        evidence = evidence.sort_values(
            ["audit_case_id", "independence_group", "proof_id"]
        ).reset_index(drop=True)
        adjudications = build_adjudications(docket, evidence)
        current_labels = build_current_labels(scenes, adjudications)
        gate = evaluate_gate(adjudications, current_labels)

        shutil.copy2(official_path, staging / "official_evidence.parquet")
        evidence_path = staging / "evidence.parquet"
        adjudications_path = staging / "adjudications.parquet"
        current_labels_path = staging / "current_labels.parquet"
        gate_path = staging / "gate_report.json"
        report_path = staging / "report.md"
        evidence.to_parquet(evidence_path, index=False)
        adjudications.to_parquet(adjudications_path, index=False)
        current_labels.to_parquet(current_labels_path, index=False)
        _json_dump(gate_path, gate)
        report_path.write_text(render_report(gate, public), encoding="utf-8")

        output_files = [
            staging / "official_evidence.parquet",
            evidence_path,
            adjudications_path,
            current_labels_path,
            gate_path,
            report_path,
            *sorted((staging / "public_raw").iterdir()),
        ]
        outputs = {
            str(path.relative_to(staging)): file_sha256(path)
            for path in output_files
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                name: {"path": str(input_paths[name]), "sha256": input_hashes[name]}
                for name in input_paths
            },
            "outputs": outputs,
            "summary": gate["metrics"],
            "verdict": gate["verdict"],
            "invariants": {
                "all_37_processed_once": True,
                "minimum_independent_evidence_groups": 2,
                "old_label_transport_to_drift": False,
                "positive_injection": False,
                "model_training_performed": False,
                "test_opened": False,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docket", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        build_artifact(
            docket_path=args.docket,
            official_dir=args.official_dir,
            scenes_path=args.scenes,
            spec_path=args.spec,
            contract_path=args.contract,
            output_root=args.output_root,
            timeout_seconds=args.timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()
