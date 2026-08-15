#!/usr/bin/env python3
"""Publish a conservative secondary same-site view of the frozen BGE cycle.

This script never changes a frozen prediction or exact-SIRET metric.  It only
adds the retrospective operational view defined in
``docs/siret_operational_equivalence_policy.md``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_CORPUS = BASE / "datasets/v4_12_neural_text_corpus/02b8668f8050c5e9"
DEFAULT_BUSINESS_DATA = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_BUSINESS_RANKER = BASE / "experiments/v4_12_learned_oof_rankers/839ef55308d5077e"
DEFAULT_BGE = BASE / "experiments/v4_12_bge_groupwise/01e1049c16af2600"
DEFAULT_STACK = BASE / "experiments/v4_12_bge_xgb_stack/8c1bce0bbf9593b5"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_bge_operational_secondary"
SCHEMA_VERSION = "sireto-v4.12-bge-operational-secondary-1"
POLICY = "docs/siret_operational_equivalence_policy.md"
FOLD = 0


ROAD_TYPES = {
    "AV": "AVENUE",
    "AVE": "AVENUE",
    "AVENUE": "AVENUE",
    "BD": "BOULEVARD",
    "BOUL": "BOULEVARD",
    "BOULEVARD": "BOULEVARD",
    "CHE": "CHEMIN",
    "CH": "CHEMIN",
    "CHEMIN": "CHEMIN",
    "IMP": "IMPASSE",
    "IMPASSE": "IMPASSE",
    "PL": "PLACE",
    "PLACE": "PLACE",
    "R": "RUE",
    "RUE": "RUE",
    "RTE": "ROUTE",
    "ROUTE": "ROUTE",
    "QU": "QUAI",
    "QUAI": "QUAI",
    "ALL": "ALLEE",
    "ALLEE": "ALLEE",
    "VC": "VOIE COMMUNALE",
}
SUFFIXES = {"B": "BIS", "BIS": "BIS", "T": "TER", "TER": "TER", "Q": "QUATER", "QUATER": "QUATER"}


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()
    return re.sub(r"\s+", " ", text)


def _normalise_street(value: str) -> str:
    tokens = _norm(value).split()
    if not tokens:
        return ""
    replacement = ROAD_TYPES.get(tokens[0])
    if replacement:
        tokens = replacement.split() + tokens[1:]
    return " ".join(tokens)


def _parse_number_and_street(value: object) -> tuple[str, str, str] | None:
    text = _norm(value)
    match = re.match(
        r"^(\d+)(?:\s*(BIS|TER|QUATER)|\s*([A-Z])(?=\s))?\s+(.+)$",
        text,
    )
    if not match:
        return None
    number = str(int(match.group(1)))
    raw_suffix = match.group(2) or match.group(3) or ""
    suffix = SUFFIXES.get(raw_suffix, raw_suffix)
    street = _normalise_street(match.group(4))
    if not street:
        return None
    return number, suffix, street


def _parse_candidate_text(text: object) -> dict[str, str]:
    value = str(text or "")
    address_match = re.search(
        r"Adresse établissement\s*:\s*(.*?)\.\s*Code commune\s*:",
        value,
    )
    insee_match = re.search(r"Code commune\s*:\s*([^.]*)\.", value)
    state_match = re.search(r"État établissement\s*:\s*([AF])\.", value)
    address_full = address_match.group(1).strip() if address_match else ""
    postal_match = re.search(r",\s*(\d{5})\s+[^,]*$", address_full)
    postcode = postal_match.group(1) if postal_match else ""
    street_address = address_full[: postal_match.start()].strip() if postal_match else address_full
    return {
        "address": street_address,
        "postcode": postcode,
        "insee": _norm(insee_match.group(1) if insee_match else ""),
        "state": state_match.group(1) if state_match else "",
    }


def _same_site_evidence(query: pd.Series, candidate_text: object) -> dict[str, Any] | None:
    crm_site = _parse_number_and_street(query.get("crm_address"))
    candidate = _parse_candidate_text(candidate_text)
    candidate_site = _parse_number_and_street(candidate["address"])
    if crm_site is None or candidate_site is None:
        return None
    number_match = crm_site[0] == candidate_site[0]
    suffix_match = crm_site[1] == candidate_site[1]
    street_match = crm_site[2] == candidate_site[2]
    crm_postcode = re.sub(r"\D", "", str(query.get("crm_postcode") or ""))
    crm_insee = _norm(query.get("crm_insee"))
    if crm_postcode and candidate["postcode"]:
        geo_match = crm_postcode == candidate["postcode"]
        geo_basis = "POSTCODE_EXACT"
    else:
        geo_match = bool(crm_insee and candidate["insee"] and crm_insee == candidate["insee"])
        geo_basis = "INSEE_EXACT_POSTCODE_MISSING"
    if not (number_match and suffix_match and street_match and geo_match):
        return None
    return {
        "crm_number": crm_site[0],
        "crm_suffix": crm_site[1],
        "crm_street_normalized": crm_site[2],
        "candidate_number": candidate_site[0],
        "candidate_suffix": candidate_site[1],
        "candidate_street_normalized": candidate_site[2],
        "crm_postcode": crm_postcode,
        "candidate_postcode": candidate["postcode"],
        "crm_insee": crm_insee,
        "candidate_insee": candidate["insee"],
        "geography_basis": geo_basis,
        "candidate_state": candidate["state"],
        "proof_strength": "NUMBER_SUFFIX_STREET_EXACT_AND_POSTCODE_OR_INSEE_EXACT",
    }


def _validate_input(root: Path, name: str) -> str:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("outputs", {}).get(name)
    if expected is not None and expected != file_sha256(root / name):
        raise ValueError(f"Input hash mismatch: {root / name}")
    return file_sha256(root / "manifest.json")


def run(args: argparse.Namespace) -> Path:
    corpus_hash = _validate_input(args.corpus, "candidates_text.parquet")
    business_data_hash = _validate_input(args.business_data, "queries.parquet")
    business_ranker_hash = _validate_input(args.business_ranker, "business_learned_oof_top1.parquet")
    bge_hash = _validate_input(args.bge, "target_top1_detail.parquet")
    stack_hash = _validate_input(args.stack, "fold0_top1_comparison.parquet")
    policy_hash = file_sha256(Path(POLICY))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "policy_sha256": policy_hash,
        "corpus_manifest_sha256": corpus_hash,
        "business_data_manifest_sha256": business_data_hash,
        "business_ranker_manifest_sha256": business_ranker_hash,
        "bge_manifest_sha256": bge_hash,
        "stack_manifest_sha256": stack_hash,
        "fold": FOLD,
        "scope": "SECONDARY_RETROSPECTIVE_OPERATIONAL_FOLD0",
        "frozen_predictions_rescored": False,
        "same_site_rule": "same_siren+number+suffix+normalized_street+postcode_or_missing_postcode_insee",
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    queries = pd.read_parquet(args.business_data / "queries.parquet")
    labels = pd.read_parquet(args.corpus / "labels.parquet")
    labels["query_id"] = labels["query_id"].astype(str)
    truth = labels[
        labels["oof_fold"].astype(int).eq(FOLD)
        & labels["label_kind"].eq("MATCH_EXACT")
    ].copy()
    query_ids = set(truth["query_id"].astype(str))
    truth_by_id = truth.set_index("query_id")
    queries["query_id"] = queries["query_id"].astype(str)
    queries = queries[queries["query_id"].isin(query_ids)].copy()
    query_by_id = queries.set_index("query_id")

    candidates = pd.read_parquet(
        args.corpus / "candidates_text.parquet",
        columns=["query_id", "candidate_siret", "candidate_siren", "candidate_text"],
    )
    candidates["query_id"] = candidates["query_id"].astype(str)
    candidates = candidates[candidates["query_id"].isin(query_ids)].copy()
    candidates["candidate_siret"] = candidates["candidate_siret"].astype(str)
    candidates["candidate_siren"] = candidates["candidate_siren"].astype(str)

    acceptable_by_query: dict[str, dict[str, dict[str, Any]]] = {}
    for row in truth.itertuples(index=False):
        query_id = str(row.query_id)
        exact = str(row.ground_truth_siret)
        acceptable_by_query[query_id] = {
            exact: {
                "reason": "EXACT_SIRET",
                "state": str(row.ground_truth_state or ""),
                "evidence": None,
            }
        }
    for row in candidates.itertuples(index=False):
        query_id = str(row.query_id)
        if query_id not in truth_by_id.index:
            continue
        label_row = truth_by_id.loc[query_id]
        if str(row.candidate_siren) != str(label_row.ground_truth_siren):
            continue
        evidence = _same_site_evidence(query_by_id.loc[query_id], row.candidate_text)
        if evidence is None:
            continue
        exact = str(label_row.ground_truth_siret)
        siret = str(row.candidate_siret)
        if siret == exact:
            acceptable_by_query[query_id][exact]["evidence"] = evidence
            continue
        reason = (
            "ACTIVE_SUCCESSOR_SAME_SITE"
            if str(label_row.ground_truth_state) == "F" and evidence["candidate_state"] == "A"
            else "OPERATIONAL_EQUIVALENT_SAME_SITE"
        )
        acceptable_by_query[query_id][siret] = {
            "reason": reason,
            "state": evidence["candidate_state"],
            "evidence": evidence,
        }

    business = pd.read_parquet(args.business_ranker / "business_learned_oof_top1.parquet")
    business["query_id"] = business["query_id"].astype(str)
    business = business[
        business["oof_fold"].astype(int).eq(FOLD)
        & business["label_kind"].eq("MATCH_EXACT")
    ][["query_id", "predicted_siret"]].copy()
    business["system"] = "BUSINESS_LEARNED"
    bge = pd.read_parquet(args.bge / "target_top1_detail.parquet")
    bge["query_id"] = bge["query_id"].astype(str)
    bge = bge[["query_id", "candidate_siret"]].rename(columns={"candidate_siret": "predicted_siret"})
    bge["system"] = "BGE_FINETUNED"
    stack = pd.read_parquet(args.stack / "fold0_top1_comparison.parquet")
    stack["query_id"] = stack["query_id"].astype(str)
    stack = stack[["query_id", "candidate_siret"]].rename(columns={"candidate_siret": "predicted_siret"})
    stack["system"] = "XGB_BGE_STACK"
    predictions = pd.concat([business, bge, stack], ignore_index=True)
    predictions["predicted_siret"] = predictions["predicted_siret"].astype(str)
    predictions = predictions.merge(
        truth[["query_id", "ground_truth_siret", "ground_truth_siren", "ground_truth_state"]],
        on="query_id",
        validate="many_to_one",
    )

    detail_rows: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        acceptable = acceptable_by_query[str(row.query_id)]
        predicted = str(row.predicted_siret)
        exact = predicted == str(row.ground_truth_siret)
        operational = predicted in acceptable
        metadata = acceptable.get(predicted, {})
        active_count = sum(1 for value in acceptable.values() if value.get("state") == "A")
        detail_rows.append(
            {
                "system": row.system,
                "query_id": str(row.query_id),
                "predicted_siret": predicted,
                "predicted_siren": predicted[:9] if len(predicted) == 14 and predicted.isdigit() else None,
                "ground_truth_siret_exact": str(row.ground_truth_siret),
                "ground_truth_siren": str(row.ground_truth_siren),
                "ground_truth_siret_state": str(row.ground_truth_state),
                "acceptable_sirets_operational": sorted(acceptable),
                "exact_siret_correct": exact,
                "operational_siret_correct": operational,
                "operational_equivalence_reason": metadata.get("reason", "NOT_OPERATIONALLY_EQUIVALENT"),
                "predicted_siret_state": metadata.get("state", ""),
                "same_site_evidence": json.dumps(metadata.get("evidence"), ensure_ascii=False, sort_keys=True),
                "multiple_active_equivalents": active_count > 1,
                "active_equivalent_count_in_candidate_pool": active_count,
            }
        )
    detail = pd.DataFrame(detail_rows).sort_values(["system", "query_id"], kind="mergesort")
    if len(detail) != 3 * len(truth):
        raise ValueError("Operational view does not contain every system/query pair")

    metric_rows = []
    for system, frame in detail.groupby("system", sort=True):
        exact_correct = int(frame["exact_siret_correct"].sum())
        operational_correct = int(frame["operational_siret_correct"].sum())
        metric_rows.append(
            {
                "system": system,
                "total": len(frame),
                "exact_siret_correct": exact_correct,
                "exact_hit_at_1": exact_correct / len(frame),
                "operational_siret_correct": operational_correct,
                "operational_hit_at_1": operational_correct / len(frame),
                "promoted_same_site": operational_correct - exact_correct,
                "active_successor_same_site": int(frame["operational_equivalence_reason"].eq("ACTIVE_SUCCESSOR_SAME_SITE").sum()),
                "multiple_active_equivalents": int(frame["multiple_active_equivalents"].sum()),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        detail.to_parquet(temporary / "fold0_operational_detail.parquet", index=False)
        metrics.to_csv(temporary / "fold0_operational_metrics.csv", index=False)
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": identity["scope"],
            "primary_exact_results_modified": False,
            "fold1_opened": False,
            "final_test_opened": False,
            "same_site_proof_is_conservative": True,
            "candidate_universe_for_equivalents": "FROZEN_TOP100_ONLY",
            "metrics": metrics.to_dict("records"),
        }
        _json_dump(temporary / "evaluation.json", evaluation)
        output_names = [
            str(path.relative_to(temporary))
            for path in temporary.rglob("*")
            if path.is_file()
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "build_identity": identity,
            "outputs": {name: file_sha256(temporary / name) for name in sorted(output_names)},
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--business-data", type=Path, default=DEFAULT_BUSINESS_DATA)
    parser.add_argument("--business-ranker", type=Path, default=DEFAULT_BUSINESS_RANKER)
    parser.add_argument("--bge", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
