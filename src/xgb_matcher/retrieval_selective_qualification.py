"""Pre-retrieval qualification from CRM fields and official dossier evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping

from rapidfuzz.distance import JaroWinkler


_TOKEN = re.compile(r"[A-Z0-9]+")


def normalize_evidence(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(_TOKEN.findall(text.upper()))


def split_portfolio(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        for raw in str(value or "").split(" | "):
            normalized = normalize_evidence(raw)
            if normalized and normalized not in seen:
                seen.add(normalized)
                output.append(normalized)
    return output


def token_jaccard(left: str, right: str) -> float:
    a, b = set(left.split()), set(right.split())
    return len(a & b) / len(a | b) if a or b else 0.0


def _maximum(left: str, values: Iterable[str], scorer: Any) -> float:
    return max((float(scorer(left, value)) for value in values), default=0.0)


@dataclass(frozen=True)
class QualificationPolicy:
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "QualificationPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != (
            "sireto-retrieval-selective-qualification-policy-v1"
        ):
            raise ValueError("Unsupported selective qualification policy")
        if payload.get("forbidden_inputs") != [
            "retrieval_hit",
            "retrieval_rank",
            "retrieval_score",
            "model_score",
            "selected_candidates",
        ]:
            raise ValueError("Qualification policy must forbid retrieval outputs")
        return cls(payload)


def qualify_direct_evidence(
    row: Mapping[str, Any], policy: QualificationPolicy
) -> dict[str, Any]:
    crm_name = normalize_evidence(row.get("crm_name"))
    crm_address = normalize_evidence(row.get("crm_address"))
    crm_number = normalize_evidence(row.get("crm_number"))
    candidate_number = normalize_evidence(row.get("candidate_number"))
    names = split_portfolio(
        row.get(field) for field in policy.payload["allowed_official_name_fields"]
    )
    addresses = split_portfolio(
        row.get(field) for field in policy.payload["allowed_official_address_fields"]
    )
    name_exact = bool(crm_name and crm_name in names)
    address_exact = bool(crm_address and crm_address in addresses)
    name_jaro = _maximum(
        crm_name, names, JaroWinkler.normalized_similarity
    )
    name_tokens = _maximum(crm_name, names, token_jaccard)
    address_jaro = _maximum(
        crm_address, addresses, JaroWinkler.normalized_similarity
    )
    name_containment = any(
        crm_name and value and (crm_name in value or value in crm_name)
        for value in names
    )
    number_compatible = bool(
        not crm_number
        or not candidate_number
        or crm_number == candidate_number
    )
    strong_name_policy = policy.payload["strong_name"]
    strong_name = bool(
        name_exact
        or (
            name_jaro >= strong_name_policy["jaro_winkler_minimum"]
            and name_tokens >= strong_name_policy["token_jaccard_minimum"]
        )
        or (
            name_containment
            and name_jaro
            >= strong_name_policy["containment_jaro_winkler_minimum"]
        )
    )
    strong_address_policy = policy.payload["strong_address"]
    strong_address = bool(
        address_exact
        or (
            address_jaro >= strong_address_policy["jaro_winkler_minimum"]
            and (
                number_compatible
                or not strong_address_policy["require_compatible_number"]
            )
        )
    )
    moderate = policy.payload["moderate_joint"]
    moderate_joint = bool(
        name_jaro >= moderate["name_jaro_winkler_minimum"]
        and name_tokens >= moderate["name_token_jaccard_minimum"]
        and address_jaro >= moderate["address_jaro_winkler_minimum"]
        and (number_compatible or not moderate["require_compatible_number"])
    )
    identifiable = strong_name or strong_address or moderate_joint
    reason = (
        "STRONG_NAME"
        if strong_name
        else "STRONG_ADDRESS"
        if strong_address
        else "MODERATE_NAME_ADDRESS_NUMBER"
        if moderate_joint
        else "INSUFFICIENT_DIRECT_EVIDENCE"
    )
    return {
        "identifiable_exact_v4": identifiable,
        "qualification_reason_v4": reason,
        "name_exact": name_exact,
        "address_exact": address_exact,
        "name_jaro_winkler": name_jaro,
        "name_token_jaccard": name_tokens,
        "address_jaro_winkler": address_jaro,
        "number_compatible": number_compatible,
        "strong_name": strong_name,
        "strong_address": strong_address,
        "moderate_joint": moderate_joint,
    }
