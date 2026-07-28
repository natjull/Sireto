"""Deterministic V4.9 site-function taxonomy and AUTO-to-REVIEW guard."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable
import unicodedata


@dataclass(frozen=True)
class FunctionDetection:
    roles: tuple[str, ...]
    matched_patterns: tuple[str, ...]
    matched_activity_codes: tuple[str, ...]

    @property
    def kind(self) -> str:
        if not self.roles:
            return "UNKNOWN"
        if len(self.roles) > 1:
            return "MULTI_ROLE"
        return self.roles[0]


@dataclass(frozen=True)
class GuardDecision:
    review: bool
    reason: str | None
    crm_roles: tuple[str, ...]
    candidate_roles: tuple[str, ...]


def normalize_function_text(value: Any) -> str:
    text = "" if value is None else str(value)
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).split())


class SiteFunctionTaxonomy:
    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != "sireto-v4.9-site-function-taxonomy-1":
            raise ValueError("Unsupported V4.9 site-function taxonomy")
        roles = payload.get("roles")
        if not isinstance(roles, list) or not roles:
            raise ValueError("V4.9 taxonomy requires ordered roles")
        self.roles: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in roles:
            role = str(raw.get("role") or "")
            if not role or role in seen:
                raise ValueError("V4.9 taxonomy role names must be unique")
            seen.add(role)
            patterns = [str(value) for value in raw.get("patterns") or []]
            codes = [str(value).upper() for value in raw.get("activity_codes") or []]
            self.roles.append(
                {
                    "role": role,
                    "patterns": patterns,
                    "compiled": [re.compile(pattern) for pattern in patterns],
                    "activity_codes": codes,
                }
            )
        self.generic_roles = {
            str(value) for value in payload.get("generic_roles") or []
        }
        self.incompatible_pairs: set[frozenset[str]] = set()
        for family in payload.get("incompatible_families") or []:
            values = [str(value) for value in family]
            for index, left in enumerate(values):
                for right in values[index + 1 :]:
                    self.incompatible_pairs.add(frozenset((left, right)))

    @classmethod
    def load(cls, path: Path) -> "SiteFunctionTaxonomy":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def detect(
        self,
        texts: Iterable[Any],
        *,
        activity_code: Any = None,
    ) -> FunctionDetection:
        normalized = " ".join(
            value for raw in texts if (value := normalize_function_text(raw))
        )
        code = str(activity_code or "").strip().upper()
        roles: list[str] = []
        matched_patterns: list[str] = []
        matched_codes: list[str] = []
        for spec in self.roles:
            pattern_hits = [
                pattern.pattern
                for pattern in spec["compiled"]
                if pattern.search(normalized)
            ]
            code_hit = code in spec["activity_codes"]
            if pattern_hits or code_hit:
                roles.append(spec["role"])
                matched_patterns.extend(
                    f"{spec['role']}:{pattern}" for pattern in pattern_hits
                )
                if code_hit:
                    matched_codes.append(f"{spec['role']}:{code}")
        specific = [role for role in roles if role not in self.generic_roles]
        if specific:
            roles = specific
        return FunctionDetection(
            roles=tuple(dict.fromkeys(roles)),
            matched_patterns=tuple(matched_patterns),
            matched_activity_codes=tuple(matched_codes),
        )

    def incompatible(self, left: str, right: str) -> bool:
        if left == right:
            return False
        return frozenset((left, right)) in self.incompatible_pairs

    def guard(
        self,
        crm: FunctionDetection,
        candidate: FunctionDetection,
    ) -> GuardDecision:
        crm_roles = tuple(role for role in crm.roles if role not in self.generic_roles)
        candidate_roles = tuple(
            role for role in candidate.roles if role not in self.generic_roles
        )
        if len(crm_roles) > 1 and any(
            self.incompatible(left, right)
            for index, left in enumerate(crm_roles)
            for right in crm_roles[index + 1 :]
        ):
            return GuardDecision(
                review=True,
                reason="SITE_FUNCTION_AMBIGUOUS",
                crm_roles=crm.roles,
                candidate_roles=candidate.roles,
            )
        if crm_roles and candidate_roles and any(
            self.incompatible(left, right)
            for left in crm_roles
            for right in candidate_roles
        ):
            return GuardDecision(
                review=True,
                reason="SITE_FUNCTION_CONFLICT",
                crm_roles=crm.roles,
                candidate_roles=candidate.roles,
            )
        return GuardDecision(
            review=False,
            reason=None,
            crm_roles=crm.roles,
            candidate_roles=candidate.roles,
        )


__all__ = [
    "FunctionDetection",
    "GuardDecision",
    "SiteFunctionTaxonomy",
    "normalize_function_text",
]
