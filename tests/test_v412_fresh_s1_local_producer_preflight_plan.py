from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN = (
    REPOSITORY
    / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def test_preflight_plan_is_canonical_and_cross_pinned() -> None:
    raw = PLAN.read_bytes()
    plan = json.loads(raw)
    assert raw == _canonical(plan)
    for role in ("contract", "execution_lock"):
        authority = plan[role]
        assert hashlib.sha256(
            (REPOSITORY / authority["path"]).read_bytes()
        ).hexdigest() == authority["sha256"]


def test_preflight_query_cannot_return_secret_or_attributes() -> None:
    plan = json.loads(PLAN.read_bytes())
    assert plan["query_exact"] == {
        "kSecClass": "kSecClassGenericPassword",
        "kSecAttrService": (
            "com.sireto.v412.fresh-s1-producer-ed25519"
        ),
        "kSecAttrAccount": "SIRETO",
        "kSecAttrSynchronizable": False,
        "kSecUseDataProtectionKeychain": True,
        "kSecUseAuthenticationUI": "kSecUseAuthenticationUIFail",
        "kSecMatchLimit": "kSecMatchLimitOne",
    }
    assert set(plan["query_forbidden_keys"]).isdisjoint(plan["query_exact"])
    assert plan["success"] == {
        "osstatus": -25300,
        "verdict": "KEYCHAIN_LOCATOR_ABSENT",
    }


def test_preflight_output_is_closed_and_real_output_absent() -> None:
    plan = json.loads(PLAN.read_bytes())
    assert plan["output"]["schema"]["exact_fields"] == [
        "schema_version",
        "verdict",
        "execution_lock_sha256",
        "query_sha256",
        "osstatus",
        "logical_time_utc",
    ]
    assert not (REPOSITORY / plan["output"]["path"]).exists()
    assert not (
        REPOSITORY
        / "config/v4_12_fresh_s1_local_producer_authorization.json"
    ).exists()
