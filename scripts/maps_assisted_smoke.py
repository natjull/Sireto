#!/usr/bin/env python3
"""Bounded, secret-safe MAPS_ASSISTED smoke runner.

This module never searches the repository for credentials.  Production secret
resolution is limited to the explicitly named environment variable and the
macOS Keychain service requested by the contract.  The smoke is disabled by
default and cannot make a network call without a positive budget and an
explicit execution flag.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request

from build_synthetic_gt_corpus import (
    canonical_json,
    max_name_similarity,
    normalize_text,
    sha256_bytes,
    text_ratio,
    token_jaccard,
)


SECRET_ENV = "SIRETO_GOOGLE_MAPS_API_KEY"
KEYCHAIN_SERVICE = "SIRETO_GOOGLE_MAPS_API_KEY"
ENDPOINT = "https://places.googleapis.com/v1/places:searchText"
FIELD_MASK = "places.id,places.displayName,places.formattedAddress,places.addressComponents"
QUERY_VERSION = "maps-text-search-new-v1-minimal-fields-2026-08"
MAX_SMOKE_CALLS = 100
ALLOWED_RESULTS = {"EXACT_HIGH_CONFIDENCE", "SILVER_AMBIGUOUS", "REJECTED"}


def _safe_status(status: str, detail: str = "") -> dict[str, str]:
    """Return a report that cannot contain the secret or subprocess output."""
    return {"status": status, "detail": detail}


def read_secret(
    environ: Mapping[str, str] | None = None,
    keychain_runner: Callable[..., Any] | None = None,
) -> tuple[str, str]:
    """Resolve the secret in memory only, with env priority then Keychain."""
    source_env = os.environ if environ is None else environ
    environment_value = source_env.get(SECRET_ENV, "")
    if environment_value:
        return environment_value, "ENVIRONMENT"

    runner = subprocess.run if keychain_runner is None else keychain_runner
    try:
        result = runner(
            [
                "security",
                "find-generic-password",
                "-a",
                getpass.getuser(),
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=5,
        )
    except Exception:
        return "", "MISSING"
    if getattr(result, "returncode", 1) != 0:
        return "", "MISSING"
    # stdout is consumed only into volatile memory and never returned in a
    # report.  Do not inspect, log, hash, or persist it elsewhere.
    value = str(getattr(result, "stdout", "") or "").strip()
    return (value, "KEYCHAIN") if value else ("", "MISSING")


def maps_preflight(
    plan: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
    keychain_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    config = plan["maps_assisted"]
    if not bool(config.get("enabled", False)):
        return {"status": "DISABLED", "calls_allowed": 0, "secret_source": "NONE"}
    max_requests = int(config.get("max_requests", 0))
    daily_quota = int(config.get("daily_quota", 0))
    max_cost = float(config.get("max_cost_eur", 0.0))
    per_request = config.get("cost_per_request_eur")
    if max_requests <= 0 or daily_quota <= 0 or max_cost <= 0 or per_request is None:
        return {"status": "BUDGET_NOT_CONFIGURED", "calls_allowed": 0, "secret_source": "NONE"}
    if max_requests > daily_quota:
        return {"status": "QUOTA_INVALID", "calls_allowed": 0, "secret_source": "NONE"}
    secret, source = read_secret(environ=environ, keychain_runner=keychain_runner)
    if not secret:
        return {"status": "NOT_CONFIGURED", "calls_allowed": 0, "secret_source": "NONE"}
    # The secret is deliberately not included in this object, its length is
    # not exposed, and no digest is computed.
    del secret
    return {
        "status": "READY",
        "calls_allowed": min(max_requests, daily_quota, MAX_SMOKE_CALLS),
        "secret_source": source,
        "field_mask": str(config.get("field_mask", FIELD_MASK)),
        "query_version": str(config.get("query_version", QUERY_VERSION)),
    }


def _safe_place(place: Mapping[str, Any]) -> dict[str, Any]:
    display_name = place.get("displayName") or {}
    return {
        "place_id": str(place.get("id", "")),
        "display_name": str(display_name.get("text", "")) if isinstance(display_name, Mapping) else "",
        "formatted_address": str(place.get("formattedAddress", "")),
        "address_components": place.get("addressComponents", []) if isinstance(place.get("addressComponents", []), list) else [],
        "provenance": "GOOGLE_PLACES_TEXT_SEARCH_NEW",
        "query_version": QUERY_VERSION,
    }


def _component(place: Mapping[str, Any], component_type: str) -> str:
    for item in place.get("address_components", []):
        if component_type in item.get("types", []):
            return str(item.get("longText") or item.get("shortText") or "")
    return ""


def _place_score(place: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[float, dict[str, bool]]:
    cp = _component(place, "postal_code")
    city = _component(place, "locality") or _component(place, "postal_town")
    route = _component(place, "route")
    number = _component(place, "street_number")
    target_name = str(target.get("name", ""))
    target_address = str(target.get("address", ""))
    target_route = str(target.get("route", ""))
    target_number = str(target.get("number", ""))
    target_cp = str(target.get("postcode", ""))
    target_city = str(target.get("city", ""))
    name_match = max_name_similarity(str(place.get("display_name", "")), [target_name] + list(target.get("names", []))) >= 0.82
    cp_match = bool(target_cp and cp and normalize_text(target_cp) == normalize_text(cp))
    city_match = bool(target_city and city and normalize_text(target_city) == normalize_text(city))
    number_match = bool(target_number and number and normalize_text(target_number) == normalize_text(number))
    route_match = bool((target_route and route and (token_jaccard(target_route, route) >= 0.8 or text_ratio(target_route, route) >= 0.85)) or (not target_route and token_jaccard(target_address, place.get("formatted_address", "")) >= 0.5))
    score = (2.0 if cp_match else 0.0) + (2.0 if city_match else 0.0) + (2.0 if number_match else 0.0) + (2.0 if route_match else 0.0) + (2.0 if name_match else 0.0)
    return score, {"cp": cp_match, "city": city_match, "number": number_match, "route": route_match, "name": name_match}


def classify_place(place: Mapping[str, Any], target: Mapping[str, Any], sibling_targets: list[Mapping[str, Any]]) -> tuple[str, float, dict[str, bool], str]:
    score, guards = _place_score(place, target)
    if not all(guards.values()):
        return "REJECTED", score, guards, "STRONG_GUARD_MISSING"
    sibling_scores = []
    for sibling in sibling_targets:
        sibling_score, _ = _place_score(place, sibling)
        sibling_scores.append((sibling_score, str(sibling.get("siret", ""))))
    better_sibling = max(sibling_scores, default=(-1.0, ""))
    if better_sibling[0] >= score and better_sibling[1] != str(target.get("siret", "")):
        return "REJECTED", score, guards, "BETTER_SIBLING"
    return "EXACT_HIGH_CONFIDENCE", score, guards, "UNIQUE_STRONG_MATCH"


def _request_places(secret: str, text_query: str, *, timeout: float = 10.0) -> list[dict[str, Any]]:
    payload = json.dumps({"textQuery": text_query, "pageSize": 5}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": secret,
            "X-Goog-FieldMask": FIELD_MASK,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(2_000_000)
        parsed = json.loads(raw.decode("utf-8"))
        return [_safe_place(place) for place in parsed.get("places", []) if isinstance(place, Mapping)]
    except Exception as exc:
        # Do not expose the exception: urllib errors may echo request headers.
        raise RuntimeError("MAPS_HTTP_ERROR") from None


def _cache_key(query: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json({"query": query, "version": QUERY_VERSION, "field_mask": FIELD_MASK}).encode("utf-8"))


def run_smoke(
    plan: Mapping[str, Any],
    seeds: list[Mapping[str, Any]],
    *,
    execute: bool = False,
    environ: Mapping[str, str] | None = None,
    keychain_runner: Callable[..., Any] | None = None,
    request_fn: Callable[[str, str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    preflight = maps_preflight(plan, environ=environ, keychain_runner=keychain_runner)
    config = plan["maps_assisted"]
    requested = min(len(seeds), int(config.get("smoke_seed_count", 100)))
    report: dict[str, Any] = {
        "status": preflight["status"],
        "preflight": {key: value for key, value in preflight.items() if key != "secret_source"},
        "requested_seed_count": requested,
        "calls_attempted": 0,
        "cache_hits": 0,
        "response_count": 0,
        "classification_counts": {"EXACT_HIGH_CONFIDENCE": 0, "SILVER_AMBIGUOUS": 0, "REJECTED": 0},
        "error_counts": {},
        "cost_estimate_eur": 0.0,
        "secret_source": preflight.get("secret_source", "NONE"),
    }
    if not execute or preflight["status"] != "READY":
        report["status"] = "NOT_EXECUTED" if preflight["status"] == "READY" and not execute else preflight["status"]
        report.pop("secret_source", None)
        return report
    if requested > MAX_SMOKE_CALLS:
        report["status"] = "STOP_MAPS_QUOTA"
        report.pop("secret_source", None)
        return report
    secret, _ = read_secret(environ=environ, keychain_runner=keychain_runner)
    cache_root = Path(config.get("cache_root", "/Volumes/CATNAT_DATA/SIRETO_RECALL100/cache/synthetic_gt_corpus/maps_assisted"))
    cache_root.mkdir(parents=True, exist_ok=True)
    request = request_fn or _request_places
    per_request_cost = float(config["cost_per_request_eur"])
    for seed in seeds[:requested]:
        query = {"name": str(seed.get("name", "")), "enseigne": str(seed.get("enseigne", "")), "address": str(seed.get("address", "")), "postcode": str(seed.get("postcode", "")), "city": str(seed.get("city", ""))}
        key = _cache_key(query)
        cache_path = cache_root / f"{key}.json"
        places: list[dict[str, Any]]
        if cache_path.is_file():
            try:
                places = json.loads(cache_path.read_text(encoding="utf-8"))
                report["cache_hits"] += 1
            except Exception:
                places = []
        else:
            if report["calls_attempted"] >= min(int(config["max_requests"]), int(config["daily_quota"]), MAX_SMOKE_CALLS):
                report["status"] = "STOP_MAPS_QUOTA"
                break
            if (report["calls_attempted"] + 1) * per_request_cost > float(config["max_cost_eur"]):
                report["status"] = "STOP_MAPS_BUDGET"
                break
            report["calls_attempted"] += 1
            try:
                places = request(secret, " ".join(value for value in [query["name"], query["address"], query["postcode"], query["city"]] if value))
            except RuntimeError as exc:
                reason = str(exc) if str(exc) == "MAPS_HTTP_ERROR" else "MAPS_REQUEST_ERROR"
                report["error_counts"][reason] = report["error_counts"].get(reason, 0) + 1
                continue
            cache_path.write_text(json.dumps(places, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        report["response_count"] += len(places)
        classifications = []
        for place in places:
            classification, score, guards, reason = classify_place(place, seed, list(seed.get("siblings", [])))
            classifications.append({"place_id": place.get("place_id", ""), "classification": classification, "score": score, "guards": guards, "reason": reason})
        if not classifications:
            report["classification_counts"]["REJECTED"] += 1
        elif len([item for item in classifications if item["classification"] == "EXACT_HIGH_CONFIDENCE"]) == 1:
            report["classification_counts"]["EXACT_HIGH_CONFIDENCE"] += 1
        elif any(item["classification"] == "SILVER_AMBIGUOUS" for item in classifications):
            report["classification_counts"]["SILVER_AMBIGUOUS"] += 1
        else:
            report["classification_counts"]["REJECTED"] += 1
    del secret
    report["cost_estimate_eur"] = round(report["calls_attempted"] * per_request_cost, 8)
    if report["status"] == preflight["status"]:
        report["status"] = "GO_MAPS_SMOKE"
    report.pop("secret_source", None)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("config/synthetic_gt_corpus_plan.json"))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    from build_synthetic_gt_corpus import load_plan

    arguments = parse_args()
    plan = load_plan(arguments.plan.resolve())
    result = run_smoke(plan, [], execute=arguments.execute)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
