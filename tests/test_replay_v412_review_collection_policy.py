from __future__ import annotations

import json
import ipaddress
from pathlib import Path

import pytest

from scripts import replay_v412_review_collection_policy as subject


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_all_frozen_collection_policy_goldens_replay_offline() -> None:
    counts = subject.validate_policy_goldens(REPO_ROOT)
    assert counts == {
        "adjudication_cases": 6,
        "ddg_charsets": 3,
        "ddg_results": 5,
        "dns_addresses": 23,
        "dns_resolutions": 6,
        "domains": 8,
        "evidence_cases": 7,
        "fact_cases": 4,
        "identifiers": 5,
        "pins": 14,
        "postopen": 8,
    }


@pytest.mark.parametrize(
    "value",
    [
        "127.0.0.1",
        "10.0.0.1",
        "64:ff9b::a00:1",
        "2002:7f00:1::1",
        "2001::1",
        "::ffff:8.8.8.8",
    ],
)
def test_forbidden_network_addresses_are_rejected(value: str) -> None:
    vectors = __import__("json").loads(
        (REPO_ROOT / "config/v4_12_review_dns_security_vectors.json").read_text()
    )
    networks = [ipaddress.ip_network(item, strict=True) for item in vectors["forbidden_cidrs"]]
    assert subject.forbidden_address(value, networks) is True


def test_mixed_dns_answer_refuses_every_connection() -> None:
    vectors = __import__("json").loads(
        (REPO_ROOT / "config/v4_12_review_dns_security_vectors.json").read_text()
    )
    networks = [ipaddress.ip_network(item, strict=True) for item in vectors["forbidden_cidrs"]]
    assert subject.evaluate_resolution(["8.8.8.8", "10.0.0.1"], networks) == {
        "expected_addresses": ["8.8.8.8", "10.0.0.1"],
        "permitted": False,
        "chosen_ip": None,
        "error_type": "PRIVATE_ADDRESS",
    }


def test_normalization_is_accent_stable() -> None:
    assert subject.normalize_value("Médiathèque — République") == "mediatheque republique"


def test_evidence_reference_id_is_domain_separated() -> None:
    observed = subject.evidence_ref_id(
        "q_exact",
        "55210055400013",
        "PUBLIC_ADMINISTRATION",
        "a" * 64,
    )
    assert observed == "e3ceaf203f772cc5735e4d5017563a16606349befae6cc5c2caf738349200b7b"


def test_adjudication_never_promotes_outside_top100_to_ranker_positive() -> None:
    observed = subject.adjudicate_vector(
        {
            "top1_siret": "78983652500020",
            "top100_sirets": ["78983652500020"],
            "supporting_groups_by_siret": {
                "55210055400013": ["ENTITY_OFFICIAL_SITE", "SIRENE_REGISTRY"]
            },
        }
    )
    assert observed["status"] == "TOP1_WRONG"
    assert observed["alternative_siret"] == "55210055400013"
    assert observed["alternative_in_top100"] is False


def _policy() -> dict:
    return json.loads((REPO_ROOT / "config/v4_12_review_collection_policy.json").read_text())


def test_policy_is_authenticated_before_following_any_path(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    payload = (REPO_ROOT / "config/v4_12_review_collection_policy.json").read_text()
    (config / "v4_12_review_collection_policy.json").write_text(
        payload.replace('"retry_count": 0', '"retry_count": 1', 1)
    )
    with pytest.raises(ValueError, match="trust-anchor"):
        subject.validate_policy_goldens(tmp_path)


def test_relative_policy_paths_are_closed_even_after_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    policy = _policy()
    policy["domain_policy"]["domain_vectors_path"] = "../secret.json"
    raw = json.dumps(policy).encode()
    (config / "v4_12_review_collection_policy.json").write_bytes(raw)
    monkeypatch.setattr(subject, "POLICY_SHA256", subject.sha256_bytes(raw))
    with pytest.raises(ValueError, match="not allowlisted"):
        subject.validate_policy_goldens(tmp_path)


def test_absolute_policy_paths_are_closed_before_any_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    policy = _policy()
    policy["network_security"]["hosts_path"] = "/tmp/untrusted-hosts"
    raw = json.dumps(policy).encode()
    (config / "v4_12_review_collection_policy.json").write_bytes(raw)
    monkeypatch.setattr(subject, "POLICY_SHA256", subject.sha256_bytes(raw))
    with pytest.raises(ValueError, match="absolute policy path is not allowlisted"):
        subject.validate_policy_goldens(tmp_path)


def test_runtime_versions_are_part_of_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    real_version = subject.importlib.metadata.version
    monkeypatch.setattr(
        subject.importlib.metadata,
        "version",
        lambda package: "0.0" if package == "idna" else real_version(package),
    )
    with pytest.raises(ValueError, match="idna_version"):
        subject.validate_policy_goldens(REPO_ROOT)


def test_fact_reconstruction_is_independent_from_expected_rows() -> None:
    policy = _policy()
    vectors = json.loads(
        (REPO_ROOT / "config/v4_12_review_fact_reconstruction_vectors.json").read_text()
    )
    for vector in vectors:
        text = "".join(
            segment.get("literal", segment.get("repeat", "") * segment.get("count", 0))
            for segment in vector["text_segments"]
        )
        facts, unqualified = subject.reconstruct_facts(
            text,
            vector["crm_name"],
            vector["crm_address"],
            vector["crm_postcode"],
            stopwords=frozenset(policy["domain_policy"]["name_stopwords"]),
            minimum_name_token_length=policy["domain_policy"]["significant_token_minimum_length"],
        )
        assert facts == vector["expected_facts"]
        assert unqualified == vector["expected_unqualified_sirets"]
        assert len(facts) % 5 == 0


def test_later_qualified_occurrence_is_not_masked_by_earlier_unqualified_one() -> None:
    siret = "55210055400013"
    text = (
        f"SIRET {siret} "
        + ("x " * 700)
        + f"Agence Alpha SIRET {siret} adresse 24 Victor 69002."
    )
    facts, unqualified = subject.reconstruct_facts(
        text,
        "Agence Alpha",
        "24 avenue Victor Hugo",
        "69002",
        stopwords=frozenset(),
        minimum_name_token_length=4,
    )
    assert len(facts) == 5
    assert {row[0] for row in facts} == {siret}
    assert unqualified == []


def test_numbered_address_requires_one_road_token_not_two() -> None:
    vector = {
        "preopen_family": "ENTITY_OFFICIAL_SITE_CANDIDATE",
        "normalized_hostname": "agence-alpha.fr",
        "crm_name": "Agence Alpha",
        "crm_address": "24 avenue Victor Hugo",
        "crm_postcode": "69002",
        "extracted_text": "Agence Alpha SIRET 55210055400013 adresse 24 Victor 69002.",
    }
    assert subject.postopen_family(vector, _policy()) == ("ENTITY_OFFICIAL_SITE", True)


@pytest.mark.parametrize(
    ("family", "hostname"),
    [
        ("PUBLIC_ADMINISTRATION", "evil.example.fr"),
        ("OFFICIAL_SECTOR_DIRECTORY", "evil.example.fr"),
    ],
)
def test_postopen_revalidates_pinned_family_suffix(family: str, hostname: str) -> None:
    vector = {
        "preopen_family": family,
        "normalized_hostname": hostname,
        "crm_name": "Agence Alpha",
        "crm_address": "24 avenue Victor Hugo",
        "crm_postcode": "69002",
        "extracted_text": "Agence Alpha SIRET 55210055400013 adresse 24 avenue Victor Hugo 69002.",
    }
    assert subject.postopen_family(vector, _policy()) == ("INADMISSIBLE_AFTER_OPEN", False)


def test_ddg_wrapper_requires_strict_single_utf8_decode() -> None:
    assert subject._resolve_ddg_href(
        "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.fr%2F%FF"
    ) is None
    assert subject._resolve_ddg_href(
        "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.fr%2F%GZ"
    ) is None


def test_ddg_wrapper_requires_one_uddg_and_ignores_tracking_parameters() -> None:
    direct = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.fr%2Fcontact"
    tracked = direct + "&rut=ignored&source=html"
    assert subject._resolve_ddg_href(direct) == "https://example.fr/contact"
    assert subject._resolve_ddg_href(tracked) == "https://example.fr/contact"
    assert subject._resolve_ddg_href(
        direct + "&uddg=https%3A%2F%2Fevil.example%2F"
    ) is None


def test_unqualified_web_relation_neither_contradicts_nor_supports_sirene() -> None:
    case = {
        "query_id": "q",
        "web_proofs": [
            ["a" * 64, "PUBLIC_ADMINISTRATION", "1" * 64, "55210055400013", "agence alpha", "24 avenue victor hugo 69002", False],
            ["b" * 64, "PUBLIC_ADMINISTRATION", "2" * 64, "73282932000074", "agence beta", "26 avenue pasteur 69002", True],
        ],
        "sirene_records": [
            ["d" * 64, "4" * 64, "55210055400013", True, "A", ["agence alpha"], "24 avenue victor hugo 69002"]
        ],
    }
    rows = subject.replay_evidence_case(case)
    assert rows[0][8:10] == [False, False]
    assert rows[1][8:10] == [False, True]
    assert rows[2][4:10] == [False, False, False, False, False, False]


def test_group_with_only_unqualified_web_relation_is_false_not_an_error() -> None:
    case = {
        "query_id": "q",
        "web_proofs": [
            [
                "a" * 64,
                "PUBLIC_ADMINISTRATION",
                "1" * 64,
                "55210055400013",
                "agence alpha",
                "24 avenue victor hugo 69002",
                False,
            ]
        ],
        "sirene_records": [],
    }
    rows = subject.replay_evidence_case(case)
    assert len(rows) == 1
    assert rows[0][4:10] == [True, True, True, False, False, False]


def test_preopen_classifier_is_closed_and_obeys_precedence() -> None:
    policy = _policy()
    suffixes = (REPO_ROOT / policy["domain_policy"]["public_suffixes_path"]).read_text().splitlines()
    unsafe = [
        "http://agence-alpha.fr/",
        "https://user@agence-alpha.fr/",
        "https://127.0.0.1/",
        "https://agence-alpha.fr:444/",
        "https://agence-alpha.fr/#fragment",
        "https://localhost/",
        "https://service.local/",
    ]
    for url in unsafe:
        assert subject.classify_preopen(url, "Agence Alpha", "", "Agence Alpha", policy, suffixes)["reason"] == "UNSAFE_URL"
    assert subject.classify_preopen(
        "https://annuaire-entreprises.data.gouv.fr/fiche.pdf",
        "Agence Alpha",
        "Agence Alpha",
        "Agence Alpha",
        policy,
        suffixes,
    )["reason"] == "SIRENE_COPY"
    assert subject.classify_preopen(
        "https://agence-alpha.fr/document.pdf",
        "Agence Alpha",
        "",
        "Agence Alpha",
        policy,
        suffixes,
    )["family"] == "DATED_PUBLIC_DOCUMENT_CANDIDATE"
