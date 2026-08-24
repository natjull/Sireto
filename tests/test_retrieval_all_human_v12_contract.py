import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/retrieval_all_human_v12.json"
CONTRACT_PATH = ROOT / "docs/retrieval_all_human_v12_contract.md"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v12_is_top100_first_with_bounded_adaptive_caps():
    config = _config()
    architecture = config["architecture"]
    ceilings = config["gates"]["candidate_ceilings"]

    assert config["schema_version"] == "sireto-retrieval-all-human-v12-config-1"
    assert architecture["normal_shortlist_siret_cap"] == 500
    assert architecture["rescue_lightweight_siret_reference_cap"] == 2000
    assert architecture["maximum_hydrated_sirets"] == 500
    assert architecture["maximum_final_sirets"] == 100
    assert architecture["hard_sites_per_siren_cap"] is None
    assert ceilings == {
        "master_references": 2000,
        "prerank_and_hydrated_sirets": 500,
        "published_sirets": 100,
        "decision_blocking": True,
    }
    assert config["retrieval"]["conditional_rescue"][
        "maximum_executions_per_query"
    ] == 1
    assert config["retrieval"]["conditional_rescue"]["automatic_retry"] is False


def test_v12_gates_make_top100_primary_and_master_diagnostic():
    gates = _config()["gates"]
    master = gates["master_exact_recall_at_2000"]
    prerank = gates["prerank_exact_recall_at_500"]
    product = gates["product_exact_recall_at_100"]
    latency = gates["warm_end_to_end_latency_ms"]

    assert master["minimum"] == 0.995
    assert master["decision_blocking"] is False
    assert master["role"] == "DIAGNOSTIC_CEILING_AND_HEADROOM_TARGET"
    assert prerank["minimum"] == 0.994
    assert prerank["failure_verdict"] == "PIVOT_GENERATION"
    assert product["minimum"] == 0.992
    assert product["wilson95_lower_minimum"] == 0.99
    assert product["minimum_each_outer_fold"] == 0.988
    assert product["role"] == "PRIMARY_PRODUCT_GATE"
    assert latency["p95_maximum"] == 1000
    assert latency["p99_maximum"] == 2000


def test_v12_preregisters_oof_locks_and_forbids_leakage_or_grid():
    config = _config()
    development = config["development"]
    text = CONTRACT_PATH.read_text(encoding="utf-8")

    assert config["folds"]["outer_oof"] == [2, 3, 4]
    assert config["folds"]["development_contaminated"] == 0
    assert config["folds"]["test_locked"] == 1
    assert development["outer_validation_fold_excluded_from_every_fit"] is True
    assert development["inner_oof_required_for_learned_intermediate_scores"] is True
    assert development["fold0_reads"] == 0
    assert development["fold1_reads_before_explicit_authorization"] == 0
    assert development["single_recipe"] is True
    assert development["variant_grid"] is False
    assert development["automatic_threshold_tuning"] is False
    assert config["retrieval"]["positive_injection"] is False
    assert config["retrieval"]["crm_alias_lookup"] is False
    assert config["retrieval"]["synthetic_labels"] is False
    assert "préenregistré, non exécuté" in text
    assert "fold 0 : contaminé" in text
    assert "fold 1 : fermé" in text
    assert "ni grille" in text
