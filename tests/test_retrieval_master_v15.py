from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.hierarchical_retrieval import InMemoryBackend, IndexedEstablishment, RetrievalQuery
from src.xgb_matcher.retrieval_master_v15 import MasterV15Recipe, compose_query, load_query_selection, rrf_parents, sha256_file, write_generation
from src.xgb_matcher.retrieval_master_v15_evaluation import CONTROL_COUNTS, MIN_RESCUES, evaluate_generation
from src.xgb_matcher.retrieval_sidecars_v11 import LocalSite, SiteQueryResult


def _record(siret: str, name: str = "ALPHA") -> IndexedEstablishment:
    return IndexedEstablishment.from_mapping({"siret": siret, "siren": siret[:9], "insee": "75056", "postcode": "75001", "names": [name], "addresses": ["10 RUE PAIX"], "number": "10"})


class _Sidecars:
    def sites_by_sirens_geo(self, **kwargs):
        return {
            siren: SiteQueryResult(
                (LocalSite(siret=f"{siren}00001", siren=siren, insee="75056", postcode="75001", street_number="10", street_number_suffix="", street_normalized="RUE PAIX", current_address_normalized="10 RUE PAIX", administrative_state="A", is_headquarters=False),),
                "SIREN_GEO_BATCH", 1, 16, False,
            )
            for siren in kwargs["sirens"]
        }


def test_query_surface_rejects_truth_columns(tmp_path: Path) -> None:
    path = tmp_path / "queries.parquet"
    pq.write_table(pa.table({"query_id": ["q"], "fold": pa.array([2], type=pa.int8()), "human_ground_truth_siret": ["12345678900001"]}), path)
    with pytest.raises(ValueError, match="only query_id and fold"):
        load_query_selection(path)


def test_rrf_and_composition_are_deterministic_label_blind_and_capped() -> None:
    backend_a = InMemoryBackend([_record("11111111100001"), _record("22222222200001")])
    backend_b = InMemoryBackend([_record("22222222200001"), _record("33333333300001")])
    query = RetrievalQuery("ALPHA", "10 RUE PAIX", "10", "75001", "75056")
    parents = rrf_parents((("a", backend_a), ("b", backend_b)), query, MasterV15Recipe())
    assert parents[0]["siren"] == "222222222"
    source = [
        {"query_id": "q", "fold": 2, "union_rank": rank, "siret": f"900000000{rank:05d}", "crm_name": "ALPHA", "crm_address": "10 RUE PAIX", "crm_number": "10", "crm_postcode": "75001", "crm_insee": "75056"}
        for rank in range(1, 1751)
    ]
    clean, clean_parents = compose_query(query_id="q", fold=2, source_rows=source, backends=(("a", backend_a), ("b", backend_b)), sidecars=_Sidecars(), recipe=MasterV15Recipe())
    labelled = [{**row, "human_ground_truth_siret": "22222222200001"} for row in source]
    with_truth, truth_parents = compose_query(query_id="q", fold=2, source_rows=labelled, backends=(("a", backend_a), ("b", backend_b)), sidecars=_Sidecars(), recipe=MasterV15Recipe())
    assert clean == with_truth
    assert clean_parents == truth_parents
    assert len(clean) <= 2000
    assert [row["candidate_rank"] for row in clean] == list(range(1, len(clean) + 1))
    assert not any("human_ground_truth_siret" in row for row in clean + clean_parents)


def _write_cohort(root: Path) -> tuple[list[dict], Path]:
    root.mkdir()
    labels: list[dict] = []
    for fold, misses in {2: 182, 3: 175, 4: 165}.items():
        for index in range(misses):
            labels.append({"query_id": f"m{fold}-{index}", "fold": fold, "human_ground_truth_siret": f"{fold}{index:08d}00001", "cohort_kind": "BASE_TOP1750_MISS"})
    for fold, controls in CONTROL_COUNTS.items():
        for index in range(controls):
            labels.append({"query_id": f"c{fold}-{index}", "fold": fold, "human_ground_truth_siret": f"8{fold}{index:07d}00001", "cohort_kind": "BASE_TOP1750_CONTROL"})
    labels_path = root / "cohort_labels.parquet"
    queries_path = root / "queries.parquet"
    pq.write_table(pa.Table.from_pylist(labels), labels_path)
    pq.write_table(pa.table({"query_id": [row["query_id"] for row in labels], "fold": pa.array([row["fold"] for row in labels], type=pa.int8())}), queries_path)
    manifest = {"files": {name: {"sha256": sha256_file(root / name), "size_bytes": (root / name).stat().st_size, "rows": len(labels)} for name in ("queries.parquet", "cohort_labels.parquet")}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return labels, root


def _write_passing_generation(root: Path, labels: list[dict]) -> Path:
    candidates, parents = [], []
    for label in labels:
        query_id, fold, truth = label["query_id"], label["fold"], label["human_ground_truth_siret"]
        candidates.append({"query_id": query_id, "fold": fold, "candidate_rank": 1, "siret": truth if label["cohort_kind"].endswith("CONTROL") else f"7{fold}999999900001", "source": "BASE_V11_TOP1750", "base_rank": 1, "parent_rank": None, "site_rank": None})
        if label["cohort_kind"].endswith("MISS"):
            index = int(query_id.split("-")[1])
            parents.append({"query_id": query_id, "fold": fold, "parent_rank": 1, "siren": truth[:9], "rrf_score": 1.0, "backend_support": 2})
            if index < MIN_RESCUES[fold]:
                candidates.append({"query_id": query_id, "fold": fold, "candidate_rank": 2, "siret": truth, "source": "OFFICIAL_PARENT_RRF_SITE_RESCUE", "base_rank": None, "parent_rank": 1, "site_rank": 1})
    write_generation(root, candidates, parents, {"query_count": len(labels), "truth_columns_opened": False})
    return root


def test_evaluator_passes_only_with_controls_fold_thresholds_and_sensitivity(tmp_path: Path) -> None:
    labels, cohort = _write_cohort(tmp_path / "cohort")
    generation = _write_passing_generation(tmp_path / "generation", labels)
    manifest = evaluate_generation(generation_dir=generation, cohort_dir=cohort, output_dir=tmp_path / "evaluation", config_sha256="a" * 64)
    raw = json.loads(manifest.read_text())
    assert raw["decision"] == "PASS_MASTER_V15_FEASIBILITY"
    assert raw["checks"] == {"controls_preserved": True, "fold_rescue_thresholds": True, "sensitivity": True}
    assert raw["variants"]["PRIMARY"]["global_projected_hits"] == 23490


def test_evaluator_rejects_tampered_generation(tmp_path: Path) -> None:
    labels, cohort = _write_cohort(tmp_path / "cohort")
    generation = _write_passing_generation(tmp_path / "generation", labels)
    with (generation / "candidates.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact hash drifted"):
        evaluate_generation(generation_dir=generation, cohort_dir=cohort, output_dir=tmp_path / "evaluation", config_sha256="a" * 64)
