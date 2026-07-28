from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import build_v46_aligned_dataset as subject
from scripts.build_v41_training_dataset import CANDIDATE_COLUMNS, CandidateWriter
from src.xgb_matcher.v41_retrieval import InputSiretState
from src.xgb_matcher.v41_split import assign_connected_siren_splits


def _queries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q1",
                "crm_record_id": "crm1",
                "crm_name": "Alpha",
                "crm_address": "1 rue Alpha",
                "crm_postcode": "75001",
                "crm_city": "Paris",
                "crm_insee": "75056",
                "crm_name_norm": "alpha",
                "crm_address_norm": "1 rue alpha",
                "crm_city_norm": "paris",
                "input_siret": "11111111100011",
                "input_siren": "111111111",
                "input_siret_state": "ACTIVE",
                "source_segment": "fit_addition",
            },
            {
                "query_id": "q2",
                "crm_record_id": "crm2",
                "crm_name": "Beta",
                "crm_address": "2 rue Beta",
                "crm_postcode": "69001",
                "crm_city": "Lyon",
                "crm_insee": "69123",
                "crm_name_norm": "beta",
                "crm_address_norm": "2 rue beta",
                "crm_city_norm": "lyon",
                "input_siret": "22222222200022",
                "input_siren": "222222222",
                "input_siret_state": "ACTIVE",
                "source_segment": "dev_new",
            },
        ]
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q1",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100011",
                "ground_truth_siren": "111111111",
            },
            {
                "query_id": "q2",
                "label_kind": "AMBIGUOUS",
                "ground_truth_siret": None,
                "ground_truth_siren": None,
            },
        ]
    )


def _assignments() -> pd.DataFrame:
    source = _queries()[
        ["query_id", "input_siret", "input_siren"]
    ].merge(_labels(), on="query_id").rename(
        columns={
            "ground_truth_siret": "target_siret",
            "ground_truth_siren": "target_siren",
        }
    )
    return assign_connected_siren_splits(
        source,
        dev_fraction=0.2,
        oof_folds=5,
        seed=42,
    )[["query_id", "siren_component_id", "split", "oof_fold"]]


def _candidate_row(query_id="q1", siret="11111111100011"):
    row = {
        "query_id": query_id,
        "candidate_siret": siret,
        "candidate_siren": siret[:9],
        "candidate_state": "A",
        "is_ground_truth": 0,
        "retrieval_rank": 1,
        "retrieval_source": "v4.2-b",
        "retrieval_channel_count": 1,
        "retrieval_agreement": 0,
    }
    row.update({feature: 0.0 for feature in subject.FEATURE_ORDER})
    return row


def test_authorization_checks_role_fields_but_not_business_text():
    allowed = _queries()
    allowed.loc[0, "crm_name"] = "Random Holdout Café"
    subject.assert_authorized_canonical_table(allowed, name="queries")

    forbidden = allowed.copy()
    forbidden.loc[0, "source_segment"] = "random_population"
    with pytest.raises(ValueError, match="forbidden roles"):
        subject.assert_authorized_canonical_table(forbidden, name="queries")


def test_frozen_split_audit_rebuilds_identical_components():
    subject.validate_frozen_assignments(
        queries=_queries(),
        labels=_labels(),
        assignments=_assignments(),
        enforce_contract_counts=False,
    )

    broken = _assignments()
    broken.loc[0, "oof_fold"] = (int(broken.loc[0, "oof_fold"]) + 1) % 5
    with pytest.raises(AssertionError):
        subject.validate_frozen_assignments(
            queries=_queries(),
            labels=_labels(),
            assignments=broken,
            enforce_contract_counts=False,
        )


@dataclass
class _SparseResult:
    gt_was_injected: bool = False
    idf_map: dict[str, float] | None = None
    default_idf: float = 1.0

    def __post_init__(self):
        self.idf_map = self.idf_map or {}


class _Retriever:
    def __init__(self, *, injected=False, state="A"):
        self.kwargs = None
        self.injected = injected
        self.state = state

    def build(self, **kwargs):
        self.kwargs = kwargs
        candidate = {
            "siret": "11111111100011",
            "siren": "111111111",
            "etat_admin": self.state,
            "nom_etablissement": "ALPHA",
            "denomination_unite_legale": "ALPHA",
            "adresse": "1 RUE ALPHA",
            "postcode": "75001",
            "city": "PARIS",
            "insee": "75056",
            "rrf_score": 0.5,
            "retrieval_channel_count": 1,
            "v41_channel_ranks": {"sparse_active": 1},
        }
        return SimpleNamespace(
            sparse_result=_SparseResult(gt_was_injected=self.injected),
            candidates=[candidate],
            input_siret=SimpleNamespace(
                normalized_siret="11111111100011",
                state=InputSiretState.ACTIVE,
            ),
        )


def test_retrieval_never_receives_truth_and_emits_frozen_64_features():
    retriever = _Retriever()

    rows, diagnostics = subject.retrieve_unlabelled_query(
        query=_queries().iloc[0].to_dict(),
        retriever=retriever,
        persistent_cache=object(),
    )

    assert retriever.kwargs["gt_siret"] is None
    assert diagnostics["candidate_count"] == 1
    assert list(rows[0]) == list(CANDIDATE_COLUMNS)
    assert len(subject.FEATURE_ORDER) == 64
    assert rows[0]["is_ground_truth"] == 0


def test_retrieval_rejects_positive_injection_and_closed_candidate():
    with pytest.raises(ValueError, match="Positive injection"):
        subject.retrieve_unlabelled_query(
            query=_queries().iloc[0].to_dict(),
            retriever=_Retriever(injected=True),
            persistent_cache=object(),
        )
    with pytest.raises(ValueError, match="non-active"):
        subject.retrieve_unlabelled_query(
            query=_queries().iloc[0].to_dict(),
            retriever=_Retriever(state="F"),
            persistent_cache=object(),
        )


def test_truth_is_joined_only_by_post_retrieval_labelling_step(tmp_path):
    unlabelled = tmp_path / "unlabelled.parquet"
    output = tmp_path / "candidates.parquet"
    writer = CandidateWriter(unlabelled)
    writer.write(
        [
            _candidate_row(),
            _candidate_row(query_id="q2", siret="22222222200022"),
        ]
    )
    writer.close()

    count = subject.label_closed_candidate_file(
        unlabelled_path=unlabelled,
        output_path=output,
        labels=_labels(),
    )
    observed = pd.read_parquet(output)

    assert count == 2
    assert observed["is_ground_truth"].tolist() == [1, 0]


def test_candidate_content_hash_covers_all_columns_and_ignores_row_index():
    first = pd.DataFrame([_candidate_row()], columns=CANDIDATE_COLUMNS)
    same = first.copy()
    same.index = [99]
    changed = first.copy()
    changed.loc[0, subject.FEATURE_ORDER[-1]] = 1.0

    assert subject.candidate_content_sha256(first) == (
        subject.candidate_content_sha256(same)
    )
    assert subject.candidate_content_sha256(first) != (
        subject.candidate_content_sha256(changed)
    )


def test_integrity_counts_retrieval_miss_as_recall_failure():
    queries = _queries()
    labels = _labels()
    assignments = _assignments()
    candidates = pd.DataFrame(
        [_candidate_row(siret="99999999900099")],
        columns=CANDIDATE_COLUMNS,
    )

    report = subject.compute_integrity(
        queries=queries,
        labels=labels,
        assignments=assignments,
        candidates=candidates,
    )
    split = assignments.set_index("query_id").loc["q1", "split"]

    assert report["recall_at_100"][split]["successes"] == 0
    assert report["candidate_content_sha256"]
    assert report["max_candidate_count"] == 1


def test_all_writable_paths_must_be_external():
    with pytest.raises(ValueError, match="must be located"):
        subject._external_path(
            Path("/tmp/sireto-v46-forbidden-write"),
            name="work_dir",
        )


def test_physical_partition_count_ignores_manifest_parquet(tmp_path):
    partition = tmp_path / "insee" / "insee=75056"
    partition.mkdir(parents=True)
    pd.DataFrame({"siret": ["11111111100011", "11111111100022"]}).to_parquet(
        partition / "part.parquet",
        index=False,
    )
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    (manifest / "postcode_counts.parquet").write_bytes(b"")

    assert subject._physical_parquet_row_count(tmp_path) == 2
