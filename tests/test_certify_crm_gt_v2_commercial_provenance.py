import json
from argparse import Namespace

import pandas as pd

from scripts.build_crm_gt_v2_population import sha256
from scripts.certify_crm_gt_v2_commercial_provenance import build


def test_commercial_provenance_certifies_without_llm_adjudication(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"query_id": ["q1", "q0"], "data_origin": ["REAL_CRM_20260817", "REAL_CRM_HISTORICAL"]}).to_parquet(
        source / "queries.parquet", index=False
    )
    pd.DataFrame(
        {
            "query_id": ["q1", "q0"],
            "data_origin": ["REAL_CRM_20260817", "REAL_CRM_HISTORICAL"],
            "exact_metric_eligible": [True, True],
            "split_role": ["PROSPECTIVE_DEV", "LEGACY_OOF"],
            "label_source": ["REAL_CRM_20260817", "OLD"],
            "validator": ["SIRENE_INSEE_CP_STRICT_V1", "OLD"],
            "reliability": ["HIGH_AUTOMATED", "OLD"],
            "label_is_human_validated": [False, False],
            "label_audit_status": ["PENDING_INDEPENDENT_REVIEW", "HISTORICAL_QUALIFIED"],
        }
    ).to_parquet(source / "labels.parquet", index=False)
    pd.DataFrame({"query_id": ["q1", "q0"], "oof_fold": [0, 2]}).to_parquet(
        source / "fold_assignments.parquet", index=False
    )
    pd.DataFrame({"query_id": ["q1"]}).to_csv(source / "crm_ok_gt_v2.csv", index=False)
    pd.DataFrame({"query_id": ["q1"]}).to_csv(
        source / "independent_audit_sample_400.csv", index=False
    )
    names = [
        "queries.parquet", "labels.parquet", "fold_assignments.parquet",
        "crm_ok_gt_v2.csv", "independent_audit_sample_400.csv",
    ]
    (source / "manifest.json").write_text(
        json.dumps({"outputs": {name: sha256(source / name) for name in names}})
    )

    destination = build(source, tmp_path / "out")
    manifest = json.loads((destination / "manifest.json").read_text())
    labels = pd.read_parquet(destination / "labels.parquet")
    assert manifest["audit_gate"]["status"] == "PASS"
    assert not manifest["qualification"]["llm_decisions_used_for_admission"]
    assert labels["label_is_human_validated"].all()
    assert labels["label_audit_status"].str.startswith("PROVENANCE_CONFIRMED").all()
    assert labels["human_label_provenance"].notna().all()
