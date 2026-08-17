import json
from argparse import Namespace

import pandas as pd

from scripts.build_crm_gt_v2_population import sha256
from scripts.certify_crm_gt_v2_commercial_provenance import build


def test_commercial_provenance_certifies_without_llm_adjudication(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"query_id": ["q1"], "data_origin": ["REAL_CRM_20260817"]}).to_parquet(
        source / "queries.parquet", index=False
    )
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "data_origin": ["REAL_CRM_20260817"],
            "exact_metric_eligible": [True],
            "split_role": ["PROSPECTIVE_DEV"],
            "label_source": ["REAL_CRM_20260817"],
            "validator": ["SIRENE_INSEE_CP_STRICT_V1"],
            "reliability": ["HIGH_AUTOMATED"],
            "label_is_human_validated": [False],
            "label_audit_status": ["PENDING_INDEPENDENT_REVIEW"],
        }
    ).to_parquet(source / "labels.parquet", index=False)
    pd.DataFrame({"query_id": ["q1"], "oof_fold": [0]}).to_parquet(
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
    assert labels.loc[0, "label_is_human_validated"]
    assert labels.loc[0, "label_audit_status"].startswith("PROVENANCE_CONFIRMED")
