import json
import pytest
from scripts import retrieval_v34_resume as v34
from src.xgb_matcher.retrieval_v20_workbench import record


def test_geo_surface_never_uses_truth_or_identity_as_feature():
    row=dict(query_id='q',oof_fold=2,crm_name='ATELIER',crm_address='12 RUE TEST',postcode='75001',insee='75101')
    a=v34.geo_surface([row]).to_pylist()[0]
    b=v34.geo_surface([dict(row,query_id='different',oof_fold=4,ground_truth_siret='12345678900001',gt_siren='123456789')]).to_pylist()[0]
    for key in ('query_id','fold'):a.pop(key);b.pop(key)
    assert a==b


def test_accelerator_code_includes_implementation_and_contract():
    assert set(v34.accelerator_code())=={'scripts/retrieval_v34_runtime.py',
      'scripts/retrieval_v34_resume.py','docs/retrieval_v34_physical_resume_contract.md'}


def test_gate_rejects_accelerator_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(v34.v33,'authority',lambda _:None)
    monkeypatch.setattr(v34.v33,'code',lambda:{'frozen':'same'})
    monkeypatch.setattr(v34,'accelerator_code',lambda:{'runtime':'new'})
    prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'manifest.json').write_text('{}')
    gate=tmp_path/'gate.json'
    gate.write_text(json.dumps(dict(status='PASS_PHYSICAL_RESUME',prepared=record(prepared/'manifest.json'),
      code={'frozen':'same'},accelerator_code={'runtime':'old'})))
    with pytest.raises(ValueError,match='Resume gate drift'):v34.gate_authority(prepared,record(gate))


def test_parity_covers_scores_ranks_pool_membership_and_all_channels():
    assert {'v23_stable/master.parquet','v30_stable/master.parquet','v23_stable/pool.parquet',
      'v30_stable/pool.parquet','ban.parquet','raw.parquet','proof.parquet',
      'structured.parquet','direct.parquet','joint.parquet'}<=set(v34.COMMON)
