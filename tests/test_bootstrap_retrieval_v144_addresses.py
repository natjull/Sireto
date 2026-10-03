import json

import pytest

import scripts.bootstrap_retrieval_v144_addresses as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_resume_retrieval_v143_addresses import setup as base_setup,finish,fake_gate


def test_minimal_reference_and_full_reference_same_identity_without_weakening_hash(tmp_path):
    p=tmp_path/'source.json';write_json(p,dict(a=1));full=record(p);minimal={k:full[k] for k in ('path','sha256')}
    assert workbench.canonical(minimal)==full and workbench.canonical(full)==full
    with pytest.raises(ValueError,match='byte size'):workbench.canonical(dict(full,size_bytes=full['size_bytes']+1))
    with pytest.raises(ValueError,match='Authority drift'):workbench.canonical(dict(minimal,sha256='0'*64))


def test_bootstrap_minimal_config_imports_and_seals_with_unchanged_backend(tmp_path,monkeypatch):
    prior,root=base_setup(tmp_path,monkeypatch);l,_=workbench.backend.load(prior);cfg=dict(l['resumption'])
    for k in ('original_launch','original_stop'):cfg[k]={p:v for p,v in cfg[k].items() if p!='size_bytes'}
    c=tmp_path/'bootstrap_config.json';write_json(c,cfg);monkeypatch.setattr(workbench,'CONFIG',c);monkeypatch.setattr(workbench,'quiet',fake_gate)
    out=tmp_path/'canonical_bootstrap';workbench.init(out);new,p=workbench.backend.load(out)
    assert new['bootstrap_revision']==144 and new['execution_backend']=='UNCHANGED_V143'
    assert 'size_bytes' in new['resumption']['original_launch'] and p['imported_part_count']==2
    assert all(new[k] is False for k in workbench.PROTECTED)
    finish(out);workbench.backend.seal(out);m=json.loads((out/'manifest.json').read_text())
    assert m['observations']==12 and len(m['parts'])==4 and sum(i['imported'] for i in m['parts'])==2
    assert all(m[k] is False for k in workbench.PROTECTED) and not m['ready_for_fold1'] and not m['complete_retrieval_index']
    assert m['status']=='SEALED_V143_ALL_ORIGINAL_OFFICIAL_AND_RAW_CURRENT_ADDRESS_PROJECTION'
