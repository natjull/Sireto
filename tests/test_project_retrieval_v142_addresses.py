import json

import numpy as np
import pyarrow.parquet as pq
import pytest

import scripts.compile_retrieval_v141_append_catalog as cat_workbench
import scripts.project_retrieval_v142_addresses as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_compile_retrieval_v141_append_catalog import setup as cat_setup
from test_retrieval_address_projection_v142 import canonical,current


def setup(tmp_path,monkeypatch):
    cat=cat_setup(tmp_path,monkeypatch);cat_workbench.worker(cat)
    official=tmp_path/'all_addresses.parquet';pq.write_table(canonical([
        dict(siren='000000001',siret='00000000100001',subject_kind='SIRET',source='RNE',raw_value='RUE A',is_current=True),
        dict(siren='000000009',siret='00000000900003',subject_kind='SIRET',source='BODACC',raw_value='RUE B',is_current=False),
        dict(siren='000000003',subject_kind='SIREN',source='RNE',raw_value='ADRESSE PARENT'),
        dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='RNE',raw_value='[ND]',normalized_value='ND ND')]),official,row_group_size=2)
    raw=tmp_path/'all_current.parquet';pq.write_table(current([dict(siren='000000001',siret='00000000100001',typeVoieEtablissement='RUE',libelleVoieEtablissement='A'),
        dict(siren='000000003',siret='00000000300001')]),raw,row_group_size=1)
    cfg=tmp_path/'address_config.json';refs=dict(official_addresses=dict(record(official),expected_rows=4),current_addresses=dict(record(raw),expected_rows=2))
    write_json(cfg,dict(catalog=record(cat/'manifest.json'),sources=refs,role_order=['official_addresses','current_addresses'],
        expected_namespace_sites=11,expected_total_observations=12,row_groups_per_part=1,batch_rows=1,scalar_first_source_rows=1,
        numpy_version=np.__version__,resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),
        max_part_output_bytes=1048576,max_total_output_bytes=10485760))
    monkeypatch.setattr(workbench,'CONFIG',cfg);out=tmp_path/'addresses';workbench.init(out);return out


def finish(out):
    _,plan=workbench.load(out)
    for role in plan['role_order']:
        for spec in plan['sources'][role]['parts']:
            workbench.directory(out,role,spec['part_id']).mkdir(parents=True);workbench.worker(out,role,spec['part_id'])


def test_all_sources_and_parts_required_before_seal_no_parent_or_mask_drop(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(out)
    finish(out);workbench.seal(out);m=json.loads((out/'manifest.json').read_text())
    assert m['source_rows']==6 and m['observations']==12 and m['lexical_observations']==4
    assert m['subject_counts']==dict(unverified_context=0,parent_context=1,explicit_site=11)
    assert {r['source']:r['observations'] for r in m['source_counts']}==dict(BODACC=1,RNE=3,SIRENE_CURRENT=8)
    assert all(m[k] is False for k in workbench.PROTECTED)
    assert not m['complete_retrieval_index'] and not m['ready_for_fold1'] and not m['all_rows_independent_scalar_oracle']
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(out)
    with pytest.raises(ValueError,match='already started'):workbench.worker(out,'official_addresses',0)
    with pytest.raises(FileExistsError):workbench.init(out)


def test_closed_fold_or_missing_scope_guard_stops_seal(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);finish(out)
    p=workbench.directory(out,'official_addresses',0)/'manifest.json';m=json.loads(p.read_text());m['fold0_opened']=True;write_json(p,m)
    with pytest.raises(ValueError,match='Protected'):workbench.seal(out)
    m['fold0_opened']=False;m['parent_addresses_spread_to_sites']=True;write_json(p,m)
    with pytest.raises(ValueError,match='Protected'):workbench.seal(out)


def test_raw_or_export_drift_and_existing_partial_never_restarted(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);_,plan=workbench.load(out)
    dest=workbench.directory(out,'official_addresses',0);dest.mkdir(parents=True)
    with pytest.raises(ValueError,match='DO_NOT_RESTART'):workbench.remaining(out)
    p=plan['sources']['official_addresses']['reference']['path'];pq.write_table(canonical([]),p)
    with pytest.raises(ValueError,match='source changed'):workbench.load(out)


def test_source_or_plan_expected_denominator_drift_stops_init(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);l,_=workbench.load(out);cfg=l['config'];cfg['expected_total_observations']=11
    p=tmp_path/'wrong_config.json';write_json(p,cfg);monkeypatch.setattr(workbench,'CONFIG',p)
    with pytest.raises(ValueError,match='planned address'):workbench.init(tmp_path/'wrong')
