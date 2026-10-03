import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.project_retrieval_v129_national_variants as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json


def test_physical_plan_covers_every_uneven_group_exactly_once(tmp_path):
    file=tmp_path/'raw.parquet';pq.write_table(pa.table({'x':list(range(17))}),file,row_group_size=3)
    parts=workbench.planned_parts(pq.ParquetFile(file),2)
    assert [(p['source_start'],p['source_stop']) for p in parts]==[(0,6),(6,12),(12,17)]
    assert [p['part_id'] for p in parts]==[0,1,2]


def setup(tmp_path,monkeypatch):
    roles=['history_ul','history_site','current_ul','current_site','official_names'];sources={}
    for role in roles:
        rows=[]
        for i in range(4):
            row=dict(siren='123456789',siret=f'123456789{i:05d}',nomUniteLegale='Durand',nomUsageUniteLegale='',
                prenomUsuelUniteLegale='Éva',prenom1UniteLegale='Eva',statutDiffusionUniteLegale='O',
                denominationUniteLegale='Alpha',dateDebut=None,dateFin=None,enseigne1Etablissement='Café',
                enseigne2Etablissement='',enseigne3Etablissement='',denominationUsuelleEtablissement='',
                subject_kind='SIREN',source='BODACC',name_kind='LEGAL',raw_value='Alpha',normalized_value='ALPHA',
                valid_from=None,valid_to=None,is_current=False)
            rows.append(row)
        file=tmp_path/(role+'.parquet');pq.write_table(pa.Table.from_pylist(rows),file,row_group_size=2)
        sources[role]=dict(record(file),expected_rows=4)
    cfg=dict(role_order=roles,sources=sources,row_groups_per_part=1,batch_rows=1,
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),
        max_part_output_bytes=1048576,max_total_output_bytes=16777216)
    path=tmp_path/'config.json';write_json(path,cfg);monkeypatch.setattr(workbench,'CONFIG',path)
    output=tmp_path/'projection';workbench.init(output);return output,roles


def test_incomplete_never_seals_then_all_source_rows_complete_and_immutable(tmp_path,monkeypatch):
    output,roles=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(output)
    assert not (output/'manifest.json').exists()
    for role in roles:
        for part in range(2):
            workbench.part_directory(output,role,part).mkdir(parents=True)
            workbench.worker(output,role,part)
    workbench.seal(output);m=json.loads((output/'manifest.json').read_text())
    assert m['source_rows']==20 and m['every_source_role_and_row_complete'] and len(m['parts'])==10
    assert m['complete_retrieval_index'] is False and m['recall_measured'] is False
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(output)
    with pytest.raises(FileExistsError):workbench.execute(output,roles[0],0)


def test_source_drift_blocks_further_projection(tmp_path,monkeypatch):
    output,roles=setup(tmp_path,monkeypatch)
    file=tmp_path/(roles[0]+'.parquet');pq.write_table(pa.table({'bad':['change']}),file)
    with pytest.raises(ValueError,match='source changed'):workbench.verify_plan(output)


def test_part_cannot_fake_closed_fold_or_lost_rows_at_seal(tmp_path,monkeypatch):
    output,roles=setup(tmp_path,monkeypatch)
    directory=workbench.part_directory(output,roles[0],0);directory.mkdir(parents=True);workbench.worker(output,roles[0],0)
    m=json.loads((directory/'manifest.json').read_text());m['source_rows']-=1;write_json(directory/'manifest.json',m)
    with pytest.raises(ValueError,match='authority/coverage'):workbench.seal(output)


def test_duplicate_role_cannot_omit_an_official_source(tmp_path,monkeypatch):
    output,roles=setup(tmp_path,monkeypatch)
    cfg=json.loads(workbench.CONFIG.read_text());cfg['role_order'][-1]=cfg['role_order'][0];cfg['sources'].pop('official_names')
    write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='All five'):workbench.init(tmp_path/'bad_plan')
