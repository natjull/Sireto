import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.project_retrieval_v129_national_variants as original
import scripts.compile_retrieval_v130_variant_dictionary as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json


def setup(tmp_path,monkeypatch):
    roles=['history_ul','history_site','current_ul','current_site','official_names'];sources={}
    for role in roles:
        row=dict(siren='123456789',siret='12345678900001',nomUniteLegale='Durand',nomUsageUniteLegale='',
            prenomUsuelUniteLegale='Éva',prenom1UniteLegale='Eva',statutDiffusionUniteLegale='O',
            denominationUniteLegale='Alpha',dateDebut=None,dateFin=None,enseigne1Etablissement='Alpha',
            enseigne2Etablissement='',enseigne3Etablissement='',denominationUsuelleEtablissement='',
            subject_kind='SIREN',source='BODACC',name_kind='LEGAL',raw_value='Alpha',normalized_value='ALPHA',
            valid_from=None,valid_to=None,is_current=False)
        path=tmp_path/(role+'.parquet');pq.write_table(pa.Table.from_pylist([row,row]),path,row_group_size=1)
        sources[role]=dict(record(path),expected_rows=2)
    resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0)
    source_cfg=tmp_path/'source_config.json';write_json(source_cfg,dict(role_order=roles,sources=sources,row_groups_per_part=2,
        batch_rows=1,resources=resources,max_part_output_bytes=1048576,max_total_output_bytes=16777216))
    monkeypatch.setattr(original,'CONFIG',source_cfg);source_root=tmp_path/'source';original.init(source_root)
    for role in roles:
        original.part_directory(source_root,role,0).mkdir(parents=True);original.worker(source_root,role,0)
    original.seal(source_root);source=json.loads((source_root/'manifest.json').read_text())
    lexical=sum(json.loads((source_root/role/'part00000/manifest.json').read_text())['lexical_observations'] for role in roles)
    cfg=tmp_path/'dictionary_config.json';write_json(cfg,dict(source_manifest=record(source_root/'manifest.json'),
        expected_source_rows=10,expected_observations=source['observations'],expected_lexical_observations=lexical,
        buckets=2,duckdb_version=duckdb.__version__,duckdb_memory='128MiB',spill_limit='32MiB',resources=resources,
        max_phase_output_bytes=1048576,max_total_output_bytes=16777216))
    monkeypatch.setattr(workbench,'CONFIG',cfg);output=tmp_path/'dictionary';workbench.init(output)
    return output,source


def stage_all(output):
    _,plan=workbench.load(output)
    for spec in plan['parts']:
        workbench.stage_dir(output,spec['part_id']).mkdir(parents=True);workbench.worker(output,'part',spec['part_id'])
    workbench.seal_parts(output)


def test_complete_global_names_and_all_observations_required_before_seals(tmp_path,monkeypatch):
    output,source=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='UNFINISHED_SOURCE'):workbench.seal_parts(output)
    with pytest.raises(FileNotFoundError):workbench.execute(output,'bucket',0)
    stage_all(output)
    with pytest.raises(ValueError,match='UNFINISHED_DICTIONARY'):workbench.seal_dictionary(output)
    for b in range(2):
        workbench.bucket_dir(output,b).mkdir(parents=True);workbench.worker(output,'bucket',b)
    workbench.seal_dictionary(output);m=json.loads((output/'manifest.json').read_text())
    assert m['original_observations']==source['observations'] and m['complete_retrieval_index'] is False and m['recall_measured'] is False
    assert m['names']>0 and m['facts']>m['names']
    assert all(m[k] is False for k in workbench.PROTECTED)
    with pytest.raises(ValueError,match='already sealed'):workbench.seal_dictionary(output)
    with pytest.raises(ValueError,match='already sealed'):workbench.seal_parts(output)
    with pytest.raises(FileExistsError):workbench.execute(output,'part',0)
    with pytest.raises(ValueError,match='already started'):workbench.worker(output,'part',0)


def test_closed_fold_flag_cannot_be_faked_at_complete_stage_seal(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);stage_all(output)
    # A completed storage part mutated after seal is not accepted by a new
    # seal attempt, even if the caller removes only that final control file.
    manifest=workbench.stage_dir(output,0)/'manifest.json';m=json.loads(manifest.read_text());m['fold1_opened']=True;write_json(manifest,m)
    # Check validator directly; no deletion/overwrite of the sealed authority.
    with pytest.raises(ValueError,match='Protected'):workbench.phase_manifest(manifest,output/'launch.json','SEALED_V130_OFFICIAL_NAME_STORAGE_PART')


def test_all_sources_guard_rejects_implementation_or_original_file_drift(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);_,plan=workbench.load(output);path=plan['parts'][0]['source']['path']
    pq.write_table(pa.table({'bad':['changed']}),path)
    workbench.stage_dir(output,0).mkdir(parents=True)
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(output,'part',0)


def test_dictionary_namespace_pins_hash_library_version(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);monkeypatch.setattr(duckdb,'__version__','not-pinned')
    with pytest.raises(ValueError,match='hash library'):workbench.load(output)
