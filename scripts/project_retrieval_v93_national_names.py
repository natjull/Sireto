"""Source-only complete national names and registry, not succession subset."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_name_facts_v85 import literal
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v93_national_names.json'


def project(c,names,sites,output,progress=lambda value:None):
    output=Path(output); phases={}
    def execute(phase,sql,params=None):
        began=time.monotonic(); result=c.execute(sql,params or [])
        phases[phase]=time.monotonic()-began; progress(dict(phase=phase,seconds=phases[phase]))
        return result
    c.execute('CREATE TEMP VIEW official_sites AS SELECT * FROM read_parquet('+literal(sites)+')')
    total,distinct,bad=execute('validate_complete_membership',"""SELECT count(*),count(DISTINCT siret),
        count(*) FILTER(WHERE regexp_full_match(siret,'[0-9]{14}') IS NOT TRUE OR
        regexp_full_match(siren,'[0-9]{9}') IS NOT TRUE OR substr(siret,1,9)<>siren)
        FROM official_sites""").fetchone()
    if bad or distinct!=total: raise ValueError('Complete official site grain/membership drift')
    execute('export_all_membership',"""COPY (SELECT CAST(siret AS UBIGINT) site_key,
        CAST(siren AS UINTEGER) parent_key,administrative_state FROM official_sites)
        TO ? (FORMAT PARQUET,COMPRESSION ZSTD,COMPRESSION_LEVEL 6)""",[str(output/'national_membership.parquet')])
    # Preserve EVERY source observation, including empty/unknown/invalid-grain
    # context. No join to a graph, query, label, selected owner, CP or commune.
    c.execute("""CREATE TEMP VIEW complete_names AS SELECT *,
        CAST(file_row_number AS UINTEGER) source_observation_index,
        (regexp_full_match(siren,'[0-9]{9}') AND
         (subject_kind='SIREN' OR (subject_kind='SIRET' AND regexp_full_match(siret,'[0-9]{14}')
         AND substr(siret,1,9)=siren))) IS TRUE name_grain_valid
        FROM read_parquet("""+literal(names)+',file_row_number=true)')
    c.execute("""CREATE TEMP VIEW scoped_names AS SELECT * EXCLUDE(file_row_number),
        CASE WHEN name_grain_valid AND subject_kind='SIRET' AND name_kind IN ('LEGAL','SIGN','TRADE','USUAL')
             THEN 'EXPLICIT_ENDPOINT_NAME'
             WHEN name_grain_valid AND subject_kind='SIREN' AND name_kind='LEGAL' THEN 'LEGAL_PERSON_NAME'
             WHEN name_grain_valid AND subject_kind='SIREN' AND name_kind IN ('USUAL','TRADE')
             THEN 'LEGAL_PERSON_COMMERCIAL_NAME' ELSE 'CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED' END lexical_scope
        FROM complete_names""")
    count=pq.ParquetFile(names).metadata.num_rows
    if not 0<count<4294967295: raise ValueError('Full observation ordinal range drift')
    execute('export_every_name_observation',"""COPY (SELECT *,
        lexical_scope<>'CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED' AND coalesce(normalized_value,'')<>'' lexical_eligible,
        FALSE alias_of_related_target,FALSE name_relation_co_validity_proven FROM scoped_names)
        TO ? (FORMAT PARQUET,COMPRESSION ZSTD,COMPRESSION_LEVEL 6)""",[str(output/'scoped_name_observations.parquet')])
    if (pq.ParquetFile(output/'scoped_name_observations.parquet').metadata.num_rows!=count or
            pq.ParquetFile(output/'national_membership.parquet').metadata.num_rows!=total):
        raise ValueError('Complete source output denominator drift')
    result=execute('describe_all_name_scopes',"""SELECT subject_kind,source,name_kind,lexical_scope,
        count(*) observations,count(*) FILTER(WHERE name_grain_valid IS NOT TRUE) invalid_grain,
        count(*) FILTER(WHERE coalesce(normalized_value,'')='') empty_normalized_names,
        count(*) FILTER(WHERE lexical_scope<>'CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED'
        AND coalesce(normalized_value,'')<>'') lexical_eligible_observations
        FROM scoped_names GROUP BY ALL ORDER BY ALL""")
    columns=[x[0] for x in result.description]; counts=[dict(zip(columns,row)) for row in result.fetchall()]
    if sum(r['observations'] for r in counts)!=count: raise ValueError('Name observation denominator drift')
    states=c.execute('SELECT administrative_state,count(*) FROM official_sites GROUP BY ALL ORDER BY ALL').fetchall()
    return dict(phases=phases,sites=total,name_observations=count,counts=counts,site_states=states,
        scope='EVERY_OFFICIAL_NAME_OBSERVATION_AND_ALL_NATIONAL_ESTABLISHMENTS',graph_membership_required=False,
        crm_label_or_query_selection=False,source_names_to_related_aliases=False,sibling_site_names_transferred=False,
        lexical_eligible_observations=sum(r['lexical_eligible_observations'] for r in counts),
        context_retained=True,all_observation_metadata_retained=True,original_observation_ordinals_retained=True,
        original_ordinals_independent_full_uniqueness_check=False)


def sources(cfg):
    manifest=json.loads(verify(cfg['dossier_manifest']).read_text())
    if manifest['policy']['crm_labels_present'] or manifest['policy']['model_scores_present']:
        raise ValueError('Official-only source required')
    paths={}
    for key,filename,expected in [('names','name_evidence.parquet',cfg['expected_name_rows']),
                                 ('sites','establishments.parquet',cfg['expected_sites'])]:
        ref=cfg[key]
        if manifest['files'][filename]['sha256']!=ref['sha256']: raise ValueError('Source authority drift')
        paths[key]=verify(ref)
        if pq.ParquetFile(paths[key]).metadata.num_rows!=expected: raise ValueError('National source count drift')
    return paths


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; c=None; budget=None
    for ref in launch['implementation'].values(): verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            paths=sources(cfg); c=duckdb.connect(); c.execute('SET threads=2'); c.execute('SET preserve_insertion_order=false')
            c.execute('SET memory_limit=?',[cfg['duckdb_memory']]); c.execute('SET temp_directory=?',[str(output/'spill')])
            c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            stats=project(c,paths['names'],paths['sites'],output,lambda row:print(json.dumps(row),flush=True))
            c.close(); c=None; sources(cfg)
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values(): verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_COMPLETE_NATIONAL_SCOPED_NAMES_AND_MEMBERSHIP',
            launch=record(output/'launch.json'),stats=stats,artifacts={p.name:record(p) for p in output.glob('*.parquet')},
            resources=budget.report(),output_bytes=size,truth_opened=False,crm_opened=False,fold0_opened=False,
            fold1_opened=False,model_fit=False,recall_measured=False,index_built=False,ready_for_fold1=False,
            identifiers_as_features=False,downstream_models_changed=False))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        if c is not None: c.close()
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,recall_measured=False)); raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v93_national_names.md',
           ROOT/'src/xgb_matcher/retrieval_name_facts_v85.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.project_retrieval_v93_national_names','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-6500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); a=p.parse_args(); (run if a.mode=='run' else worker)(a.output)
