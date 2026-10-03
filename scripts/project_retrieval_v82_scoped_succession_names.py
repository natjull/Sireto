"""Factor official names by actual succession endpoints and legal owners."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v82_scoped_succession_names.json'


def project(c,bindings,names,output,progress=lambda value:None):
    phases={}; output=Path(output)
    def execute(name,sql,params=None):
        began=time.monotonic(); c.execute(sql,params or [])
        phases[name]=time.monotonic()-began; progress(dict(phase=name,seconds=phases[name]))
    execute('endpoint_scope','''CREATE TEMP TABLE endpoints AS
        SELECT CAST(predecessor_siret AS UBIGINT) id,CAST(predecessor_siren AS UINTEGER) parent
        FROM read_parquet(?) UNION SELECT CAST(successor_siret AS UBIGINT),CAST(successor_siren AS UINTEGER)
        FROM read_parquet(?)''',[str(bindings),str(bindings)])
    if c.execute('SELECT count(*)-count(DISTINCT id) FROM endpoints').fetchone()[0]:
        raise ValueError('Conflicting official parent for an endpoint')
    execute('parent_scope','CREATE TEMP TABLE parents AS SELECT DISTINCT parent FROM endpoints')
    execute('export_endpoint_scope','''COPY (SELECT lpad(CAST(id AS VARCHAR),14,'0') siret,
        lpad(CAST(parent AS VARCHAR),9,'0') siren FROM endpoints) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',
        [str(output/'endpoint_scope.parquet')])
    execute('official_names_view',"CREATE TEMP VIEW official_names AS SELECT * FROM read_parquet('"+
            str(names).replace("'","''")+"')")
    # Two precise SEMI joins instead of a SIREN-only OR join that could leak
    # a sibling trade name. No direction/path duplication of name observations.
    selection='''SELECT n.* FROM official_names n SEMI JOIN endpoints e ON
        TRY_CAST(n.siret AS UBIGINT)=e.id AND TRY_CAST(n.siren AS UINTEGER)=e.parent
        WHERE n.subject_kind='SIRET' AND coalesce(n.normalized_value,'')<>''
        UNION ALL SELECT n.* FROM official_names n SEMI JOIN parents p ON TRY_CAST(n.siren AS UINTEGER)=p.parent
        WHERE n.subject_kind='SIREN' AND coalesce(n.normalized_value,'')<>\'\' '''
    execute('export_scoped_names','''COPY (SELECT *,
        CASE WHEN subject_kind='SIRET' AND name_kind IN ('LEGAL','SIGN','TRADE','USUAL') THEN 'EXPLICIT_ENDPOINT_NAME'
             WHEN subject_kind='SIREN' AND name_kind='LEGAL' THEN 'LEGAL_PERSON_NAME'
             WHEN subject_kind='SIREN' AND name_kind IN ('USUAL','TRADE') THEN 'LEGAL_PERSON_COMMERCIAL_NAME'
             ELSE 'CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED' END lexical_scope,
        FALSE alias_of_related_target,FALSE name_relation_co_validity_proven
        FROM ('''+selection+''')) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',
        [str(output/'scoped_name_observations.parquet')])
    endpoint_count=c.execute('SELECT count(*) FROM endpoints').fetchone()[0]
    parent_count=c.execute('SELECT count(*) FROM parents').fetchone()[0]
    execute('release_scopes','DROP TABLE endpoints'); c.execute('DROP TABLE parents')
    result=c.execute('''SELECT subject_kind,source,name_kind,lexical_scope,count(*) observations,
        count(*) FILTER(WHERE valid_from IS NULL) unknown_valid_from,
        count(*) FILTER(WHERE valid_to IS NULL) unknown_valid_to FROM read_parquet(?)
        GROUP BY ALL ORDER BY subject_kind,source,name_kind,lexical_scope''',[str(output/'scoped_name_observations.parquet')])
    columns=[x[0] for x in result.description]; counts=[dict(zip(columns,r)) for r in result.fetchall()]
    return dict(phases=phases,endpoint_sites=endpoint_count,legal_owners=parent_count,
        name_observations=sum(r['observations'] for r in counts),counts=counts,
        scope='ALL_OFFICIAL_ENDPOINT_NAMES_AND_THEIR_LEGAL_PERSON_NAMES',
        sibling_site_names_transferred=False,relation_aliases_created=False,
        repeated_observations_retained=True,unique_lexical_fact_count_not_measured=True)


def sources(cfg):
    bm=json.loads(verify(cfg['bindings']).read_text()); bl=json.loads(verify(bm['launch']).read_text())
    for ref in bl['implementation'].values(): verify(ref)
    if (bm['status']!='SEALED_NATIONAL_OFFICIAL_SUCCESSION_SITE_BINDINGS' or bm['truth_opened'] or bm['crm_opened']
        or bm['stats']['output_observations']!=cfg['expected_bound_observations'] or bm['stats']['names_transferred']):
        raise ValueError('Complete official-only binding source required')
    bound=verify(bm['artifacts']['succession_bindings.parquet'])
    dossier=json.loads(verify(cfg['dossier_manifest']).read_text()); names=verify(cfg['names'])
    if (dossier['policy']['crm_labels_present'] or dossier['policy']['model_scores_present']
        or dossier['files']['name_evidence.parquet']['sha256']!=cfg['names']['sha256']
        or pq.ParquetFile(names).metadata.num_rows!=cfg['expected_name_rows']
        or pq.ParquetFile(bound).metadata.num_rows!=cfg['expected_bound_observations']):
        raise ValueError('National name/binding source denominator drift')
    return bound,names


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; c=None
    for ref in launch['implementation'].values(): verify(ref)
    budget=Budget(cfg['resources'])
    try:
        with budget:
            bound,names=sources(cfg)
            c=duckdb.connect(); c.execute('SET threads=2'); c.execute('SET preserve_insertion_order=false')
            c.execute('SET memory_limit=?',[cfg['duckdb_memory']]); c.execute('SET temp_directory=?',[str(output/'spill')])
            c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            stats=project(c,bound,names,output,lambda value:print(json.dumps(value),flush=True))
            c.close(); c=None; sources(cfg)
            output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if output_bytes>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values(): verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_FACTORIZED_SCOPED_OFFICIAL_SUCCESSION_NAMES',
            launch=record(output/'launch.json'),bindings_source=cfg['bindings'],stats=stats,
            artifacts={p.name:record(p) for p in output.glob('*.parquet')},resources=budget.report(),output_bytes=output_bytes,
            truth_opened=False,crm_opened=False,model_fit=False,index_built=False,recall_measured=False,
            ready_for_fold1=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False,
            identifiers_as_features=False,official_names_not_yet_scored=True))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=output_bytes)),flush=True)
    except BaseException as exc:
        if c is not None: c.close()
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v82_scoped_succession_names.md',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.project_retrieval_v82_scoped_succession_names','worker','--output',str(output)],
                check=True,timeout=cfg['supervisor_seconds'],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-8500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); args=p.parse_args()
    (run if args.mode=='run' else worker)(args.output)
