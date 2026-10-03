"""Compile complete official succession endpoints, not an entity search pool."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG=ROOT/'config/retrieval_v81_succession_bindings.json'


def compile_bindings(c,successions,sites,output,progress=lambda value:None):
    phases={}
    def execute(name,sql,params=None):
        began=time.monotonic(); c.execute(sql,params or [])
        phases[name]=time.monotonic()-began; progress(dict(phase=name,seconds=phases[name]))
    execute('raw_view',"CREATE TEMP VIEW raw AS SELECT * FROM read_parquet('"+
            str(successions).replace("'","''")+"')")
    execute('eligible_observations','''CREATE TEMP TABLE edges AS SELECT
        CAST(siretEtablissementPredecesseur AS UBIGINT) predecessor,
        CAST(siretEtablissementSuccesseur AS UBIGINT) successor,
        dateLienSuccession reported_effective_date,transfertSiege headquarters_transfer,
        dateDernierTraitementLienSuccession source_updated_at
        FROM raw WHERE continuiteEconomique IS TRUE
        AND regexp_full_match(siretEtablissementPredecesseur,'[0-9]{14}')
        AND regexp_full_match(siretEtablissementSuccesseur,'[0-9]{14}')
        AND siretEtablissementPredecesseur<>siretEtablissementSuccesseur''')
    execute('endpoint_keys','CREATE TEMP TABLE endpoint_ids AS SELECT predecessor id FROM edges UNION SELECT successor FROM edges')
    execute('official_endpoint_sites','''CREATE TEMP TABLE sites AS SELECT
        CAST(s.siret AS UBIGINT) id,s.siren,s.administrative_state,s.is_headquarters,
        s.insee,s.postcode,s.street_number,s.current_address_normalized
        FROM read_parquet(?) s SEMI JOIN endpoint_ids e ON CAST(s.siret AS UBIGINT)=e.id''',[str(sites)])
    execute('release_endpoint_keys','DROP TABLE endpoint_ids')
    if c.execute('SELECT count(*)-count(DISTINCT id) FROM sites').fetchone()[0]:
        raise ValueError('Official endpoint identity duplicate; no arbitrary scaffold')
    output=Path(output)
    # All eligible observations remain in this narrow annotated table until
    # exhaustive coverage is exported; no fixed query-specific reservoir.
    execute('site_bindings','''CREATE TEMP TABLE bindings AS SELECT e.*,
        p.id predecessor_found,s.id successor_found,
        p.siren predecessor_siren,s.siren successor_siren,
        p.administrative_state predecessor_state,s.administrative_state successor_state,
        p.is_headquarters predecessor_headquarters,s.is_headquarters successor_headquarters,
        p.insee predecessor_insee,s.insee successor_insee,p.postcode predecessor_postcode,s.postcode successor_postcode,
        p.street_number predecessor_number,s.street_number successor_number,
        p.current_address_normalized predecessor_address,s.current_address_normalized successor_address
        FROM edges e LEFT JOIN sites p ON p.id=e.predecessor LEFT JOIN sites s ON s.id=e.successor''')
    execute('release_edges','DROP TABLE edges'); c.execute('DROP TABLE sites')
    execute('coverage','''COPY (
        SELECT status,count(*) observation_rows FROM (
            SELECT CASE
                WHEN siretEtablissementPredecesseur IS NULL OR siretEtablissementSuccesseur IS NULL
                    OR NOT regexp_full_match(siretEtablissementPredecesseur,'[0-9]{14}')
                    OR NOT regexp_full_match(siretEtablissementSuccesseur,'[0-9]{14}') THEN 'INVALID_IDENTIFIER'
                WHEN continuiteEconomique IS NOT TRUE THEN 'NO_EXPLICIT_ECONOMIC_CONTINUITY'
                WHEN siretEtablissementPredecesseur=siretEtablissementSuccesseur THEN 'SELF_RELATION'
                ELSE NULL END status FROM raw) WHERE status IS NOT NULL GROUP BY status
        UNION ALL SELECT CASE WHEN predecessor_found IS NULL AND successor_found IS NULL THEN 'BOTH_ENDPOINTS_ABSENT'
            WHEN predecessor_found IS NULL THEN 'PREDECESSOR_ABSENT'
            WHEN successor_found IS NULL THEN 'SUCCESSOR_ABSENT' ELSE 'BOUND_OFFICIAL_ENDPOINTS' END status,
            count(*) observation_rows FROM bindings GROUP BY status
        ) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',[str(output/'coverage.parquet')])
    execute('export_paths','''COPY (SELECT
        lpad(CAST(predecessor AS VARCHAR),14,'0') predecessor_siret,
        lpad(CAST(successor AS VARCHAR),14,'0') successor_siret,* EXCLUDE(predecessor,successor,predecessor_found,successor_found),
        'SIRENE' relation_source,'ESTABLISHMENT_SUCCESSION' relation_type,
        'SOURCE_EFFECTIVE_DATE' date_semantics,TRUE economic_continuity,
        CASE WHEN coalesce(predecessor_address,'')='' OR coalesce(successor_address,'')=''
            OR (coalesce(predecessor_insee,'')='' AND coalesce(predecessor_postcode,'')='')
            OR (coalesce(successor_insee,'')='' AND coalesce(successor_postcode,'')='')
            THEN 'CURRENT_SITE_BINDING_INCOMPLETE'
          WHEN predecessor_address=successor_address AND
            ((coalesce(predecessor_insee,'')<>'' AND predecessor_insee=successor_insee)
             OR (coalesce(predecessor_postcode,'')<>'' AND predecessor_postcode=successor_postcode))
            THEN 'CURRENT_ADDRESS_GEO_CONCORDANT_NOT_HISTORICAL_PROOF'
          ELSE 'CURRENT_LOCATION_DIFFERS_OR_GEO_DISAGREES' END site_binding,
        FALSE historical_site_coexistence_proven
        FROM bindings WHERE predecessor_found IS NOT NULL AND successor_found IS NOT NULL
        ) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',[str(output/'succession_bindings.parquet')])
    source_rows=c.execute('SELECT count(*) FROM raw').fetchone()[0]
    coverage=c.execute('SELECT status,observation_rows FROM read_parquet(?) ORDER BY status',[str(output/'coverage.parquet')]).fetchall()
    if sum(n for _,n in coverage)!=source_rows:
        raise ValueError('Exhaustive raw-source coverage denominator drift')
    classes=c.execute('SELECT site_binding,count(*) FROM read_parquet(?) GROUP BY site_binding ORDER BY site_binding',
                      [str(output/'succession_bindings.parquet')]).fetchall()
    return dict(phases=phases,source_rows=source_rows,
                coverage=[dict(status=s,observations=n) for s,n in coverage],
                site_binding_classes=[dict(binding=s,observations=n) for s,n in classes],
                output_observations=sum(n for _,n in classes),semantic_hops=1,
                names_transferred=False,global_relation_deduplication_claimed=False)


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; c=None
    for ref in launch['implementation'].values(): verify(ref)
    budget=Budget(cfg['resources'])
    try:
        with budget:
            for k in ('successions','dossier_manifest','establishments'): verify(cfg[k])
            m=json.loads(Path(cfg['dossier_manifest']['path']).read_text())
            if (m['policy']['crm_labels_present'] or m['policy']['model_scores_present']
                or m['counts']['establishments']!=cfg['expected_sites']
                or m['files']['establishments.parquet']['sha256']!=cfg['establishments']['sha256']
                or pq.ParquetFile(cfg['successions']['path']).metadata.num_rows!=cfg['expected_source_rows']):
                raise ValueError('Official source scope drift')
            c=duckdb.connect(); c.execute('SET threads=2'); c.execute('SET preserve_insertion_order=false')
            c.execute('SET memory_limit=?',[cfg['duckdb_memory']]); c.execute('SET temp_directory=?',[str(output/'spill')])
            c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            stats=compile_bindings(c,cfg['successions']['path'],cfg['establishments']['path'],output,
                lambda value:print(json.dumps(value),flush=True))
            c.close(); c=None
            if stats['source_rows']!=cfg['expected_source_rows']:
                raise ValueError('Source denominator changed')
            for k in ('successions','dossier_manifest','establishments'): verify(cfg[k])
            output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if output_bytes>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values(): verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_NATIONAL_OFFICIAL_SUCCESSION_SITE_BINDINGS',
            launch=record(output/'launch.json'),artifacts={p.name:record(p) for p in output.glob('*.parquet')},
            stats=stats,resources=budget.report(),output_bytes=output_bytes,
            truth_opened=False,crm_opened=False,recall_measured=False,index_built=False,model_fit=False,
            ready_for_fold1=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False,
            all_source_observations_accounted_for=True,historical_site_coexistence_proven=False))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=output_bytes)),flush=True)
    except BaseException as exc:
        if c is not None: c.close()
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),recall_measured=False,truth_opened=False))
        raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v81_succession_bindings.md',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v81_succession_bindings','worker','--output',str(output)],
                check=True,timeout=cfg['supervisor_seconds'],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-5000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); args=p.parse_args()
    (run if args.mode=='run' else worker)(args.output)
