"""Label-free, bounded audit of current site binding for official successions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb

from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG=ROOT/'config/retrieval_v80_succession_site_audit.json'


def stats(connection, succession_path, establishment_path, sample_size):
    if sample_size!=64:
        raise ValueError('Preregistered source sample scope drift')
    connection.execute('CREATE TEMP VIEW raw_edges AS SELECT * FROM read_parquet(\''+
                       str(succession_path).replace("'","''")+'\')')
    valid="regexp_full_match(siretEtablissementPredecesseur,'[0-9]{14}') AND regexp_full_match(siretEtablissementSuccesseur,'[0-9]{14}')"
    def row(sql, params=None):
        result=connection.execute(sql,params or [])
        return dict(zip([x[0] for x in result.description],result.fetchone()))
    full=row(f'''SELECT count(*) raw_rows,
        count(*) FILTER(WHERE continuiteEconomique IS TRUE) continuity_true_rows,
        count(*) FILTER(WHERE continuiteEconomique IS NULL) continuity_unknown_rows,
        count(*) FILTER(WHERE NOT ({valid}) OR siretEtablissementPredecesseur IS NULL OR siretEtablissementSuccesseur IS NULL) invalid_identifier_rows,
        count(dateLienSuccession) reported_effective_dates,
        count(dateDernierTraitementLienSuccession) observed_timestamps,
        count(*) FILTER(WHERE siretEtablissementPredecesseur=siretEtablissementSuccesseur) self_rows
        FROM raw_edges''')
    # A streaming TOP512 of raw source rows avoids a millions-row hash GROUP.
    # If it contains >=64 distinct pairs, its first64 unique keys are exactly
    # the global first64 pair keys; otherwise STOP, never silently shrink it.
    connection.execute(f'''CREATE TEMP TABLE sample_keys AS SELECT DISTINCT predecessor,successor FROM (
        SELECT siretEtablissementPredecesseur predecessor,siretEtablissementSuccesseur successor
        FROM raw_edges WHERE continuiteEconomique IS TRUE AND ({valid})
        AND siretEtablissementPredecesseur<>siretEtablissementSuccesseur
        ORDER BY sha256(predecessor||'|'||successor),predecessor,successor LIMIT 512)
        ORDER BY sha256(predecessor||'|'||successor),predecessor,successor LIMIT {sample_size}''')
    if connection.execute('SELECT count(*) FROM sample_keys').fetchone()[0]!=sample_size:
        raise ValueError('Insufficient unique pair keys in diagnostic TOP512; STOP')
    connection.execute(f'''CREATE TEMP TABLE sampled_pairs AS
        SELECT siretEtablissementPredecesseur predecessor,siretEtablissementSuccesseur successor,
          count(*) source_rows,min(dateLienSuccession) earliest_date,max(dateLienSuccession) latest_date,
          bool_or(transfertSiege) any_headquarters_transfer
        FROM raw_edges r SEMI JOIN sample_keys k ON
        r.siretEtablissementPredecesseur=k.predecessor AND r.siretEtablissementSuccesseur=k.successor
        WHERE continuiteEconomique IS TRUE GROUP BY 1,2''')
    # The sample is a diagnostic only; no index or corpus is produced from it.
    connection.execute('CREATE TEMP TABLE endpoint_ids AS SELECT predecessor siret FROM sampled_pairs UNION SELECT successor FROM sampled_pairs')
    connection.execute('''CREATE TEMP TABLE sample_sites AS SELECT e.siret,e.siren,e.insee,e.postcode,
        e.current_address_normalized,e.administrative_state FROM read_parquet(?) e
        SEMI JOIN endpoint_ids ids USING(siret)''',[str(establishment_path)])
    connection.execute('''CREATE TEMP VIEW paired AS SELECT r.*,p.siret predecessor_found,s.siret successor_found,
        p.siren predecessor_parent,s.siren successor_parent,
        p.current_address_normalized predecessor_address,s.current_address_normalized successor_address,
        p.insee predecessor_insee,s.insee successor_insee,p.postcode predecessor_postcode,s.postcode successor_postcode
        FROM sampled_pairs r LEFT JOIN sample_sites p ON p.siret=r.predecessor
        LEFT JOIN sample_sites s ON s.siret=r.successor''')
    compatible="coalesce(predecessor_address,'')<>'' AND predecessor_address=successor_address AND ((coalesce(predecessor_insee,'')<>'' AND predecessor_insee=successor_insee) OR (coalesce(predecessor_postcode,'')<>'' AND predecessor_postcode=successor_postcode))"
    binding=row(f'''SELECT count(*) sampled_pairs,sum(source_rows) sampled_source_rows,
        count(*) FILTER(WHERE predecessor_found IS NOT NULL AND successor_found IS NOT NULL) both_endpoints_found,
        count(*) FILTER(WHERE predecessor_found IS NULL) predecessor_absent,
        count(*) FILTER(WHERE successor_found IS NULL) successor_absent,
        count(*) FILTER(WHERE predecessor_parent=successor_parent) same_parent_pairs,
        count(*) FILTER(WHERE predecessor_parent<>successor_parent) cross_parent_pairs,
        count(*) FILTER(WHERE ({compatible})) current_address_geo_compatible_pairs,
        count(*) FILTER(WHERE any_headquarters_transfer) headquarters_transfer_pairs
        FROM paired''')
    if binding['sampled_pairs']!=sample_size:
        raise ValueError('Complete diagnostic sample required')
    return dict(raw_source=full,sample_current_binding=binding,
                sample_selection='SHA256_OF_OFFICIAL_ENDPOINT_PAIR_NO_CRM_OR_LABEL',
                sample_is_not_search_corpus=True,historical_site_coexistence_proven=False)


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); config=launch['config']
    for ref in launch['implementation'].values(): verify(ref)
    with Budget(config['resources']) as budget:
        for key in ('successions','dossier_manifest','establishments'): verify(config[key])
        dossier=json.loads(Path(config['dossier_manifest']['path']).read_text())
        if (dossier['files']['establishments.parquet']['sha256']!=config['establishments']['sha256']
                or dossier['counts']['establishments']!=43896818):
            raise ValueError('National official establishment authority drift')
        connection=duckdb.connect(); connection.execute("SET memory_limit='512MB'")
        connection.execute('SET threads=2'); connection.execute("SET max_temp_directory_size='0B'")
        result=stats(connection,config['successions']['path'],config['establishments']['path'],config['sample_pairs'])
        connection.close()
        for key in ('successions','dossier_manifest','establishments'): verify(config[key])
    for ref in launch['implementation'].values(): verify(ref)
    write_json(output/'manifest.json',dict(status='AUDITED_SUCCESSION_CURRENT_SITE_BINDING_NOT_RETRIEVAL',
        launch=record(output/'launch.json'),stats=result,resources=budget.report(),
        truth_opened=False,crm_opened=False,recall_measured=False,model_fit=False,index_built=False,
        full_source_counts=True,full_site_binding_audit=False,ready_for_fold1=False,
        identifiers_as_features=False,downstream_models_changed=False,fold0_opened=False,fold1_opened=False))
    print(json.dumps(dict(stats=result,resources=budget.report(),recall_measured=False)),flush=True)


def run(output):
    output.mkdir(parents=True,exist_ok=False); config=json.loads(CONFIG.read_text())
    implementation={str(p):record(p) for p in (Path(__file__),CONFIG,ROOT/'docs/retrieval_v80_succession_site_audit.md')}
    write_json(output/'launch.json',dict(config=config,implementation=implementation,truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.audit_retrieval_v80_succession_site','worker','--output',str(output)],
                timeout=config['supervisor_seconds'],check=True,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-4000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False,truth_opened=False)); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=['run','worker'])
    parser.add_argument('--output',required=True,type=Path); args=parser.parse_args()
    (run if args.mode=='run' else worker)(args.output)
