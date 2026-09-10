"""Diagnose label-blind succession witness instability; no recipe modification."""
import argparse
import json
from pathlib import Path

import duckdb
from scripts.retrieval_v31_replay_preflight import DATA,ROOT,RESOURCE
from scripts.run_retrieval_parent_discovery_v18 import _write_structured_signals
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json


def run(source,output):
    cfg=json.loads((ROOT/'config/retrieval_all_human_v18_parent_discovery_generation.json').read_text())
    a=cfg['generation_authorities'];records=[]
    temporal=json.loads(verify(dict(path=Path(a['temporal_sources']['index_path'])/'manifest.json',sha256=a['temporal_sources']['manifest_sha256'])).read_text())
    exact_paths=[]
    for s in temporal['sources']:
        p=verify(dict(path=Path(a['temporal_sources']['source_root'])/s['projection_build_id']/'succession_edges.parquet',sha256=s['successions_sha256']))
        exact_paths.append(str(p));records.append(record(p))
    actual_paths=sorted(str(p) for p in Path(a['temporal_sources']['source_root']).glob('*/succession_edges.parquet'))
    if actual_paths!=sorted(exact_paths):raise ValueError('Unsealed temporal glob file')
    c=duckdb.connect();c.execute("SET memory_limit='2GB'");c.execute('SET threads=2')
    c.execute('CREATE TABLE raw AS SELECT * FROM read_parquet(?)',[str(source/'raw.parquet')])
    ids=tuple(c.execute('SELECT DISTINCT query_id,fold FROM raw ORDER BY query_id').fetchall())
    if len(ids)>16 or any(f not in (2,3,4) for _,f in ids):raise ValueError('Preflight scope')
    paths=dict(raw_union=source/'raw.parquet',v17_parent_presence=source/'parent_presence.parquet',v17_parent_candidates=source/'parent_candidates.parquet',
      rne_roles=verify(a['rne_roles']),bodacc_relations=verify(a['bodacc_relations']),temporal_glob=Path(a['temporal_sources']['source_root'])/'*'/'succession_edges.parquet')
    for threads in (1,2,8):
        cfg['resources']['duckdb_threads']=threads
        _write_structured_signals(config=cfg,query_rows=ids,paths=paths,output_path=output/f'threads{threads}.parquet')
    oldpath=verify(json.loads((ROOT/'config/retrieval_all_human_v19_parent_fusion.json').read_text())['authorities']['structured_signals'])
    c.execute('CREATE TABLE old AS SELECT * FROM read_parquet(?) SEMI JOIN (SELECT DISTINCT query_id,fold FROM raw) USING(query_id,fold)',[str(oldpath)])
    c.execute('CREATE TABLE new AS SELECT * FROM read_parquet(?)',[str(source/'structured.parquet')])
    cols='query_id,fold,channel,channel_rank,siren,source_siret'
    repeat={}
    for threads in (1,2,8):
        c.execute('CREATE OR REPLACE TABLE again AS SELECT * FROM read_parquet(?)',[str(output/f'threads{threads}.parquet')])
        repeat[str(threads)]={}
        for baseline in ('old','new'):
            repeat[str(threads)][baseline]=c.execute(f'SELECT channel,count(*) FROM ((SELECT {cols} FROM {baseline}) EXCEPT ALL (SELECT {cols} FROM again)) GROUP BY channel').fetchall()
    c.execute('CREATE TABLE edges AS SELECT * FROM read_parquet(?)',[exact_paths])
    # Enumerate only naturally eligible succession witnesses from actual seeds.
    c.execute('''CREATE TABLE eligible AS
      WITH emitted AS (
        SELECT r.query_id,r.fold,e.successor_siren siren,e.successor_siret siret,
          r.union_rank seed_rank,'PREDECESSOR_TO_SUCCESSOR' direction
        FROM raw r JOIN edges e ON r.siret=e.predecessor_siret
        WHERE r.union_rank<=250 AND e.economic_continuity AND length(e.successor_siren)=9
        UNION ALL
        SELECT r.query_id,r.fold,e.predecessor_siren siren,e.predecessor_siret siret,
          r.union_rank seed_rank,'SUCCESSOR_TO_PREDECESSOR' direction
        FROM raw r JOIN edges e ON r.siret=e.successor_siret
        WHERE r.union_rank<=250 AND e.economic_continuity AND length(e.predecessor_siren)=9
      ) SELECT DISTINCT * FROM emitted QUALIFY seed_rank=min(seed_rank) OVER(PARTITION BY query_id,fold,siren)''')
    c.execute('''CREATE TABLE changed AS SELECT o.query_id,o.fold,o.siren,o.source_siret old_siret,n.source_siret new_siret,
      o.seed_rank old_seed_rank,n.seed_rank new_seed_rank,o.relation_direction old_direction,n.relation_direction new_direction
      FROM old o JOIN new n USING(query_id,fold,channel,siren)
      WHERE o.channel='sirene_economic_continuity' AND o.source_siret<>n.source_siret''')
    changed=c.execute('SELECT count(*),count(DISTINCT query_id),count(*) FILTER(WHERE old_seed_rank<>new_seed_rank),count(*) FILTER(WHERE old_direction<>new_direction) FROM changed').fetchone()
    unsupported={}
    for side in ('old','new'):
        unsupported[side]=c.execute(f'''SELECT count(*) FROM changed t ANTI JOIN eligible e ON
          t.query_id=e.query_id AND t.fold=e.fold AND t.siren=e.siren AND t.{side}_siret=e.siret
          AND t.{side}_seed_rank=e.seed_rank AND t.{side}_direction=e.direction''').fetchone()[0]
    ties=c.execute('''SELECT count(*) FROM (SELECT t.query_id,t.fold,t.siren,count(DISTINCT e.siret) n FROM changed t
      JOIN eligible e USING(query_id,fold,siren) GROUP BY 1,2,3 HAVING n>1)''').fetchone()[0]
    master={}
    for version,oldmaster in [('v23',DATA/'audits/retrieval_v23_oof_master/bb12058773f80d6f/master.parquet'),
      ('v30',DATA/'audits/retrieval_v30_master/8cb4ad779f11ac84/master.parquet')]:
        c.execute('CREATE OR REPLACE TABLE om AS SELECT * FROM read_parquet(?) SEMI JOIN (SELECT DISTINCT query_id,fold FROM raw) USING(query_id,fold)',[str(oldmaster)])
        c.execute('CREATE OR REPLACE TABLE nm AS SELECT * FROM read_parquet(?)',[str(source/version/'master.parquet')])
        master[version]=dict(removed_pairs=c.execute('SELECT count(*) FROM om ANTI JOIN nm USING(query_id,fold,siret)').fetchone()[0],
          added_pairs=c.execute('SELECT count(*) FROM nm ANTI JOIN om USING(query_id,fold,siret)').fetchone()[0],
          changed_ranks=c.execute('SELECT count(*) FROM om JOIN nm USING(query_id,fold,siret) WHERE om.candidate_rank<>nm.candidate_rank').fetchone()[0])
    explained=changed[0]>0 and ties==changed[0] and not any(unsupported.values())
    replay_differs=any(bool(v['new']) for v in repeat.values())
    status='STOP_NONDETERMINISTIC_SUCCESSION_WITNESS' if explained and replay_differs else 'STOP_PARITY_CAUSE_NOT_FULLY_ESTABLISHED'
    return dict(status=status,queries=len(ids),changed_witnesses=changed[0],affected_queries=changed[1],
      changed_seed_ranks=changed[2],changed_directions=changed[3],changed_with_multiple_min_seed_targets=ties,
      unsupported_witnesses=unsupported,replays=repeat,master_membership_changes=master,
      temporal_files_verified=len(records),temporal_sources=records,truth_columns_opened=False,quality_measured=False,
      raw_manifest=record(source/'manifest.json'),code=record(Path(__file__)),
      recommendation='Freeze an explicit deterministic tie-break under a new recipe; do not claim unchanged v23/v30 or reuse per-query witnesses.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with Budget(RESOURCE) as budget:result=run(a.source,a.output)
    result['resources']=budget.report();write_json(a.output/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='temporal_sources'}),flush=True)


if __name__=='__main__':main()
