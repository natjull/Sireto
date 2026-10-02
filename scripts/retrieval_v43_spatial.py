"""Full official spatial index and label-blind direct geographic top100 pilot."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts.run_retrieval_geo_v16_workbench import _resolve_ban
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_geo_v16 import normalize_street
from src.xgb_matcher.retrieval_spatial_v43 import build_tree,nearest_sites,validate_arrays
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v43_spatial.json'
PREDICATE="""regexp_full_match(codeCommuneEtablissement,'[0-9AB]{5}')
 AND left(codeCommuneEtablissement,2) NOT IN ('97','98')
 AND isfinite(x) AND isfinite(y) AND x BETWEEN 0 AND 1300000 AND y BETWEEN 6000000 AND 7300000"""


def implementation():
    paths=[Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_spatial_v43.py',
           ROOT/'src/xgb_matcher/retrieval_geo_v16.py',ROOT/'scripts/run_retrieval_geo_v16_workbench.py',
           Path(v42.__file__),Path(v38.__file__),ROOT/'src/xgb_matcher/hierarchical_retrieval.py',
           ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    return {str(p):record(p) for p in paths}


def build(output):
    import duckdb
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    cfg=json.loads(CONFIG.read_text())
    try:
        with Budget(cfg['resources_build']) as budget:
            source=verify(cfg['source']); before=source.stat()
            conn=duckdb.connect(); conn.execute("SET threads=2"); conn.execute("SET memory_limit='512MB'")
            conn.execute("SET max_temp_directory_size='0B'"); conn.execute('SET enable_progress_bar=false')
            sql=str(source).replace("'","''")
            relation=f"""SELECT siret,siren,codeCommuneEtablissement,
              try_cast(coordonneeLambertAbscisseEtablissement AS DOUBLE) x,
              try_cast(coordonneeLambertOrdonneeEtablissement AS DOUBLE) y FROM read_parquet('{sql}')"""
            counts=conn.execute(f"""SELECT count(*) n,count(*) FILTER(WHERE isfinite(x) AND isfinite(y)) finite_pairs,
                count(*) FILTER(WHERE {PREDICATE}) compatible_pairs FROM ({relation})""").fetchone()
            total,finite,count=map(int,counts)
            if total!=cfg['source_rows'] or count<=262144: raise ValueError('Official corpus count drift')
            xy=np.lib.format.open_memmap(output/'xy.npy',mode='w+',dtype=np.float64,shape=(count,2))
            keys=np.lib.format.open_memmap(output/'keys.npy',mode='w+',dtype=np.uint64,shape=(count,))
            parents=np.lib.format.open_memmap(output/'parents.npy',mode='w+',dtype=np.uint32,shape=(count,))
            cursor=0
            batches=conn.execute(f"""SELECT cast(siret AS UBIGINT) siret,cast(siren AS UINTEGER) siren,x,y
                FROM ({relation}) WHERE {PREDICATE}""").fetch_record_batch(rows_per_batch=262144)
            for batch in batches:
                n=batch.num_rows
                keys[cursor:cursor+n]=batch.column(0).to_numpy(zero_copy_only=False)
                parents[cursor:cursor+n]=batch.column(1).to_numpy(zero_copy_only=False)
                xy[cursor:cursor+n,0]=batch.column(2).to_numpy(zero_copy_only=False)
                xy[cursor:cursor+n,1]=batch.column(3).to_numpy(zero_copy_only=False)
                cursor+=n
            conn.close()
            if cursor!=count: raise ValueError('Coordinate projection denominator drift')
            order=np.argsort(keys)
            keys[:]=keys[order]; parents[:]=parents[order]; xy[:]=xy[order]
            del order
            xy.flush(); keys.flush(); parents.flush(); validate_arrays(xy,keys,parents)
            # Label-blind cost preflight on real official coordinates, not truth sites.
            t=time.monotonic(); sample=build_tree(np.asarray(xy[:262144]).copy()); sample_seconds=time.monotonic()-t
            if sample_seconds>2.: raise ValueError('STOP_SPATIAL_BUILD_PREFLIGHT')
            del sample
            t=time.monotonic(); tree=build_tree(xy); tree_seconds=time.monotonic()-t
            with (output/'tree.pickle').open('xb') as handle: pickle.dump(tree,handle,protocol=5)
            del tree
            after=source.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise ValueError('Source changed')
            refs={n:record(output/n) for n in ('xy.npy','keys.npy','parents.npy','tree.pickle')}
        write_json(output/'manifest.json',dict(status='SEALED_OFFICIAL_SPATIAL_INDEX',schema='sireto-v43-spatial-index-1',
            source=cfg['source'],source_rows=total,finite_coordinate_pairs=finite,compatible_coordinate_pairs=count,
            missing_or_incompatible_coordinates=total-count,files=refs,sample_tree_seconds=sample_seconds,
            full_tree_seconds=tree_seconds,resources=budget.report(),seconds=time.monotonic()-began,
            implementation=implementation(),crm_opened=False,truth_opened=False,closed_sites_kept=True,
            coordinate_scope=cfg['coordinate_scope'],siret_order='ASCENDING_NUMERIC_FOR_JOINS_AND_TIES_ONLY'))
        print(json.dumps({k:json.loads((output/'manifest.json').read_text())[k] for k in
            ('source_rows','finite_coordinate_pairs','compatible_coordinate_pairs','sample_tree_seconds','full_tree_seconds','resources','seconds')}))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,truth_opened=False)); raise


def prepare(output,index):
    import duckdb
    cfg=json.loads(CONFIG.read_text())
    base=json.loads(verify(cfg['query_generation']).read_text())
    if base['status']!='SEALED_INITIAL_TOP100' or base['queries']!=128: raise ValueError('Query authority drift')
    base_launch=json.loads(verify(base['launch']).read_text())
    queries=base_launch['queries']
    if len(queries)!=128 or len({r['query_id'] for r in queries})!=128 or any(
        set(r)!=v38.QUERY_FIELDS or r['fold'] not in (2,3,4) for r in queries):
        raise ValueError('Forbidden query population')
    sm=json.loads((index/'manifest.json').read_text())
    if sm['status']!='SEALED_OFFICIAL_SPATIAL_INDEX' or sm['truth_opened'] or sm['crm_opened']:
        raise ValueError('Unsealed coordinate corpus')
    for ref in sm['files'].values(): verify(ref)
    ban=verify(cfg['ban'])
    surface=pa.Table.from_pylist([dict(query_id=r['query_id'],fold=r['fold'],crm_number=r['crm_number'],
        crm_street_norm=normalize_street(r['crm_street']),crm_postcode=r['crm_postcode'],crm_insee=r['crm_insee']) for r in queries])
    began=time.monotonic()
    with Budget(cfg['resources_build']) as budget:
        conn=duckdb.connect(); conn.execute("SET threads=2"); conn.execute("SET memory_limit='512MB'")
        conn.execute("SET max_temp_directory_size='0B'"); conn.execute('SET enable_progress_bar=false')
        geo=_resolve_ban(conn,surface,ban); conn.close()
    rows=geo.to_pylist()
    if len(rows)!=128 or {r['query_id'] for r in rows}!={r['query_id'] for r in queries}:
        raise ValueError('BAN resolution denominator drift')
    for r,q in zip(rows,queries):
        if r['query_id']!=q['query_id'] or r['fold']!=q['fold']: raise ValueError('BAN order drift')
        # Other CRSs remain a missing spatial feature, not a removed query.
        r['spatial_crs_compatible']=bool(q['crm_insee'] and not q['crm_insee'].startswith(('97','98'))
            and r['ban_resolved'] and 0<=r['ban_x_l93']<=1300000 and 6000000<=r['ban_y_l93']<=7300000)
    write_json(output/'geo_queries.json',rows)
    im=v38.load(v38.INDEX/'manifest.json',v38.INDEX_SHA)
    launch=dict(queries=queries,query_sources=base_launch['query_sources'],query_generation=cfg['query_generation'],
        geo_queries=record(output/'geo_queries.json'),ban=cfg['ban'],spatial_manifest=record(index/'manifest.json'),
        spatial_files=sm['files'],spatial_sites=sm['compatible_coordinate_pairs'],models=cfg['models'],
        index_manifest=record(v38.INDEX/'manifest.json'),index_meta=record(v38.INDEX/'tantivy/meta.json'),
        inventory=v38.inventory(),index_documents=im['num_documents'],index_sites=im['num_establishment_documents'],
        implementation=implementation(),collector_limit=100,fixed_reservoir=None,truth_opened=False,
        full_national_source_rows=sm['source_rows'],development_panel_reused=True,
        ban_resolution_resources=budget.report(),ban_resolution_seconds=time.monotonic()-began,
        resources=cfg['resources_worker'])
    write_json(output/'launch.json',launch)


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    geo={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}
    models=json.loads(verify(launch['models']).read_text()); rows=[]
    with Budget(launch['resources']) as budget:
        paths=launch['spatial_files']
        for ref in paths.values(): verify(ref)
        with Path(paths['tree.pickle']['path']).open('rb') as handle: tree=pickle.load(handle)
        # The tree owns its coordinates after deserialization; preserve their
        # alignment with the independently hashed identity arrays.
        xy=tree.data
        keys=np.load(paths['keys.npy']['path'],mmap_mode='r',allow_pickle=False)
        parents=np.load(paths['parents.npy']['path'],mmap_mode='r',allow_pickle=False)
        if len(xy)!=launch['spatial_sites']: raise ValueError('Spatial count drift')
        backend=TantivyBackend(v38.INDEX)
        for raw in launch['queries'][start:start+8]:
            g=geo[raw['query_id']]; model=models[str(raw['fold'])]
            if model['heldout_fold']!=raw['fold'] or raw['fold'] in model['training_folds']:
                raise ValueError('Fallback model OOF drift')
            t=time.monotonic(); learned=v42.search(backend,v42.as_query(raw),model['weights']); lexical_seconds=time.monotonic()-t
            t=time.monotonic()
            if g['spatial_crs_compatible']:
                spatial,trace=nearest_sites(tree,xy,keys,parents,[g['ban_x_l93'],g['ban_y_l93']])
                source='SPATIAL_DIRECT'; primary_seconds=time.monotonic()-t
            else:
                spatial=learned; trace={}; source='FROZEN_V42_NO_COMPATIBLE_QUERY_POSITION'; primary_seconds=lexical_seconds
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=spatial,comparator=learned),
                seconds=dict(primary=primary_seconds,comparator=lexical_seconds),primary_source=source,
                ban_resolved=g['ban_resolved'],ban_declared_weak=g['ban_declared_weak'],spatial_trace=trace))
    v38.verify_launch(launch)
    write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def run(output,index):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        prepare(output,index); refs=[]; rows=[]
        for start in range(0,128,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v43_spatial','worker',
                    '--output',str(output),'--start',str(start)],stdout=log,stderr=subprocess.STDOUT,check=True,
                    timeout=min(90,max(1,900-(time.monotonic()-began))),
                    env={**os.environ,'RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            p=output/f'batch{start:03d}.json'; batch=json.loads(p.read_text()); rows.extend(batch['rows']); refs.append(record(p))
            if len(batch['rows'])!=8 or not batch['resources']['pass_resource_policy']: raise ValueError('Worker scope/resource drift')
            print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                cost=sum(sum(r['seconds'].values()) for r in rows)/16
                write_json(output/'cost_gate.json',dict(queries=16,mean_pair_seconds=cost,
                    passed=cost<=4.,quality_opened=False))
                if cost>4.: raise ValueError('STOP_COST_PREFLIGHT')
        launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
        if len(rows)!=128 or {r['query_id'] for r in rows}!={r['query_id'] for r in launch['queries']}:
            raise ValueError('Generation denominator drift')
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,launch=record(output/'launch.json'),
            batches=refs,seconds=time.monotonic()-began,truth_opened=False,model_used=True,
            full_corpus=False,full_prescribed_spatial_corpus=True,full_lexical_corpus=True,
            full_query_population=False,ready_for_fold1=False,
            spatial_eligible_corpus_not_full_catalog=True,spatial_rows=launch['spatial_sites'],
            spatial_queries=sum(r['primary_source']=='SPATIAL_DIRECT' for r in rows)))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,
            truth_opened=False,ready_for_fold1=False)); raise


def evaluate(output):
    """All-human paired measurement, only after complete label-blind generation."""
    dest=output/'evaluation.json'
    if dest.exists(): raise ValueError('Do not overwrite an evaluation')
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']:
        raise ValueError('Complete sealed generation required')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    generated=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    ids={r['query_id'] for r in launch['queries']}
    if len(generated)!=128 or {r['query_id'] for r in generated}!=ids: raise ValueError('Evaluation population drift')
    ev=v38.load(v38.EVAL,v38.EVAL_SHA); label_side=json.loads(verify(ev['label_side']).read_text())
    for ref in label_side['files'].values(): verify(ref)
    labels={r['query_id']:r for r in pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),
        filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    catalog={r['query_id']:r for r in pq.read_table(verify(ev['files']['official_catalog_sidecar.parquet']),
        filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    acceptable={}
    for r in pq.read_table(verify(ev['files']['acceptable_operational.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist():
        if r['query_id'] in ids: acceptable.setdefault(r['query_id'],set()).add(r['siret'])
    if set(labels)!=ids or set(catalog)!=ids or set(acceptable)!=ids: raise ValueError('Incomplete official label side')
    outcomes=[]
    for r in generated:
        label=labels[r['query_id']]
        if r['fold']!=label['fold'] or r['fold'] not in (2,3,4): raise ValueError('Evaluation fold drift')
        row=dict(query_id=r['query_id'],**{c:label[c] for c in v38.SEGMENTS},
            primary_source=r['primary_source'],ban_declared_weak=r['ban_declared_weak'])
        for mode in v38.MODES:
            candidates=r['results'][mode]; found={c['siret'] for c in candidates}
            if len(found)!=len(candidates) or len(found)>100: raise ValueError('Invalid initial cap')
            hit=label['human_ground_truth_siret'] in found
            row[mode+'_exact']=hit; row[mode+'_operational']=bool(found&acceptable[r['query_id']])
            row[mode+'_loss']='HIT' if hit else 'NOT_RETRIEVED_AT100' if catalog[r['query_id']]['official_catalog_present'] else 'OFFICIAL_CATALOG_ABSENT'
        outcomes.append(row)
    segments={}
    for column in (*v38.SEGMENTS,'primary_source','ban_declared_weak'):
        groups={}
        for r in outcomes:
            value=r[column]; key='UNKNOWN' if value is None or value=='' else str(value)
            groups.setdefault(key,[]).append(r)
        segments[column]={key:{mode:v38.summarize(group,mode) for mode in v38.MODES} for key,group in groups.items()}
    spatial_rows=[r for r in generated if r['primary_source']=='SPATIAL_DIRECT']
    result=dict(status='MEASURED_SPATIAL_INITIAL_TOP100_PILOT',decision='PIVOT_GENERATION',
        primary=v38.summarize(outcomes,'primary'),comparator=v38.summarize(outcomes,'comparator'),segments=segments,
        gains=sum(r['primary_exact'] and not r['comparator_exact'] for r in outcomes),
        losses=sum(r['comparator_exact'] and not r['primary_exact'] for r in outcomes),
        loss_attribution={mode:dict(Counter(r[mode+'_loss'] for r in outcomes)) for mode in v38.MODES},
        latency={mode:dict(zip(('p50','p95','p99'),np.quantile([r['seconds'][mode] for r in generated],[.5,.95,.99]).tolist())) for mode in v38.MODES},
        spatial_search_latency=None if not spatial_rows else dict(zip(('p50','p95','p99'),
            np.quantile([r['seconds']['primary'] for r in spatial_rows],[.5,.95,.99]).tolist())),
        latency_scope='Loaded tree native search or lexical fallback; excludes batch BAN resolution and index loads; cache uncontrolled, NOT production SLO',
        ban_resolution_seconds=launch['ban_resolution_seconds'],spatial_queries=len(spatial_rows),
        spatial_sites=launch['spatial_sites'],national_catalog_sites=launch['full_national_source_rows'],
        spatial_index_has_missing_sites=True,geo_only_is_not_joint_metric=True,development_panel_reused=True,
        excluded_missing_human_labels=ev['excluded_missing_human_labels'],generation=record(output/'manifest.json'),
        ledger=record(v38.LEDGER),label_side=ev['label_side'],outcomes=outcomes,fine_loss_attribution_measured=False,
        ready_for_fold1=False,fold0_opened=False,fold1_opened=False,model_used=True,full_query_population=False,
        metric_family='DIRECT_SPATIAL100_OR_FROZEN_V42_INITIAL_SEARCH',implementation=record(Path(__file__)))
    if result['primary']['exact_recall']>=.992:
        result['decision']='STOP'; result['decision_reason']='PILOT_ONLY_NO_FULL_OOF_GENERALIZATION_SLO_GATE'
    write_json(dest,result)
    print(json.dumps({k:result[k] for k in ('primary','comparator','gains','losses','latency','spatial_search_latency','spatial_queries','decision')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['build','run','worker','evaluate'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--index',type=Path); p.add_argument('--start',type=int)
    args=p.parse_args()
    if args.mode=='build': build(args.output)
    elif args.mode=='evaluate': evaluate(args.output)
    elif args.mode=='run':
        if args.index is None: p.error('--index required')
        run(args.output,args.index)
    else:
        if args.start not in range(0,128,8): p.error('Invalid start')
        worker(args.output,args.start)
