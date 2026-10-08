"""Reusable NATIONAL initial100 baseline on ALL validation queries.

This is not the dense-family result. Generation opens only query-only census,
official index and implementation hashes. All 2212 outputs must precede labels.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v10_representative as population
from scripts import embeddinggemma2_v3_compare as paired
from scripts import retrieval_v38_initial_top100 as lexical
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend, RetrievalQuery
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v12'


def generate(out):
    import psutil
    launch = io.read(out / 'launch.json')
    for ref in launch['implementation']: verify(ref)
    panel = io.read(verify(launch['panel']))
    if len(panel) != 2212 or any(q['fold'] not in (2,3,4) for q in panel): raise ValueError('Entire safe population')
    backend = TantivyBackend(io.INDEX)
    if backend.searcher.num_docs != 82772691: raise ValueError('National index drift')
    before = paired.old.inventory(); started = time.monotonic(); refs = []; rows = []
    for lo in range(0, len(panel), 16):
        seal = out / f'batch_{lo:04d}.seal.json'
        if seal.exists():
            ref = io.read(seal); shard = io.read(verify(ref))
        else:
            free, level = io.pressure()
            if free < 20 or level not in (0,1,2) or psutil.Process().memory_info().rss > 8*1024**3:
                raise RuntimeError('Real resource protection; shards retained for resume')
            shard = []
            for q in panel[lo:lo+16]:
                rq = RetrievalQuery(
                    name=paired.data.safe(q['crm_name']),
                    address=' '.join(paired.data.safe(q[k]) for k in ('crm_number','crm_suffix','crm_street')),
                    number=q['crm_number'], insee=q['crm_insee'], postcode=q['crm_postcode'])
                t = time.monotonic(); cs = lexical.search(backend, rq, 'primary')
                if len(cs)>100 or len({r['siret'] for r in cs})!=len(cs): raise ValueError('Initial100 cap')
                shard.append(dict(query_id=q['query_id'], fold=q['fold'], candidates=cs, seconds=time.monotonic()-t))
            path = out / f'batch_{lo:04d}.json'; io.write(path, shard); ref=record(path); io.write(seal, ref)
        if [r['query_id'] for r in shard] != [q['query_id'] for q in panel[lo:lo+16]]: raise ValueError('Resume query binding')
        refs.append(ref); rows.extend(shard)
        io.write(out / 'state.json', dict(status='FULL_VALIDATION_NATIONAL_LEXICAL_INITIAL100',
            queries=len(rows), total=len(panel), seconds=time.monotonic()-started, dense_measured=False))
    if before != paired.old.inventory(): raise ValueError('Index drift')
    for ref in launch['implementation']: verify(ref)
    io.write(out / 'outputs.json', rows)
    io.write(out / 'generation.json', dict(status='SEALED_ALL2212_NATIONAL_LEXICAL_INITIAL100',
        outputs=record(out / 'outputs.json'), shards=refs, queries=len(rows), truth_opened=False,
        test_opened=False, dense_measured=False, seconds=time.monotonic()-started))


def evaluate(out):
    ex=io.read(out / 'owned_exit.json'); gen=io.read(out / 'generation.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed'] or gen['queries']!=2212:
        raise ValueError('All outputs and owned exit required before labels')
    rows=io.read(verify(gen['outputs'])); scope={r['query_id'] for r in rows}
    manifest=io.read(population.SOURCE / 'manifest.json'); membership=population.SOURCE / 'validation_membership.json'
    if io.digest(membership)!=manifest['split_hashes']['validation']: raise ValueError('Label drift')
    labels={r['query_id']:r for r in io.read(membership)}
    if set(labels)!=scope or len(rows)!=2212: raise ValueError('Complete denominator')
    cfg=io.read(io.ROOT / 'config/retrieval_v358_postseal_evaluation.json')
    ev=io.read(verify(cfg['evaluation'])); side=io.read(verify(ev['label_side']))
    if not side['never_used_in_retrieval'] or side['ledger']!=cfg['ledger']: raise ValueError('Official sidecar')
    import pyarrow.parquet as pq
    accepted={q:{labels[q]['siret']} for q in scope}
    for r in pq.read_table(verify(side['files']['acceptable_operational.parquet']),
            filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))],use_threads=False).to_pylist(): accepted[r['query_id']].add(r['siret'])
    meta={r['query_id']:r for r in pq.read_table(verify(cfg['ledger']),
        filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))],use_threads=False).to_pylist()}
    outcomes=[]; roles=io.read(verify(io.read(out/'launch.json')['roles']))
    for row in rows:
        q=row['query_id']; cs=row['candidates']; keys={c['siret'] for c in cs}; gt=labels[q]
        outcomes.append(dict(query_id=q, siren_group=gt['siren'], fold=gt['fold'],
            exact=gt['siret'] in keys, siren=gt['siren'] in {c['siren'] for c in cs},
            operational=bool(keys & accepted[q]), role=roles[q]))
    segments={}
    for field in ('fold','role','qualification_v2','qualification_v3','data_origin','ground_truth_state'):
        groups=defaultdict(list)
        for r in outcomes: groups[str(r.get(field,meta.get(r['query_id'],{}).get(field,'UNKNOWN')))].append(r)
        segments[field]={k:paired.summarize(v) for k,v in groups.items()}
    # Company-group bootstrap, retaining all sites whenever a company is drawn.
    import numpy as np
    groups=defaultdict(list)
    for r in outcomes: groups[r['siren_group']].append(r['exact'])
    counts=np.array([(sum(v),len(v)) for v in groups.values()]); rng=np.random.default_rng(20261008)
    draws=[]
    for _ in range(2000):
        sampled=counts[rng.integers(0,len(counts),len(counts))].sum(axis=0); draws.append(sampled[0]/sampled[1])
    io.write(out/'outcomes.json',outcomes)
    io.write(out/'report.json',dict(status='MEASURED_FULL2212_NATIONAL_LEXICAL_BASELINE_NOT_DENSE_FUSION',
        metrics=paired.summarize(outcomes), siren_cluster_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
        segments=segments, dense_family_measured=False, all_queries_in_denominator=True,
        validation_used_for_checkpoint_selection=True, pristine_pipeline_holdout=False,
        national_dense_99percent_validated=False,test_opened=False,fold0_opened=False,fold1_opened=False,
        downstream_models_changed=False))
    io.write(out/'state.json',dict(status='COMPLETE_FULL_VALIDATION_LEXICAL_BASELINE',report=record(out/'report.json')))


def run(census, out):
    source=io.read(census/'census.json'); verify(source['sources']['panel'])
    if not out.exists():
        out.mkdir(parents=True)
        io.write(out/'launch.json',dict(census=record(census/'census.json'),panel=source['sources']['panel'],
            roles=source['sources']['roles'],implementation=[record(Path(__file__)),record(Path(lexical.__file__))],
            test_opened=False,dense_measured=False))
    elif io.read(out/'launch.json')['census']!=record(census/'census.json'): raise ValueError('Resume census binding')
    command=[sys.executable,'-u','-m','scripts.embeddinggemma2_v12_full_validation_lexical','generate','--output',str(out)]
    with (out/'generation.log').open('a') as log: child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    io.write(out/'owned_exit.json',dict(exit_code=child.returncode,own_child_exit_confirmed=True,command=command))
    child.check_returncode(); evaluate(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['run','generate']); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--census',type=Path)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated V12 namespace')
    if args.action=='run': run(args.census,args.output)
    else: generate(args.output)
