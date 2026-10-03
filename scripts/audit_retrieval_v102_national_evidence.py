"""Post-seal all128 point audit of complete national V99/V101 scope."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import audit_retrieval_v91_scoped_evidence as core
from scripts.retrieval_v101_national_paths import official_sources
from src.xgb_matcher.retrieval_national_paths_v101 import QueryWordPostings
from src.xgb_matcher.retrieval_national_paths_v99 import FullRegistryWithRelations,NationalScopedPaths
from src.xgb_matcher.retrieval_national_membership_v96 import NationalMembership

BATCH=16


def branch_reason(proof,query_geo_available,xy_available):
    if proof['outside_national_site_scope']: return 'OUTSIDE_FULL_NATIONAL_REGISTRY'
    if not proof['score']: return 'NO_MATCHING_LITERAL_WORD_PROOF_IN_FULL_SCOPED_NAMES'
    if not query_geo_available: return 'QUERY_BAN_UNAVAILABLE_OR_INCOMPATIBLE'
    if not xy_available: return 'TARGET_XY_ABSENT'
    return 'POSITIVE_NATIONAL_SCOPED_WORD_SITE_PROOF'


def check_point(result,gt,full):
    hit,cutoff=core.check_point(result,gt,full)
    if (not hit and len(result)==100 and full>0 and full==cutoff and
            gt<result[-1]['siret']):
        raise ValueError('STOP_CANONICAL_TIE_POINT_AUDIT_CONTRADICTION')
    return hit,cutoff



def population(generation):
    manifest=json.loads((generation/'manifest.json').read_text())
    evaluation=json.loads((generation/'evaluation.json').read_text())
    launch=json.loads(core.verify(manifest['launch']).read_text())
    core.v38.verify_launch(launch)
    rows=[r for ref in manifest['batches'] for r in json.loads(core.verify(ref).read_text())['rows']]
    core.validate_population(manifest,evaluation,generation,launch['queries'],rows)
    if any(r['results']['primary']!=sorted(r['results']['primary'],key=lambda x:(-x['score'],x['siret'])) for r in rows):
        raise ValueError('Initial100 global canonical output order drift')
    return evaluation,launch,{r['query_id']:r for r in rows}


def worker(output,start):
    audit=json.loads((output/'launch.json').read_text())
    for ref in audit['implementation'].values(): core.verify(ref)
    generation=Path(core.verify(audit['generation']).parent)
    core.verify(audit['evaluation'])
    evaluation,launch,emitted=population(generation)
    # No label may be opened before the complete blind population above passes.
    labels={r['query_id']:r for r in core.pq.read_table(core.verify(evaluation['ledger']),
            filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in emitted}
    if set(labels)!=set(emitted): raise ValueError('All-human audit denominator drift')
    models=json.loads(core.verify(launch['models']).read_text())
    terms={r['query_id']:r['terms'] for r in json.loads(core.verify(launch['expansions']).read_text())['rows']}
    geos={r['query_id']:r for r in json.loads(core.verify(launch['geo_queries']).read_text())}
    rows=[]; began=time.monotonic(); budget=None
    try:
        with core.Budget(dict(max_seconds=80,max_rss_bytes=6442450944,max_swapout_pages_delta=0)) as budget:
            words,_,_=official_sources(launch)
            index=QueryWordPostings(words['postings'],[core.v42.as_query(q).name for q in launch['queries'][start:start+BATCH]])
            owners=core.OwnerFacts(Path(words['owner_directory']))
            graph=FullRegistryWithRelations(NationalMembership(Path(launch['membership_manifest']['path']).parent),
                core.SuccessionCSR(Path(launch['graph_manifest']['path']).parent))
            graph.validate_relation_membership()
            tree,keys,_=core.v46.load_tree(launch); base=core.TantivyBackend(core.v38.INDEX)
            if base.searcher.num_docs!=launch['index_documents']: raise ValueError('Native corpus drift')
            for q in launch['queries'][start:start+BATCH]:
                label=labels[q['query_id']]; gt=label['human_ground_truth_siret']; parent=label['human_ground_truth_siren']
                if label['fold']!=q['fold'] or not gt.startswith(parent): raise ValueError('Official label membership/fold drift')
                scores,_=core.literal_scores(index,core.v42.as_query(q).name); paths=NationalScopedPaths(owners,graph,scores)
                proof=paths.best(gt); g=geos[q['query_id']]; point,_,reliable=core.geo_parameters(g)
                factor=1. if reliable else .5
                pos=core.typed_position(keys,int(gt)); xy=pos<len(keys) and keys[pos]==int(gt)
                site=float(core.geo_features(keys,tree.data,[gt],g)[0,0]*factor)
                m=models[str(q['fold'])]
                native=core.NativeStream(base,core.native_query(base,core.v42.as_query(q),m['weights'],terms[q['query_id']],m['alpha']))
                point_began=time.monotonic(); ns=native.complete([gt])[gt]; native_seconds=time.monotonic()-point_began
                graph_score=launch['path_weight']*proof['score']*site; full=max(ns,graph_score)
                hit,cutoff=check_point(emitted[q['query_id']]['results']['primary'],gt,full)
                rows.append(dict(query_id=q['query_id'],fold=q['fold'],human_siret=gt,human_siren=parent,
                    exact_hit=hit,branch_zero_or_positive_reason=branch_reason(proof,point is not None,xy),
                    native_score=ns,path_name_score=proof['score'],site_score=site,graph_score=graph_score,initial_score=full,
                    initial100_cutoff=cutoff,target_xy_available=bool(xy),query_geo_available=point is not None,
                    ban_reliability_factor=factor,scoped_name_proof=proof,
                    own_site_name_facts=len(owners.references('site',int(gt))),
                    scoped_parent_name_facts=len(owners.references('parent',int(parent))),
                    target_score_at_boundary=full==cutoff,native_point_seconds=native_seconds))
                print(json.dumps(dict(audited=start+len(rows),native_point_seconds=native_seconds)),flush=True)
        core.v38.verify_launch(launch); official_sources(launch)
        for ref in audit['implementation'].values(): core.verify(ref)
        core.write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report(),
            generation=audit['generation'],evaluation=audit['evaluation'],seconds=time.monotonic()-began,
            truth_scope='EVALUATOR_ONLY_AFTER_SEALED128',generation_modified=False))
    except BaseException as exc:
        core.write_json(output/f'worker_stop{start:03d}.json',dict(status='STOP',reason=str(exc),
            completed=len(rows),resources=budget.report() if budget is not None else None,
            generation_modified=False,truth_scope='EVALUATOR_ONLY')); raise


def assemble(queries,evaluation,batches,generation_ref,evaluation_ref):
    rows=[]
    for b in batches:
        if (not b['resources']['pass_resource_policy'] or b['generation']!=generation_ref or
                b['evaluation']!=evaluation_ref or b['generation_modified'] or
                b['truth_scope']!='EVALUATOR_ONLY_AFTER_SEALED128' or len(b['rows'])!=BATCH):
            raise ValueError('Audit batch authority/resource drift')
        rows.extend(b['rows'])
    if (len(rows)!=128 or [(r['query_id'],r['fold']) for r in rows]!=[(q['query_id'],q['fold']) for q in queries]
            or sum(r['exact_hit'] for r in rows)!=evaluation['primary']['exact_hits']):
        raise ValueError('Point audit population/hits drift')
    return rows


def run(generation,output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    evaluation,launch,_=population(generation)
    audit=dict(generation=core.record(generation/'manifest.json'),evaluation=core.record(generation/'evaluation.json'),
        implementation={str(p):core.record(p) for p in (Path(__file__),Path(core.__file__),
            Path(__file__).parents[1]/'src/xgb_matcher/retrieval_national_paths_v99.py',
            Path(__file__).parents[1]/'src/xgb_matcher/retrieval_national_paths_v101.py',
            Path(__file__).parents[1]/'docs/retrieval_v102_national_evidence_audit.md')},
        queries=launch['queries'],batch_size=BATCH,worker_seconds=80,supervisor_seconds=90,controller_seconds=900,
        generation_modified=False,truth_scope='EVALUATOR_ONLY_AFTER_SEALED128')
    core.write_json(output/'launch.json',audit); batches=[]; refs=[]
    try:
        for start in range(0,128,BATCH):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.audit_retrieval_v102_national_evidence','worker',
                    '--output',str(output),'--start',str(start)],stdout=log,stderr=subprocess.STDOUT,check=True,
                    timeout=min(90,max(1,900-(time.monotonic()-began))),
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','RAYON_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
            ref=core.record(output/f'batch{start:03d}.json'); refs.append(ref)
            batches.append(json.loads(core.verify(ref).read_text()))
            print(json.dumps(dict(audited=start+BATCH,seconds=time.monotonic()-began)),flush=True)
        rows=assemble(launch['queries'],evaluation,batches,audit['generation'],audit['evaluation'])
        core.verify(audit['generation']); core.verify(audit['evaluation']); core.v38.verify_launch(launch)
        official_sources(launch)
        core.write_json(output/'report.json',dict(status='SEALED_EVALUATION_ONLY_POINT_PROOF_AUDIT',queries=128,
            misses=sum(not r['exact_hit'] for r in rows),rows=rows,batches=refs,
            miss_branch_reasons=dict(Counter(r['branch_zero_or_positive_reason'] for r in rows if not r['exact_hit'])),
            loss_attribution_scope='THIS_SCOPED_BRANCH_ONLY_NOT_ABSENCE_OF_ANY_OFFICIAL_SOURCE_EVIDENCE',
            generation=audit['generation'],evaluation=audit['evaluation'],launch=core.record(output/'launch.json'),
            seconds=time.monotonic()-began,truth_opened=True,truth_scope='EVALUATOR_ONLY_AFTER_SEALED128',
            fold0_opened=False,fold1_opened=False,generation_modified=False,model_fit=False,
            outputs_promoted_to_features=False,ready_for_fold1=False))
        print(json.dumps(dict(misses=sum(not r['exact_hit'] for r in rows),miss_branch_reasons=Counter(
            r['branch_zero_or_positive_reason'] for r in rows if not r['exact_hit']))),flush=True)
    except BaseException as exc:
        core.write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,
            generation_modified=False,truth_scope='EVALUATOR_ONLY')); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--generation',type=Path); p.add_argument('--output',required=True,type=Path); p.add_argument('--start',type=int)
    a=p.parse_args()
    if a.mode=='worker':
        if a.start not in range(0,128,BATCH): p.error('Invalid audit worker start')
        worker(a.output,a.start)
    else:
        if a.generation is None: p.error('Generation required')
        run(a.generation,a.output)
