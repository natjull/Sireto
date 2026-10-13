"""Label-side diagnosis after COMPLETE V81; never retrieval inputs or a pool.

Every526 retained. Candidate concentration is measured on already sealed100;
GT feature lookup is explicitly diagnostic and cannot modify any recipe.
"""
from collections import Counter
from pathlib import Path
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v32_learned_query as label_source
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE=io.HOME/'v81/indirect_lexical_query_20261013_a'
BASE=io.HOME/'v32/global_public_parent_names_20261012_a'
OUTPUT=io.HOME/'v85/postseal_proof_gap_20261013_a'


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh label-side audit')
    refs={}
    for name,root in [('baseline',BASE),('source',SOURCE)]:
        generation=io.read(root/'generation.json');owned=io.read(root/'generate_exit.json')
        if generation['status']!='COMPLETE_SEALED_ALL526' or owned['exit_code'] or not owned['own_child_exit_confirmed']:raise ValueError('Complete owned generation before any diagnostic labels')
        verify(generation['outputs']);refs[name]=dict(generation=record(root/'generation.json'),owned_exit=record(root/'generate_exit.json'),outcomes=record(root/'outcomes.json'))
    raw,controls,label_refs=label_source.calibration_sources();labels={r['query_id']:r for r in raw}
    launch=io.read(SOURCE/'launch.json');panel={r['query_id']:r for r in io.read(verify(launch['panel']))}
    source={r['query_id']:r for r in io.read(verify(refs['source']['outcomes']))['learned_native_ooc']}
    baseline={r['query_id']:r for r in io.read(verify(refs['baseline']['outcomes']))['learned_native_ooc']}
    outputs=io.read(verify(io.read(SOURCE/'generation.json')['outputs']))['learned_native_ooc']
    feature=io.read(SOURCE/'features_generation.json');cal=io.read(SOURCE/'calibration/generation.json')
    assignment=io.read(verify(cal['assignment']));models={k:io.read(verify(v)) for k,v in cal['models'].items()}
    if set(panel)!=set(source) or len(source)!=526:raise ValueError('All526 exact denominator')
    rows=[]
    for qid,r in source.items():
        mandate.guard(mandate.DEADLINE)
        if labels[qid]['fold'] not in (2,3,4):raise ValueError('Closed fold')
        gt=int(labels[qid]['siret']);cs=outputs[qid];parents=Counter(c['siren'] for c in cs)
        detail=dict(query_id=qid,query=panel[qid],label_siret=str(gt).zfill(14),
            exact=r['exact'],siren=r['siren'],operational=r['operational'],baseline_exact=baseline[qid]['exact'],
            weight=r['calibrated_query_weight'],stratum=r['stratum'],fold=r['fold'],
            output100_distinct_sirens=len(parents),output100_max_sites_one_siren=max(parents.values(),default=0),
            gt_siren_sites_in_output=parents.get(str(gt//100000).zfill(9),0),
            output100_cut_score=cs[-1]['score'] if cs else None,diagnostic_only_never_search_input=True)
        if not r['exact'] or r['exact']!=baseline[qid]['exact']:
            with np.load(verify(feature['features'][qid]),allow_pickle=False) as f:
                keys=f['keys'];at=int(np.searchsorted(keys,gt));inside=at<len(keys) and keys[at]==gt
                x=f['features'][at].copy() if inside else np.zeros(f['features'].shape[1],dtype=np.float32)
            m=models[str(assignment[qid])];coef=np.asarray(m['coefficients'][-len(x):]);fields=m['feature_fields'][-len(x):]
            detail.update(in_complete_nonlexical_scope=bool(inside),gt_numerical_score=float(x@coef),
                gt_proofs={name:dict(value=float(v),coefficient=float(w),contribution=float(v*w)) for name,v,w in zip(fields,x,coef)})
        rows.append(detail)
    summary={}
    for name,wanted in [('hits',True),('misses',False)]:
        rs=[r for r in rows if r['exact']==wanted]
        summary[name]=dict(queries=len(rs),
            distinct_sirens_quantiles=np.quantile([r['output100_distinct_sirens'] for r in rs],[0,.5,.9,1]).tolist(),
            max_same_siren_sites_quantiles=np.quantile([r['output100_max_sites_one_siren'] for r in rs],[0,.5,.9,1]).tolist(),
            same_siren_known=sum(r['siren'] for r in rs),operational_correct=sum(r['operational'] for r in rs))
    OUTPUT.mkdir(parents=True);io.write(OUTPUT/'cases.json',rows)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_POSTSEAL_LABELSIDE_PROOF_AND_CONCENTRATION_AUDIT',
        implementation=record(Path(__file__)),sources=refs,label_sources=label_refs,cases=record(OUTPUT/'cases.json'),
        summary=summary,all526_retained=True,no_retrieval_or_model_modified=True,no_positive_injected_into_retrieval=True,
        no_new_recall_result=True,closed_test_opened=False,observed_unix=time.time()))
    print(summary,flush=True)


if __name__=='__main__':main()
