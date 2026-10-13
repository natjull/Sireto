"""Post-seal attribution of COMPLETE V57/V68; no active experiment labels.

Positive-feature lookup here is label-side diagnosis, never search input.
New cue presence or a numerical contribution is not a causal recall claim:
lexical coefficients, negatives and the admission threshold also changed.
"""
from collections import defaultdict
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v32_learned_query as labels_source
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v76/postseal_source_attribution_20261013_a'
SOURCES={'baseline':io.HOME/'v32/global_public_parent_names_20261012_a',
    'corporate':io.HOME/'v57/corporate_site_nullsafe_20261013_a',
    'filled_geo':io.HOME/'v68/filled_own_site_geo_20261013_a'}


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh post-seal diagnosis')
    sources={}
    for name,path in SOURCES.items():
        generation=io.read(path/'generation.json');ex=io.read(path/'generate_exit.json')
        if generation['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Every output complete before labels')
        sources[name]=dict(generation=record(path/'generation.json'),owned_exit=record(path/'generate_exit.json'),
            outputs=io.read(verify(generation['outputs']))['learned_native_ooc'],
            outcomes={r['query_id']:r for r in io.read(path/'outcomes.json')['learned_native_ooc']},
            features=io.read(path/'features_generation.json'),calibration=io.read(path/'calibration/generation.json'))
    labels,controls,refs=labels_source.calibration_sources();labels={r['query_id']:r for r in labels}
    scope=set(sources['baseline']['outputs'])
    if len(scope)!=526 or any(set(s['outputs'])!=scope for s in sources.values()):raise ValueError('All same526 denominator')
    OUTPUT.mkdir(parents=True);rows=[];assignments={};models={}
    for name,s in sources.items():
        assignments[name]=io.read(verify(s['calibration']['assignment']))
        models[name]={k:io.read(verify(r)) for k,r in s['calibration']['models'].items()}
    for qid in sorted(scope):
        mandate.guard(mandate.DEADLINE);gt=int(labels[qid]['siret']);detail={}
        if labels[qid]['fold'] not in (2,3,4):raise ValueError('Closed folds')
        for name,s in sources.items():
            with np.load(verify(s['features']['features'][qid]),allow_pickle=False) as f:
                keys=f['keys'];at=int(np.searchsorted(keys,gt));inside=at<len(keys) and keys[at]==gt
                x=f['features'][at].copy() if inside else np.zeros(f['features'].shape[1],dtype=np.float32)
            model=models[name][str(assignments[name][qid])];numeric=np.asarray(model['coefficients'][-len(x):])
            fields=model['feature_fields'][-len(x):];extra=x[22:];cs=s['outputs'][qid]
            selected=next((r for r in cs if int(r['siret'])==gt),None)
            detail[name]=dict(in_full_proof_universe=bool(inside),exact=s['outcomes'][qid]['exact'],
                original_own_geo=float(x[5]),new_feature_max=float(extra.max()) if len(extra) else 0.,
                full_numerical_score=float(x@numeric),new_numerical_contribution=float(extra@numeric[22:]),
                own_address_proof=float(x[3]),site_cosine=float(x[0]),parent_cosine=float(x[1]),
                gt_output_score=selected['score'] if selected else None,
                output100_cut_score=cs[-1]['score'] if cs else None,
                new_features={n:float(v) for n,v in zip(fields[22:],extra)},
                lexical_coefficients=model['coefficients'][:-len(x)])
        rows.append(dict(query_id=qid,label_siret=str(gt).zfill(14),label_siren=labels[qid]['siren'],
            fold=labels[qid]['fold'],calibrated_query_weight=sources['baseline']['outcomes'][qid]['calibrated_query_weight'],
            variants=detail,postseal_only_never_retrieval_input=True))
    summary={}
    for name in ('corporate','filled_geo'):
        groups={}
        for segment,wanted in [('baseline_misses',False),('baseline_hits',True)]:
            rs=[r for r in rows if r['variants']['baseline']['exact']==wanted]
            groups[segment]=dict(queries=len(rs),new_cue_present=sum(r['variants'][name]['new_feature_max']>0 for r in rs),
                new_cue_contributes_positive_score=sum(r['variants'][name]['new_numerical_contribution']>0 for r in rs),
                original_geo_zero_with_new_cue=sum(r['variants'][name]['original_own_geo']==0 and r['variants'][name]['new_feature_max']>0 for r in rs),
                genuine_new_geo_present=sum(r['variants'][name]['new_features'].get('filled_own_site_geo',0)>0 for r in rs))
        gain=[r['query_id'] for r in rows if not r['variants']['baseline']['exact'] and r['variants'][name]['exact']]
        loss=[r['query_id'] for r in rows if r['variants']['baseline']['exact'] and not r['variants'][name]['exact']]
        summary[name]=dict(segments=groups,gains=gain,losses=loss,
            lost_cases_have_new_positive_cue=sum(r['variants'][name]['new_numerical_contribution']>0 for r in rows if r['query_id'] in loss))
    io.write(OUTPUT/'cases.json',rows)
    io.write(OUTPUT/'summary.json',dict(status='COMPLETE_POSTSEAL_SOURCE_ATTRIBUTION_NOT_NEW_RETRIEVAL',
        implementation=record(__import__('pathlib').Path(__file__)),summary=summary,cases=record(OUTPUT/'cases.json'),
        source_generations={k:s['generation'] for k,s in sources.items()},label_sources=refs,
        every526_retained=True,source_presence_not_causal_recall_proof=True,
        no_active_v71_v73_v75_labels_read=True,no_prediction_or_model_changed=True,
        no_positive_injected_in_search=True,observed_unix=time.time()))
    print(summary,flush=True)


if __name__=='__main__':main()
