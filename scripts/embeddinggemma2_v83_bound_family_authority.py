"""Five outcome-blind cached queries: can exact lazy scoring avoid encoding?

Not a retrieval run or a speed promise. Keep all parent identity scores known
(optimistic after a separate parent encoding pass), replace every unencoded
own-site cosine by its rigorous bound1, propagate the ORIGINAL direct edges.
The final100 cut is from a complete sealed old generation, NOT ground truth.
Count mandatory potential refinements even with lexical scores omitted: this
is a LOWER bound on workload, not a topk prepool or recall/speed measurement.
"""
import hashlib
from pathlib import Path
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v23_native_features as original
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v83/bound_family_authority_20261013_a'
BASE=io.HOME/'v32/global_public_parent_names_20261012_a'


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh label-free feasibility diagnosis')
    generation=io.read(BASE/'generation.json');ex=io.read(BASE/'generate_exit.json')
    if ex['exit_code'] or not ex['own_child_exit_confirmed'] or generation['status']!='COMPLETE_SEALED_ALL526':raise ValueError('Complete old score authority')
    launch=io.read(BASE/'launch.json');features=io.read(BASE/'features_generation.json')
    panel=io.read(verify(launch['panel']));cache=io.read(verify(launch['feature_launch']))
    source=io.read(verify(launch['source_launch']));sampling=io.read(verify(source['sampling']))
    chosen=[]
    for stratum in ('SMALL','MEDIUM','DENSE','LARGE','VERY_LARGE'):
        candidates=[q for q in panel if sampling['zone_weights'][q['crm_insee']]['stratum']==stratum]
        if not candidates:raise ValueError('Every density stratum retained')
        q=min(candidates,key=lambda q:hashlib.sha256(('lazy-bound-density-blind-v1|'+q['query_id']).encode()).digest())
        chosen.append((stratum,q))
    outputs=io.read(verify(generation['outputs']))['learned_native_ooc']
    cal=io.read(BASE/'calibration/generation.json');assignment=io.read(verify(cal['assignment']))
    geo=io.read(verify(launch['geo_sources']))
    family=io.read(verify(features['family_sources']));projected=family['compiled_links'];verify(projected)
    edges=original.EdgeViews(launch,projected);rows=[]
    try:
        for stratum,q in chosen:
            mandate.guard(mandate.DEADLINE);qid=q['query_id']
            directory=original.CACHE/'zones'/(q['crm_insee'] or 'UNKNOWN')
            binding=io.read(engine.sealed(directory/'features_binding.json'))
            own_keys=np.asarray(binding['sirets'],dtype=np.uint64)
            with np.load(verify(features['features'][qid]),allow_pickle=False) as f:keys=f['keys'];x=f['features'];u=x.copy()
            own=np.searchsorted(keys,own_keys)
            if (own>=len(keys)).any() or not np.array_equal(keys[own],own_keys):raise ValueError('Every unpruned own corpus leaf included')
            u[own,0]=1.;u[:,6]=u[:,0]*u[:,5]
            geo_keys=np.asarray([r[0] for r in geo[qid]],dtype=np.uint64)
            source_family=original.expand_families(edges.keys,np.union1d(own_keys//100000,geo_keys//100000))
            selection=edges.select(source_family)
            for col in (14,15,21):u[:,col]=0.
            if len(selection):
                fr=np.searchsorted(keys,edges.sources[selection]);to=np.searchsorted(keys,edges.targets[selection])
                if (fr>=len(keys)).any() or (to>=len(keys)).any():raise ValueError('Original direct edge closure')
                dense=.75*u[fr,0]+.25*u[fr,1]
                for col,values in ((14,dense),(15,dense*edges.close[selection]),(21,u[fr,0])):np.maximum.at(u[:,col],to,values)
            if np.any(u+1e-6<x):raise ValueError('Numerical bound must dominate EVERY original proof value')
            m=io.read(verify(cal['models'][str(assignment[qid])]))
            w=np.asarray(m['coefficients'][-22:]);numeric_upper=u.astype(float)@w
            cut=float(outputs[qid][-1]['score']);error=float(w.sum()/((1<<original.RECIPE['quantization_bits'])-1))
            # No lexical term can decrease a nonnegative weighted query score.
            mandatory=int((numeric_upper[own]>=cut+2*error+1e-5).sum())
            rows.append(dict(query_id=qid,stratum=stratum,full_own_sites=len(own_keys),
                unique_own_parents=len(np.unique(own_keys//100000)),full_nonlexical_leaves=len(keys),
                complete_old_cut100=cut,upper_bound_dominates_all_original_values=True,
                minimum_potential_own_refinements_with_parent_scores_already_known=mandatory,
                minimum_fraction=mandatory/len(own_keys) if len(own_keys) else None,
                lexical_scores_omitted_makes_workload_bound_optimistic=True,
                online_threshold_unknown_further_increases_workload=True,no_labels_read=True))
    finally:edges.close_files()
    OUTPUT.mkdir(parents=True);io.write(OUTPUT/'diagnostic.json',dict(implementation=record(Path(__file__)),
        source_generation=record(BASE/'generation.json'),source_features=record(BASE/'features_generation.json'),
        rows=rows,selection='ONE_HASH_PER_DENSITY_STRATUM_BEFORE_SCORE_INSPECTION',
        all_parent_encoding_cost_not_included=True,exact_original_projected_edges=projected,no_claim_of_exact_lazy_runtime_or_speedup=True,
        no_new_recall=True,no_new_candidates=True,no_pristine_validation_opened=True,observed_unix=time.time()))
    print(rows,flush=True)


if __name__=='__main__':main()
