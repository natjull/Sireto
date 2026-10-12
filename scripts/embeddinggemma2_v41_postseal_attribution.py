"""Label-side diagnosis after COMPLETE owned generation; never retrieval input."""
from collections import Counter
import hashlib
from pathlib import Path
import time
import numpy as np
from scripts import embeddinggemma2_v25_learned_query as run
from scripts import embeddinggemma2_v25_native_features as proof

SOURCE=proof.OUTPUT
OUTPUT=run.io.HOME/'v41/postseal_v25_attribution_20261012_a'

def main():
    if OUTPUT.exists():raise ValueError('Fresh audit namespace required')
    launch=proof.authority(SOURCE)
    ex=run.io.read(SOURCE/'generate_exit.json');generation=run.io.read(SOURCE/'generation.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed'] or generation['status']!='COMPLETE_SEALED_ALL526':
        raise ValueError('Owned complete output before labels')
    run.verify(generation['outputs'])
    label_path=run.io.HOME/'v20/deep_audit_20261011_c/rows.json'
    labels={r['query_id']:r['label'] for r in run.io.read(label_path)}
    panel=run.io.read(run.verify(launch['panel']));queries={q['query_id']:q for q in panel}
    if set(labels)!=set(queries) or any(r['fold'] not in (2,3,4) for r in labels.values()):raise ValueError('Same all526 folds234')
    outcomes=run.io.read(SOURCE/'outcomes.json')['learned_native_ooc']
    features=run.io.read(SOURCE/'features_generation.json')
    backend=run.TantivyBackend(run.io.INDEX)
    OUTPUT.mkdir(parents=True)
    run.io.write(OUTPUT/'launch.json',dict(implementation=run.record(Path(__file__)),
        source_generation=run.record(SOURCE/'generation.json'),source_exit=run.record(SOURCE/'generate_exit.json'),
        labels=run.record(label_path),purpose='POSTSEAL_DIAGNOSIS_ONLY_NEVER_RETRIEVAL_FEATURES',
        all526_denominator_unchanged=True,no_label_correction=True,deadline_unix=launch['deadline_unix']))
    rows=[]
    for outcome in outcomes:
        if outcome['exact']:continue
        proof.guard(launch['deadline_unix']);qid=outcome['query_id'];q=queries[qid];label=labels[qid]
        path=SOURCE/'native'/(hashlib.sha256(qid.encode()).hexdigest()+'.json')
        part=run.io.read(run.engine.sealed(path));model=run.io.read(run.verify(part['model']))
        with np.load(run.verify(features['features'][qid]),allow_pickle=False) as data:
            keys=data['keys'];x=data['features'];r=proof.coherence(x)
            at=np.searchsorted(keys,int(label['siret']));covered=at<len(keys) and keys[at]==int(label['siret'])
            target=x[at].copy() if covered else np.zeros(len(proof.source.PROOF_FIELDS))
        n=len(proof.BASE_FEATURE_FIELDS);w=np.asarray(model['coefficients']);effective=r*w[:n]+(1-r)*w[n:]
        lexical=run.masked_lexical_features(backend,q,np.asarray([int(label['siret'])],dtype=np.int64))[0]
        values=np.concatenate((lexical,target));contributions=values*effective
        cutoff=min(z['score'] for z in part['candidates']) if part['candidates'] else None
        row=dict(query_id=qid,crm=q,label=label,calibrated_query_weight=outcome['calibrated_query_weight'],
            target_in_full_proof_stream=bool(covered),query_coherence=r,
            target_features=dict(zip(proof.BASE_FEATURE_FIELDS,map(float,values))),
            target_contributions=dict(zip(proof.BASE_FEATURE_FIELDS,map(float,contributions))),
            target_unquantized_score=float(contributions.sum()),emitted_cutoff_score=cutoff,
            approximate_cutoff_deficit=float(cutoff-contributions.sum()) if cutoff is not None else None,
            approximation_not_a_national_rank=True,source_native=run.record(path))
        rows.append(row)
        run.io.write(OUTPUT/'state.json',dict(status='POSTSEAL_AUDIT_ACTIVE',cases=len(rows),updated_unix=time.time()))
    run.io.write(OUTPUT/'cases.json',rows)
    run.io.write(OUTPUT/'summary.json',dict(status='COMPLETE_POSTSEAL_V25_ATTRIBUTION',
        misses=len(rows),siren_groups=len({r['label']['siren'] for r in rows}),
        absent_from_full_proof_stream=sum(not r['target_in_full_proof_stream'] for r in rows),
        below_native_cutoff_unquantized=sum(r['approximate_cutoff_deficit'] is not None and r['approximate_cutoff_deficit']>1e-4 for r in rows),
        dominant_target_contributions=dict(Counter(max(r['target_contributions'],key=r['target_contributions'].get) for r in rows)),
        cases=run.record(OUTPUT/'cases.json'),label_side_only=True,no_recipe_changed=True))
    print(run.io.read(OUTPUT/'summary.json'),flush=True)

if __name__=='__main__':main()
