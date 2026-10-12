"""Compare real sealed retrieval outputs without evaluation labels or unit tests."""
import hashlib
from pathlib import Path
import numpy as np
from scripts import embeddinggemma2_v40_learned_query as run
from scripts import embeddinggemma2_v40_native_features as proof

def main():
    out=proof.OUTPUT
    run.calibrate(out)
    proof.bind_runtime(out)
    launch=proof.authority(out)
    gen=run.io.read(out/'features_generation.json')
    panel=run.io.read(run.verify(launch['panel']))
    old=run.io.HOME/'v37/global_identity_existing_heads_20261012_a/native'
    backend=run.TantivyBackend(run.io.INDEX)
    comparisons=[]
    for q in panel[:8]:
        path=old/(hashlib.sha256(q['query_id'].encode()).hexdigest()+'.json')
        sealed=run.engine.sealed(path)
        if not sealed: raise ValueError('Source real output not sealed')
        prior=run.io.read(sealed)
        model=run.io.read(run.verify(prior['model']))
        with np.load(run.verify(gen['features'][q['query_id']]),allow_pickle=False) as f:
            rows,cost=proof.native_search(backend,q,f['keys'],f['features'],model['coefficients'])
        same_ids=[r['siret'] for r in rows]==[r['siret'] for r in prior['candidates']]
        delta=max((abs(a['score']-b['score']) for a,b in zip(rows,prior['candidates'])),default=0.)
        result=dict(query_id=q['query_id'],same_ordered_sirets=same_ids,max_abs_score_delta=delta,
            source_seconds=prior['cost']['total_seconds'],new_cost=cost,source=run.record(path))
        comparisons.append(result)
        run.io.write(out/'real_equivalence.json',dict(status='RUNNING',labels_read=False,comparisons=comparisons))
        print(result,flush=True)
        if not same_ids or delta>1e-5: raise ValueError('Real retrieval equivalence rejected')
    run.io.write(out/'real_equivalence.json',dict(status='PASS_REAL_SEALED8',labels_read=False,
        new_hypothesis=False,comparisons=comparisons))

if __name__=='__main__':main()
