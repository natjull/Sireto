import inspect
import numpy as np
import pytest
from tests.test_retrieval_proof_v56 import backend
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_proof_v56 import point_features
from src.xgb_matcher.retrieval_native_v57 import LEXICAL,fit,search

def test_native_initial100_equals_complete_lexical_oracle(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA' if i%3 else 'BETA',addresses='12 RUE BLEUE') for i in range(250)])
    q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',number='12',insee='75056')
    w=np.arange(1,12,dtype=float)*.1; ids=[f'{100000000+i}00001' for i in range(250)]
    _,features=point_features(b,q,ids); found,trace=search(b,q,w)
    scores={s:float(features[i]@w) for i,s in enumerate(ids)}
    cutoff=sorted(scores.values(),reverse=True)[99]
    assert len(found)==100 and len({r['siret'] for r in found})==100
    assert all(np.isclose(r['score'],scores[r['siret']],atol=1e-5) and scores[r['siret']]>=cutoff for r in found)
    assert trace['fixed_reservoir'] is None and trace['native_exact_top100']

def test_fit_native_has_no_geo_coordinates_and_excludes_entire_heldout_entities():
    registry={str(f):[str(100000000+f*100+i) for i in range(20)] for f in (2,3,4)}
    scenes=[]
    for f in (2,3,4):
        for i,parent in enumerate(registry[str(f)]):
            scenes.append(dict(query_id=f'q{f}_{i}',fold=f,positive_siren=parent,positive_features=np.ones(11).tolist(),
                negatives=[dict(siren='900000000',features=np.zeros(11).tolist()),dict(siren=registry['2'][0],features=np.zeros(11).tolist())]))
    model=fit(scenes,2,registry)
    assert model['features']==list(LEXICAL) and len(model['weights'])==11
    assert model['training_queries']==40 and model['training_pairs']==40 and model['heldout_entity_overlap']==0
    assert all(not q.startswith('q2_') for q in model['query_ids'])
    with pytest.raises(ValueError): fit(scenes,0,registry)

def test_serving_never_reads_training_scenes_or_geometry_or_previous_candidates():
    from scripts.retrieval_v57_native_proof import run,worker
    code=inspect.getsource(run)+inspect.getsource(worker)
    assert all(s not in code for s in ('training_rows','cached_features','by_siret','geo_queries','BASELINE'))
