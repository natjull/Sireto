import inspect
import numpy as np
import pytest
from scripts import retrieval_v66_head_initial as v66
from src.xgb_matcher.retrieval_initialisms_v65 import FEATURES,query
from src.xgb_matcher.retrieval_sparse_flat_v61 import query as flat
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from tests.test_retrieval_proof_v56 import backend


def test_selected_native_reduction_preserves_every_active_component(tmp_path):
    weights=np.r_[np.arange(1,12)*.1,np.zeros(4),1.7]
    m=dict(weights=weights.tolist(),features=list(FEATURES),family='HEAD_WORD',heldout_fold=2,training_folds=[3,4],query_ids=['training'])
    reduced=v66.reduce_model(m)
    b=backend(tmp_path,[dict(legal_names='ALPHA BETA',addresses='12 RUE BLEUE'),dict(legal_names='COLLEGE',addresses='13 RUE VERTE')])
    q=RetrievalQuery(name='AB',address='12 RUE BLEUE',number='12',insee='75056'); terms=[('alpha',.25),('college',.75)]
    def scores(p): return {b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s) for s,loc in b.searcher.search(p,limit=100,count=False).hits}
    a=scores(query(b,q,weights,terms)); c=scores(flat(b,q,reduced['weights'],terms,reduced['alpha']))
    assert a.keys()==c.keys()
    np.testing.assert_allclose(list(a.values()),[c[s] for s in a],atol=1e-5,rtol=0)
    m['weights'][11]=1.
    with pytest.raises(ValueError,match='declared HEAD_WORD'): v66.reduce_model(m)


def test_cache_only_preparation_and_unchanged_cost_guards():
    import json
    cfg=json.loads(v66.CONFIG.read_text())
    assert cfg['controller_seconds']==360 and cfg['primary_mean16_seconds']==4
    assert cfg['resources']['max_rss_bytes']==6442450944 and cfg['resources']['max_swapout_pages_delta']==0
    prepare=inspect.getsource(v66.prepare)
    assert 'human_ground_truth' not in prepare and "['batches']" not in prepare
    assert "cached['queries']!=queries" in prepare and "encoded['vectors']" in prepare
    assert v66.BASELINE['sha256']=='11151f230e9616ee799dffbc47aa588bdbb6685ccebc357c6b326e61d3bd0384'
