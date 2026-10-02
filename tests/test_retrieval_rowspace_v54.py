import numpy as np

from src.xgb_matcher.retrieval_contrastive_v52 import fit
from src.xgb_matcher.retrieval_rowspace_v54 import solve
from tests.test_retrieval_contrastive_v52 import fixture


def test_rowspace_and_full_solve_agree_on_convex_objective():
    q,p,n,m,r=fixture()
    a,full=fit(q,p,n,m,2,r,ridge=1.,maxiter=200)
    b,reduced=solve(q,p,n,m,2,r,'FULL',1.,10.,8,200)
    assert np.isclose(full['objective_after'],reduced['objective_after'],atol=1e-7)
    assert np.allclose(a,b,atol=1e-3) and reduced['same_full_objective']
    assert full['query_ids']==reduced['query_ids'] and full['training_pairs']==reduced['training_pairs']


def test_unobserved_query_directions_keep_identity():
    q,p,n,m,r=fixture(); q[:,4:]=0.
    w,report=solve(q,p,n,m,2,r,'FULL',1.,10.,8,100)
    # QR may include surplus orthogonal columns on rank deficiency. Their
    # train loss gradient is zero and regularization keeps the optimum0.
    assert np.allclose(w[4:,:],np.eye(8)[4:,:],atol=1e-5)
