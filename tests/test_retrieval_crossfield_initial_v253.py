from types import SimpleNamespace
import pytest

from src.xgb_matcher.retrieval_crossfield_initial_v253 import search
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from test_retrieval_corroborated_initial_v217 import tiny_native


def fixture():
    b=tiny_native();calls=[];original=b.searcher
    def queried(parsed,**kw):calls.append(kw);return original.search(parsed,**kw)
    b.searcher=SimpleNamespace(search=queried,doc=original.doc)
    guard=lambda:None
    e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=lambda text:(set(),set(),{}),
        related=lambda parents,sites:(set(),{}),radial_proofs=lambda geo:([],{}))
    return b,e,calls,guard


def test_initial_collection_uses_new_proofs_before_one_global100():
    b,e,calls,guard=fixture()
    found,trace=search(b,e,RetrievalQuery(name='ROAD',address='ROAD',postcode='11111',number='1'),{},
        [1.]*11,[],0.,[.2,.3,.4])
    assert calls==[dict(limit=100,count=False)] and len(found)<=100
    assert len(found)==len({r['siret'] for r in found})
    assert trace['fixed_reservoir'] is None and not trace['score_after_fixed_pool']
    assert trace['query_time_training'] is False and trace['original11_head_fixed_v228']
    assert trace['cross_name_address_is_not_legal_identity_or_alias'] and trace['label_lookup'] is False
    assert e.files.guard is guard


def test_guard_restored_on_failed_extra_coefficients():
    b,e,calls,guard=fixture()
    with pytest.raises(ValueError):search(b,e,RetrievalQuery(name='ROAD'),{},[1.]*11,[],0.,[float('nan')]*3)
    assert not calls and e.files.guard is guard
