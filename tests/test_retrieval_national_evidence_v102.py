import pytest
from scripts.audit_retrieval_v102_national_evidence import branch_reason
from scripts.audit_retrieval_v91_scoped_evidence import check_point

@pytest.mark.parametrize('outside,score,ban,xy,expected',[
    (True,0.,True,True,'OUTSIDE_FULL_NATIONAL_REGISTRY'),
    (False,0.,True,True,'NO_MATCHING_LITERAL_WORD_PROOF_IN_FULL_SCOPED_NAMES'),
    (False,1.,False,True,'QUERY_BAN_UNAVAILABLE_OR_INCOMPATIBLE'),
    (False,1.,True,False,'TARGET_XY_ABSENT'),
    (False,1.,True,True,'POSITIVE_NATIONAL_SCOPED_WORD_SITE_PROOF')])
def test_branch_reasons_do_not_confuse_graph_endpoints_with_full_registry(outside,score,ban,xy,expected):
    proof=dict(outside_national_site_scope=outside,outside_succession_endpoint_scope=True,score=score)
    assert branch_reason(proof,ban,xy)==expected

def test_inconsistent_points_remain_fail_closed():
    with pytest.raises(ValueError,match='CONTRADICTION'):
        check_point([dict(siret='12345678900001',score=1.)],'12345678900002',2.)
    with pytest.raises(ValueError,match='CONTRADICTION'):
        check_point([dict(siret='12345678900001',score=1.)],'12345678900001',.5)
