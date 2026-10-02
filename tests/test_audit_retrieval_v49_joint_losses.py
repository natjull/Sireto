import pytest

from scripts.audit_retrieval_v49_joint_losses import reason


@pytest.mark.parametrize('present,text,geo,score,cutoff,expected',[
    (False,False,False,0.,2.,'INDEX_DOCUMENT_ABSENT'),
    (True,False,False,0.,2.,'NO_TEXT_OR_COMPATIBLE_GEO_SIGNAL'),
    (True,False,True,1.,2.,'BELOW_JOINT_SCORE_CUTOFF'),
    (True,True,False,2.,2.,'TIE_AT_CAP'),
])
def test_loss_reasons(present,text,geo,score,cutoff,expected):
    assert reason(present,text,geo,score,cutoff)==expected


def test_certificate_violation_fails_closed():
    with pytest.raises(ValueError,match='above certified'): reason(True,True,True,3.,2.)


def test_incomplete_cutoff_fails_closed():
    with pytest.raises(ValueError,match='complete100'): reason(True,True,False,1.,None)
