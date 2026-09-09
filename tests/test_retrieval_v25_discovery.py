from scripts.audit_retrieval_v25_discovery import classify


def test_diagnosis_precedence_and_no_truth_injection_claim():
    r=dict(official_sites=1,indexed_parent_sites=1,indexed_geo_sites=1,matching_channels=['address_word'],parent_in_evidence=False)
    assert classify(r)=='TEXT_REACHABLE_BUT_NOT_RETRIEVED'
    assert classify({**r,'parent_in_evidence':True})=='ALREADY_RETRIEVED'
    assert classify({**r,'matching_channels':[]})=='NO_V18_TEXT_MATCH_IN_GEO'
    assert classify({**r,'indexed_geo_sites':0})=='OUTSIDE_EFFECTIVE_INDEX_GEO'
    assert classify({**r,'indexed_parent_sites':0,'indexed_geo_sites':0})=='NO_INDEXED_PARENT_SITE'
    assert classify({**r,'official_sites':0})=='NO_OFFICIAL_PARENT_SITE'
