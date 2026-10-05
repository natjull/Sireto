"""Prepared input conformance helper, never a producer or admission authority.

Same frozen V52 mask. All controls/labels stay in the all-human denominator.
No alias/identifier lookup, fit, candidate pool, model or score changes.
"""
from .hierarchical_retrieval import RetrievalQuery
from .retrieval_contrastive_v52 import surface
from .retrieval_variants_v118 import normalized


def feature_query(raw):
    def clean(value):
        return surface([value], '')[0][2:]
    return RetrievalQuery(name=normalized(clean(raw['crm_name'])),
        address=normalized(clean(' '.join(raw[k] for k in ('crm_number','crm_suffix','crm_street')))),
        number=clean(raw['crm_number']),postcode=raw['crm_postcode'],insee=raw['crm_insee'])
