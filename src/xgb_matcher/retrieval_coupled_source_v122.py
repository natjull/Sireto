"""Preserve official self-links as context; do not expand to the same endpoint."""
from src.xgb_matcher.retrieval_coupled_source_v121 import (
    build_sites as original_build_sites, family_batches, source_coordinate,
)
from src.xgb_matcher.retrieval_variants_v118 import clean


def build_sites(current_rows,names,facts,observation_rows,extra_name_rows=(),address_rows=(),edges=()):
    effective=[];self_links=[];total=0
    for ordinal,edge in enumerate(edges):
        total+=1
        a=clean(edge.get('siretEtablissementPredecesseur'));b=clean(edge.get('siretEtablissementSuccesseur'))
        if any(len(s)!=14 or not s.isascii() or not s.isdecimal() for s in (a,b)):
            raise ValueError('Invalid explicit official succession endpoints')
        if a==b:
            self_links.append(dict(source_row_ordinal=int(edge.get('source_row_ordinal',ordinal)),
                reason='SELF_LINK_NO_NEW_ENDPOINT',original_observation_retained=True))
        else:
            effective.append(edge)
    sites,stats=original_build_sites(current_rows,names,facts,observation_rows,extra_name_rows,address_rows,effective)
    if len(effective)+len(self_links)!=total:
        raise ValueError('Official succession input ledger drift')
    stats['succession_input_ledger']=dict(input_observations=total,effective_observations=len(effective),
        self_link_observations=len(self_links),self_link_context=self_links,
        all_original_records_retained_in_source_snapshot=True,
        identifiers_or_site_denominator_changed=False)
    return sites,stats
