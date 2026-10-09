"""Evidence synthesis of the completed post-seal audit; never a retrieval rule."""
from collections import Counter
import sqlite3

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts.embeddinggemma2_v16_failure_audit import SOURCE, OUTPUT, tokens, overlap
from src.xgb_matcher.retrieval_v20_workbench import record, verify

STREET_TYPES = {'RUE','ROUTE','AVENUE','AV','BOULEVARD','BD','CHEMIN','CHE','PLACE','PL',
                'ROND','POINT','LIEUDIT','IMPASSE','ALLEE','QUAI','R','RTE','TRAVERSE'}


def distinctive_address_overlap(crm, addresses):
    terms=tokens(crm)-STREET_TYPES
    return any(terms & (tokens(v)-STREET_TYPES) for v in addresses)


def main():
    import pyarrow.parquet as pq
    audit=io.read(OUTPUT/'launch.json'); verify(audit['implementation'])
    done=io.read(OUTPUT/'dense_completion.json')
    if done['status']!='COMPLETE_POSTSEAL_DENSE_ATTRIBUTION' or done['cases']!=45:
        raise ValueError('No incomplete diagnostic synthesis')
    dense={r['query_id']:r for r in io.read(verify(done['dense_cases']))}
    rows=io.read(verify(audit['cases'])); summary=io.read(OUTPUT/'summary.json')
    if set(dense)!={r['query_id'] for r in rows}: raise ValueError('Full audit denominator')
    cfg=io.read(io.ROOT/'config/retrieval_v358_postseal_evaluation.json')
    ledger={r['query_id']:r for r in pq.read_table(verify(cfg['ledger']),
        filters=[('query_id','in',list(dense))],use_threads=False).to_pylist()}
    database=io.read(SOURCE/'names/generation.json')['database']
    con=sqlite3.connect('file:'+database['path']+'?mode=ro&immutable=1',uri=True)
    related=sorted({p for r in rows for p in r['encoded_target_document']['related_parents']})
    names=engine.lookup(con,1,related)
    expanded=[]
    try:
        for r in rows:
            q=r['query_id']; name=r['crm']['crm_name']; parent=int(r['label']['gt_siren'])
            sibling_names=sorted({v[0] for v in con.execute(
                'SELECT name FROM aliases WHERE kind=2 AND owner_key>=? AND owner_key<?',
                (parent*100000,(parent+1)*100000))})
            before=r['evidence']['name_token_coverage'] or 0
            sibling_matches=[n for n in sibling_names if (overlap(name,[n]) or 0)>before]
            related_matches={p:[n for n in sorted(names[int(p)]) if (overlap(name,[n]) or 0)>before]
                for p in r['encoded_target_document']['related_parents']}
            related_matches={k:v for k,v in related_matches.items() if v}
            flags=[]
            address_match=distinctive_address_overlap(r['crm']['crm_street'],r['evidence']['official_addresses'])
            if before==0 and not address_match: flags.append('NO_LITERAL_NAME_OR_DISTINCTIVE_STREET_OVERLAP')
            if r['evidence']['official_addresses'] and all(set(tokens(a))<={'ND'} for a in r['evidence']['official_addresses']):
                flags.append('MASKED_ADDRESS_ND_MARKER')
            geo=dict(r['geo'])
            if geo['status']=='WITHIN_RADIUS_AT_OR_BEFORE_CUTOFF' and 'geo' not in r['trace']['channel_ranks']:
                geo['status']='WITHIN_RADIUS_TIE_AT_NEAREST100_CUTOFF'
            expanded.append(dict(**r,dense=dense[q],label_provenance=ledger[q],
                distinctive_street_token_overlap=address_match,review_flags=flags,
                geo_refined=geo,same_siren_sibling_name_context=sibling_matches,
                official_related_parent_name_context=related_matches,
                relationship_types_not_inferred_from_untyped_links=True))
    finally: con.close()
    missing=[r for r in expanded if r['attribution']=='ALL_CHANNELS_MISS_AT100']
    def bucket(rank):
        if rank<=200: return '101_200'
        if rank<=1000: return '201_1000'
        return 'ABOVE_1000'
    masked=[r for r in expanded if 'MASKED_ADDRESS_ND_MARKER' in r['review_flags']]
    evidence_summary=dict(**summary,
        full_dense_replay_verified=True,dense_replay_max_score_error=max(r['dense']['max_score_error'] for r in expanded),
        all_channel_misses_dense_rank_buckets=dict(Counter(bucket(r['dense']['dense_rank']) for r in missing)),
        all_channel_misses_no_literal_name_overlap=sum(r['evidence']['name_token_coverage']==0 for r in missing),
        all_channel_misses_no_distinctive_street_overlap=sum(not r['distinctive_street_token_overlap'] for r in missing),
        all_channel_misses_neither_literal_evidence=sum('NO_LITERAL_NAME_OR_DISTINCTIVE_STREET_OVERLAP' in r['review_flags'] for r in missing),
        overlap_is_descriptive_not_semantic_proof=True,
        all_channel_misses_geo_causes=dict(Counter(r['geo_refined']['status'] for r in missing)),
        fusion_evictions_by_single_source=dict(Counter(next(iter(r['trace']['channel_ranks'])) for r in expanded if r['attribution']=='FUSION_EVICTION')),
        geo_causes_refined=dict(Counter(r['geo_refined']['status'] for r in expanded)),
        masked_address_cases=len(masked),
        removed_index_name_values=sorted({n for r in expanded for n in r['evidence']['names_lost_from_index_to_doc']}),
        non_placeholder_index_name_losses=sum(any(n!='ND' for n in r['evidence']['names_lost_from_index_to_doc']) for r in expanded),
        sibling_context_improves_literal_overlap_cases=sum(bool(r['same_siren_sibling_name_context']) for r in expanded),
        strongest_sibling_context_example='13765: FOYER TY BREIZ exists on sibling official site names, not target site names',
        sibling_context_is_not_own_site_alias=True,
        no_measured_graph_or_perplexity_retrieval_gain=True,
        label_provenance_counts=dict(Counter(r['label_provenance']['human_ground_truth_source'] for r in expanded)),
        same_name_NAOS_misses=sum(r['crm']['crm_name']=='NAOS' for r in expanded),
        same_name_NAOS_parent_sirens=sorted({r['label']['gt_siren'] for r in expanded if r['crm']['crm_name']=='NAOS'}),
        label_errors_proven_after_review=0,denominator_unchanged=True)
    io.write(OUTPUT/'evidence_cases.json',expanded)
    io.write(OUTPUT/'evidence_summary.json',evidence_summary)
    io.write(OUTPUT/'review_completion.json',dict(status='COMPLETE_EVIDENCE_REVIEW_ALL45',
        implementation=record(__file__),cases=record(OUTPUT/'evidence_cases.json'),
        summary=record(OUTPUT/'evidence_summary.json'),dense_completion=record(OUTPUT/'dense_completion.json'),
        labels_used_only_for_postseal_audit=True,test_opened=False,fold0_opened=False,fold1_opened=False))
    print(evidence_summary)


if __name__=='__main__': main()
