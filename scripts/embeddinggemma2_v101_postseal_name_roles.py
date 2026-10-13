"""Post-seal typed lexical name-proof audit; GT lookups never search inputs.

Measures which indexed name field actually matches CRM at the HUMAN target.
Does not decide correctness, rerank, generate a pool or change any model.
"""
from pathlib import Path
import time
from collections import Counter
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v32_learned_query as labels_source
from scripts import embeddinggemma2_v23_native_features as original
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend,normalize_text
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCE=io.HOME/'v99/initial_alignment_native_20261013_a'
OUTPUT=io.HOME/'v101/postseal_name_roles_20261013_a'


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh diagnostic')
    generation=io.read(SOURCE/'generation.json');ex=io.read(SOURCE/'generate_exit.json')
    if generation['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned all526 before GT lookup')
    verify(generation['outputs']);outcomes={r['query_id']:r for r in io.read(SOURCE/'outcomes.json')['learned_native_ooc']}
    labels,controls,refs=labels_source.calibration_sources();labels={r['query_id']:r for r in labels}
    launch=io.read(SOURCE/'launch.json');panel=io.read(verify(launch['panel']))
    status={r['query_id']:r for r in io.read(io.HOME/'v47/postseal_stock_status_20261012_a/cases.json')}
    backend=TantivyBackend(io.INDEX);tv=backend._tantivy
    if backend.searcher.num_docs!=82772691:raise ValueError('Original national index')
    fields=original.lexical.NAMES;rows=[];start=time.monotonic();OUTPUT.mkdir(parents=True)
    for q in panel:
        mandate.guard(mandate.DEADLINE);qid=q['query_id'];label=labels[qid]
        if label['fold'] not in (2,3,4):raise ValueError('Closed fold')
        s=status.get(qid,{});protected=s.get('site_diffusion')=='P' or s.get('owner_diffusion')=='P'
        scores={};token_list=original.lexical.tokens(normalize_text(original.data.safe(q['crm_name'])))
        if not protected:
            lookup=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'siret',label['siret'],'basic'),0.)
            leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
            for field in fields:
                part=original.lexical.mean_query(backend,field,token_list)
                if part is None:scores[field]=0.;continue
                query=tv.Query.boolean_query([(tv.Occur.Must,lookup),(tv.Occur.Must,leaf),(tv.Occur.Should,part)])
                hits=backend.searcher.search(query,limit=2,count=False).hits
                if len(hits)>1:raise ValueError('One national leaf per SIRET')
                scores[field]=float(hits[0][0]) if hits else 0.
        rows.append(dict(query_id=qid,query=q,label_siret=label['siret'],exact=outcomes[qid]['exact'],
            siren=outcomes[qid]['siren'],operational=outcomes[qid]['operational'],
            protected_no_backfill=protected,national_field_bm25_scores=scores,
            all_fields_maximum=max(scores.values(),default=0.),
            lexical_keyword_overlap_not_entity_identity_proof=True,label_side_only=True))
        if len(rows)%32==0:io.write(OUTPUT/'state.json',dict(status='POSTSEAL_NAME_FIELD_DIAGNOSIS',queries=len(rows),total=526,updated_unix=time.time()))
    summary={}
    for name,flag in [('hits',True),('misses',False)]:
        rs=[r for r in rows if r['exact']==flag]
        summary[name]=dict(queries=len(rs),protected_no_lookup=sum(r['protected_no_backfill'] for r in rs),
            no_nonzero_word_name_field=sum(not r['protected_no_backfill'] and r['all_fields_maximum']==0 for r in rs),
            nonzero_field_counts={f:sum(r['national_field_bm25_scores'].get(f,0)>0 for r in rs) for f in fields})
    io.write(OUTPUT/'cases.json',rows)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_POSTSEAL_NAME_FIELD_AUDIT_NOT_RETRIEVAL',
        implementation=record(Path(__file__)),source_generation=record(SOURCE/'generation.json'),
        source_exit=record(SOURCE/'generate_exit.json'),label_sources=refs,cases=record(OUTPUT/'cases.json'),
        summary=summary,all526_retained=True,no_positive_injected_into_retrieval=True,
        no_source_alias_or_model_changed=True,no_recall_claim=True,elapsed_s=time.monotonic()-start))
    print(summary,flush=True)


if __name__=='__main__':main()
