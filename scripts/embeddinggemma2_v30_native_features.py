"""V30: unchanged sealed V23 full proofs; only training objective/mining changes."""
from pathlib import Path
import argparse
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from src.xgb_matcher.retrieval_v20_workbench import record, verify

PREPARED=io.HOME/'v23/hierarchical_native_20261012_a'
OUTPUT=io.HOME/'v30/legal_role_context_20261012_a'
ROLE_SOURCE=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/rne_legal_entity_roles_v1/legal_entity_roles.parquet')
# The four codes with >100000 observations in the COMPLETE source, never labels.
ROLE_CODES=['75','73','99','71','OTHER']
PROOF_FIELDS=list(source.PROOF_FIELDS)+[field for code in ROLE_CODES for field in
    ('rne_actor_name_code_'+code,'rne_actor_name_x_own_geo_code_'+code)]
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
RECIPE=dict(source.RECIPE,feature_fields=FEATURE_FIELDS,
    full_rne_company_roles='LEGAL_ENTITY_CORPORATE_ACTOR_NAMES_ONLY',
    role_codes=ROLE_CODES,role_code_bins='GLOBAL_SOURCE_FREQUENCY_GT100000_NO_LABELS',
    role_context_is_not_own_alias_group_ownership_or_current_identity=True,
    related_siren_required_for_company_entity_only=True,
    remove_representative_name_suffixes=True,physical_persons_beneficial_owners_not_read=True,
    unknown_effective_dates_are_unknown_not_current_claim=True,relationship_hops=1,
    scope='ALL_ORIGINAL_QUERY_PROOF_FAMILIES_NOT_GT_OR_TOPK_SEEDS')

def initialize(out,deadline):
    source.initialize(out,deadline)
    previous=source.authority(PREPARED)
    generation=io.read(PREPARED/'features_generation.json')
    ex=io.read(PREPARED/'prepare_exit.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed'] or generation['queries']!=526:
        raise ValueError('Owned complete original features only')
    launch=io.read(out/'launch.json')
    if previous['panel']!=launch['panel']:raise ValueError('Same complete query panel')
    io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_launch=record(PREPARED/'launch.json'),prepared_generation=record(PREPARED/'features_generation.json'),
        prepared_exit=record(PREPARED/'prepare_exit.json'))
    manifest=io.read(ROLE_SOURCE.with_suffix('.parquet.manifest.json'))
    if io.digest(ROLE_SOURCE)!=manifest['output_sha256']:raise ValueError('Original corporate role source hash')
    launch.update(role_source=record(ROLE_SOURCE),role_manifest=record(ROLE_SOURCE.with_suffix('.parquet.manifest.json')))
    launch['helpers'].extend([record(Path(source.__file__)),record(io.ROOT/'src/xgb_matcher/rne_legal_entity_roles.py')])
    io.write(out/'launch.json',launch)

def authority(out):
    launch=source.authority(out)
    for name in ('prepared_launch','prepared_generation','prepared_exit','role_source','role_manifest'):verify(launch[name])
    return launch


def prepare(out):
    import pyarrow.parquet as pq
    from collections import defaultdict
    import re
    launch=authority(out);original=io.read(verify(launch['prepared_generation']))
    table=pq.read_table(verify(launch['role_source']),columns=['subject_siren','related_siren','related_denomination','role_code','effective_date','observed_at'],use_threads=False)
    related=defaultdict(lambda:defaultdict(set));skipped_identifier=0;future=0;unknown_date=0;stripped=0
    for row in table.to_pylist():
        subject=row['subject_siren'];actor=row['related_siren']
        if not subject or len(subject)!=9 or not subject.isdigit() or not actor or len(actor)!=9 or not actor.isdigit() or subject==actor:
            skipped_identifier+=1;continue
        observed=str(row['observed_at'] or '')[:10];effective=str(row['effective_date'] or '')[:10]
        if (observed and observed>'2026-08-01') or (effective and effective>'2026-08-01'):future+=1;continue
        if not effective:unknown_date+=1
        name=structural.normalize(data.safe(row['related_denomination']))
        cleaned=re.split(r'\b(?:REPRESENTE|REPRESENTEE|REPRESENTES|REPRESENTEES|REPR)\s+(?:PAR|P[.]?)\b',name,maxsplit=1)[0].strip()
        if cleaned!=name:stripped+=1
        if not cleaned:continue
        code=str(row['role_code']);bucket=code if code in ROLE_CODES[:-1] else 'OTHER'
        related[int(subject)][bucket].add(cleaned)
    del table
    source_count=sum(len(v) for groups in related.values() for v in groups.values())
    io.write(out/'role_source_receipt.json',dict(source=launch['role_source'],source_records=1754075,
        represented_parents=len(related),distinct_context_names_by_code=source_count,
        invalid_or_missing_related_identifier_excluded=skipped_identifier,future_excluded=future,
        unknown_effective_date_retained=unknown_date,representative_suffixes_removed=stripped,
        relation_is_not_ownership_identity_or_current_status=True))
    panel=io.read(verify(launch['panel']));refs={};total=0;started=time.monotonic()
    for q in panel:
        guard(launch['deadline_unix']);qid=q['query_id'];path=feature_path(out,qid);saved=engine.sealed(path)
        if saved:
            with np.load(saved,allow_pickle=False) as f:n=len(f['keys'])
        else:
            with np.load(verify(original['features'][qid]),allow_pickle=False) as f:keys=f['keys'];old=f['features']
            x=np.zeros((len(keys),len(PROOF_FIELDS)),dtype=np.float32);x[:,:len(source.PROOF_FIELDS)]=old
            groups,group=np.unique(keys//100000,return_inverse=True)
            name=structural.normalize(data.safe(q['crm_name']))
            for col,code in enumerate(ROLE_CODES):
                values=np.asarray([structural.similarity(name,tuple(related.get(int(p),{}).get(code,()))) for p in groups],dtype=np.float32)
                x[:,len(source.PROOF_FIELDS)+2*col]=values[group]
                x[:,len(source.PROOF_FIELDS)+2*col+1]=values[group]*x[:,5]
            if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():raise ValueError('Bounded role context')
            temp=path.with_suffix('.pending.npz');np.savez(temp,keys=keys,features=x);os.replace(temp,path);engine.seal(path)
            n=len(keys);del keys,old,x;gc.collect()
        refs[qid]=record(path);total+=n
        engine.progress(out,'FULL_RNE_LEGAL_ACTOR_CONTEXT',len(refs),total=len(panel),full_feature_rows=total,elapsed_s=time.monotonic()-started)
    if len(refs)!=526:raise ValueError('All526 complete')
    io.write(out/'features_generation.json',dict(status='COMPLETE_SEALED_QUERY_ONLY_ALL526_FULL_PROOFS',
        features=refs,queries=526,full_feature_rows=total,commune_sites=original['commune_sites'],
        role_source=launch['role_source'],role_source_receipt=record(out/'role_source_receipt.json'),
        all_original_query_proof_families_before100=True,whole_source_scanned=True,
        source_names_are_related_actor_context_not_own_aliases=True,relation_hops=1,
        labels_used=False,no_candidate_budget=True,no_gt_directed_edges=True))


def native_query(backend,q,keys,features,coefficients):
    """ONE native national query, learned lexical + ALL numeric proof views.

    Bit planes attach a *computed proof score* to its official reference. IDs
    have no learned coefficient and are never numerical model inputs. No top-k
    prepool, no sampled subset, no capped channel union is materialized here.
    """
    coefficients=np.asarray(coefficients,dtype=float)
    if coefficients.shape!=(len(FEATURE_FIELDS),) or not np.isfinite(coefficients).all() or (coefficients<0).any():
        raise ValueError('Finite learned nonnegative typed coefficients')
    tv=backend._tantivy;parts=[]
    for component,weight in zip(lexical_parts(backend,q),coefficients[:5]):
        if component is not None and weight>0:parts.append(tv.Query.boost_query(component,float(weight)))
    weights=coefficients[5:];bound=float(weights.sum());bits=RECIPE['quantization_bits'];maximum=(1<<bits)-1
    if bound>0 and len(keys):
        scores=features.astype(np.float64)@weights
        if not np.isfinite(scores).all() or (scores<0).any() or (scores>bound+1e-5).any():raise ValueError('Full numeric proof score bound')
        integers=np.floor(np.clip(scores/bound,0,1)*maximum).astype(np.uint32)
        refs=[f'{int(k):014d}' for k in keys]
        for bit in range(bits):
            selected=np.flatnonzero(integers & (1<<bit))
            if len(selected):
                mask=tv.Query.term_set_query(backend.index.schema,'siret',[refs[i] for i in selected])
                parts.append(tv.Query.const_score_query(mask,float(bound*(1<<bit)/maximum)))
        del scores,integers,refs
    if not parts:return tv.Query.empty_query(),bound/maximum
    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    return tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))]),bound/maximum


def native_search(backend,q,keys,features,coefficients):
    started=time.monotonic();query,error=native_query(backend,q,keys,features,coefficients)
    built=time.monotonic();rows=[]
    for score,loc in backend.searcher.search(query,limit=100,count=False).hits:
        raw=backend.searcher.doc(loc).to_dict();siret=backend._first(raw,'siret');siren=backend._first(raw,'siren')
        if not siret.isdigit() or len(siret)!=14 or len(siren)!=9 or not siret.startswith(siren) or not np.isfinite(score):
            raise ValueError('Valid official native leaf and score')
        rows.append(dict(siret=siret,siren=siren,score=float(score)))
    if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows):raise ValueError('Native unique absolute100')
    rows.sort(key=lambda r:(-r['score'],r['siret']))
    return rows,dict(query_build_seconds=built-started,search_seconds=time.monotonic()-built,
        total_seconds=time.monotonic()-started,quantization_abs_error_bound=error,
        full_nonlexical_leaves=len(keys),national_lexical_sites=43896818,no_prepool=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--deadline-unix',type=float);a=p.parse_args()
    if a.action=='initialize':initialize(a.output,a.deadline_unix)
    else:prepare(a.output)


