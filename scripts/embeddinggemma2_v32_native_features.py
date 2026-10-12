"""V32: unchanged sealed V23 full proofs; only training objective/mining changes."""
from pathlib import Path
import argparse
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from src.xgb_matcher.retrieval_v20_workbench import record, verify

PREPARED=io.HOME/'v23/hierarchical_native_20261012_a'
OUTPUT=io.HOME/'v32/global_public_parent_names_20261012_a'
SECTOR=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sector_official_v1/official_sector_evidence.parquet')
LEXICAL_FIELDS=list(source.LEXICAL_FIELDS)+['global_published_public_parent_name']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
RECIPE=dict(source.RECIPE,feature_fields=FEATURE_FIELDS,feature_values_changed=False,
    public_parent_scope='ALL_PUBLISHED_SIREN_OWNER_NAMES_BROADCAST_TO_ALL_NATIONAL_MEMBER_LEAVES',
    source_siret_site_names_not_used_as_parent_aliases=True,
    parent_reference_is_lookup_not_identifier_feature=True,all_source_scores_before_one_native100=True,
    parent_name_score='MAX_WRATIO_ALL_PUBLISHED_OWNER_ALIASES_LEARNED_WEIGHT',
    no_topk_parent_selection=True,source_date_cutoff='2026-08-01_UNKNOWN_RETAINED')

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
    manifest=io.read(SECTOR.with_suffix('.parquet.manifest.json'))
    if io.digest(SECTOR)!=manifest['output_sha256']:raise ValueError('Original public source hash')
    launch.update(sector_source=record(SECTOR),sector_manifest=record(SECTOR.with_suffix('.parquet.manifest.json')))
    launch['helpers'].extend([record(Path(source.__file__)),record(io.ROOT/'src/xgb_matcher/official_sector_evidence.py')])
    io.write(out/'launch.json',launch)

def authority(out):
    launch=source.authority(out)
    for name in ('prepared_launch','prepared_generation','prepared_exit','sector_source','sector_manifest'):verify(launch[name])
    return launch


PUBLIC=None
LAST_PART=None
ACTIVE_OUTPUT=OUTPUT
def bind_runtime(out):
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT=out

def prepare(out):
    import pyarrow.parquet as pq
    from collections import defaultdict
    launch=authority(out)
    table=pq.read_table(verify(launch['sector_source']),columns=['subject_kind','siren','name','observed_at'],use_threads=False)
    parents=defaultdict(set);future=0;undated=0
    for row in table.to_pylist():
        if row['subject_kind']!='SIREN':continue
        observed=str(row['observed_at'] or '')[:10]
        if observed and observed>'2026-08-01':future+=1;continue
        if not observed:undated+=1
        key=row['siren'];name=structural.normalize(data.safe(row['name']))
        if not key or len(key)!=9 or not key.isdigit():raise ValueError('Public typed parent identifier')
        if name:parents[key].add(name)
    path=out/'published_parent_names.json'
    io.write(path,{k:sorted(v) for k,v in sorted(parents.items())});engine.seal(path)
    io.write(out/'parent_source_receipt.json',dict(source=launch['sector_source'],parents=len(parents),
        aliases=sum(map(len,parents.values())),future_excluded=future,undated_retained=undated,
        source_is_published_owner_name_not_borrowed_site_alias=True,source_scanned_without_labels=True))
    gen=io.read(verify(launch['prepared_generation']))
    io.write(out/'features_generation.json',dict(gen,immutable_shared_source=launch['prepared_generation'],
        published_parent_names=record(path),public_parent_receipt=record(out/'parent_source_receipt.json'),labels_used=False))

def parent_component(backend,q):
    from rapidfuzz import process,fuzz
    global PUBLIC,LAST_PART
    name=structural.normalize(data.safe(q['crm_name']))
    if LAST_PART and LAST_PART[0] is backend and LAST_PART[1]==name:return LAST_PART[2]
    if PUBLIC is None:
        launch=authority(ACTIVE_OUTPUT);gen=io.read(ACTIVE_OUTPUT/'features_generation.json')
        rows=io.read(verify(gen['published_parent_names']))
        aliases=sorted({v for values in rows.values() for v in values});positions={v:i for i,v in enumerate(aliases)}
        owners=list(rows);alias_at=[];owner_at=[]
        for i,key in enumerate(owners):
            for text in rows[key]:alias_at.append(positions[text]);owner_at.append(i)
        PUBLIC=(aliases,owners,np.asarray(alias_at,dtype=np.int64),np.asarray(owner_at,dtype=np.int64))
    aliases,owners,alias_at,owner_at=PUBLIC
    if not name or not aliases:return None
    values=process.cdist([name],aliases,scorer=fuzz.WRatio,processor=None,workers=1,dtype=np.float32)[0]/100
    scores=np.zeros(len(owners),dtype=np.float32);np.maximum.at(scores,owner_at,values[alias_at])
    maximum=(1<<RECIPE['quantization_bits'])-1
    integers=np.floor(np.clip(scores,0,1)*maximum).astype(np.uint32);tv=backend._tantivy;parts=[]
    for bit in range(RECIPE['quantization_bits']):
        indices=np.flatnonzero(integers & (1<<bit))
        if len(indices):
            lookup=tv.Query.term_set_query(backend.index.schema,'siren',[owners[i] for i in indices])
            parts.append(tv.Query.const_score_query(lookup,float((1<<bit)/maximum)))
    result=tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]) if parts else None
    LAST_PART=(backend,name,result)
    return result

def lexical_parts(backend,q):
    return source.lexical_parts(backend,q)+[parent_component(backend,q)]


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
    for component,weight in zip(lexical_parts(backend,q),coefficients[:len(LEXICAL_FIELDS)]):
        if component is not None and weight>0:parts.append(tv.Query.boost_query(component,float(weight)))
    weights=coefficients[len(LEXICAL_FIELDS):];bound=float(weights.sum());bits=RECIPE['quantization_bits'];maximum=(1<<bits)-1
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
        full_nonlexical_leaves=len(keys),national_lexical_sites=43896818,no_prepool=True,
        public_parent_quantization_error_bound=float(coefficients[5])/((1<<RECIPE['quantization_bits'])-1),
        all_published_parents_nationally_before100=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--deadline-unix',type=float);a=p.parse_args()
    if a.action=='initialize':initialize(a.output,a.deadline_unix)
    else:prepare(a.output)


