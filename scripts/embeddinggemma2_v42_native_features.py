"""V28: unchanged sealed V23 full proofs; only training objective/mining changes."""
from pathlib import Path
import argparse
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from src.xgb_matcher.retrieval_v20_workbench import record, verify

PREPARED=io.HOME/'v28/official_sector_native_20261012_a'
OUTPUT=io.HOME/'v42/temporal_economic_chains_20261012_a'
SECTOR=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sector_official_v1/official_sector_evidence.parquet')
PROOF_FIELDS=list(source.PROOF_FIELDS)+['sector_own_name','sector_own_street','sector_own_number',
    'sector_own_street_x_number','sector_own_name_x_geo','sector_parent_name',
    'sector_parent_name_x_geo','sector_own_administrative','sector_own_same_observation_address']
PROOF_FIELDS += ['forward2_geo_context','reverse2_geo_context','forward2_site_context','reverse2_site_context',
    'forward2_name_geo_context','reverse2_name_geo_context','forward2_recent_geo_context','reverse2_recent_geo_context']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
RECIPE=dict(source.RECIPE,feature_fields=FEATURE_FIELDS,
    public_sector_sources='FULL_ORIGINAL_FINNESS_EDUCATION_DINUM_STRUCTURED_IDS',
    direct_site_source_all_national_sites_before100=True,
    owner_source_scope='PARENT_CONTEXT_ON_EXISTING_FULL_QUERY_PROOF_FAMILIES',
    source_date_cutoff='2026-08-01_UNKNOWN_DATES_RETAINED_AS_UNKNOWN',
    same_source_observation_address_required=True,
    sector_owner_names_are_not_site_aliases=True,feature_values_changed=True,
    temporal_economic_chains='TWO_CONSECUTIVE_FORWARD_LINKS_DATE1_LE_DATE2_AND_REVERSE_SAME_CHAIN',
    source_family_then_one_succession_hop_then_target_family_no_second_link=False,
    no_chain_address_or_name_copied_as_own_alias=True,no_fixed_chain_pool=True)

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
    if io.digest(SECTOR)!=manifest['output_sha256']:raise ValueError('Original public-sector source hash')
    launch.update(sector_source=record(SECTOR),sector_manifest=record(SECTOR.with_suffix('.parquet.manifest.json')))
    launch['helpers'].extend([record(Path(source.__file__)),record(io.ROOT/'src/xgb_matcher/official_sector_evidence.py'),
        record(io.ROOT/'scripts/embeddinggemma2_v19_proof_retrieval.py')])
    io.write(out/'launch.json',launch)

def authority(out):
    launch=source.authority(out)
    for name in ('prepared_launch','prepared_generation','prepared_exit','sector_source','sector_manifest'):verify(launch[name])
    return launch


def prepare(out):
    import duckdb,pyarrow.parquet as pq
    launch=authority(out);original=io.read(verify(launch['prepared_generation']))
    panel=io.read(verify(launch['panel']))
    compiled=engine.sealed(out/'chains.parquet')
    if not compiled:
        con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='1GB'")
        tmp=out/'duck_tmp';tmp.mkdir(exist_ok=True);con.execute('SET temp_directory=?',[str(tmp)])
        raw=verify(launch['raw_successions'])
        con.execute("""CREATE TEMP TABLE links AS
          SELECT DISTINCT CAST(siretEtablissementPredecesseur AS UBIGINT) p,
          CAST(siretEtablissementSuccesseur AS UBIGINT) s,dateLienSuccession d
          FROM read_parquet(?) WHERE continuiteEconomique IS TRUE
          AND dateLienSuccession<=DATE '2026-08-01'
          AND regexp_full_match(siretEtablissementPredecesseur,'[0-9]{14}')
          AND regexp_full_match(siretEtablissementSuccesseur,'[0-9]{14}')
          AND siretEtablissementPredecesseur<>siretEtablissementSuccesseur""",[str(raw)])
        con.execute("""COPY (SELECT DISTINCT a.p src,a.s intermediate,b.s dst,a.d date1,b.d date2
          FROM links a JOIN links b ON a.s=b.p AND a.d<=b.d
          WHERE a.p<>b.s) TO ? (FORMAT PARQUET)""",[str(out/'chains.pending.parquet')])
        con.close();os.replace(out/'chains.pending.parquet',out/'chains.parquet');engine.seal(out/'chains.parquet')
        compiled=engine.sealed(out/'chains.parquet')
    table=pq.read_table(compiled,use_threads=False)
    a=np.asarray(table['src'].to_numpy(),dtype=np.uint64);b=np.asarray(table['dst'].to_numpy(),dtype=np.uint64)
    d1=table['date1'].to_numpy().astype('datetime64[D]');d2=table['date2'].to_numpy().astype('datetime64[D]')
    if np.isnat(d1).any() or np.isnat(d2).any() or (d1>d2).any() or (d2>np.datetime64('2026-08-01')).any():
        raise ValueError('Monotonic dated original economic chains')
    age=np.maximum(0,(np.datetime64('2026-08-01')-d1).astype(float)/365.25)
    recent=(1/(1+age)).astype(np.float32)
    sources=np.concatenate((a,b));targets=np.concatenate((b,a))
    reverse=np.concatenate((np.zeros(len(a),dtype=bool),np.ones(len(a),dtype=bool)))
    recency=np.tile(recent,2);order=np.argsort(sources,kind='stable')
    sources=sources[order];targets=targets[order];reverse=reverse[order];recency=recency[order]
    del table,a,b,d1,d2,age,recent,order
    registry=np.load(verify(launch['keys']),mmap_mode='r',allow_pickle=False)
    at=np.searchsorted(registry,targets);valid=at<len(registry)
    valid[valid]&=registry[at[valid]]==targets[valid]
    sources=sources[valid];targets=targets[valid];reverse=reverse[valid];recency=recency[valid]
    registry._mmap.close()
    refs={};total=0;started=time.monotonic()
    for q in panel:
        guard(launch['deadline_unix']);qid=q['query_id'];path=feature_path(out,qid)
        saved=engine.sealed(path)
        if saved:
            with np.load(saved,allow_pickle=False) as f:n=len(f['keys'])
        else:
            with np.load(verify(original['features'][qid]),allow_pickle=False) as f:old_keys=f['keys'];old=f['features']
            lo=np.searchsorted(sources,old_keys,'left');hi=np.searchsorted(sources,old_keys,'right');counts=hi-lo
            edge=np.arange(int(counts.sum()))+np.repeat(lo-np.cumsum(counts)+counts,counts)
            keys=np.union1d(old_keys,targets[edge]);n=len(keys)
            x=np.zeros((n,len(PROOF_FIELDS)),dtype=np.float32);x[np.searchsorted(keys,old_keys),:old.shape[1]]=old
            if len(edge):
                fr=np.searchsorted(old_keys,sources[edge]);to=np.searchsorted(keys,targets[edge]);back=reverse[edge]
                for col,values in ((31,old[fr,5]),(33,old[fr,0]),(35,old[fr,10]),(37,old[fr,5]*recency[edge])):
                    np.maximum.at(x[:,col],to[~back],values[~back])
                    np.maximum.at(x[:,col+1],to[back],values[back])
            if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():raise ValueError('Bounded official typed chain proof')
            temp=path.with_suffix('.pending.npz');np.savez(temp,keys=keys,features=x)
            os.replace(temp,path);engine.seal(path)
            del old_keys,old,keys,x,lo,hi,counts,edge;gc.collect()
        refs[qid]=record(path);total+=n
        engine.progress(out,'TEMPORAL_CHAIN_FULL_STREAM',len(refs),total=len(panel),full_feature_rows=total,elapsed_s=time.monotonic()-started)
    authority(out)
    io.write(out/'features_generation.json',dict(status='COMPLETE_SEALED_QUERY_ONLY_ALL526_FULL_PROOFS',
        features=refs,queries=526,full_feature_rows=total,commune_sites=2554848,
        immutable_source=launch['prepared_generation'],chains=record(compiled),chain_observations=len(sources)//2,
        labels_used=False,no_candidate_budget=True,no_gt_directed_edges=True,no_alias_copy=True))

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
