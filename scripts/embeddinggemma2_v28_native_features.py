"""V28: unchanged sealed V23 full proofs; only training objective/mining changes."""
from pathlib import Path
import argparse
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from src.xgb_matcher.retrieval_v20_workbench import record, verify

PREPARED=io.HOME/'v23/hierarchical_native_20261012_a'
OUTPUT=io.HOME/'v28/official_sector_native_20261012_a'
SECTOR=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sector_official_v1/official_sector_evidence.parquet')
PROOF_FIELDS=list(source.PROOF_FIELDS)+['sector_own_name','sector_own_street','sector_own_number',
    'sector_own_street_x_number','sector_own_name_x_geo','sector_parent_name',
    'sector_parent_name_x_geo','sector_own_administrative','sector_own_same_observation_address']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
RECIPE=dict(source.RECIPE,feature_fields=FEATURE_FIELDS,
    public_sector_sources='FULL_ORIGINAL_FINNESS_EDUCATION_DINUM_STRUCTURED_IDS',
    direct_site_source_all_national_sites_before100=True,
    owner_source_scope='PARENT_CONTEXT_ON_EXISTING_FULL_QUERY_PROOF_FAMILIES',
    source_date_cutoff='2026-08-01_UNKNOWN_DATES_RETAINED_AS_UNKNOWN',
    same_source_observation_address_required=True,
    sector_owner_names_are_not_site_aliases=True,feature_values_changed=True)

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
    import pyarrow.parquet as pq
    from collections import defaultdict
    from scripts import embeddinggemma2_v19_proof_retrieval as address_tools
    launch=authority(out);original=io.read(verify(launch['prepared_generation']))
    panel=io.read(verify(launch['panel']))
    table=pq.read_table(verify(launch['sector_source']),use_threads=False)
    site=defaultdict(list);parent=defaultdict(set);future=0;undated=0
    for row in table.to_pylist():
        observed=str(row['observed_at'] or '')[:10]
        if observed and observed>'2026-08-01':future+=1;continue
        if not observed:undated+=1
        name=structural.normalize(data.safe(row['name']))
        if row['subject_kind']=='SIRET':
            key=row['siret']
            if not key or len(key)!=14 or not key.isdigit():raise ValueError('Structured public site identifier')
            raw=data.safe(row['address']);number=data.safe(row['number'])
            parsed=address_tools.address(raw)
            if not parsed[0] and number:parsed=address_tools.address(number+' '+raw)
            site[int(key)].append((name,parsed,str(row['insee'] or ''),str(row['postcode'] or '')))
        elif row['subject_kind']=='SIREN':
            key=row['siren']
            if not key or len(key)!=9 or not key.isdigit():raise ValueError('Structured public owner identifier')
            if name:parent[int(key)].add(name)
        else:raise ValueError('Typed site/owner source only')
    del table
    registry=np.load(verify(launch['keys']),mmap_mode='r',allow_pickle=False)
    xy=np.load(verify(launch['xy']),mmap_mode='r',allow_pickle=False)
    sector_keys=np.asarray(sorted(site),dtype=np.uint64);at=np.searchsorted(registry,sector_keys)
    found=at<len(registry);found[found]&=registry[at[found]]==sector_keys[found]
    absent=int((~found).sum());sector_keys=sector_keys[found];coords=np.array(xy[at[found]],dtype=float)
    registry._mmap.close();xy._mmap.close()
    valid=np.isfinite(coords).all(axis=1)&(coords[:,0]>=0)&(coords[:,0]<=1300000)&(coords[:,1]>=6000000)&(coords[:,1]<=7300000)
    geometries={r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    refs={};total=0;started=time.monotonic()
    for q in panel:
        guard(launch['deadline_unix']);qid=q['query_id'];path=feature_path(out,qid)
        saved=engine.sealed(path)
        if saved:
            with np.load(saved,allow_pickle=False) as f:n=len(f['keys'])
        else:
            with np.load(verify(original['features'][qid]),allow_pickle=False) as f:old_keys=f['keys'];old=f['features']
            keys=np.union1d(old_keys,sector_keys);x=np.zeros((len(keys),len(PROOF_FIELDS)),dtype=np.float32)
            x[np.searchsorted(keys,old_keys),:len(source.PROOF_FIELDS)]=old
            loc=np.searchsorted(keys,sector_keys);geometry=geometries[qid]
            geo=np.zeros(len(sector_keys),dtype=np.float32)
            if geometry['spatial_crs_compatible']:
                distance=np.hypot(coords[:,0]-geometry['ban_x_l93'],coords[:,1]-geometry['ban_y_l93'])
                reliability=.5 if geometry['ban_declared_weak'] else 1.
                geo[valid]=reliability/(1+distance[valid]/(300+max(0,geometry['ban_uncertainty_m'])))
            # Existing V23 values stay unchanged; new physical leaves get OWN XY only.
            old_at=np.searchsorted(old_keys,sector_keys);inside=old_at<len(old_keys)
            inside[inside]&=old_keys[old_at[inside]]==sector_keys[inside]
            x[loc[~inside],5]=geo[~inside]
            name=structural.normalize(data.safe(q['crm_name']));qa=address_tools.query_address(q)
            for j,key in enumerate(sector_keys):
                aliases=site[int(key)];scores=x[loc[j],len(source.PROOF_FIELDS):]
                scores[0]=structural.similarity(name,tuple(r[0] for r in aliases if r[0]))
                for n,addr,zone,cp in aliases:
                    street=address_tools.street_similarity(qa[2],addr[2])
                    number=address_tools.number_grade(qa[0],qa[1],addr[0],addr[1])
                    administrative=float(bool(q['crm_insee'] and zone==q['crm_insee']) or bool(q['crm_postcode'] and cp==q['crm_postcode']))
                    scores[1]=max(scores[1],street);scores[2]=max(scores[2],number)
                    scores[3]=max(scores[3],street*number);scores[7]=max(scores[7],administrative)
                    # Street+number+admin must be same source observation, not Frankenaddress.
                    scores[8]=max(scores[8],street*number*administrative)
                scores[4]=scores[0]*geo[j]
            groups,group=np.unique(keys//100000,return_inverse=True)
            owner=np.asarray([structural.similarity(name,tuple(parent.get(int(p),()))) for p in groups],dtype=np.float32)
            x[:,len(source.PROOF_FIELDS)+5]=owner[group]
            x[:,len(source.PROOF_FIELDS)+6]=owner[group]*x[:,16]
            if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():raise ValueError('Bounded official role-separated fields')
            temp=path.with_suffix('.pending.npz');np.savez(temp,keys=keys,features=x);os.replace(temp,path);engine.seal(path)
            n=len(keys);del old_keys,old,keys,x;gc.collect()
        refs[qid]=record(path);total+=n
        engine.progress(out,'FULL_PUBLIC_SECTOR_SOURCE_PROOFS',len(refs),total=len(panel),full_feature_rows=total,elapsed_s=time.monotonic()-started)
    if len(refs)!=526:raise ValueError('All526 complete')
    io.write(out/'features_generation.json',dict(status='COMPLETE_SEALED_QUERY_ONLY_ALL526_FULL_PROOFS',
        features=refs,queries=526,full_feature_rows=total,commune_sites=original['commune_sites'],
        sector_source=launch['sector_source'],sector_manifest=launch['sector_manifest'],
        complete_national_sector_site_count=len(sector_keys),source_sector_site_absent_catalog=absent,
        source_records_future_excluded=future,source_records_undated=undated,
        whole_source_scanned=True,public_owner_names_are_context_not_site_aliases=True,
        source_date_cutoff='2026-08-01',labels_used=False,no_candidate_budget=True,no_gt_directed_edges=True))


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


