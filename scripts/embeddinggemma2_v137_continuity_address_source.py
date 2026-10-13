"""Query-only feasibility: economic continuity plus BOTH own public addresses.

Coordinates, names and addresses are never copied to the linked endpoint.
Exact house identity and mere same-street support are distinct proof levels.
Only already-published supplier brand context selects the source scope; no
human target, predictions, held-out fold or metrics are read.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import re
import time

import numpy as np
from scripts.embeddinggemma2_v131_mechanism_audit import HOME, BASE, read, write, record, verify
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from scripts.embeddinggemma2_v133_supplier_site_context import normalize as name_tokens

OUTPUT=HOME/'v137/continuity_address_source_20261013_night'
PREPARED=HOME/'v133/supplier_site_context_20261013_night'
RECIPE=dict(hypothesis='DATED_ECONOMIC_CONTINUITY_PLUS_BOTH_OWN_PUBLIC_ADDRESSES',
    scope='ALL_SOURCE_LINKS_TOUCHING_PUBLISHED_CLIENT_OWNER_NAME_MATCHES_NO_GT_SELECTION',
    publication_scope='ONE_PUBLISHED_SUPPLIER_SECTION_DIAGNOSTIC_PARTIAL_NOT_NATIONAL_GENERALIZATION',
    public_diffusion='BOTH_ENDPOINT_SITES_AND_BOTH_LEGAL_OWNERS_O_BEFORE_ADDRESS_READ',
    link='DIRECT_ECONOMIC_CONTINUITY_TRUE_DATED_ON_OR_BEFORE_2026_08_01',
    direction='BIDIRECTIONAL_TYPED_CONTINUITY_CONTEXT_NOT_IDENTITY',
    strong_address='EXACT_COMMUNE_AND_NORMALIZED_OWN_STREET_AND_VALID_HOUSE_NUMBER_SUFFIX',
    weak_address='EXACT_COMMUNE_AND_NORMALIZED_OWN_STREET_IF_ONE_OR_BOTH_NUMBERS_ABSENT',
    distinct_known_numbers='NO_MATCH',malformed_or_range_house_numbers='NO_MATCH',
    strong_address_strength=1.,weak_address_strength=.5,
    weak_strength_reuses_existing_half_reliability_not_tuned=True,
    query_source_support='MAX_SOURCE_OWN_GEO_SOURCE_FILLED_OWN_GEO_EXACT_CRM_SOURCE_ADDRESS_STRENGTH',
    context_confidence='FROZEN_PUBLICATION_05_TIMES_QUERY_SOURCE_SUPPORT_TIMES_ENDPOINT_ADDRESS_STRENGTH',
    coefficient='UNCHANGED_HELDOUT_CONTEXT_COEFFICIENT_W6_IF_EVER_ABLATED',
    missing_target_xy_allowed_but_not_inferred=True,no_geometry_transferred=True,
    own_names_or_addresses_not_transferred=True,duplicate_source_links='MAX_NOT_SUM',
    all526_query_only=True,no_index_search=True,no_native_retrieval=True,no_recall=True,
    no_training=True,no_label_change=True,production_promotion_allowed=False,
    eligibility='DIAGNOSTIC_ONLY_INHERITED_PARTIAL_PUBLISHED_SUPPLIER_SOURCE_AND_POSTCUTOFF_BASE')


def house(value,suffix):
    raw=' '.join(str(v).strip() for v in (value or '',suffix or '') if str(v).strip())
    if not raw:return None,False
    text=normalize(raw)
    match=re.fullmatch(r'([0-9]+)(?:\s*(BIS|TER|QUATER|[A-Z]))?',text)
    if not match:return None,True
    return (str(int(match.group(1))),match.group(2) or ''),False


def address_strength(a,b):
    if not a or not b or not a['insee'] or a['insee']!=b['insee']:
        return 0.,'NO_EXACT_COMMUNE'
    if not a['street'] or a['street']!=b['street']:
        return 0.,'NO_EXACT_STREET'
    ah,bh=house(a['number'],a['suffix']),house(b['number'],b['suffix'])
    if ah[1] or bh[1]:return 0.,'UNSUPPORTED_HOUSE_FORMAT'
    if ah[0] and bh[0]:
        return (1.,'EXACT_PUBLIC_HOUSE') if ah[0]==bh[0] else (0.,'DIFFERENT_KNOWN_HOUSE')
    return .5,'SAME_PUBLIC_STREET_NUMBER_ABSENT_NOT_CERTIFIED_BUILDING'


def main(out,deadline):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    if out.exists():raise ValueError('Fresh source hypothesis namespace required')
    if not time.time()<deadline<=time.time()+8*3600:raise ValueError('Authorized absolute deadline')
    previous,owned=read(PREPARED/'generation.json'),read(PREPARED/'owned_tool_exit.json')
    if previous['status']!='COMPLETE_QUERY_ONLY_ALL526_SUPPLIER_SITE_CONTEXT' or owned['exit_code'] or not owned['own_tool_exit_confirmed']:
        raise ValueError('Complete observed prior source only, not outcomes')
    launch=read(BASE/'launch.json');fg=read(BASE/'features_generation.json')
    panel=read(verify(launch['panel']))
    if len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('Complete safe query scope')
    indirect=read(verify(previous['source_brands']))
    provider=name_tokens(indirect['provider_published_brand'])
    owners=np.array(sorted({int(s) for b in indirect['client_brands'] for s in b['owner_refs']}),dtype=np.uint64)
    out.mkdir(parents=True);(out/'spill').mkdir();(out/'shards').mkdir()
    write(out/'recipe.json',RECIPE)
    write(out/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prior_source=record(PREPARED/'generation.json'),prior_owned_exit=record(PREPARED/'owned_tool_exit.json'),
        query_panel=launch['panel'],source_links=previous['source_links'],source_brands=previous['source_brands'],
        source_stock=launch['stock_source'],source_legal_units=previous['source_legal_units'],
        helpers=[record(Path(normalize.__code__.co_filename)),record(Path(name_tokens.__code__.co_filename))],
        deadline_unix=deadline,rules_fixed_before_source_counts=True,labels_read=False,created_unix=time.time()))
    table=pq.read_table(verify(previous['source_links']),columns=['siretEtablissementPredecesseur',
        'siretEtablissementSuccesseur','dateLienSuccession','continuiteEconomique'],use_threads=False)
    a=np.asarray(table['siretEtablissementPredecesseur'].to_pylist(),dtype=np.uint64)
    b=np.asarray(table['siretEtablissementSuccesseur'].to_pylist(),dtype=np.uint64)
    date=np.asarray(table['dateLienSuccession'].to_numpy(),dtype='datetime64[D]')
    continuity=np.asarray(table['continuiteEconomique'].to_pylist(),dtype=bool)
    if not continuity.all() or np.isnat(date).any() or np.any(date>np.datetime64('2026-08-01')):
        raise ValueError('Explicit unchanged dated continuity source')
    left=np.concatenate((a,b));right=np.concatenate((b,a));dates=np.concatenate((date,date))
    match=np.isin(left//100000,owners);left,right,dates=left[match],right[match],dates[match]
    del table,a,b,date,continuity,match;gc.collect()
    refs=np.union1d(left,right)
    con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='512MB'")
    con.execute('SET temp_directory=?',[str(out/'spill')]);con.register('wanted',pa.table({'site_key':refs}))
    rows=con.execute("""SELECT CAST(e.siret AS UBIGINT) site_key,e.codeCommuneEtablissement insee,
      e.numeroVoieEtablissement number,e.indiceRepetitionEtablissement suffix,
      e.typeVoieEtablissement street_type,e.libelleVoieEtablissement street
      FROM read_parquet(?) e JOIN wanted w ON CAST(e.siret AS UBIGINT)=w.site_key
      JOIN read_parquet(?) u USING(siren)
      WHERE e.statutDiffusionEtablissement='O' AND u.statutDiffusionUniteLegale='O'""",
      [launch['stock_source']['path'],previous['source_legal_units']['path']]).fetch_arrow_table().to_pylist()
    con.close()
    own={r['site_key']:dict(insee=r['insee'] or '',number=r['number'],suffix=r['suffix'],
        street=normalize(' '.join(str(r[k] or '') for k in ('street_type','street')),street=True)) for r in rows}
    facts=[];compatible=[]
    for i,(a,b) in enumerate(zip(left,right)):
        if int(a) not in own or int(b) not in own:continue
        strength,kind=address_strength(own[int(a)],own[int(b)])
        facts.append(dict(source_siret=f'{int(a):014d}',linked_siret=f'{int(b):014d}',date=str(dates[i]),
            source_address=own[int(a)],linked_address=own[int(b)],compatibility=kind,strength=strength,
            own_endpoint_facts_not_transferred=True))
        if strength:compatible.append((int(a),int(b),strength,kind))
    src=np.array([r[0] for r in compatible],dtype=np.uint64)
    dst=np.array([r[1] for r in compatible],dtype=np.uint64)
    power=np.array([r[2] for r in compatible],dtype=np.float64)
    official=np.load(launch['keys']['path'],mmap_mode='r',allow_pickle=False)
    xy=np.load(launch['xy']['path'],mmap_mode='r',allow_pickle=False)
    pos=np.searchsorted(official,dst);present=pos<len(official);present[present]&=official[pos[present]]==dst[present]
    target_xy_valid=np.zeros(len(dst),dtype=bool)
    if present.any():
        point=np.asarray(xy[pos[present]])
        okay=np.isfinite(point).all(axis=1)&(point[:,0]>=0)&(point[:,0]<=1300000)&(point[:,1]>=6000000)&(point[:,1]<=7300000)
        target_xy_valid[np.flatnonzero(present)]=okay
    official._mmap.close();xy._mmap.close()
    extras={};coverage=[]
    for q in panel:
        if time.time()>=deadline:raise TimeoutError('Hard deadline; preserve source artifacts')
        qid=q['query_id']
        if not provider<=name_tokens(q['crm_name']):
            extras[qid]=None;continue
        with np.load(verify(fg['features'][qid]),allow_pickle=False) as f:
            keys=f['keys'];x=f['features']
            if x.shape!=(len(keys),51):raise ValueError('Unchanged full numeric proof schema')
            source_at=np.searchsorted(keys,src);target_at=np.searchsorted(keys,dst)
            good=(source_at<len(keys))&(target_at<len(keys));at=np.flatnonzero(good)
            at=at[(keys[source_at[at]]==src[at])&(keys[target_at[at]]==dst[at])]
            source_geo=np.maximum(x[source_at[at],5],x[source_at[at],42]).astype(np.float64)
            query_address=dict(insee=q['crm_insee'],number=q['crm_number'],suffix=q['crm_suffix'],
                street=normalize(q['crm_street'],street=True))
            source_address=np.array([address_strength(query_address,own[int(src[j])])[0] for j in at])
            support=np.maximum(source_geo,source_address)
            proof=.5*support*power[at]
            scores=np.zeros(len(keys),dtype=np.float64);np.maximum.at(scores,target_at[at],proof)
            if not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>.500001):
                raise ValueError('Finite bounded frozen source confidence')
            chosen=np.flatnonzero(scores>0)
            path=out/'shards'/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')
            np.savez_compressed(path,keys=keys[chosen],confidence=scores[chosen])
            ref=record(path);write(path.with_suffix('.seal.json'),ref);extras[qid]=ref
            coverage.append(dict(query_id=qid,query=q,compatible_public_pairs_in_full_proof_universe=len(at),
                positive_context_leaves=len(chosen),positive_address_supported_edges=int(np.count_nonzero(proof)),
                positive_edges_target_without_original_xy=int(np.count_nonzero((proof>0)&~target_xy_valid[at])),
                strong_positive_edges=int(np.count_nonzero((proof>0)&(power[at]==1))),
                weak_same_street_positive_edges=int(np.count_nonzero((proof>0)&(power[at]==.5)))))
        gc.collect()
    if len(extras)!=526:raise ValueError('No denominator filtering')
    write(out/'public_address_pair_facts.json',facts);write(out/'query_extras.json',extras);write(out/'coverage.json',coverage)
    from collections import Counter
    summary=dict(public_direct_observations=len(facts),compatibility=dict(Counter(r['compatibility'] for r in facts)),
        provider_queries=len(coverage),queries_with_positive_context=sum(r['positive_context_leaves']>0 for r in coverage),
        positive_context_leaves=sum(r['positive_context_leaves'] for r in coverage),
        positive_edges_to_missing_xy=sum(r['positive_edges_target_without_original_xy'] for r in coverage))
    for name in ('public_address_pair_facts.json','query_extras.json','coverage.json'):
        p=out/name;write(p.with_suffix('.seal.json'),record(p))
    write(out/'generation.json',dict(status='COMPLETE_QUERY_ONLY_ALL526_CONTINUITY_ADDRESS_FEASIBILITY_NOT_RECALL',
        implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),query_panel=launch['panel'],
        facts=record(out/'public_address_pair_facts.json'),extras=record(out/'query_extras.json'),coverage=record(out/'coverage.json'),
        summary=summary,no_xy_names_or_addresses_inferred_or_transferred=True,no_target_labels_read=True,
        no_retrieval_or_fitting=True,production_promotion_allowed=False,observed_unix=time.time()))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--deadline-unix',required=True,type=float);args=p.parse_args()
    main(args.output,args.deadline_unix)
