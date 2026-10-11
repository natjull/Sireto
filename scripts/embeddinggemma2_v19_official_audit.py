"""POST-V18-SEAL audit, never a retrieval feature/source/candidate generator.

Original name/address evidence and official relations only. Inspect every V18
miss including 26 all-channel misses; no label correction or denominator edit.
Never imports rejected V320 outputs or runs the unapproved V321 projection.
"""
import argparse
from collections import Counter,defaultdict
from datetime import date,datetime
from pathlib import Path
import sqlite3
import time

from scripts import embeddinggemma2_v18_identity_retrieval as previous
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v17_structural_retrieval as structural
from src.xgb_matcher.retrieval_v20_workbench import record,verify

DOSSIER=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/siren_dossier_v3/01eb624cca97c6f3/manifest.json',
    sha256='6d8ed0053b78862adafd76d0b2ac46e06ac5fc735e81723bb21229528fcea198',size_bytes=3417)
SUCCESSION=io.ROOT/'config/retrieval_v133_all_succession.json'


def serial(rows):
    return [{k:v.isoformat() if isinstance(v,(date,datetime)) else v for k,v in row.items()} for row in rows]


def categories(q,target,channels):
    ranks={tag:next((i for i,c in enumerate(values[q],1) if c['siret']==target),None) for tag,values in channels.items()}
    return ('FUSION_EVICTION' if any(ranks[k] for k in ('joint','lexical','oldgeo')) else
            'OTHER_CHANNEL_ONLY' if any(ranks.values()) else 'ALL_CHANNELS_MISS'),ranks


def run(out):
    import duckdb,pyarrow as pa,pyarrow.parquet as pq
    from scripts import embeddinggemma2_v15_probability_pilot as pilot
    from src.xgb_matcher.official_evidence import OfficialRelation
    from scripts import embeddinggemma2_v2_data as data
    if out.exists():raise ValueError('Immutable fresh audit namespace, no silent overwrite')
    state=io.read(previous.OUTPUT/'state.json');report=io.read(verify(state['report']))
    ex=io.read(previous.OUTPUT/'generate_exit.json');gen=io.read(verify(report['generation']))
    if state['status']!='COMPLETE_IDENTITY_SITE_COMPARISON' or ex['exit_code']!=0 or not ex['own_child_exit_confirmed'] or gen['status']!='COMPLETE_SEALED_ALL526':
        raise ValueError('Complete V18 seal/owned exit0 before audit labels')
    source=engine.authority(previous.SOURCE); manifest=io.read(verify(source['training_manifest']))
    membership=pilot.population.SOURCE/'validation_membership.json'
    if io.digest(membership)!=manifest['split_hashes']['validation']:raise ValueError('Original label authority')
    outcomes=io.read(previous.OUTPUT/'outcomes.json')['joint_lexical_oldgeo']
    if len(outcomes)!=526:raise ValueError('All-human original denominator')
    missed={r['query_id']:r for r in outcomes if not r['exact']}
    labels={r['query_id']:r for r in io.read(membership) if r['query_id'] in missed}
    if set(labels)!=set(missed) or any(r['fold'] not in (2,3,4) for r in labels.values()):raise ValueError('Only safe-fold missed labels, no test')
    panel={q['query_id']:q for q in io.read(verify(source['panel']))}
    outputs=io.read(verify(gen['outputs']));old=io.read(verify(io.read(previous.SOURCE/'generation.json')['outputs']))
    structural_gen=io.read(previous.GEO_SOURCE/'generation.json');v17=io.read(verify(structural_gen['outputs']))
    channels={'joint':outputs['joint'],'lexical':old['lexical'],'oldgeo':old['geo'],
        'site':old['gemma2'],'newgeo':v17['geo'],'family':v17['family']}
    parent_keys=sorted({l['siren'] for l in labels.values()});site_keys={l['siret'] for l in labels.values()}
    out.mkdir(parents=True);started=time.monotonic(); refs=[];epochs={}
    def pinned(ref):
        p=verify(ref);epochs[str(p)]=(p.stat().st_size,p.stat().st_mtime_ns);refs.append(ref);return p
    dossier=io.read(pinned(DOSSIER));root=Path(DOSSIER['path']).parent
    def source_file(name):return pinned(dict(path=str(root/name),**dossier['files'][name]))
    c=duckdb.connect();c.execute('SET threads=1');c.execute("SET memory_limit='512MB'")
    tmp=out/'duck_tmp';tmp.mkdir();c.execute('SET temp_directory=?',[str(tmp)])
    c.register('wanted',pa.table({'siren':parent_keys}))
    evidence={};
    for name,count in (('name_evidence.parquet',162537492),('address_evidence.parquet',142096607)):
        engine.guard();path=source_file(name)
        if pq.ParquetFile(path).metadata.num_rows!=count:raise ValueError('Complete canonical source count')
        rows=c.execute('SELECT * FROM read_parquet(?) WHERE siren IN (SELECT siren FROM wanted)',[str(path)]).fetch_arrow_table().to_pylist()
        evidence[name]=serial(rows)
        io.write(out/'state.json',dict(status='ORIGINAL_OFFICIAL_SOURCE_AUDIT',source=name,matched_records=len(rows),elapsed_s=time.monotonic()-started))
    relation_path=source_file('relations.parquet')
    if pq.ParquetFile(relation_path).metadata.num_rows!=692010:raise ValueError('Complete raw original relations')
    rows=c.execute('''SELECT * FROM read_parquet(?) WHERE
        (from_kind='SIREN' AND from_identifier IN (SELECT siren FROM wanted)) OR
        (to_kind='SIREN' AND to_identifier IN (SELECT siren FROM wanted))''',[str(relation_path)]).fetch_arrow_table().to_pylist()
    relations=serial(rows)
    for row,original in zip(rows,relations):
        canonical=OfficialRelation.from_dict(row).to_dict()
        if any(canonical[k]!=v for k,v in original.items()):raise ValueError('Exact official relation values/types, no repair')
    succession_cfg=io.read(SUCCESSION);succession_path=pinned(succession_cfg['source'])
    if pq.ParquetFile(succession_path).metadata.num_rows!=9730972:raise ValueError('Whole raw succession source')
    rows=c.execute('''SELECT * FROM read_parquet(?) WHERE
        substr(siretEtablissementPredecesseur,1,9) IN (SELECT siren FROM wanted) OR
        substr(siretEtablissementSuccesseur,1,9) IN (SELECT siren FROM wanted)''',[str(succession_path)]).fetch_arrow_table().to_pylist()
    successions=serial(rows);c.close()
    io.write(out/'original_source_records.json',dict(names=evidence['name_evidence.parquet'],addresses=evidence['address_evidence.parquet'],relations=relations,successions=successions,
        originals_preserved=True,transfer_is_not_identity=True,sibling_addresses_not_own_addresses=True))
    dbref=io.read(previous.SOURCE/'names/generation.json')['database'];con=sqlite3.connect(f'file:{pinned(dbref)}?mode=ro&immutable=1',uri=True)
    adjacency=defaultdict(set)
    for r in relations:
        if r['from_kind']=='SIREN' and r['to_kind']=='SIREN' and r['from_identifier']!=r['to_identifier']:
            adjacency[r['from_identifier']].add(r['to_identifier']);adjacency[r['to_identifier']].add(r['from_identifier'])
    neighbors={p:sorted(engine.lookup(con,1,[p])[int(p)]) for p in sorted({v for values in adjacency.values() for v in values})}
    # Query name lookup in ALL official owners, not an inferred NAOS/AUBERT alias.
    official_naos=[dict(kind=r[0],owner_key=str(r[1]),name=r[2]) for r in con.execute("SELECT kind,owner_key,name FROM aliases WHERE name='NAOS'")]
    con.close();byparent=defaultdict(list);addresses=defaultdict(list);succ=defaultdict(list)
    for row in evidence['name_evidence.parquet']:byparent[row['siren']].append(row)
    for row in evidence['address_evidence.parquet']:addresses[row['siren']].append(row)
    for row in successions:
        for p in {row['siretEtablissementPredecesseur'][:9],row['siretEtablissementSuccesseur'][:9]}:succ[p].append(row)
    findings=[]
    for q in sorted(missed):
        label=labels[q];p=label['siren'];target=label['siret'];category,ranks=categories(q,target,channels)
        name=structural.normalize(data.safe(panel[q]['crm_name']))
        own=[r for r in byparent[p] if r['subject_kind']=='SIREN' or r['siret']==target]
        siblings=[r for r in byparent[p] if r['subject_kind']=='SIRET' and r['siret']!=target]
        own_addresses=[r for r in addresses[p] if r['subject_kind']=='SIRET' and r['siret']==target]
        parent_addresses=[r for r in addresses[p] if r['subject_kind']=='SIREN']
        natural={candidate['siren'] for values in channels.values() for candidate in values[q]}
        neighbor_names=[n for neighbor in adjacency[p] for n in neighbors[neighbor]]
        normalize=lambda values:tuple(dict.fromkeys(structural.normalize(data.safe(v)) for v in values if data.safe(v)))
        findings.append(dict(query_id=q,crm=panel[q],label=label,category=category,ranks=ranks,
            own_parent_official_name_similarity=structural.similarity(name,normalize([r['raw_value'] for r in own])),
            sibling_context_name_similarity=structural.similarity(name,normalize([r['raw_value'] for r in siblings])),
            related_parent_name_similarity=structural.similarity(name,normalize(neighbor_names)),
            own_address_records=own_addresses,parent_address_context_records=parent_addresses,
            linked_parent_context_names={other:neighbors[other] for other in sorted(adjacency[p])},
            naturally_retrieved_related_parent_seeds=sorted(adjacency[p]&natural),
            succession_records=succ[p],same_siren_found=missed[q]['siren'],operational_found=missed[q]['operational'],
            label_error_proven=False,no_alias_or_identity_inferred=True))
    for path,epoch in epochs.items():
        p=Path(path)
        if (p.stat().st_size,p.stat().st_mtime_ns)!=epoch:raise ValueError('Official source epoch drift')
    io.write(out/'findings.json',findings)
    summary=dict(status='POSTSEAL_V18_SOURCE_DIAGNOSTIC_NOT_RETRIEVAL',misses=len(findings),all_human_denominator=526,
        categories=dict(Counter(r['category'] for r in findings)),
        misses_with_transfer_context=sum(bool(r['linked_parent_context_names']) for r in findings),
        misses_with_natural_transfer_seed=sum(bool(r['naturally_retrieved_related_parent_seeds']) for r in findings),
        misses_with_succession_context=sum(bool(r['succession_records']) for r in findings),
        official_naos_owner_records=official_naos,
        name_sources=dict(Counter(r['source'] for r in evidence['name_evidence.parquet'])),
        address_sources=dict(Counter(r['source'] for r in evidence['address_evidence.parquet'])),
        findings=record(out/'findings.json'),original_records=record(out/'original_source_records.json'),
        source_report=state['report'],sources=refs,implementation=record(Path(__file__)),elapsed_s=time.monotonic()-started,
        label_errors_proven=0,labels_corrected=0,denominator_changed=False,no_retrieval_generation=True,
        no_claim_of_expansion_gain=True,never_use_as_retrieval_features=True,fold0_opened=False,fold1_opened=False,test_opened=False)
    io.write(out/'summary.json',summary);io.write(out/'completion.json',dict(status='COMPLETE_POSTSEAL_OFFICIAL_AUDIT',summary=record(out/'summary.json')))
    print({k:v for k,v in summary.items() if k in ('misses','categories','misses_with_transfer_context','misses_with_natural_transfer_seed','misses_with_succession_context','elapsed_s')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.output)
    except Exception as e:
        if a.output.exists():io.write(a.output/'execution_error.json',dict(error=repr(e),updated_unix=time.time()))
        raise
