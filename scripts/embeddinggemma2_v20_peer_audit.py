"""Post-seal DIRECT succession-neighbour proof audit, never generation.

Inspect actual endpoint/parent names and own addresses of neighbours instead
of mistaking parent-wide links or an ASSET_TRANSFER for target-site identity.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import time

import numpy as np

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v17_structural_retrieval as structural
from scripts import embeddinggemma2_v19_official_audit as official
from scripts import embeddinggemma2_v19_proof_retrieval as proof
from scripts import embeddinggemma2_v20_deep_audit as audit
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE=io.HOME/'v20/deep_audit_20261011_c'
OUTPUT=io.HOME/'v20/direct_peer_audit_20261011_a'


def run(out):
    import duckdb
    import pyarrow as pa
    if out.exists():raise ValueError('Fresh audit namespace only')
    started=time.monotonic();summary=io.read(verify(io.read(SOURCE/'state.json')['summary']))
    rows=[r for r in io.read(verify(summary['rows'])) if r['query_id'] in summary['unresolved_all_seven_channels']]
    if len(rows)!=26 or any(r['crm']['fold'] not in (2,3,4) for r in rows):raise ValueError('All26 previously sealed development misses')
    out.mkdir(parents=True);direct={}
    for r in rows:
        t=r['label']['siret'];direct[r['query_id']]=sorted({s['siretEtablissementSuccesseur'] if s['siretEtablissementPredecesseur']==t else s['siretEtablissementPredecesseur'] for s in r['official_audit']['direct_target_successions']})
    peers=sorted({p for values in direct.values() for p in values});parents=sorted({p[:9] for p in peers})
    dossier=io.read(verify(official.DOSSIER));root=Path(official.DOSSIER['path']).parent
    c=duckdb.connect();c.execute('SET threads=1');c.execute("SET memory_limit='512MB'")
    tmp=out/'duck_tmp';tmp.mkdir();c.execute('SET temp_directory=?',[str(tmp)])
    c.register('wanted',pa.table({'siren':parents}));c.register('endpoints',pa.table({'siret':peers}))
    refs=[];facts={}
    for filename in ('name_evidence.parquet','address_evidence.parquet'):
        ref=dict(path=str(root/filename),**dossier['files'][filename]);p=verify(ref);refs.append(ref)
        sql='''SELECT * FROM read_parquet(?) WHERE
            (subject_kind='SIREN' AND siren IN (SELECT siren FROM wanted)) OR
            (subject_kind='SIRET' AND siret IN (SELECT siret FROM endpoints))'''
        facts[filename]=official.serial(c.execute(sql,[str(p)]).fetch_arrow_table().to_pylist())
    c.close();io.write(out/'original_peer_source_records.json',facts)
    launch=io.read(io.HOME/'v14/representative_20261008_b/launch.json')
    arrays={k:np.load(verify(launch['geographic'][k]),mmap_mode='r',allow_pickle=False) for k in ('keys','xy')}
    def xy(key):
        loc=np.searchsorted(arrays['keys'],np.uint64(key))
        if loc>=len(arrays['keys']) or int(arrays['keys'][loc])!=int(key):return None
        x,y=arrays['xy'][loc]
        return [float(x),float(y)] if math.isfinite(x) and math.isfinite(y) and 0<=x<=1300000 and 6000000<=y<=7300000 else None
    names=defaultdict(list);addresses=defaultdict(list)
    for f in facts['name_evidence.parquet']:names[f['siren'] if f['subject_kind']=='SIREN' else f['siret']].append(f)
    for f in facts['address_evidence.parquet']:addresses[f['siren'] if f['subject_kind']=='SIREN' else f['siret']].append(f)
    output=[]
    for r in rows:
        qid=r['query_id'];t=r['label']['siret'];tx=xy(t);g=r['official_audit']['geometry'];qx=[g['ban_x_l93'],g['ban_y_l93']] if g['spatial_crs_compatible'] else None
        candidates=[]
        for peer in direct[qid]:
            nf=names[peer[:9]]+names[peer];ns=sorted(audit.normalized(v['raw_value'] for v in nf)-{'','ND'})
            af=addresses[peer];avs=[audit.source_address(v) for v in af];px=xy(peer)
            similarity=structural.similarity(structural.normalize(r['crm']['crm_name']),tuple(ns))
            a,st=proof.address_proof(proof.query_address(r['crm']),tuple(proof.address(v) for v in avs),similarity)
            candidates.append(dict(siret=peer,official_parent_and_own_names=ns,official_own_addresses=af,
                name_query_similarity=similarity,coherent_address_heuristic=a,street_query_similarity=st,
                peer_xy=px,target_to_peer_m=math.dist(tx,px) if tx and px else None,
                crm_to_peer_m=math.dist(qx,px) if qx and px else None,
                direct_successions=[s for s in r['official_audit']['direct_target_successions'] if peer in (s['siretEtablissementPredecesseur'],s['siretEtablissementSuccesseur'])],
                already_naturally_retrieved=peer in r['official_audit']['naturally_retrieved_direct_succession_sites']))
        output.append(dict(query_id=qid,crm=r['crm'],label=r['label'],target_xy=tx,
            crm_to_target_m=math.dist(qx,tx) if qx and tx else None,peers=candidates,
            graph_is_context_not_identity=True,label_error_proven=False))
    for a in arrays.values():a._mmap.close()
    io.write(out/'cases.json',output)
    report=dict(status='COMPLETE_POSTSEAL_DIRECT_PEER_AUDIT_NOT_RETRIEVAL',queries=26,with_direct_links=sum(bool(r['peers']) for r in output),
        with_peer_name_similarity85=sum(any(p['name_query_similarity']>=.85 for p in r['peers']) for r in output),
        with_peer_street_similarity90=sum(any(p['street_query_similarity']>=.90 for p in r['peers']) for r in output),
        with_peer_within300m_query=sum(any(p['crm_to_peer_m'] is not None and p['crm_to_peer_m']<=300 for p in r['peers']) for r in output),
        with_peer_and_target_within300m=sum(any(p['target_to_peer_m'] is not None and p['target_to_peer_m']<=300 for p in r['peers']) for r in output),
        cases=record(out/'cases.json'),source_refs=refs,source_summary=record(SOURCE/'summary.json'),implementation=record(Path(__file__)),
        no_retrieval_generation=True,no_fit=True,no_label_correction=True,closed_folds_opened=False,
        no_claim_of_recall_gain=True,elapsed_s=time.monotonic()-started)
    io.write(out/'summary.json',report);print({k:v for k,v in report.items() if k not in ('cases','source_refs','source_summary','implementation')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);run(p.parse_args().output)
