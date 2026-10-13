"""Whole-source corporate incidence for typed contextual retrieval, not identity.

No CRM or labels. Distinct public corporate subjects may share a declared RNE
corporate actor. This is NOT ownership, equivalence or current affiliation.
Only identifiers with published public corporate Stock UL nodes are linked;
unresolved foreign actor names are retained as direct published context only.
"""
from collections import defaultdict
from pathlib import Path
import re
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v23_native_features as original
from scripts import embeddinggemma2_v30_native_features as roles
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT=io.HOME/'v72/corporate_incidence_20261013_a'
STOCK=mandate.audit_utils.STOCK
CUTOFF='2026-08-01'


def actor_name(value):
    name=original.structural.normalize(value or '')
    return re.split(r'\b(?:REPRESENTE|REPRESENTEE|REPRESENTES|REPRESENTEES|REPR)\s+(?:PAR|P[.]?)\b',name,maxsplit=1)[0].strip()


def main():
    import duckdb
    import pyarrow.parquet as pq
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh immutable source namespace')
    OUTPUT.mkdir(parents=True);(OUTPUT/'spill').mkdir()
    launch=dict(implementation=record(Path(__file__)),roles=record(roles.ROLE_SOURCE),stock=record(STOCK),
        cutoff=CUTOFF,deadline_unix=mandate.DEADLINE,crm_read=False,labels_read=False,
        context_not_identity_or_ownership=True,current_status_not_asserted=True)
    io.write(OUTPUT/'launch.json',launch);start=time.monotonic()
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.execute('SET temp_directory=?',[str(OUTPUT/'spill')])
    c.execute("""CREATE TABLE r AS SELECT * FROM read_parquet(?)
        WHERE regexp_full_match(subject_siren,'[0-9]{9}') AND subject_siren<>'000000000'
        AND (effective_date IS NULL OR substr(CAST(effective_date AS VARCHAR),1,10)<=?)
        AND (observed_at IS NULL OR substr(CAST(observed_at AS VARCHAR),1,10)<=?)""",
        [str(verify(launch['roles'])),CUTOFF,CUTOFF])
    c.execute("""CREATE TABLE u AS SELECT s.* FROM read_parquet(?) s WHERE
        s.siren IN (SELECT subject_siren FROM r UNION SELECT related_siren FROM r)""",[str(verify(launch['stock']))])
    c.execute("""CREATE TABLE nodes AS SELECT DISTINCT siren,denominationUniteLegale,
        denominationUsuelle1UniteLegale,denominationUsuelle2UniteLegale,
        denominationUsuelle3UniteLegale,sigleUniteLegale FROM u WHERE
        statutDiffusionUniteLegale='O' AND categorieJuridiqueUniteLegale<>'1000'
        AND coalesce(denominationUniteLegale,'')<>'' ORDER BY siren""")
    nodes=c.execute('SELECT * FROM nodes').fetchall()
    ids=np.asarray([int(r[0]) for r in nodes],dtype=np.uint64)
    names={str(r[0]):sorted({original.structural.normalize(v) for v in r[1:] if v and original.structural.normalize(v)}) for r in nodes}
    edges=c.execute("""SELECT DISTINCT r.subject_siren,r.related_siren,r.role_code
        FROM r JOIN nodes s ON r.subject_siren=s.siren JOIN nodes a ON r.related_siren=a.siren
        WHERE r.subject_siren<>r.related_siren
        ORDER BY r.subject_siren,r.related_siren,r.role_code""").fetchall()
    if not edges:raise ValueError('Actual nonempty public typed corporate graph')
    edge_subject=np.searchsorted(ids,np.asarray([int(r[0]) for r in edges],dtype=np.uint64))
    edge_actor=np.searchsorted(ids,np.asarray([int(r[1]) for r in edges],dtype=np.uint64))
    # One incidence per subject/actor; duplicate roles remain in the source
    # Parquet but must never inflate graph degree or independent support.
    pairs=np.unique(np.column_stack((edge_subject,edge_actor)),axis=0)
    unknown=defaultdict(set)
    unknown_rows=c.execute("""SELECT r.subject_siren,r.related_denomination,r.related_siren
        FROM r JOIN nodes s ON r.subject_siren=s.siren LEFT JOIN u a ON r.related_siren=a.siren
        WHERE (NOT coalesce(regexp_full_match(r.related_siren,'[0-9]{9}'),FALSE)
               OR r.related_siren='000000000')
        AND coalesce(a.statutDiffusionUniteLegale,'O')<>'P'""").fetchall()
    for subject,name,actor in unknown_rows:
        cleaned=actor_name(name)
        if cleaned:unknown[subject].add(cleaned)
    c.close()
    np.savez(OUTPUT/'incidence.npz',ids=ids,subject=pairs[:,0].astype(np.int32),actor=pairs[:,1].astype(np.int32))
    io.write(OUTPUT/'names.json',names);io.write(OUTPUT/'unresolved_actor_names.json',{s:sorted(v) for s,v in sorted(unknown.items())})
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist([dict(subject_siren=s,actor_siren=a,role_code=k) for s,a,k in edges]),OUTPUT/'role_edges.parquet',compression='zstd')
    for ref in (launch['roles'],launch['stock']):verify(ref)
    degree=np.bincount(pairs[:,1],minlength=len(ids))
    artifacts={n:record(OUTPUT/n) for n in ('incidence.npz','names.json','unresolved_actor_names.json','role_edges.parquet')}
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_WHOLE_SOURCE_PUBLIC_CORPORATE_INCIDENCE',
        launch=record(OUTPUT/'launch.json'),artifacts=artifacts,nodes=len(ids),role_edges=len(edges),
        distinct_incidences=len(pairs),actors_with_multiple_subjects=int((degree>=2).sum()),
        maximum_actor_degree=int(degree.max()),unresolved_actor_subjects=len(unknown),
        unresolved_actor_source_rows=len(unknown_rows),unresolved_names_never_entity_joined=True,
        public_corporate_nodes_only=True,physical_people_not_read=True,dates_unknown_retained_as_unknown=True,
        closed_companies_retained=True,no_current_affiliation_claim=True,
        identifiers_reference_joins_only_not_predictive_values=True,labels_read=False,crm_read=False,
        elapsed_s=time.monotonic()-start))


class Incidence:
    """Exact max across all distinct peer subjects, with self exclusion.

    Hub attenuation sqrt(degree-1) is source-structural, not fitted to misses.
    It prevents e.g. common fiduciaries from masquerading as company identity.
    Query similarity uses published corporate names only, no numeric ID feature.
    """
    def __init__(self):
        self.generation=io.read(OUTPUT/'generation.json')
        refs=self.generation['artifacts']
        with np.load(verify(refs['incidence.npz']),allow_pickle=False) as f:
            self.ids=f['ids'];self.subject=f['subject'];self.actor=f['actor']
        names=io.read(verify(refs['names.json']));self.unknown=io.read(verify(refs['unresolved_actor_names.json']))
        self.aliases=sorted({v for values in names.values() for v in values})
        positions={v:i for i,v in enumerate(self.aliases)}
        owner=[];alias=[]
        for i,s in enumerate(self.ids):
            for name in names[str(int(s))]:owner.append(i);alias.append(positions[name])
        self.owner_at=np.asarray(owner,dtype=np.int32);self.alias_at=np.asarray(alias,dtype=np.int32)
        self.degree=np.bincount(self.actor,minlength=len(self.ids))

    def query(self,name):
        from rapidfuzz import process,fuzz
        name=original.structural.normalize(name or '')
        if not name:return np.zeros(len(self.ids),dtype=np.float32)
        similarities=process.cdist([name],self.aliases,scorer=fuzz.WRatio,processor=None,workers=1,dtype=np.float32)[0]/100
        own=np.zeros(len(self.ids),dtype=np.float32);np.maximum.at(own,self.owner_at,similarities[self.alias_at])
        edge=own[self.subject];first=np.zeros(len(self.ids),dtype=np.float32);np.maximum.at(first,self.actor,edge)
        maximal=edge==first[self.actor]
        ties=np.bincount(self.actor[ maximal ],minlength=len(self.ids))
        second=np.zeros(len(self.ids),dtype=np.float32);np.maximum.at(second,self.actor[~maximal],edge[~maximal])
        peer=first[self.actor].copy();unique_self=maximal&(ties[self.actor]==1)
        peer[unique_self]=second[self.actor[unique_self]]
        # Actors with only one subject supply no distinct peer evidence.
        peer[self.degree[self.actor]<2]=0
        peer/=np.sqrt(np.maximum(1,self.degree[self.actor]-1))
        result=np.zeros(len(self.ids),dtype=np.float32);np.maximum.at(result,self.subject,peer)
        if not np.isfinite(result).all() or (result<0).any() or (result>1.000001).any():raise ValueError('Finite peer context')
        return result


if __name__=='__main__':main()
