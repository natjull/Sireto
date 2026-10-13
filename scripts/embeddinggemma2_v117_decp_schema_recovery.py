"""Recover DECP container/actor wrappers without redownloading immutable bodies.

Explicitly report absence of declared name fields rather than join Sirene names
and falsely claim new aliases. No source-selected GT or retrieval prediction.
"""
from collections import Counter
from pathlib import Path
import time

from scripts import embeddinggemma2_v116_decp_name_probe as prior
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v117/decp_schema_recovery_20261013_a'


def actors(row):
    if isinstance(row.get('acheteur'),dict):yield 'DECLARED_BUYER',row['acheteur']
    for key in ('titulaires','concessionnaires'):
        values=row.get(key) or []
        if isinstance(values,dict):values=[values]
        for value in values:
            if not isinstance(value,dict):continue
            yield 'DECLARED_HOLDER',value.get('titulaire',value.get('concessionnaire',value))


def records(path):
    import ijson
    with path.open('rb') as stream:yield from ijson.items(stream,'marches.marche.item',use_float=True)


def main():
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    engine.guard()
    if OUTPUT.exists():raise ValueError('Fresh schema recovery namespace')
    old=io.read(prior.OUTPUT/'launch.json');verify(old['implementation']);verify(old['catalog'])
    artifacts=[]
    for r in old['resources']:
        p=prior.OUTPUT/'raw'/(r['id']+'.json')
        if p.stat().st_size!=r['filesize']:raise ValueError('Complete downloaded canonical body')
        artifacts.append(dict(catalog=r,artifact=record(p)))
    ids=set();counts=Counter();coverage=[];fields=Counter()
    for item in artifacts:
        n=0
        for row in records(verify(item['artifact'])):
            n+=1
            for role,actor in actors(row):
                if not isinstance(actor,dict):continue
                counts[role]+=1;fields.update(role+'.'+k for k in actor)
                ref=prior.refkey(actor)
                if ref:ids.add(ref)
                else:counts['unknown_actor_identifier']+=1
        coverage.append(dict(resource=item['catalog']['id'],title=item['catalog']['title'],contracts=n))
    if not sum(r['contracts'] for r in coverage):raise ValueError('No valid contracts, don't publish fake completion')
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='1GB'")
    con.register('wanted',pa.table({'ref':sorted(ids),'siren':[r[:9] for r in sorted(ids)]}))
    public={r[0] for r in con.execute("""SELECT w.ref FROM wanted w JOIN read_parquet(?) u USING(siren)
        LEFT JOIN read_parquet(?) e ON w.ref=e.siret WHERE u.statutDiffusionUniteLegale='O'
        AND u.categorieJuridiqueUniteLegale<>'1000'
        AND (length(w.ref)=9 OR e.statutDiffusionEtablissement='O')""",
        [str(prior.STOCK/'StockUniteLegale.parquet'),str(prior.STOCK/'StockEtablissement.parquet')]).fetchall()};con.close()
    labels={};links=set()
    for item in artifacts:
        engine.guard()
        for row in records(verify(item['artifact'])):
            date=str(row.get('datePublicationDonnees') or '')[:10]
            if date and date>prior.CUTOFF:continue
            pairs=list(actors(row));buyers=[prior.refkey(a) for role,a in pairs if role=='DECLARED_BUYER' and prior.refkey(a) in public]
            holders=[prior.refkey(a) for role,a in pairs if role=='DECLARED_HOLDER' and prior.refkey(a) in public]
            for b in buyers:
                for h in holders:
                    if b[:9]!=h[:9]:links.add((b,h))
            for role,actor in pairs:
                ref=prior.refkey(actor)
                if ref not in public:continue
                name=actor.get('denominationSociale') or actor.get('nom') or ''
                if isinstance(name,str) and name.strip():labels[(ref,role,name.strip())]=dict(subject_ref=ref,role=role,declared_label=name.strip())
    OUTPUT.mkdir(parents=True)
    schema=pa.schema([('subject_ref',pa.string()),('role',pa.string()),('declared_label',pa.string())])
    pq.write_table(pa.Table.from_pylist(list(labels.values()),schema=schema),OUTPUT/'declared_actor_labels.parquet',compression='zstd')
    pq.write_table(pa.table({'buyer_ref':[b for b,h in sorted(links)],'holder_ref':[h for b,h in sorted(links)]}),
        OUTPUT/'declared_buyer_holder_links.parquet',compression='zstd')
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_PRIMARY_DECP_SCHEMA_AND_PUBLIC_RELATION_AUDIT',
        implementation=record(Path(__file__)),old_launch=record(prior.OUTPUT/'launch.json'),sources=artifacts,
        coverage=coverage,contracts=sum(r['contracts'] for r in coverage),actor_counts=dict(counts),actor_fields=dict(fields),
        distinct_ids=len(ids),public_corporate_ids=len(public),declared_actor_names=len(labels),
        declared_buyer_holder_pairs=len(links),names=record(OUTPUT/'declared_actor_labels.parquet'),
        links=record(OUTPUT/'declared_buyer_holder_links.parquet'),
        sirene_name_join_not_counted_as_new_source_names=True,procurement_relation_not_identity_or_ownership=True,
        all_source_snapshots_reused_no_redownload=True,no_gt_or_retrieval_or_model_changed=True,
        incomplete_month_coverage_not_global_absence_claim=True,source_admission_not_granted=True,observed_unix=time.time()))
    print('COMPLETE',sum(r['contracts'] for r in coverage),len(labels),len(links),flush=True)


if __name__=='__main__':main()
