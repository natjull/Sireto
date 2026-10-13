"""Primary DECP snapshots: declared public corporate actor labels, not aliases.

Select whole monthly versions by catalog dates, never GT subjects. Missing
months and all provenance bounds are published. No contacts/contract objects
or personal names extracted, no retrieval/labels changed by source diagnosis.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import re
import shutil
import time
import urllib.request

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v116/decp_name_probe_20261013_a'
CATALOG=io.HOME/'v113/open_data_inventory_20261013_a/datasets.json'
STOCK=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01')
CUTOFF='2026-08-01'


def download(r):
    target=OUTPUT/'raw'/(r['id']+'.json');temp=target.with_suffix('.pending')
    with urllib.request.urlopen(r['url'],timeout=60) as response,temp.open('wb') as f:
        headers={k:response.headers.get(k) for k in ('Date','Last-Modified','ETag','Content-Length')}
        shutil.copyfileobj(response,f,1024*1024)
    if r['filesize'] and temp.stat().st_size!=r['filesize']:raise ValueError('Catalog body size mismatch')
    temp.rename(target)
    return dict(catalog=r,artifact=record(target),http=headers,collected_unix=time.time())


def actors(row):
    buyer=row.get('acheteur')
    if isinstance(buyer,dict):yield 'DECLARED_BUYER',buyer
    for key in ('titulaires','concessionnaires'):
        xs=row.get(key) or []
        if isinstance(xs,dict):xs=[xs]
        for x in xs:
            if isinstance(x,dict):yield 'DECLARED_HOLDER',x


def refkey(actor):
    ref=actor.get('id')
    # Do not invent omitted zeroes or resolve non-French identifiers by names.
    if not isinstance(ref,str):return None
    ref=ref.strip();kind=str(actor.get('typeIdentifiant') or '').upper()
    if kind not in ('','SIRET','SIREN'):return None
    if not re.fullmatch('[0-9]{14}|[0-9]{9}',ref) or set(ref)=={'0'}:return None
    return ref


def records(path):
    import ijson
    with path.open('rb') as stream:yield from ijson.items(stream,'marches.item',use_float=True)


def main():
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    engine.guard()
    if OUTPUT.exists():raise ValueError('Fresh source namespace')
    ds=next(d for d in io.read(CATALOG) if d['id']=='5cd57bf68b4c4179299eb0e9')
    if ds['organization']!='Ministères économiques et financiers':raise ValueError('Primary publisher required')
    resources=[r for r in ds['resources'] if r['format']=='json'
        and (r['title'] or '').startswith(('decp-2025-','decp-2026-'))
        and (r['last_modified'] or '')[:10]<=CUTOFF]
    if not resources:raise ValueError('No pre-cutoff publisher resource versions')
    OUTPUT.mkdir(parents=True);(OUTPUT/'raw').mkdir();(OUTPUT/'spill').mkdir()
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),catalog=record(CATALOG),
        dataset=ds['id'],organization=ds['organization'],resources=resources,
        scope='ALL_AVAILABLE_PRECUTOFF_PUBLISHER_VERSIONS_NOTIFICATION_MONTHS_2025_2026',
        declared_cutoff=CUTOFF,snapshot_metadata_not_cryptographic_asof_proof=True,
        missing_months_not_absence_of_contracts=True,no_labels_read=True,external_free_primary_data=True))
    artifacts=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(download,r):r for r in resources}
        for f in as_completed(futures):
            item=f.result();artifacts.append(item);print(item['catalog']['title'],'downloaded',flush=True)
    ids=set();coverage=[];counts=Counter()
    for a in artifacts:
        n=0
        for row in records(verify(a['artifact'])):
            n+=1
            for role,actor in actors(row):
                ref=refkey(actor)
                if ref:ids.add(ref)
                else:counts['unknown_or_nonstandard_actor_identifier']+=1
        coverage.append(dict(source=a,contracts=n))
    if not any(x['contracts'] for x in coverage):raise ValueError('Wrong/empty DECP container, no fake completion')
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='1GB'")
    con.execute('SET temp_directory=?',[str(OUTPUT/'spill')])
    con.register('wanted',pa.table({'ref':sorted(ids),'siren':[r[:9] for r in sorted(ids)]}))
    allowed=con.execute("""SELECT w.ref FROM wanted w JOIN read_parquet(?) u USING(siren)
        LEFT JOIN read_parquet(?) e ON e.siret=w.ref WHERE u.statutDiffusionUniteLegale='O'
        AND u.categorieJuridiqueUniteLegale<>'1000'
        AND (length(w.ref)=9 OR e.statutDiffusionEtablissement='O')""",
        [str(STOCK/'StockUniteLegale.parquet'),str(STOCK/'StockEtablissement.parquet')]).fetchall()
    con.close();public={r[0] for r in allowed};facts={}
    for item in artifacts:
        for row in records(verify(item['artifact'])):
            engine.guard()
            date=str(row.get('datePublicationDonnees') or '')[:10]
            if date and date>CUTOFF:counts['future_declared_publication']+=1;continue
            for role,actor in actors(row):
                ref=refkey(actor)
                if ref not in public:continue
                label=actor.get('denominationSociale') or actor.get('nom') or ''
                if not isinstance(label,str):continue
                label=' '.join(label.split())
                if not label:continue
                key=(ref,role,label)
                facts.setdefault(key,dict(subject_ref=ref,owner_siren=ref[:9],
                    subject_kind='SIRET' if len(ref)==14 else 'SIREN',role=role,declared_label=label,
                    declared_publication_date=date or None,source_resource_id=item['catalog']['id'],
                    actor_label_not_automatic_own_site_alias=True))
    rows=list(facts.values())
    if not rows:raise ValueError('No valid public declared actor labels, keep raw artifacts')
    pq.write_table(pa.Table.from_pylist(rows),OUTPUT/'declared_actor_labels.parquet',compression='zstd')
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_PRIMARY_DECP_PUBLIC_DECLARED_ACTOR_NAMES_NOT_RETRIEVAL',
        implementation=record(Path(__file__)),launch=record(OUTPUT/'launch.json'),coverage=coverage,
        artifacts=artifacts,labels=record(OUTPUT/'declared_actor_labels.parquet'),
        contracts=sum(x['contracts'] for x in coverage),distinct_identifiers=len(ids),
        public_corporate_identifiers=len(public),distinct_declared_labels=len(rows),rejections=dict(counts),
        role_counts=dict(Counter(r['role'] for r in rows)),no_new_recall_or_alias=True,
        no_full_national_history_coverage_claim=True,no_source_admission_yet=True,observed_unix=time.time()))
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
