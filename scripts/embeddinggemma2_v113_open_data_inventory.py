"""Archive public open-data catalog evidence, no source admission or labels.

Discover all declared search results by pagination, bounded catalog queries.
Metadata alone never proves record correctness or availability at a cutoff.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import time
import urllib.parse
import urllib.request

from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record

OUTPUT=io.HOME/'v113/open_data_inventory_20261013_a'
SEARCHES=['BANATIC','données essentielles commande publique fichiers consolidés',
          'annuaire administration base données locales','répertoire national associations',
          'FINESS Structures','annuaire éducation','DATA ES','répertoire national élus']
API='https://www.data.gouv.fr/api/1/datasets/'


def fetch(term):
    pages=[];rows=[];url=API+'?'+urllib.parse.urlencode(dict(q=term,page_size=100,page=1))
    while url:
        with urllib.request.urlopen(url,timeout=30) as response:
            data=json.load(response);headers={k:response.headers.get(k) for k in ('Date','ETag','Last-Modified')}
        pages.append(dict(url=url,total=data['total'],headers=headers,received=len(data['data'])))
        rows.extend(data['data']);url=data.get('next_page')
    return dict(query=term,pages=pages,datasets=rows,no_cutoff_or_source_precision_claim=True)


def main():
    if OUTPUT.exists():raise ValueError('Fresh catalog namespace')
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),queries=SEARCHES,
        catalog_api=API,all_catalog_pages=True,external_public_read_only_free=True,
        no_retrieval_or_labels=True,started_unix=time.time()))
    results={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,q):q for q in SEARCHES}
        for future in as_completed(futures):
            term=futures[future];result=future.result();index=SEARCHES.index(term)
            path=OUTPUT/f'catalog_{index}.json';io.write(path,result);results[term]=record(path)
            print(term,len(result['datasets']),flush=True)
    datasets={}
    for ref in results.values():
        for d in io.read(Path(ref['path']))['datasets']:
            datasets[d['id']]=dict(id=d['id'],title=d['title'],page=d.get('page'),license=d.get('license'),
                organization=(d.get('organization') or {}).get('name'),last_modified=d.get('last_modified'),
                resources=[dict(id=r['id'],title=r.get('title'),url=r.get('url'),format=r.get('format'),
                    filesize=r.get('filesize'),last_modified=r.get('last_modified')) for r in d.get('resources',[])])
    io.write(OUTPUT/'datasets.json',list(datasets.values()))
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_PUBLIC_CATALOG_SOURCE_INVENTORY_NOT_SOURCE_ADMISSION',
        implementation=record(Path(__file__)),searches=results,datasets=record(OUTPUT/'datasets.json'),
        distinct_datasets=len(datasets),queries=len(SEARCHES),no_recall_or_feature_changed=True,
        publisher_and_cutoff_require_case_specific_review=True,observed_unix=time.time()))


if __name__=='__main__':main()
