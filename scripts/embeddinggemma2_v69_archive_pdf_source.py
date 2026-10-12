"""Pin an unchanged historical primary corporate PDF for source-only review."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import time
import urllib.request
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--timestamp',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    mandate.guard(mandate.DEADLINE)
    stamp=datetime.strptime(a.timestamp,'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    if stamp>=datetime(2026,8,2,tzinfo=timezone.utc):raise ValueError('Pre-cutoff historical primary source only')
    if a.output.exists():raise ValueError('Fresh source directory')
    a.output.mkdir(parents=True)
    url='https://web.archive.org/web/'+a.timestamp+'id_/'+a.url
    with urllib.request.urlopen(url,timeout=45) as response:
        headers=dict(response.headers);effective=response.geturl();body=response.read(30*1024**2+1)
    if len(body)>30*1024**2 or not body.startswith(b'%PDF'):raise ValueError('Bounded actual PDF, not redirect HTML')
    memento=headers.get('memento-datetime') or headers.get('Memento-Datetime')
    if memento!=stamp.strftime('%a, %d %b %Y %H:%M:%S GMT'):raise ValueError('Exact archive capture datetime')
    (a.output/'primary.pdf').write_bytes(body)
    io.write(a.output/'source.json',dict(implementation=record(Path(__file__)),source=record(a.output/'primary.pdf'),
        original_publisher_url=a.url,archived_url=url,effective_url=effective,
        memento_datetime=memento,observed_unix=time.time(),source_before_cutoff=True,
        headers={k:v for k,v in headers.items() if k.lower()!='set-cookie'},
        crm_read=False,labels_read=False,source_not_yet_indexed=True,
        tables_not_yet_visually_verified=True,no_current_ownership_claim=True,
        external_spend=0,unchanged_original_pdf=True))
    print(a.output/'source.json',flush=True)


if __name__=='__main__':main()
