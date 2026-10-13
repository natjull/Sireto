"""Archive the complete primary published Cegedim report, no label-selected rows.

Publication date was observed in publisher's report catalog. Consolidation
table interpretation needs visual review; no ownership/alias/candidate output.
"""
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record

OUTPUT=io.HOME/'v120/primary_group_report_20261013_a'
PAGE='https://www.cegedim.fr/finance/documentation/Pages/rapports.aspx'
URL='https://www.cegedim.fr/Docs_Communiques/CEGEDIM_URD_2025_ASSEMBLAGE_FR_202604171735.pdf'


def main():
    if OUTPUT.exists():raise ValueError('Fresh primary document namespace')
    OUTPUT.mkdir(parents=True)
    with urllib.request.urlopen(PAGE,timeout=30) as response:page=response.read().decode()
    if URL.split('/')[-1] not in page:raise ValueError('Document must be linked by primary publisher')
    with urllib.request.urlopen(URL,timeout=45) as response,(OUTPUT/'primary.pdf').open('wb') as f:
        headers={k:response.headers.get(k) for k in ('Date','Last-Modified','Content-Length')}
        shutil.copyfileobj(response,f,1024*1024)
    subprocess.run(['pdftotext','-layout',str(OUTPUT/'primary.pdf'),str(OUTPUT/'layout.txt')],check=True)
    subprocess.run(['pdfinfo',str(OUTPUT/'primary.pdf')],check=True,stdout=subprocess.DEVNULL)
    io.write(OUTPUT/'archive.json',dict(status='COMPLETE_PRIMARY_DOCUMENT_ARCHIVE_PENDING_TABLE_REVIEW',
        implementation=record(Path(__file__)),publisher_catalog=PAGE,url=URL,
        observed_publisher_publication_date='2026-04-17',financial_period='2025',
        pdf=record(OUTPUT/'primary.pdf'),layout=record(OUTPUT/'layout.txt'),http=headers,
        no_human_labels_read=True,no_subsidiary_id_resolution_or_alias=True,no_recall_claim=True,
        collected_unix=time.time()))
    print(record(OUTPUT/'primary.pdf'),flush=True)


if __name__=='__main__':main()
