"""Archive primary supplier/customer evidence; no aliases, predictions or GT.

A corporate supplier relationship is NOT company identity or ownership.
Capture all published client-image references, not one label-selected client.
Source is historical primary website, held for source-policy review only.
"""
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import re
import time
import urllib.parse
import urllib.request

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record

OUTPUT=io.HOME/'v66/archived_provider_evidence_20261013_a'
URL='https://web.archive.org/web/20260618075451id_/https://naos-it.com/'


class PublishedImages(HTMLParser):
    def __init__(self):super().__init__();self.clients=[]
    def handle_starttag(self,tag,attrs):
        if tag!='img':return
        attrs=dict(attrs);src=attrs.get('src','')
        if '/clients/' in src:self.clients.append(dict(src=src,alt=attrs.get('alt','')))


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh source corroboration namespace')
    OUTPUT.mkdir(parents=True)
    with urllib.request.urlopen(URL,timeout=30) as response:
        body=response.read();headers=dict(response.headers);effective=response.geturl()
    html=body.decode('utf-8',errors='strict')
    source_time=headers.get('memento-datetime') or headers.get('Memento-Datetime')
    if source_time!='Thu, 18 Jun 2026 07:54:51 GMT':raise ValueError('Exact pre-cutoff archived capture required')
    (OUTPUT/'primary_archive.html').write_bytes(body)
    parser=PublishedImages();parser.feed(html)
    if not parser.clients:raise ValueError('No complete published client section image references')
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_PRIMARY_ARCHIVE_SOURCE_CORROBORATION_NOT_RETRIEVAL',
        implementation=record(Path(__file__)),source=record(OUTPUT/'primary_archive.html'),
        requested_url=URL,effective_url=effective,memento_datetime=source_time,
        observed_unix=time.time(),headers=headers,all_published_client_images=parser.clients,
        explicit_application_brand_mention=bool(re.search('applications spécifiques AUBERT',html)),
        relation_type='SUPPLIER_CUSTOMER_PUBLISHED_REFERENCE_NOT_OWNERSHIP_OR_IDENTITY',
        client_images_not_yet_visually_verified=True,legal_entity_ids_not_resolved=True,
        source_before_20260801=True,crm_read=False,labels_read=False,
        source_policy_not_yet_admitted=True,index_or_score_changed=False,
        not_a_recall_measurement=True,no_external_compute_or_expense=True))


if __name__=='__main__':main()
