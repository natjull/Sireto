"""Visually auditable whole published client section; not a legal-identity join."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
import time
import urllib.parse
import urllib.request
from PIL import Image
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v74/archived_client_visuals_20261013_a'
SOURCE=io.HOME/'v66/archived_provider_evidence_20261013_a'


def capture(item):
    mandate.guard(mandate.DEADLINE)
    original=urllib.parse.urljoin('https://naos-it.com/',item['src'])
    request='https://web.archive.org/web/20260618075451id_/'+original
    with urllib.request.urlopen(request,timeout=35) as response:
        body=response.read(4*1024**2+1);headers=dict(response.headers);effective=response.geturl()
    if len(body)>4*1024**2:raise ValueError('Bounded image source')
    memento=headers.get('memento-datetime') or headers.get('Memento-Datetime')
    if not memento or parsedate_to_datetime(memento)>=datetime(2026,8,2,tzinfo=timezone.utc):raise ValueError('No pre-cutoff actual image capture')
    name=Path(urllib.parse.urlparse(original).path).name
    image=Image.open(BytesIO(body));image.load()
    target=OUTPUT/name;target.write_bytes(body)
    preview=OUTPUT/(Path(name).stem+'.png');image.convert('RGBA').save(preview)
    return dict(original_url=original,requested_url=request,effective_url=effective,
        memento_datetime=memento,source=record(target),preview=record(preview),
        dimensions=image.size,semantic_brand_not_yet_visually_verified=True)


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh immutable whole-section corroboration')
    source=io.read(SOURCE/'generation.json');verify(source['source'])
    items=source['all_published_client_images'];OUTPUT.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(capture,items))
    if len(rows)!=len(items):raise ValueError('Every declared client image, no selected subset')
    io.write(OUTPUT/'generation.json',dict(implementation=record(Path(__file__)),primary_html=source['source'],
        images=rows,whole_client_section=True,relation='PUBLISHED_SUPPLIER_CUSTOMER_REFERENCE_ONLY',
        legal_entity_ids_not_resolved=True,crm_read=False,labels_read=False,
        source_policy_not_admitted=True,no_prediction_or_alias_or_score_created=True,observed_unix=time.time()))


if __name__=='__main__':main()
