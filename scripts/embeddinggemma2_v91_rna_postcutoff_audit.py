"""Whole official RNA mirror availability audit, POST-CUTOFF NEVER INPUT.

August primary archive refused403. October mirror is a different publicly
available official dataset, not a bypass. Mixed dates explicitly ineligible
for the dated native retrieval experiments; no labels or candidate outputs.
"""
from pathlib import Path
import time
import urllib.request
import json
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record

OUTPUT=io.HOME/'v91/postcutoff_rna_availability_20261013_a'
CATALOG='https://www.data.gouv.fr/api/1/datasets/67164a231c2a52cc06b9b618/'
URL='https://data-pipeline-open.s3.sbg.io.cloud.ovh.net/rna/waldec.parquet'


def main():
    import duckdb
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh audit namespace')
    OUTPUT.mkdir(parents=True)
    with urllib.request.urlopen(CATALOG,timeout=20) as response:catalog=json.load(response)
    resource=next(r for r in catalog['resources'] if r['url']==URL)
    io.write(OUTPUT/'catalog.json',catalog)
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),catalog=record(OUTPUT/'catalog.json'),
        resource=resource,source_date='2026-10-01',stock_date='2026-08-01',source_cutoff='2026-08-01',
        status='POST_CUTOFF_SOURCE_DIAGNOSIS_ONLY_NEVER_NATIVE_INPUT',labels_read=False,
        whole_national_source_not_target_selected=True,source_eligibility=False,
        no_contact_or_person_fields_selected=True,no_address_imported=True))
    path=OUTPUT/'whole_public_waldec_20261001.parquet';temporary=path.with_suffix('.pending.parquet');count=0
    with urllib.request.urlopen(URL,timeout=40) as response,temporary.open('xb') as f:
        while True:
            mandate.guard(mandate.DEADLINE);data=response.read(4*1024**2)
            if not data:break
            f.write(data);count+=len(data)
            if count%(16*1024**2)==0:io.write(OUTPUT/'state.json',dict(status='READING_POSTCUTOFF_PUBLIC_MIRROR',bytes=count,expected=resource['filesize'],updated_unix=time.time()))
    if count!=resource['filesize']:raise ValueError('Actual whole-file size')
    temporary.rename(path);engine.seal(path)
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    columns=[r[0] for r in c.execute('DESCRIBE SELECT * FROM read_parquet(?)',[str(path)]).fetchall()]
    if not {'id','titre','titre_court'}<=set(columns):raise ValueError(('Actual source name columns',columns))
    dest=OUTPUT/'public_declared_owner_names.parquet'
    c.execute("""COPY (SELECT u.siren,u.identifiantAssociationUniteLegale rna,
        r.titre title,r.titre_court short_title FROM read_parquet(?) r JOIN read_parquet(?) u
        ON r.id=u.identifiantAssociationUniteLegale
        WHERE u.statutDiffusionUniteLegale='O') TO ? (FORMAT PARQUET,COMPRESSION ZSTD)""",
        [str(path),str(mandate.audit_utils.STOCK),str(dest)])
    total=c.execute('SELECT count(*) FROM read_parquet(?)',[str(path)]).fetchone()[0]
    joined=c.execute('SELECT count(*),count(DISTINCT rna),count(DISTINCT siren),count(*) FILTER(WHERE coalesce(short_title,\'\')<>\'\') FROM read_parquet(?)',[str(dest)]).fetchone()
    c.close();engine.seal(dest)
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_POST_CUTOFF_RNA_AVAILABILITY_ONLY',
        launch=record(OUTPUT/'launch.json'),archive=record(path),names=record(dest),stock=record(mandate.audit_utils.STOCK),
        national_rows=total,public_joined_rows=joined[0],public_joined_rna=joined[1],public_joined_sirens=joined[2],
        nonempty_declared_short_titles=joined[3],source_date='2026-10-01',stock_date='2026-08-01',
        source_eligibility=False,not_a_point_in_time_reconstruction=True,
        no_title_or_sigle_learned_from_crm=True,no_native_model_input_or_score_changed=True,
        no_addresses_or_person_contact_fields_selected=True,no_recall_claim=True,labels_read=False))
    print(joined,flush=True)


if __name__=='__main__':main()
