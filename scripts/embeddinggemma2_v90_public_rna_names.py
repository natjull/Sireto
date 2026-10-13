"""Whole national official RNA name source, exact public SIRENE RNA joins.

Not GT seeded. Only declared association titles/short titles are admitted to
diagnostic source; no managers/contact/person fields, no HQ address broadcast.
RNA identifiers join only, never predictive text. Old/no-RNA imports unjoined.
"""
import csv
import hashlib
import io as streamio
from pathlib import Path
import re
import time
import urllib.request
import zipfile
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v90/public_rna_names_20261013_a'
CATALOG='https://www.data.gouv.fr/api/1/datasets/repertoire-national-des-associations/'
TITLE='rna_waldec_20260801.zip'
STOCK=mandate.audit_utils.STOCK


def main():
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh whole-source namespace')
    OUTPUT.mkdir(parents=True)
    with urllib.request.urlopen(CATALOG,timeout=30) as response:catalog=__import__('json').load(response)
    resource=next(r for r in catalog['resources'] if r['title']==TITLE)
    if resource['last_modified'][:10]>'2026-08-01' or not resource['url'].startswith('https://media.interieur.gouv.fr/rna/'):raise ValueError('Primary observed source before cutoff')
    io.write(OUTPUT/'catalog.json',catalog)
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),catalog_url=CATALOG,
        catalog=record(OUTPUT/'catalog.json'),selected_resource=resource,stock=record(STOCK),
        whole_national_waldec_not_target_selected=True,cutoff='2026-08-01',
        name_fields_only_no_contact_or_person_data=True,addresses_not_imported=True,
        old_import_without_rna_not_name_joined=True,identifiers_join_only=True,labels_read=False,
        source_policy='DIAGNOSTIC_ONLY_NEW_OFFICIAL_SOURCE_NOT_SERVING_PROMOTION'))
    target=OUTPUT/TITLE;temporary=target.with_suffix('.pending.zip');count=0;start=time.monotonic()
    with urllib.request.urlopen(resource['url'],timeout=45) as response,temporary.open('xb') as f:
        while True:
            mandate.guard(mandate.DEADLINE);part=response.read(4*1024**2)
            if not part:break
            f.write(part);count+=len(part)
            if count%(16*1024**2)==0:io.write(OUTPUT/'state.json',dict(status='READING_WHOLE_PUBLIC_PRIMARY_ZIP',bytes=count,expected=resource['filesize'],updated_unix=time.time()))
    if count!=resource['filesize']:raise ValueError('Full primary file exact catalog size')
    temporary.rename(target);engine.seal(target)
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    rows=c.execute("SELECT siren,identifiantAssociationUniteLegale FROM read_parquet(?) WHERE statutDiffusionUniteLegale='O' AND regexp_matches(identifiantAssociationUniteLegale,'^W[A-Z0-9]{9}$')",[str(STOCK)]).fetchall();c.close()
    public={}
    for siren,rna in rows:public.setdefault(rna,set()).add(siren)
    schema=pa.schema([('siren',pa.string()),('rna',pa.string()),('title',pa.string()),('short_title',pa.string()),
        ('source_member',pa.string()),('source_line',pa.int64()),('snapshot',pa.string())])
    path=OUTPUT/'public_owner_names.parquet';writer=pq.ParquetWriter(path,schema,compression='zstd')
    total=joined=0;headers={};unknown=0;covered=set();names=0
    try:
        with zipfile.ZipFile(target) as z:
            files=sorted(n for n in z.namelist() if n.lower().endswith('.csv'))
            if not files:raise ValueError('Actual CSV source members required')
            for member in files:
                mandate.guard(mandate.DEADLINE)
                with z.open(member) as f:
                    first=f.readline();encoding='utf-8-sig'
                    try:header=first.decode(encoding)
                    except UnicodeDecodeError:encoding='iso-8859-15';header=first.decode(encoding)
                fields=next(csv.reader([header],delimiter=';'));fields=[v.strip().lower() for v in fields]
                if not {'id','titre','titre_court'}<=set(fields):raise ValueError(('Actual official RNA columns',member,fields))
                headers[member]=dict(columns=fields,encoding=encoding)
                indices=[fields.index(k) for k in ('id','titre','titre_court')];batch=[]
                with z.open(member) as raw,streamio.TextIOWrapper(raw,encoding=encoding,newline='') as text:
                    reader=csv.reader(text,delimiter=';');next(reader)
                    for line,row in enumerate(reader,2):
                        total+=1
                        if len(row)!=len(fields):raise ValueError(('Malformed uncropped source row',member,line,len(row),len(fields)))
                        rna,title,short=[row[i].strip() for i in indices]
                        if rna not in public:unknown+=1;continue
                        covered.add(rna)
                        if not title and not short:continue
                        for siren in sorted(public[rna]):
                            batch.append(dict(siren=siren,rna=rna,title=title,short_title=short,
                                source_member=member,source_line=line,snapshot='2026-08-01'));joined+=1
                            names+=bool(title)+bool(short)
                        if len(batch)>=8192:
                            writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[];mandate.guard(mandate.DEADLINE)
                    if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema))
                io.write(OUTPUT/'state.json',dict(status='WHOLE_PUBLIC_RNA_TITLE_JOIN',member=member,
                    scanned=total,joined_rows=joined,updated_unix=time.time()))
    finally:writer.close()
    engine.seal(path)
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_WHOLE_PUBLIC_RNA_TITLE_JOIN',implementation=record(Path(__file__)),
        launch=record(OUTPUT/'launch.json'),archive=record(target),names=record(path),headers=headers,
        national_rna_rows_scanned=total,public_sirene_rna_references=len(public),public_sirene_owners=len(rows),
        joined_rows=joined,joined_rna_ids=len(covered),declared_name_fields_present=names,
        rna_ids_with_multiple_public_sirens=sum(len(v)>1 for v in public.values()),
        associations_without_public_siren_not_inferred=unknown,whole_source_scanned=True,
        no_contact_or_individual_names_selected=True,no_source_address_inherited=True,labels_read=False,
        no_new_recall_or_prediction=True,source_policy='DIAGNOSTIC_ONLY',elapsed_s=time.monotonic()-start))
    print({k:v for k,v in io.read(OUTPUT/'generation.json').items() if not isinstance(v,dict)},flush=True)


if __name__=='__main__':main()
