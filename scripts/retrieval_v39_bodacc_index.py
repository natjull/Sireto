"""Search compiled BODACC paths directly, with one document per target SIRET.

This is a complete official transfer component, not the full national product
corpus. No CRM pilot or recall result can validate initial retrieval with it.
"""
import argparse
import itertools
import json
from pathlib import Path
import re
import time

import pyarrow.parquet as pq
import tantivy

from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json


def schema():
    builder=tantivy.SchemaBuilder()
    for key in ('anchor_names','event_address'):
        builder.add_text_field(key,stored=False)
    for key in ('postcode','insee'):
        builder.add_text_field(key,tokenizer_name='raw',index_option='basic',stored=False)
    # IDs and provenance are not indexed as text or used in query scoring.
    builder.add_bytes_field('identity',stored=True,indexed=False,fast=False)
    builder.add_bytes_field('proofs',stored=True,indexed=False,fast=False)
    return builder.build()


def grouped_rows(path):
    def records():
        previous=''
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                if row['siret']<previous: raise ValueError('Proof corpus is not ordered by SIRET')
                previous=row['siret']; yield row
    for siret,group in itertools.groupby(records(),key=lambda row:row['siret']):
        rows=list(group)
        site={(r['target_siren'],r['target_address'],r['target_postcode'],r['target_insee']) for r in rows}
        if len(site)!=1: raise ValueError('Conflicting official site bindings')
        siren,address,postcode,insee=site.pop()
        names=sorted({n['text'] for r in rows for n in r['anchor_names']})
        proofs=[{k:r[k] for k in ('relation_id','relation_record_id','event_id','anchor_siren',
            'address_evidence_id','address_record_id','reported_date','date_semantics','direction','site_binding')}
            for r in rows]
        yield dict(siret=siret,siren=siren,names=names,address=address,postcode=postcode,insee=insee,proofs=proofs)


def build(path,dest):
    dest.mkdir()
    index=tantivy.Index(schema(),path=str(dest),reuse=False)
    writer=index.writer(heap_size=32*1024**2,num_threads=1)
    sites=0; paths=0
    for row in grouped_rows(path):
        doc=tantivy.Document()
        for name in row['names']: doc.add_text('anchor_names',name)
        doc.add_text('event_address',row['address'])
        for key in ('postcode','insee'):
            if row[key]: doc.add_text(key,row[key])
        doc.add_bytes('identity',json.dumps({k:row[k] for k in ('siret','siren')}).encode())
        doc.add_bytes('proofs',json.dumps(row['proofs'],default=str,sort_keys=True).encode())
        writer.add_document(doc); sites+=1; paths+=len(row['proofs'])
    writer.commit(); writer.wait_merging_threads(); index.reload()
    if index.searcher().num_docs!=sites: raise ValueError('Unique site document count drift')
    return dict(sites=sites,proof_paths=paths)


def search(index,name,address,postcode='',insee=''):
    clauses=[]
    for field,text in (('anchor_names',name),('event_address',address)):
        words=sorted(set(re.findall('[a-z0-9]+',text.lower())))
        if not words: return []
        terms=[(tantivy.Occur.Should,tantivy.Query.term_query(index.schema,field,word,'basic')) for word in words]
        clauses.append((tantivy.Occur.Must,tantivy.Query.boost_query(tantivy.Query.boolean_query(terms),1/len(words))))
    for field,value in (('postcode',postcode),('insee',insee)):
        if value:
            clauses.append((tantivy.Occur.Should,tantivy.Query.const_score_query(
                tantivy.Query.term_query(index.schema,field,value,'basic'),1.)))
    query=tantivy.Query.boolean_query(clauses); searcher=index.searcher(); result=[]
    for score,location in searcher.search(query,limit=100,count=False).hits:
        doc=searcher.doc(location).to_dict()
        identity=json.loads(bytes(doc['identity'][0]))
        result.append(dict(**identity,score=score,proofs=json.loads(bytes(doc['proofs'][0]))))
    if len(result)>100 or len({r['siret'] for r in result})!=len(result):
        raise ValueError('Unique top100 collector contract violated')
    return result


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    budget=Budget(dict(max_seconds=180,max_rss_bytes=3*1024**3,max_swapout_pages_delta=0))
    try:
        manifest=json.loads(source.read_text())
        if manifest['status']!='SEALED_BODACC_SOURCE_PROOF_COMPILATION' or manifest['truth_opened'] or manifest['crm_opened']:
            raise ValueError('Invalid source compilation')
        with budget:
            path=verify(manifest['artifacts']['proof_paths.parquet'])
            counts=build(path,output/'tantivy')
            expected=manifest['stats']['proofs'][0]
            if counts!=dict(sites=expected['target_sites'],proof_paths=expected['proof_paths']):
                raise ValueError('Incomplete official proof component')
            files={str(p.relative_to(output)):record(p) for p in (output/'tantivy').rglob('*')
                   if p.is_file() and not p.name.endswith('.lock')}
        write_json(output/'manifest.json',dict(status='SEALED_BODACC_INITIAL_SEARCH_COMPONENT',
            source_manifest=record(source),implementation=record(Path(__file__)),counts=counts,
            resources=budget.report(),files=files,collector_limit=100,unique_siret_documents=True,
            identifiers_indexed=False,truth_opened=False,crm_opened=False,recall_measured=False,
            full_national_site_corpus=False,ready_for_fold1=False,
            decision='STOP_PRODUCT_INTEGRATION_AND_FULL_CORPUS_RECALL_NOT_MEASURED'))
        print(json.dumps(dict(counts=counts,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),ready_for_fold1=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); run(args.source,args.output)
