"""Source name-inventory coverage after all526 sealed, not a model input.

Current diffusion is checked for ALL526 before any raw-name lookup. Only
counts and overlapping CRM words are exported, no person/contact names.
Zero exact-token overlap is not proof of semantic impossibility or false GT.
"""
from pathlib import Path
import re
import sqlite3
import time
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v23_native_features as original
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCE=io.HOME/'v101/postseal_name_roles_20261013_a'
OUTPUT=io.HOME/'v103/raw_alias_coverage_20261013_a'


def main():
    import duckdb
    import pyarrow as pa
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh source coverage audit')
    source=io.read(SOURCE/'report.json');verify(source['cases'])
    rows=io.read(verify(source['cases']));pipeline=io.read(io.HOME/'v99/initial_alignment_native_20261013_a/launch.json')
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.register('wanted',pa.Table.from_pylist([dict(qid=r['query_id'],siret=r['label_siret'],siren=r['label_siret'][:9]) for r in rows]))
    status=c.execute("""SELECT w.qid,e.statutDiffusionEtablissement site_diffusion,
        u.statutDiffusionUniteLegale owner_diffusion FROM wanted w
        LEFT JOIN read_parquet(?) e USING(siret) LEFT JOIN read_parquet(?) u ON w.siren=u.siren""",
        [str(mandate.STOCK),str(mandate.audit_utils.STOCK)]).fetch_arrow_table().to_pylist();c.close()
    statuses={r['qid']:r for r in status}
    if len(rows)!=526 or len(statuses)!=526:raise ValueError('All526 status authority')
    db=sqlite3.connect(f"file:{verify(pipeline['names'])}?mode=ro&immutable=1",uri=True)
    result=[]
    for r in rows:
        mandate.guard(mandate.DEADLINE);s=statuses[r['query_id']];protected=s['site_diffusion']!='O' or s['owner_diffusion']!='O'
        p=int(r['label_siret'][:9]);key=int(r['label_siret']);counts={};overlap=set()
        if not protected:
            tokens=set(re.findall('[A-Z0-9]+',original.structural.normalize(original.data.safe(r['query']['crm_name']))))
            aliases=db.execute('SELECT kind,name FROM aliases WHERE (kind=1 AND owner_key=?) OR (kind=2 AND owner_key=?)',(p,key)).fetchall()
            for kind in (1,2):counts[str(kind)]=sum(k==kind for k,name in aliases)
            for kind,name in aliases:
                overlap|=tokens&set(re.findall('[A-Z0-9]+',original.structural.normalize(original.data.safe(name))))
        result.append(dict(query_id=r['query_id'],exact=r['exact'],protected_no_lookup=protected,
            raw_published_legal_and_own_site_alias_counts=counts,overlapping_query_words=sorted(overlap),
            indexed_name_word_score=r['all_fields_maximum'],no_semantic_impossibility_claim=True))
    db.close();OUTPUT.mkdir(parents=True);io.write(OUTPUT/'cases.json',result);io.write(OUTPUT/'diffusion_status.json',status)
    public_misses=[r for r in result if not r['exact'] and not r['protected_no_lookup']]
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_RAW_OWN_NAME_INVENTORY_COVERAGE_NOT_RETRIEVAL',
        implementation=record(Path(__file__)),source=record(SOURCE/'report.json'),names=pipeline['names'],
        stock_sites=record(mandate.STOCK),stock_owners=record(mandate.audit_utils.STOCK),
        cases=record(OUTPUT/'cases.json'),statuses=record(OUTPUT/'diffusion_status.json'),all526_retained=True,
        protected_all526=sum(r['protected_no_lookup'] for r in result),public_misses=len(public_misses),
        public_misses_zero_raw_query_word_overlap=sum(not r['overlapping_query_words'] for r in public_misses),
        public_misses_raw_word_overlap_but_indexed_zero=sum(bool(r['overlapping_query_words']) and not r['indexed_name_word_score'] for r in public_misses),
        no_alias_created=True,no_model_or_retrieval_changed=True,no_recall_claim=True,observed_unix=time.time()))
    print(len(public_misses),sum(not r['overlapping_query_words'] for r in public_misses),flush=True)


if __name__=='__main__':main()
