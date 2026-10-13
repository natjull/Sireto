"""Full primary announcement context for outcome-blind source QA, not retrieval."""
from pathlib import Path
import time
import duckdb
import pyarrow as pa
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as source
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v80/bodacc_precision_context_20261013_a'
QA=io.HOME/'v79/bodacc_disjoint_precision_20261013_a'


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh immutable QA context')
    gen=io.read(QA/'generation.json');sample=io.read(verify(gen['sample']))
    c=duckdb.connect();c.execute('SET threads=1');c.execute("SET memory_limit='512MB'")
    c.register('sample_ids',pa.table({'id':[r['announcement_id'] for r in sample]}))
    result={}
    for file in sorted({r['source_file'] for r in sample}):
        mandate.guard(mandate.DEADLINE)
        matched=c.execute("""SELECT a.id,a.dateparution,a.listepersonnes,
            json_extract_string(a.acte,'$.descriptif') AS body
            FROM read_parquet(?) a JOIN sample_ids s USING(id)""",[file]).fetchall()
        for announcement,date,people,body in matched:
            if announcement in result:raise ValueError('Unambiguous primary source ID')
            result[announcement]=dict(announcement_id=announcement,date=str(date)[:10],body=body,
                announced_subjects=[dict(siren=source.siren(source.unwrap(p)),names=source.names(source.unwrap(p))) for p in source.decoded(people)
                    if source.siren(source.unwrap(p)) in {r['subject_siren'] for r in sample if r['announcement_id']==announcement}])
    c.close()
    if set(result)!={r['announcement_id'] for r in sample}:raise ValueError('Every blind sample has complete primary context')
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'contexts.json',[dict(sample=r,primary=result[r['announcement_id']]) for r in sample])
    io.write(OUTPUT/'generation.json',dict(implementation=record(Path(__file__)),sample_generation=record(QA/'generation.json'),
        contexts=record(OUTPUT/'contexts.json'),source_files=[record(Path(p)) for p in sorted({r['source_file'] for r in sample})],
        complete_primary_contexts=32,crm_read=False,labels_read=False,no_source_or_model_changed=True,
        scope='QA_ONLY_NOT_SOURCE_INPUT_ADMISSION',observed_unix=time.time()))


if __name__=='__main__':main()
