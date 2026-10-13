"""Resume whole public October RNA audit only, named SQL binds, no download.

Post-cutoff remains ineligible; source archive exact same sealed bytes. Never
feed these titles into dated native runs or reconstruct protected labels.
"""
from pathlib import Path
import time
from scripts import embeddinggemma2_v91_rna_postcutoff_audit as source
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v93/postcutoff_rna_join_20261013_a'


def main():
    import duckdb
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh resume namespace')
    launch=io.read(source.OUTPUT/'launch.json');verify(launch['catalog']);verify(launch['implementation'])
    archive=engine.sealed(source.OUTPUT/'whole_public_waldec_20261001.parquet')
    if not archive or archive.stat().st_size!=launch['resource']['filesize']:raise ValueError('Sealed whole public archive')
    OUTPUT.mkdir(parents=True);dest=OUTPUT/'public_declared_owner_names.parquet'
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.execute("""COPY (SELECT u.siren,u.identifiantAssociationUniteLegale rna,
        r.titre title,r.titre_court short_title FROM read_parquet($rna) r JOIN read_parquet($stock) u
        ON r.id=u.identifiantAssociationUniteLegale WHERE u.statutDiffusionUniteLegale='O')
        TO $dest (FORMAT PARQUET,COMPRESSION ZSTD)""",
        dict(rna=str(archive),stock=str(mandate.audit_utils.STOCK),dest=str(dest)))
    total=c.execute('SELECT count(*) FROM read_parquet(?)',[str(archive)]).fetchone()[0]
    joined=c.execute("SELECT count(*),count(DISTINCT rna),count(DISTINCT siren),count(*) FILTER(WHERE coalesce(short_title,'')<>'') FROM read_parquet(?)",[str(dest)]).fetchone()
    c.close();engine.seal(dest)
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_POST_CUTOFF_RNA_AVAILABILITY_ONLY',
        implementation=record(Path(__file__)),original_launch=record(source.OUTPUT/'launch.json'),
        archive=record(archive),names=record(dest),stock=record(mandate.audit_utils.STOCK),
        national_rows=total,public_joined_rows=joined[0],public_joined_rna=joined[1],public_joined_sirens=joined[2],
        nonempty_declared_short_titles=joined[3],source_date='2026-10-01',stock_date='2026-08-01',
        source_eligibility=False,not_a_point_in_time_reconstruction=True,no_redownload=True,
        no_title_or_sigle_learned_from_crm=True,no_native_model_input_or_score_changed=True,
        no_addresses_or_person_contact_fields_selected=True,no_recall_claim=True,labels_read=False,observed_unix=time.time()))
    print(joined,flush=True)


if __name__=='__main__':main()
