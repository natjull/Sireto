"""Whole-source, query-only typed BODACC activity context; not entity aliases.

Strict single-person/single-establishment JSON object scope. Public corporate
subject checked before activity is returned. Complex records counted/skipped,
never falsely attributed to all parties. IDs are references, not score text.
"""
from collections import Counter,defaultdict
from pathlib import Path
import math
import re
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as words
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v112/bodacc_activity_context_20261013_a'
BASE=io.HOME/'v99/initial_alignment_native_20261013_a'
CUTOFF='2026-08-01'


def main():
    import duckdb
    engine.guard()
    if OUTPUT.exists():raise ValueError('Fresh source namespace')
    base=io.read(BASE/'launch.json');panel=io.read(verify(base['panel']))
    if len(panel)!=526:raise ValueError('Same query-only panel')
    qs={r['query_id']:words.tokens(r['crm_name']) for r in panel}
    wanted=set().union(*qs.values())
    pattern='(^|[^A-Z0-9])('+'|'.join(re.escape(w) for w in sorted(wanted))+')([^A-Z0-9]|$)'
    OUTPUT.mkdir(parents=True);(OUTPUT/'spill').mkdir()
    recipe=dict(scope='ALL_LOCAL_BODACC_YEARS_STRICT_SINGLE_CORPORATE_SUBJECT_ESTABLISHMENT',
        source_field='listeetablissements.etablissement.activite',single_object_only=True,
        source_cutoff=CUTOFF,public_current_corporate_subject_only=True,related_people_not_read=True,
        publication_not_effective_date=True,activity_is_parent_context_not_alias_or_site_identity=True,
        matching='QUERY_INFORMATIVE_TOKEN_COVERAGE_SOURCE_OWNER_IDF',confidence_cap=.5,
        owner_idf_formula='LOG((N+1)/(DF+1))/LOG(N+1)',deduplicate='MAXIMUM_NOT_SUM_ACROSS_ANNOUNCEMENTS',
        all_source_subjects_no_topk=True,no_siret_or_siren_as_feature=True,no_labels_read=True,
        snapshot_archiving_not_automatically_asof_reproduction=True)
    io.write(OUTPUT/'recipe.json',recipe)
    manifests=sorted(words.SOURCES.glob('*/manifest.json'))
    refs=[]
    for path in manifests:
        item=io.read(path)['sources'][0]
        source=Path(item['path'])
        if source.stat().st_size!=item['size_bytes']:raise ValueError('Canonical source size drift')
        refs.append(dict(manifest=record(path),source=record(source)))
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(OUTPUT/'recipe.json'),
        query_panel=base['panel'],sources=refs,public_units=record(words.STOCK),renewed_human_mandate=True,
        historical_night_not_resumed=True,labels_read=False))
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='1GB'")
    con.execute('SET temp_directory=?',[str(OUTPUT/'spill')])
    con.read_parquet(str(words.STOCK)).filter("statutDiffusionUniteLegale='O' AND categorieJuridiqueUniteLegale<>'1000'").project('siren').create_view('public_units')
    owners=set();by_token=defaultdict(set);evidence=defaultdict(set);seen=set();coverage=[]
    started=time.monotonic()
    for ref in refs:
        engine.guard();path=verify(ref['source'])
        con.read_parquet(str(path)).create_view('ann',replace=True)
        counts=con.execute("""SELECT count(*) total,count(*) FILTER(WHERE dateparution<=CAST(? AS DATE)) before_cutoff,
            count(*) FILTER(WHERE dateparution<=CAST(? AS DATE) AND json_extract_string(listepersonnes,'$.personne.typePersonne')='pm'
                AND json_type(listeetablissements,'$.etablissement')='OBJECT') strict_single_objects
            FROM ann""",[CUTOFF,CUTOFF]).fetchone()
        con.execute("""CREATE OR REPLACE TEMP TABLE eligible AS SELECT a.id,a.dateparution,
            regexp_replace(json_extract_string(a.listepersonnes,'$.personne.numeroImmatriculation.numeroIdentification'),'[^0-9]','','g') subject,
            json_extract_string(a.listeetablissements,'$.etablissement.activite') activity
            FROM ann a JOIN public_units u ON u.siren=regexp_replace(
              json_extract_string(a.listepersonnes,'$.personne.numeroImmatriculation.numeroIdentification'),'[^0-9]','','g')
            WHERE a.dateparution<=CAST(? AS DATE)
            AND json_extract_string(a.listepersonnes,'$.personne.typePersonne')='pm'
            AND json_type(a.listeetablissements,'$.etablissement')='OBJECT'
            AND coalesce(json_extract_string(a.listeetablissements,'$.etablissement.activite'),'')<>''""",[CUTOFF])
        owners.update(r[0] for r in con.execute('SELECT DISTINCT subject FROM eligible').fetchall())
        rows=con.execute("SELECT id,dateparution,subject,activity FROM eligible WHERE regexp_matches(upper(strip_accents(activity)),?)",[pattern]).fetchall()
        for ann,date,parent,text in rows:
            key=(str(ann),str(date),parent)
            if key in seen:continue
            seen.add(key);ts=words.tokens(text)&wanted
            for token in ts:
                by_token[token].add(parent);evidence[(token,parent)].add((str(ann),str(date)))
        coverage.append(dict(source=ref['source'],total_rows=counts[0],before_cutoff=counts[1],strict_single_objects=counts[2],
            eligible_public_activity_rows=con.execute('SELECT count(*) FROM eligible').fetchone()[0],query_token_matching_rows=len(rows)))
        io.write(OUTPUT/'state.json',dict(status='COMPILING_WHOLE_SOURCE_TYPED_ACTIVITY_CONTEXT',years=len(coverage),
            source_owners=len(owners),owner_token_pairs=sum(len(v) for v in by_token.values()),elapsed_s=time.monotonic()-started,
            updated_unix=time.time()))
        print(Path(path).stem,len(rows),len(owners),flush=True)
    con.close();n=len(owners)
    idfs={t:math.log((n+1)/(len(by_token.get(t,()))+1))/math.log(n+1) if n else 0. for t in wanted}
    output={}
    for qid,tokens in qs.items():
        denom=sum(idfs[t] for t in tokens);matches=defaultdict(set)
        for t in tokens:
            for parent in by_token.get(t,()):matches[parent].add(t)
        entries=[]
        for parent,shared in sorted(matches.items()):
            confidence=.5*sum(idfs[t] for t in shared)/denom if denom else 0.
            if confidence<=0:continue
            entries.append(dict(siren_ref=parent,confidence=confidence,shared_query_tokens=sorted(shared),
                complete_informative_query=tokens<=shared,source_activity_context_not_identity=True,
                publication_ids=sorted({a for t in shared for a,d in evidence[(t,parent)]})))
        output[qid]=entries
    io.write(OUTPUT/'activity_context.json',output);engine.seal(OUTPUT/'activity_context.json')
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_WHOLE_SOURCE_QUERY_ONLY_BODACC_ACTIVITY_CONTEXT',
        implementation=record(Path(__file__)),launch=record(OUTPUT/'launch.json'),recipe=record(OUTPUT/'recipe.json'),
        query_panel=base['panel'],context=record(OUTPUT/'activity_context.json'),coverage=coverage,
        source_owners=n,source_owner_token_pairs=sum(len(v) for v in by_token.values()),
        query_count=len(output),queries_with_context=sum(bool(v) for v in output.values()),
        cumulative_owner_references=sum(map(len,output.values())),maximum_owner_references=max(map(len,output.values())),
        no_prepool_or_label_based_source_selection=True,known_parser_scope_limits_published=True,
        no_activity_text_name_or_person_exported=True,no_alias_created=True,no_recall_measured=True,
        elapsed_seconds=time.monotonic()-started))
    print('COMPLETE',n,sum(map(len,output.values())),flush=True)


if __name__=='__main__':main()
