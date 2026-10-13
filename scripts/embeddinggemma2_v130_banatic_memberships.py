"""Whole archived BANATIC membership graph, XML streaming, query-only.

Group/member/adhesion SIRENs are references, never features. Relationship is
context, not identity or site ownership. President/contact fields excluded.
August source is post-cutoff and diagnostic only. No labels or predictions.
"""
from collections import defaultdict
from pathlib import Path
import re
import resource as process_resource
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as naming
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT=io.HOME/'v130/banatic_memberships_20261013_night_b'
SOURCE=Path('/Volumes/CATNAT_DATA/oubusiness/raw/banatic_public_service_competences/2026-08-05_current-v1')
STOCK=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockUniteLegale.parquet')
BASE=io.HOME/'v115/streamed_activity_ablation_20261013_a'
NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
COLUMNS={'D':'N° SIREN','E':'Nom du groupement','AU':'Siren membre','AV':'Nom membre',
         'AW':'Catégorie des membres du groupement','FU':'Adhésion siren','FV':'Adhésion nom'}
RECIPE=dict(hypothesis='PUBLIC_EXPLICIT_GROUP_MEMBER_ADHESION_ONE_HOP_CONTEXT',
    scope='WHOLE_ARCHIVED_FRANCE_EXPORT_NOT_CASE_SELECTED',source_observed='2026-08-05',
    source_eligibility='POST_CUTOFF_DIAGNOSTIC_ONLY_NO_RETROACTIVE_ADMISSION',
    source_columns=COLUMNS, private_contact_president_fields_excluded=True,
    join='EXACT_PUBLIC_CORPORATE_SIRENE_REFERENCE_BOTH_ENDPOINTS',
    seeds='SOURCE_PUBLISHED_GROUP_MEMBER_ADHESION_NAMES_AND_CURRENT_PUBLIC_LEGAL_OWNER_NAMES',
    name_match='ALL_INFORMATIVE_QUERY_TOKENS_MIN_TWO_OR_EQUAL_SINGLE_EXISTING_STOP',
    graph='ONE_UNDIRECTED_CONTEXT_HOP_ROLE_AND_DIRECTION_RETAINED_NO_IDENTITY',
    confidence='0.5/SQRT_DISTINCT_NEIGHBOUR_DEGREE_OF_MATCHED_SOURCE_OWNER',
    duplicate_votes='MAX_NOT_SUM',query_admin_corroboration_required=True,
    no_site_alias_or_address_inheritance=True,no_labels_or_native_retrieval=True)


def tokens(value):return frozenset(naming.tokens(value))


def run():
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    if OUTPUT.exists():raise ValueError('Fresh immutable whole-source namespace')
    if shutil.disk_usage(io.HOME).free<80*1024**3:raise ValueError('CATNAT reserve80GiB')
    start=time.monotonic();OUTPUT.mkdir(parents=True);(OUTPUT/'spill').mkdir()
    def resource_guard():
        peak=process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
        peak_bytes=int(peak if sys.platform=='darwin' else peak*1024)
        if peak_bytes>512*1024**2:raise ValueError('Owned BANATIC source RSS cap512MiB')
        if shutil.disk_usage(io.HOME).free<80*1024**3:raise ValueError('CATNAT reserve80GiB')
        return peak_bytes
    metadata=io.read(SOURCE/'dataset.json')
    resource=next(r for r in metadata['resources'] if r.get('format')=='xlsx')
    if resource['last_modified'][:10]<='2026-08-01':raise ValueError('Observed August postcutoff resource expected')
    workbook=SOURCE/'banatic_france.xlsx';launch=io.read(BASE/'launch.json')
    panel=io.read(verify(launch['panel']))
    if len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('Same whole query-only development panel')
    io.write(OUTPUT/'recipe.json',RECIPE)
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(OUTPUT/'recipe.json'),
        source=record(workbook),metadata=record(SOURCE/'dataset.json'),resource=resource,stock=record(STOCK),
        query_panel=launch['panel'],labels_read=False))
    names=defaultdict(set);edges=set();rows=0
    def identifier(s):
        s=(s or '').strip()
        return s if re.fullmatch('[0-9]{9}',s) else None
    with zipfile.ZipFile(workbook) as z:
        # Shared strings selected lazily are non-person organisational headers
        # and names. The actual archived workbook uses inlineStr for rows.
        with z.open('xl/worksheets/sheet1.xml') as f:
            parser=ET.iterparse(f,events=('start','end'));event,root=next(parser)
            for event,e in parser:
                if event!='end' or e.tag!=NS+'row':continue
                selected={}
                for c in e.findall(NS+'c'):
                    col=re.sub('[0-9]','',c.attrib['r'])
                    if col not in COLUMNS:continue
                    v=c.find(NS+'v');inline=c.find(NS+'is')
                    if c.attrib.get('t')=='s':raise ValueError('Expected observed inline string organisation columns')
                    selected[col]=(''.join(inline.itertext()) if inline is not None else v.text if v is not None else '').strip()
                if rows==0:
                    if selected!=COLUMNS:raise ValueError(('Observed required organisation header',selected))
                else:
                    group,member,adhesion=[identifier(selected.get(k)) for k in ('D','AU','FU')]
                    for key,name_col in ((group,'E'),(member,'AV'),(adhesion,'FV')):
                        if key and selected.get(name_col):names[key].add(selected[name_col])
                    if group and member and group!=member:
                        edges.add((member,group,'MEMBER_OF_GROUP',selected.get('AW','')))
                    if group and adhesion and group!=adhesion:
                        edges.add((group,adhesion,'GROUP_ADHERES_TO_GROUP','GROUP_ADHESION'))
                rows+=1;e.clear();root.clear()
                if rows%4096==0:
                    resource_guard()
                    io.write(OUTPUT/'state.json',dict(status='STREAMING_WHOLE_BANATIC_PUBLIC_COLUMNS',rows=rows,
                        subjects=len(names),distinct_edges=len(edges),updated_unix=time.time()))
    resource_guard()
    con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='128MB'")
    con.execute('SET temp_directory=?',[str(OUTPUT/'spill')])
    con.register('wanted',pa.table({'siren':sorted(names)}))
    owners=con.execute("""SELECT u.siren,u.denominationUniteLegale,u.sigleUniteLegale,
        u.denominationUsuelle1UniteLegale,u.denominationUsuelle2UniteLegale,u.denominationUsuelle3UniteLegale
        FROM read_parquet(?) u JOIN wanted USING(siren)
        WHERE u.statutDiffusionUniteLegale='O' AND u.categorieJuridiqueUniteLegale<>'1000'""",[str(STOCK)]).fetchall();con.close()
    public={r[0] for r in owners}
    for r in owners:
        names[r[0]].update(v for v in r[1:] if v)
    safe_edges=sorted(e for e in edges if e[0] in public and e[1] in public)
    table=pa.Table.from_pylist([dict(from_siren=a,to_siren=b,role=c,member_category=d) for a,b,c,d in safe_edges])
    pq.write_table(table,OUTPUT/'public_relationships.parquet',compression='zstd')
    io.write(OUTPUT/'public_organisation_names.json',{k:sorted(names[k]) for k in sorted(public)})
    neighbours=defaultdict(list)
    for a,b,role,category in safe_edges:
        neighbours[a].append((b,role,'FORWARD',category));neighbours[b].append((a,role,'REVERSE',category))
    qtokens={q['query_id']:tokens(q['crm_name']) for q in panel};index=defaultdict(set)
    for qid,qt in qtokens.items():
        for token in qt:index[token].add(qid)
    matched=defaultdict(set)
    for owner in public:
        for name in names[owner]:
            nt=tokens(name)
            possible=set().union(*(index.get(t,set()) for t in nt))
            for qid in possible:
                qt=qtokens[qid]
                if qt and qt<=nt and (len(qt)>=2 or qt==nt):matched[qid].add(owner)
    context={}
    for q in panel:
        by_owner={}
        for seed in matched[q['query_id']]:
            degree=len({n[0] for n in neighbours[seed]})
            if not degree:continue
            confidence=.5/degree**.5
            for other,role,direction,category in neighbours[seed]:
                if other in matched[q['query_id']]:continue
                item=by_owner.setdefault(other,dict(siren_ref=other,confidence=0.,evidence=[],
                    administrative_corroboration_required=True,no_alias_or_address_inherited=True))
                item['confidence']=max(item['confidence'],confidence)
                item['evidence'].append(dict(source_owner_ref=seed,relationship_role=role,direction=direction,
                    member_category=category,source_degree=degree,not_identity_or_ownership=True))
        context[q['query_id']]=sorted(by_owner.values(),key=lambda r:r['siren_ref'])
    io.write(OUTPUT/'context.json',context)
    for file in ('public_relationships.parquet','public_organisation_names.json','context.json'):engine.seal(OUTPUT/file)
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_SEALED_WHOLE_BANATIC_PUBLIC_RELATIONS_QUERY_ONLY_NOT_RECALL',
        implementation=record(Path(__file__)),launch=record(OUTPUT/'launch.json'),recipe=record(OUTPUT/'recipe.json'),
        relationships=record(OUTPUT/'public_relationships.parquet'),names=record(OUTPUT/'public_organisation_names.json'),
        context=record(OUTPUT/'context.json'),source_rows_including_header=rows,all_source_edges=len(edges),
        public_edges=len(safe_edges),public_organisations=len(public),all_queries=526,
        queries_with_context=sum(bool(v) for v in context.values()),owner_query_references=sum(map(len,context.values())),
        largest_query_owner_degree=max(map(len,context.values())),labels_read=False,
        source_eligibility=False,no_raw_contact_president_fields_exported=True,no_new_recall_or_predictions=True,
        coordination='ROOT_EXPLICIT_LIGHT_STREAM_EXCEPTION_DURING_OWNED_OSM_HEAVY_WORKER',
        cpu_threads=1,max_rss_bytes=resource_guard(),rss_cap_bytes=512*1024**2,
        elapsed_seconds=time.monotonic()-start,observed_unix=time.time()))
    print({k:v for k,v in io.read(OUTPUT/'generation.json').items() if not isinstance(v,dict)},flush=True)


if __name__=='__main__':run()
