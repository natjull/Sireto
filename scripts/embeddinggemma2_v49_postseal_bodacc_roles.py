"""Post-seal source audit only: no retrieval, aliases, training or label edits.

Searches the already consumed 526-case pilot's official parent announcements.
Registry matching is subsequently checked against structured subject IDs.
Director administration remains typed context, never an own-company alias.
No births, private addresses or beneficial-owner objects are extracted.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import time
import unicodedata

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
ROOT = BASE/'v49/postseal_bodacc_roles_20261012_a'
SOURCES = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/official_evidence/bodacc_yearly_v1')
STOCK = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockUniteLegale.parquet')
CUTOFF = '2026-08-01'
STOP = set('DE DU DES LA LE LES ET EN AU AUX A D L SA SAS SASU SARL SCI EURL STE SOCIETE ETS ETABLISSEMENTS GROUPE GROUP SERVICES SERVICE CENTRE COMMUNE COMMUNAUTE MAIRIE INSTITUT'.split())


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')


def reference(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=h.hexdigest())


def norm(value):
    s = unicodedata.normalize('NFKD', str(value or ''))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(re.sub('[^A-Za-z0-9]+', ' ', s).upper().split())


def tokens(s):
    return {t for t in norm(s).split() if len(t)>=3 and t not in STOP and not t.isdigit()}


def decoded(value):
    if not value: return []
    if isinstance(value, str):
        try: value = json.loads(value)
        except ValueError: return []
    if isinstance(value, dict):
        for key in ('items', 'liste', 'values'):
            if isinstance(value.get(key), list): return value[key]
        return [value]
    return value if isinstance(value, list) else []


def unwrap(value):
    if not isinstance(value, dict): return {}
    for k in ('personne', 'personneMorale', 'personnePhysique', 'personnepm', 'personnepp',
              'etablissement', 'precedentProprietairePM', 'precedentProprietairePP',
              'precedentExploitantPM', 'precedentExploitantPP'):
        if isinstance(value.get(k), dict): return value[k]
    return value


def siren(person):
    number = person.get('numeroImmatriculation', person.get('numeroimmatriculation', {}))
    if isinstance(number, dict): number = number.get('numeroIdentification', number.get('numeroidentification', ''))
    number = re.sub('[^0-9]', '', str(number or person.get('siren', '') or person.get('numeroIdentification', '')))
    return number[:9] if len(number) in (9, 14) else ''


def names(person):
    # EI identity only when explicitly the announced subject, not a director.
    result = [str(person[k]) for k in ('denomination', 'nomCommercial', 'enseigne') if person.get(k)]
    if not result and str(person.get('typePersonne', '')).lower()=='pp':
        individual = ' '.join(str(person.get(k, '')) for k in ('prenom', 'nom')).strip()
        if individual: result.append(individual)
    return result


def main():
    if ROOT.exists(): raise ValueError('Fresh immutable audit namespace required')
    source = BASE/'v32/global_public_parent_names_20261012_a'
    generation, exit_record = read(source/'generation.json'), read(source/'generate_exit.json')
    if generation['status']!='COMPLETE_SEALED_ALL526' or exit_record['exit_code']!=0 or not exit_record['own_child_exit_confirmed']:
        raise ValueError('Full owned output required before labels')
    if reference(Path(generation['outputs']['path']))['sha256']!=generation['outputs']['sha256']:
        raise ValueError('Sealed generation hash changed')
    pilot = read(BASE/'v20/deep_audit_20261011_c/rows.json')
    outcomes = {r['query_id']:r for r in read(source/'outcomes.json')['learned_native_ooc']}
    if len(pilot)!=526 or len(outcomes)!=526 or any(r['label']['fold'] not in (2,3,4) for r in pilot):
        raise ValueError('Only same consumed folds2/3/4 all526')
    ROOT.mkdir(parents=True)
    con = duckdb.connect(); con.execute('SET threads=2'); con.execute("SET memory_limit='1GB'")
    con.execute('SET temp_directory=?', [str(ROOT/'spill')])
    wanted = sorted({r['label']['siren'] for r in pilot})
    con.register('wanted', pa.Table.from_pylist([dict(siren=s) for s in wanted]))
    statuses = dict(con.execute('SELECT w.siren,u.statutDiffusionUniteLegale FROM wanted w LEFT JOIN read_parquet(?) u USING(siren)',[str(STOCK)]).fetchall())
    by_parent = defaultdict(list)
    for r in pilot: by_parent[r['label']['siren']].append(r)
    sql = sqlite3.connect(f'file:{BASE}/v14/full_validation_20261008_a/names/owners.sqlite?mode=ro', uri=True)
    known = {s:{norm(n[0]) for n in sql.execute('SELECT name FROM aliases WHERE kind=1 AND owner_key=?', (int(s),))} for s in wanted}
    sql.close()
    accum = {r['query_id']:dict(query_id=r['query_id'], crm_name=r['crm']['crm_name'],
        siren=r['label']['siren'], siret=r['label']['siret'], exact_hit=outcomes[r['query_id']]['exact'],
        owner_diffusion=statuses[r['label']['siren']], announcements=0, own_subject_announcements=0,
        administration_announcements=0, predecessor_announcements=0, new_own_names=[],
        administration_overlap=[], ignored_field_overlap=[], structured_predecessors=[]) for r in pilot}
    coverage=[]; start=time.monotonic(); seen=set()
    pattern='('+'|'.join(wanted)+')'
    for mf in sorted(SOURCES.glob('*/manifest.json')):
        m=read(mf); path=Path(m['sources'][0]['path']); schema=pq.read_schema(path).names
        selected=['id','dateparution','registre','listepersonnes','listeetablissements',
                  'listeprecedentexploitant','listeprecedentproprietaire']
        extra=[c for c in ('acte','modificationsgenerales','divers') if c in schema]
        cols=', '.join('"'+c+'"' for c in selected+extra)
        rows=con.execute(f'SELECT {cols} FROM read_parquet(?) WHERE dateparution <= CAST(? AS DATE) AND regexp_matches(replace(registre,\' \',\'\'), ?)', [str(path),CUTOFF,pattern]).fetch_arrow_table().to_pylist()
        coverage.append(dict(year=mf.parent.name,source=reference(mf),snapshot_path=str(path),
            snapshot_bytes=path.stat().st_size,rows=pq.ParquetFile(path).metadata.num_rows,
            columns=schema,selected_announcements=len(rows)))
        for ann in rows:
            key=(str(ann['id']), str(ann['dateparution']))
            if key in seen: continue
            seen.add(key)
            people=[unwrap(p) for p in decoded(ann['listepersonnes'])]
            pred=[unwrap(p) for f in ('listeprecedentexploitant','listeprecedentproprietaire') for p in decoded(ann[f])]
            own={siren(p):p for p in people if siren(p)}
            reg={s for s in re.findall('[0-9]{9}', (ann['registre'] or '').replace(' ','')) if s in by_parent}
            for subject in reg:
                person=own.get(subject)
                for query in by_parent[subject]:
                    item=accum[query['query_id']]; item['announcements']+=1
                    if not person: continue
                    item['own_subject_announcements']+=1
                    if statuses[subject]=='P': continue  # no recovery of masked identity
                    base=dict(id=ann['id'],date=str(ann['dateparution']),year=mf.parent.name)
                    official_names=names(person)
                    if len(own)==1:
                        official_names += [n for e in decoded(ann['listeetablissements']) for n in names(unwrap(e))]
                    for name in official_names:
                        if norm(name) not in known[subject]:
                            item['new_own_names'].append(dict(base, name=name))
                    admin=str(person.get('administration') or '')
                    if admin: item['administration_announcements']+=1
                    qt=tokens(query['crm']['crm_name'])
                    shared=sorted(qt & tokens(admin))
                    if shared:
                        item['administration_overlap'].append(dict(base,shared_tokens=shared,
                            complete_informative_query=bool(qt and qt<=tokens(admin)), administration=admin))
                    for field in extra:
                        value=str(ann.get(field) or '')
                        overlap=sorted(qt & tokens(value))
                        if overlap:
                            item['ignored_field_overlap'].append(dict(base,field=field,shared_tokens=overlap,
                                complete_informative_query=bool(qt and qt<=tokens(value))))
                    predecessors=[dict(siren=siren(p), names=names(p)) for p in pred if siren(p) and siren(p)!=subject]
                    if predecessors:
                        item['predecessor_announcements']+=1
                        item['structured_predecessors'].append(dict(base,predecessors=predecessors))
        print(json.dumps(dict(year=mf.parent.name,selected=len(rows),elapsed=round(time.monotonic()-start,1))),flush=True)
        write(ROOT/'progress.json',dict(year=mf.parent.name,elapsed=time.monotonic()-start))
    con.close()
    cases=list(accum.values()); write(ROOT/'cases.json',cases)
    def segment(rows):
        return dict(queries=len(rows),parent_announcements=sum(bool(r['announcements']) for r in rows),
            structured_own_subject=sum(bool(r['own_subject_announcements']) for r in rows),
            with_administration=sum(bool(r['administration_announcements']) for r in rows),
            with_new_own_name=sum(bool(r['new_own_names']) for r in rows),
            administration_any_token_overlap=sum(bool(r['administration_overlap']) for r in rows),
            administration_full_query_overlap=sum(any(o['complete_informative_query'] for o in r['administration_overlap']) for r in rows),
            ignored_field_full_query_overlap=sum(any(o['complete_informative_query'] for o in r['ignored_field_overlap']) for r in rows),
            structured_predecessors=sum(bool(r['structured_predecessors']) for r in rows))
    summary=dict(status='COMPLETE_POSTSEAL_SOURCE_AUDIT_NOT_RETRIEVAL',cutoff=CUTOFF,
        all526=segment(cases),miss31=segment([r for r in cases if not r['exact_hit']]),
        hit495=segment([r for r in cases if r['exact_hit']]),coverage=coverage,
        cases=reference(ROOT/'cases.json'),implementation=reference(Path(__file__)),
        limitations=['Registry filter not an exhaustive reverse predecessor search',
          'Source presence and token overlaps are not recovery, causal gains or director identity proof',
          'No dates of birth, personal addresses or beneficial owners read or retained',
          'Protected owner identities not reconstructed',
          'Already consumed development pilot, no closed folds or new independent test'],
        no_retrieval_change=True,no_training=True,no_label_edit=True,elapsed_seconds=time.monotonic()-start)
    write(ROOT/'summary.json',summary);print(json.dumps({k:summary[k] for k in ('status','all526','miss31','hit495','elapsed_seconds')}),flush=True)


if __name__=='__main__': main()
