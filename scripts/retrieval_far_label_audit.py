"""Read-only, preregistered sample audit of distant human labels; no retrieval."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.retrieval_site_operational_metric import (
    address_match, candidate_point, crm_point, distance, fingerprint, index,
    number, read, record, threshold, verified, wilson,
)

PROTOCOL = ROOT/'docs/retrieval_far_label_audit_protocol_20261015.md'
PRIOR = ROOT/'docs/retrieval_site_operational_metric_result_20261014.json'
RESULT = ROOT/'docs/retrieval_far_label_audit_20261015.json'
SEED = 20261015
CATEGORIES = ('SIEGE', 'AUTRE_SITE', 'AUTRE_ENTREPRISE', 'ADRESSE_CRM_DOUTEUSE',
              'COORD_IMPRECISES', 'INDETERMINE')
ADDRESS_COLS = ['numeroVoieEtablissement', 'indiceRepetitionEtablissement',
                'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement',
                'typeVoieEtablissement', 'libelleVoieEtablissement', 'codePostalEtablissement',
                'libelleCommuneEtablissement', 'codeCommuneEtablissement']
SITE_COLS = ['siren', 'siret', 'etablissementSiege', 'etatAdministratifEtablissement',
             'enseigne1Etablissement', 'enseigne2Etablissement', 'enseigne3Etablissement',
             'denominationUsuelleEtablissement', 'complementAdresseEtablissement'] + ADDRESS_COLS
NAME_COLS = ['siren', 'denominationUniteLegale', 'sigleUniteLegale',
             'denominationUsuelle1UniteLegale', 'denominationUsuelle2UniteLegale',
             'denominationUsuelle3UniteLegale']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true', help='Collect evidence only, no published results')
    parser.add_argument('--review', type=Path, default=RESULT, help='Reviewed annotations; default: existing result JSON')
    args = parser.parse_args()
    started = time.monotonic()
    process = psutil.Process()
    peak = 0
    stopped = threading.Event()

    def guard():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        if peak >= int(3.5*1024**3):
            print(f'STOP RSS >=3.5 GiB: {peak}', file=sys.stderr, flush=True)
            os._exit(2)

    def watch():
        while not stopped.wait(.05):
            guard()

    threading.Thread(target=watch, daemon=True).start()
    rel = str(PROTOCOL.relative_to(ROOT))
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', rel], cwd=ROOT, text=True).strip()
    if not commit or subprocess.check_output(['git', 'show', f'{commit}:{rel}'], cwd=ROOT) != PROTOCOL.read_bytes():
        raise ValueError('STOP: protocol not committed or changed')
    prior_ref = record(PRIOR, '54988149ccfe156c176780c930b0d608b64317024ae7f45022d6aede47836f16')
    prior = read(PRIOR)
    far = sorted(r['query_id'] for r in prior['per_query'] if r['exact'] and r['label_site_diagnostic']['label_far_from_crm'] is True)
    if len(far) != 86:
        raise ValueError('Far population changed')
    selection = random.Random(SEED).sample(far, 30)
    sealed_selection = json.loads(re.search(r'```json\n(.*?)\n```', PROTOCOL.read_text(), re.S)[1])
    if selection != sealed_selection:
        raise ValueError('STOP: selection differs from committed protocol')
    scope = set(selection)
    refs = prior['source_refs']
    used = dict(protocol=record(PROTOCOL), implementation=record(Path(__file__)), previous_result=prior_ref)
    for key in ('panel', 'geometry', 'outputs', 'ledger', 'implementation', 'normalizer'):
        verified(refs[key])
        used['previous_implementation' if key == 'implementation' else key] = refs[key]
    panel = {r['query_id']: r for r in read(refs['panel']['path']) if r['query_id'] in scope}
    geometry = {r['query_id']: r['geometry'] for r in read(refs['geometry']['path']) if r['query_id'] in scope}
    outputs = {q: cs for q, cs in read(refs['outputs']['path'])['learned_native_ooc'].items() if q in scope}
    labels = index(pq.read_table(refs['ledger']['path'], columns=['query_id', 'fold', 'human_ground_truth_siret', 'human_ground_truth_siren'],
                               filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', selection)], use_threads=False).to_pylist())
    if not all(set(x) == scope for x in (panel, geometry, outputs, labels)):
        raise ValueError('Incomplete selected scope')
    if any(panel[q]['fold'] not in (2, 3, 4) or labels[q]['fold'] != panel[q]['fold'] for q in selection):
        raise ValueError('Forbidden or mismatched fold')
    if any(len(cs) > 100 or len({c['siret'] for c in cs}) != len(cs) for cs in outputs.values()):
        raise ValueError('Invalid candidate lists')
    stock = Path(refs['stock']['path'])
    study = Path(refs['study']['path'])
    legal = stock.with_name('StockUniteLegale.parquet')
    legal_config = ROOT/'config/retrieval_v129_national_variants.json'
    used['legal_binding_config'] = record(legal_config)
    legal_ref = read(legal_config)['sources']['current_ul']
    if Path(legal_ref['path']) != legal:
        raise ValueError('Unexpected legal stock binding')
    large = {k: fingerprint(p) for k, p in (('stock', stock), ('study', study), ('legal_stock', legal))}
    for key in ('stock', 'study'):
        if large[key] != {k: refs[key][k] for k in large[key]}:
            raise ValueError(f'Large source changed since prior metric: {key}')
        used[key] = refs[key]
    used['legal_stock'] = dict(large['legal_stock'], semantic_date='2026-08-01',
                              inherited_sha256=legal_ref['sha256'], sha256_recomputed=False)
    con = duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '512MB', 'max_temp_directory_size': '0B'})
    sirens = sorted({r['human_ground_truth_siren'] for r in labels.values()})
    con.register('requested_sirens', pa.table({'siren': sirens}))
    pairs = [{'insee': panel[q]['crm_insee'], 'num': number(panel[q]['crm_number'], panel[q]['crm_suffix'])[0]}
             for q in selection if number(panel[q]['crm_number'], panel[q]['crm_suffix']) and panel[q]['crm_insee']]
    con.register('address_pairs', pa.Table.from_pylist(pairs))
    sites, occupants = {}, {q: [] for q in selection}
    address_scope = {}
    for q in selection:
        num = number(panel[q]['crm_number'], panel[q]['crm_suffix'])
        if num and panel[q]['crm_insee']:
            address_scope.setdefault((panel[q]['crm_insee'], num[0]), []).append(q)
    sql = ('SELECT ' + ','.join('s.'+c for c in SITE_COLS)
           + ' FROM read_parquet(?) s WHERE siren IN (SELECT siren FROM requested_sirens) OR EXISTS '
           '(SELECT 1 FROM address_pairs a WHERE a.insee=s.codeCommuneEtablissement '
           'AND a.num=try_cast(s.numeroVoieEtablissement AS BIGINT))')
    cursor = con.execute(sql, [str(stock)])
    while batch := cursor.fetchmany(2048):
        for values in batch:
            site = dict(zip(SITE_COLS, values))
            if site['siren'] in sirens:
                if site['siret'] in sites:
                    raise ValueError('Duplicate stock SIRET')
                sites[site['siret']] = site
            num = number(site['numeroVoieEtablissement'], site['indiceRepetitionEtablissement'])
            for q in address_scope.get((site['codeCommuneEtablissement'], num[0] if num else None), []):
                if address_match(panel[q], site):
                    occupants[q].append(site)
        guard()
    print('FILTERED_STOCK', len(sites), 'same-SIREN sites', sum(map(len, occupants.values())), 'strict occupants', flush=True)
    con.register('requested_sites', pa.table({'siret': sorted(sites)}))
    cols = ['siret', 'x', 'y', 'qualite_xy', 'epsg', 'distance_precision']
    cursor = con.execute('SELECT '+','.join('s.'+c for c in cols)+' FROM read_parquet(?) s SEMI JOIN requested_sites r USING(siret)', [str(study)])
    points = {}
    while batch := cursor.fetchmany(4096):
        for values in batch:
            row = dict(zip(cols, values))
            if row['siret'] in points:
                raise ValueError('Duplicate point SIRET')
            points[row['siret']] = row
        guard()
    all_sirens = sorted(set(sirens) | {s['siren'] for group in occupants.values() for s in group})
    con.unregister('requested_sirens')
    con.register('requested_sirens', pa.table({'siren': all_sirens}))
    names = index(con.execute('SELECT '+','.join('s.'+c for c in NAME_COLS)+' FROM read_parquet(?) s SEMI JOIN requested_sirens r USING(siren)', [str(legal)]).fetch_arrow_table().to_pylist(), key='siren')
    con.close()
    old_rows = index(prior['per_query'])
    rows = []
    for q in selection:
        label = labels[q]
        siret, siren = label['human_ground_truth_siret'], label['human_ground_truth_siren']
        if siret not in sites or siret not in {c['siret'] for c in outputs[q]}:
            raise ValueError('Missing exact label')
        cp = crm_point(geometry[q])
        ranks = {c['siret']: rank for rank, c in enumerate(outputs[q], 1)}
        evidence = []
        for site in sorted(sites.values(), key=lambda s: s['siret']):
            if site['siren'] != siren:
                continue
            point = candidate_point(points.get(site['siret']))
            d, t = distance(cp, point), threshold(cp, point)
            am = address_match(panel[q], site)
            dm = d is not None and t is not None and d <= t
            evidence.append(dict(site=site, point=point, distance_m=d, threshold_m=t,
                                 address_match=am, distance_match=dm, at_crm_by_rule=am or dm,
                                 emitted_rank=ranks.get(site['siret'])))
        le = next(e for e in evidence if e['site']['siret'] == siret)
        if le['distance_m'] != old_rows[q]['label_site_diagnostic']['distance_m'] or le['threshold_m'] != old_rows[q]['label_site_diagnostic']['threshold_m']:
            raise ValueError('Label metric not reproduced')
        emitted_matches = {e['site']['siret'] for e in evidence if e['at_crm_by_rule'] and e['emitted_rank'] is not None}
        if bool(emitted_matches) != old_rows[q]['site_operational']:
            raise ValueError('Original same-SIREN top100 site measure not reproduced')
        rows.append(dict(query_id=q, fold=label['fold'], crm=panel[q], crm_geometry=geometry[q], crm_point=cp,
                         label_siret=siret, label_siren=siren, official_legal_names=names.get(siren),
                         label_evidence=le, same_siren_sites_scanned=len(evidence),
                         same_siren_at_crm=any(e['at_crm_by_rule'] for e in evidence),
                         same_siren_at_crm_in_top100=bool(emitted_matches), same_siren_evidence=evidence,
                         strict_address_occupants=[dict(site=s, legal_names=names.get(s['siren']), emitted_rank=ranks.get(s['siret'])) for s in occupants[q]]))
    for key, ref in large.items():
        if fingerprint(ref['path']) != ref:
            raise ValueError(f'Large source mutated: {key}')
    for ref in used.values():
        if 'sha256' in ref:
            verified(ref)
    guard()
    stopped.set()
    result = dict(protocol_commit=commit, source_refs=used, seed=SEED, selection=selection,
                  source_population=86, exact_measurable_population=440, sample_size=30,
                  closed_sets_opened=False, labels_modified=False, sources_mutated=False,
                  resources=dict(peak_guarded_rss_bytes=peak, elapsed_seconds=time.monotonic()-started,
                                 duckdb_memory_limit='512MB', threads=1), per_query=rows)
    if args.inspect:
        path = Path('/tmp/sireto_far_label_evidence_20261015.json')
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        for r in rows:
            le = r['label_evidence']
            print('\nQUERY', r['query_id'], 'CRM', r['crm'])
            print('LABEL', r['official_legal_names'], le['site'], 'distance', round(le['distance_m'], 1), 'rank', le['emitted_rank'])
            for e in r['same_siren_evidence']:
                if e['at_crm_by_rule']:
                    print('MATCH', e['site'], 'distance', e['distance_m'], 'address', e['address_match'], 'rank', e['emitted_rank'])
            print('OTHER_OCCUPANTS', [(s['site']['siret'], s['site']['enseigne1Etablissement'], s['legal_names']) for s in r['strict_address_occupants'] if s['site']['siren'] != r['label_siren']])
        print('EVIDENCE', path, 'RESOURCES', result['resources'])
    else:
        publish(result, args.review)


def publish(result, review_path):
    review_ref = record(review_path)
    reviewed = read(review_path)
    annotations = index(reviewed['per_query'])
    if set(annotations) != set(result['selection']):
        raise ValueError('Review must contain exactly the committed 30 query_id')
    review_fields = ('category', 'text_relation', 'phrase', 'measured', 'supposed', 'unknown', 'crm_defects')
    sealed_review = []
    for row in result['per_query']:
        annotation = annotations[row['query_id']].get('review', annotations[row['query_id']])
        annotation = {k: annotation[k] for k in review_fields if k in annotation}
        if not all(annotation.get(k) for k in review_fields[:6]):
            raise ValueError('Incomplete qualitative review')
        category = annotation['category']
        if category not in CATEGORIES or len(annotation['phrase'].split()) > 20:
            raise ValueError(f'Invalid category or >20-word phrase: {row["query_id"]}')
        site = row['label_evidence']['site']
        matches = [e for e in row['same_siren_evidence'] if e['at_crm_by_rule']]
        if category == 'SIEGE' and (site['etablissementSiege'] is not True or annotation['text_relation'] != 'different'):
            raise ValueError('SIEGE requires an official seat and different textual address')
        if category == 'AUTRE_SITE' and (site['etablissementSiege'] is not False or annotation['text_relation'] != 'different'
                                         or not any(e['site']['siret'] != row['label_siret'] for e in matches)):
            raise ValueError('AUTRE_SITE requires another site and a non-seat label')
        if category == 'AUTRE_ENTREPRISE' and (matches or not row['strict_address_occupants']):
            raise ValueError('AUTRE_ENTREPRISE requires absence of same-SIREN match and official occupant evidence')
        if category == 'ADRESSE_CRM_DOUTEUSE' and not annotation.get('crm_defects'):
            raise ValueError('CRM defect must be explicit')
        if category == 'COORD_IMPRECISES' and annotation['text_relation'] not in ('same', 'quasi'):
            raise ValueError('Coordinate category requires textual concordance')
        row['review'] = annotation
        row['category'] = category
        row['same_siren_matches'] = matches
        row['same_siren_match_counts'] = dict(total=len(matches),
            strict_address=sum(e['address_match'] for e in matches),
            distance_only=sum(not e['address_match'] for e in matches),
            in_top100=sum(e['emitted_rank'] is not None for e in matches),
            active=sum(e['site']['etatAdministratifEtablissement'] == 'A' for e in matches),
            closed=sum(e['site']['etatAdministratifEtablissement'] == 'F' for e in matches),
            unknown_state=sum(e['site']['etatAdministratifEtablissement'] not in ('A', 'F') for e in matches))
        full = row['same_siren_evidence']
        row['same_siren_scan_sirets'] = [e['site']['siret'] for e in full]
        row['same_siren_full_evidence_sha256'] = hashlib.sha256(json.dumps(full, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        nearest = sorted((e for e in full if e['distance_m'] is not None), key=lambda e: (e['distance_m'], e['site']['siret']))[:3]
        keep = {row['label_siret']} | {e['site']['siret'] for e in matches+nearest}
        row['same_siren_evidence'] = [e for e in full if e['site']['siret'] in keep]
        row['same_siren_evidence_scope'] = 'Label, all rule matches and three nearest measurable; all same-SIREN sites scanned before selection, inventory retained.'
        sealed_review.append(dict(query_id=row['query_id'], review=annotation))
    verified(review_ref)
    result['review_method'] = reviewed['review_method']
    result['review_sha256'] = hashlib.sha256(json.dumps(sealed_review, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    result['status'] = 'DESCRIPTIVE_LOCAL_SAMPLE_AUDIT_NOT_LABEL_CORRECTION'
    counts = Counter(r['category'] for r in result['per_query'])
    summary = {}
    for cat in CATEGORIES:
        n = counts[cat]
        p = n/30
        lo, hi = wilson(n, 30)
        summary[cat] = dict(count=n, sample_size=30, proportion=p, wilson95=[lo, hi],
                           extrapolated_expected_among_86=86*p, extrapolated_interval_among_86=[86*lo, 86*hi],
                           extrapolated_expected_in_far_stratum_of_440=86*p,
                           extrapolated_share_of_440=86*p/440, extrapolated_share_interval_of_440=[86*lo/440, 86*hi/440])
    result['category_summary'] = summary
    rows = result['per_query']
    result['diagnostics'] = dict(
        unique_queries=len({r['query_id'] for r in rows}),
        unique_label_sirens=len({r['label_siren'] for r in rows}),
        unique_same_siren_sites_scanned=len({s for r in rows for s in r['same_siren_scan_sirets']}),
        queries_with_same_siren_match=sum(r['same_siren_at_crm'] for r in rows),
        queries_with_same_siren_match_in_top100=sum(r['same_siren_at_crm_in_top100'] for r in rows),
        queries_with_strict_address_match=sum(r['same_siren_match_counts']['strict_address'] > 0 for r in rows),
        queries_with_distance_only_matches=sum(r['same_siren_at_crm'] and not r['same_siren_match_counts']['strict_address'] for r in rows),
        queries_with_any_closed_match=sum(r['same_siren_match_counts']['closed'] > 0 for r in rows),
        closed_labels=sum(r['label_evidence']['site']['etatAdministratifEtablissement'] == 'F' for r in rows),
        exact_label_distances_reproduced=30, prior_top100_site_booleans_reproduced=30,
        folds=dict(Counter(str(r['fold']) for r in rows)),
        label_xy_quality=dict(Counter(r['label_evidence']['point']['raw_quality'] for r in rows)))
    result['extrapolation_method'] = 'For each category: 86*p expected among 86 and within their stratum of 440; 86*p/440 share. No inference on other 354 measurable or 67 unmeasurable labels. Wilson without finite population correction; not a national or weighted confidence interval.'
    lines = ['# Audit des 86 labels éloignés — échantillon du 15 octobre 2026', '',
             f'Protocole figé avant les cas : `{result["protocol_commit"]}`.',
             '**30 requêtes tirées sans remise parmi 86**, graine 20261015, ordre lexicographique puis `random.sample`.',
             'Folds 2/3/4 exclusivement ; sources officielles locales, aucun web, retrieval ou label modifié.',
             result['review_method'], '', '## Catégories observées et extrapolations indicatives', '',
             '| Catégorie | Observé /30 | % | Wilson 95 % | Attendu sur 86 (intervalle indicatif) | Part des 440 (intervalle indicatif) |',
             '|---|---:|---:|---:|---:|---:|']
    for cat, m in summary.items():
        lo, hi = (100*x for x in m['wilson95'])
        elo, ehi = m['extrapolated_interval_among_86']
        slo, shi = (100*x for x in m['extrapolated_share_interval_of_440'])
        lines.append(f'| {cat} | {m["count"]}/30 | {100*m["proportion"]:.1f} | [{lo:.1f} ; {hi:.1f}] | {m["extrapolated_expected_among_86"]:.1f} [{elo:.1f} ; {ehi:.1f}] | {100*m["extrapolated_share_of_440"]:.1f} % [{slo:.1f} ; {shi:.1f}] |')
    d = result['diagnostics']
    lines += ['', '**Les valeurs sur 86 et 440 sont des extrapolations, pas des comptes observés.**',
              'Sur les 440 mesurables, le nombre attendu est le même que sur les 86 : seule la strate éloignée est extrapolée.',
              'Calcul : 86 × (compte/30), puis division par 440 pour la part ; jamais 440 × (compte/30).',
              'Les 354 autres mesurables et les 67 non mesurables ne sont pas audités. Wilson brut sans correction de population finie,',
              'sans pondération ; intervalles indicatifs, non simultanés, ne couvrant pas l’erreur de classement qualitatif.',
              '', '## Lecture métier et limites', '',
              f'- **{counts["SIEGE"]+counts["AUTRE_SITE"]}/30** constats administratifs SIEGE ou AUTRE_SITE, pas autant d’erreurs humaines certifiées.',
              '- SIEGE est prioritaire uniquement si une autre adresse textuelle est étayée : un siège à la même adresse n’est pas classé SIEGE.',
              '- Une différence de numéro, une autre entrée ou une adresse postale peuvent expliquer un constat SIEGE sans autre site physique.',
              f'- **{counts["COORD_IMPRECISES"]}/30** adresses concordantes ou quasi : l’écart XY ne suffit donc pas à accuser le label.',
              '- Les quasi-concordances restent des hypothèses (numéro SIRENE absent, ancienne route, avenue/allée) : la règle stricte n’est pas assouplie.',
              '- ADRESSE_CRM_DOUTEUSE signifie adresse insuffisante ou malformée, pas mauvaise cible humaine démontrée.',
              '- INDETERMINE conserve les absences de preuve ; AUTRE_ENTREPRISE exige davantage que zéro concordance du SIREN.',
              '- Sites et sièges : stock du 2026-08-01 ; points SIRENE du 2026-07-01 ; BAN CRM v17 existant.',
              '- Point numéro/CFE n’est pas une entrée fibre certifiée ; sans contrôle terrain, aucune source de point fautive n’est identifiée.',
              '- Ni chronologie du CRM, ni occupation à la date fibre : les labels fermés peuvent refléter une vérité historique.',
              '', '## Même SIREN à l’adresse CRM selon la règle préexistante', '',
              f'- Scan exhaustif des **{d["unique_same_siren_sites_scanned"]} établissements** des **{d["unique_label_sirens"]} SIREN** du tirage, pas seulement le top100.',
              f'- **{d["queries_with_same_siren_match"]}/30** avec concordance ; **{d["queries_with_same_siren_match_in_top100"]}/30** avec concordance déjà dans le top100.',
              f'- **{d["queries_with_strict_address_match"]}/30** avec adresse stricte ; **{d["queries_with_distance_only_matches"]}/30** uniquement par distance.',
              f'- **{d["closed_labels"]}/30** labels fermés ; **{d["queries_with_any_closed_match"]}/30** avec au moins une concordance fermée.',
              '- « Oui » peut signifier proximité à 200/500 m, pas occupant exact : preuve et état sont détaillés dans le JSON.',
              '- « Non » = aucune concordance dans ce stock selon cette règle, jamais absence réelle de l’entreprise.',
              '- Rangs émis à base 1 ; A=actif, F=fermé, ?=état inconnu ; HT100=non émis. Tous les témoins sont dans le JSON.',
              '', '| query_id | Catégorie | Même SIREN au CRM ? rang / état / preuve | Phrase (≤20 mots) |',
              '|---|---|---|---|']
    for row in rows:
        matches = sorted(row['same_siren_matches'], key=lambda e: (e['emitted_rank'] is None, e['emitted_rank'] or 0, e['site']['siret']))
        entries = []
        for e in matches:
            rank = str(e['emitted_rank']) if e['emitted_rank'] is not None else 'HT100'
            state = e['site']['etatAdministratifEtablissement'] or '?'
            entries.append(f'{rank}/{state}/{"adresse" if e["address_match"] else "distance"}')
        cell = 'Oui : ' + ', '.join(entries) if entries else 'Non ; rang/état sans objet'
        lines.append(f'| {row["query_id"]} | {row["category"]} | {cell} | {row["review"]["phrase"]} |')
    lines += ['', '## Éléments manquants et nuances par cas', '']
    for row in rows:
        review = row['review']
        lines.append(f'- `{row["query_id"]}` — **Mesuré** : {review["measured"]} **Supposé** : {review["supposed"]} **Inconnu** : {review["unknown"]}')
    resource = result['resources']
    lines += ['', '## Preuve et reproductibilité', '',
              '- Distances des labels et booléens site top100 historiques reproduits : **30/30 et 30/30**.',
              f'- Folds du tirage : `{json.dumps(d["folds"], sort_keys=True)}` ; qualités XY des labels : `{json.dumps(d["label_xy_quality"], sort_keys=True)}`.',
              '- Lecture filtrée avant matérialisation du ledger, sources grandes vérifiées taille/mtime/inode avant/après.',
              '- Stock unités légales daté du même 2026-08-01 : seules dénominations/sigles officiels, aucun nom personnel ajouté.',
              '- JSON : annotations qualitatives réutilisables, empreinte de revue, adresses et points probants, inventaire des SIRET scannés.',
              f'- Durée du scan final : **{resource["elapsed_seconds"]:.2f} s** ; RSS gardée maximale **{resource["peak_guarded_rss_bytes"]/1024**2:.1f} Mio**, DuckDB 512 Mio/un thread.',
              '- Reproduction réelle : `.venv/bin/python scripts/retrieval_far_label_audit.py` (réutilise les annotations du JSON, rescane les sources).',
              '- Inspection sans publication : même commande avec `--inspect` (preuves temporaires sous `/tmp`).',
              '- Le rapport d’agent antérieur demandé était vide ; aucune information n’en a été déduite.']
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    RESULT.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(category_counts=dict(counts), diagnostics=d, resources=resource), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
