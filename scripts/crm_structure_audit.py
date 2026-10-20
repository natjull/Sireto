"""Read-only CRM structure audit; permitted G0 labels only, no model fitting.

Raw label columns are projected to presence flags only. --smoke scores 100
SHA-selected real queries against the same permitted sister universe; it writes
nothing. The full run writes only docs/crm_structure_audit.{md,json}.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time
import unicodedata

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import duckdb
import numpy as np
import psutil
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
R = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100')
H = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
CRM = R/'datasets/crm_new_data_20260817/source/plateforme_new_data_to_cure.csv'
HIST = ROOT/'data/crm_ok_gt.csv'
POP = R/'datasets/crm_gt_v2_commercial_certified_population/73328b7611f37a06'
OPTION = H/'step2_option_d_20261017/full'
RISK = H/'step2_risk_model_v1_20261020_c/full'
FT = H/'step2_gemma_finetune_10h_20261019/full'
BENCH = H/'step2_g0_bench_20261017/full'
OUT = ROOT/'docs/crm_structure_audit'


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def verify(expected):
    actual = ref(expected['path'])
    if actual['sha256'] != expected['sha256']:
        raise ValueError('Source hash mismatch: '+expected['path'])
    return actual


def norm(value):
    value = unicodedata.normalize('NFKD', value or '').encode('ascii', 'ignore').decode().upper()
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in value).split())


def masked(value):
    # No plaintext name/contact/address/identifier is published, even in JSON.
    return ''.join('A' if c.isalpha() else '9' if c.isdigit() else c for c in value)[:64]


def rate(n, d):
    return dict(n=int(n), denominator=int(d), percent=100*n/d if d else None)


class Guard:
    def __init__(self):
        self.started = time.monotonic()
        self.peak = 0
        self.done = threading.Event()
        threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        proc = psutil.Process()
        while not self.done.wait(.05):
            self.peak = max(self.peak, proc.memory_info().rss)
            if self.peak >= 3.85*1024**3 or time.monotonic()-self.started > 3500:
                print('STOP own RSS/time cap', self.peak, flush=True)
                os._exit(75)


def connection():
    return duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '256MB', 'max_temp_directory_size': '0B'})


USAGE = {
    'stc_service_id': ('non (jointure seulement)', 'Dédupliquer les services avant décision collective ; identifiant technique, pas un compte et jamais feature.'),
    'stc_reference': ('non', 'Jointure/doublons seulement tant que la sémantique client/contrat est inconnue ; identifiant instable, pas feature.'),
    'stc_categorie_produit': ('non', 'Stratifier erreurs et routage par offre ; risque de proxy métier/temps et de faible transférabilité.'),
    'stc_state': ('non', 'Stratifier le cycle de vie du service ; risque de fuite si statut postérieur au rattachement.'),
    'client_final': ('oui (nom)', 'Clé de groupe exacte normalisée, sans alias ; homonymes, franchises et noms génériques mélangent des SIREN.'),
    'stc_zipcode': ('oui', 'Adresse et filtrage géographique déjà utilisés.'),
    'stc_city': ('amont seulement, pas entrée directe G0', 'Contexte commune conservé en population ; G0 reçoit INSEE, CP, numéro et voie, pas la ville.'),
    'order_siret': ('non (vérité en amont pour les lignes admises)', 'Ancre vérifiée possible en production, pas feature ; fuite directe si vérité de la cible ou sœur fermée.'),
    'stc_street_number': ('oui', 'Numéro et suffixe déjà utilisés.'),
    'stc_street_name': ('oui', 'Voie déjà utilisée.'),
    'stc_insee_commune': ('oui', 'Commune déjà utilisée.'),
    'crm_name': ('oui', 'Nom historique.'), 'crm_cp': ('oui', 'CP historique.'),
    'crm_insee': ('oui', 'Commune historique.'), 'crm_id': ('non (jointure seulement)', 'Identifiant historique, pas compte client démontré.'),
    'crm_commune': ('amont seulement', 'Ville historique, pas entrée directe G0.'),
    'gt_siret': ('non (label)', 'Vérité historique ; présence agrégée uniquement dans cet inventaire.'),
    'crm_adresse': ('oui', 'Adresse historique parsée en numéro/suffixe/voie.'),
    'SITE_CLI_COMMUNE': ('non directement', 'Champ commune auxiliaire ; redondance probable, sémantique à confirmer, jamais vérité officielle.'),
    'sirene_insee': ('non côté CRM', 'Enrichissement de validation label-side, interdit comme nouveau signal de la cible.'),
    'sirene_cp': ('non côté CRM', 'Enrichissement de validation label-side, risque de fuite.'),
    'sirene_etat': ('non côté CRM', 'État du label, pas du service ; état candidat officiel déjà utilisé, risque de fuite directe ici.'),
    'loc_match_type': ('non (qualification)', 'Qualification issue de la comparaison au label ; audit uniquement, fuite si feature.'),
}


def schema_inventory(con, path, label):
    with path.open(encoding='utf-8-sig') as stream:
        columns = next(csv.reader(stream, delimiter=';'))
    n = con.execute("SELECT count(*) FROM read_csv(?,delim=';',all_varchar=true)", [str(path)]).fetchone()[0]
    result = []
    for col in columns:
        quoted = '"'+col.replace('"', '""')+'"'
        filled = con.execute(f"SELECT count(*) FROM read_csv(?,delim=';',all_varchar=true) WHERE trim(coalesce({quoted},''))<>''", [str(path)]).fetchone()[0]
        if col == label or col in ('sirene_insee', 'sirene_cp', 'sirene_etat', 'loc_match_type'):
            # Labels and label-derived validation fields: aggregate presence only.
            example = '[label/validation masqué ; présence seule]'
        else:
            val = con.execute(f"SELECT {quoted} FROM read_csv(?,delim=';',all_varchar=true) WHERE trim(coalesce({quoted},''))<>'' LIMIT 1", [str(path)]).fetchone()
            example = masked(val[0]) if val else '[vide]'
        usage, interest = USAGE[col]
        result.append(dict(column=col, filled=rate(filled, n), example_masked=example, used=usage, interest_risk=interest))
    return dict(rows=n, columns=result)


def size_summary(groups):
    sizes = np.array([len(v) for v in groups.values()], dtype=int)
    return dict(groups=len(groups), rows=int(sizes.sum()), singleton_groups=int((sizes == 1).sum()),
                multi_groups=int((sizes > 1).sum()), rows_in_multi_groups=int(sizes[sizes > 1].sum()),
                quantiles=dict(zip(('min', 'p50', 'p90', 'p95', 'p99', 'max'), np.quantile(sizes, [0, .5, .9, .95, .99, 1]).tolist()))) if len(sizes) else dict(groups=0, rows=0)


def raw_inventory(con):
    # No raw label value is returned by DuckDB, only a boolean presence flag.
    rows = con.execute("""SELECT stc_service_id,stc_reference,client_final,stc_categorie_produit,stc_state,
        trim(coalesce(order_siret,''))<>'' AS has_siret
        FROM read_csv(?,delim=';',all_varchar=true)""", [str(CRM)]).fetchall()
    groups = {k: defaultdict(list) for k in ('service', 'reference', 'name_norm')}
    service_ref, ref_service = defaultdict(set), defaultdict(set)
    for i, (service, reference, name, category, state, has) in enumerate(rows):
        for key, value in (('service', service), ('reference', reference), ('name_norm', norm(name))):
            if value:
                groups[key][value].append(i)
        service_ref[service].add(reference)
        ref_service[reference].add(service)
    n_missing = sum(not row[-1] for row in rows)
    distinct_inputs = con.execute("SELECT count(DISTINCT (client_final,stc_street_number,stc_street_name,stc_zipcode,stc_city,stc_insee_commune)) FROM read_csv(?,delim=';',all_varchar=true)", [str(CRM)]).fetchone()[0]
    assert len(rows) == 131749 and n_missing == 88465 and distinct_inputs == 102495
    result = dict(rows=len(rows), siret_present=len(rows)-n_missing, siret_absent=n_missing,
                  distinct_raw_query_inputs=distinct_inputs, distinct_product_categories=len({r[3] for r in rows}),
                  distinct_service_states=len({r[4] for r in rows}))
    for key, gg in groups.items():
        mixed = sum(sum(not rows[i][-1] for i in indices) for indices in gg.values() if any(rows[i][-1] for i in indices))
        result[key] = dict(**size_summary(gg), missing_siret_with_present_sister_inventory_only=rate(mixed, n_missing))
    result['reference_relation'] = dict(services_with_multiple_references=sum(len(v) > 1 for v in service_ref.values()),
                                      references_with_multiple_services=sum(len(v) > 1 for v in ref_service.values()))
    return rows, groups, result


def choose2(n):
    return n*(n-1)//2


def coherence(indices, truths, addresses, population, partition):
    pairs = same_siren = address_pairs = same_siret = 0
    mixed = cross_pop = cross_cv = rows_cross_pop = rows_cross_cv = 0
    for ii in indices.values():
        sc = Counter(truths[i][:9] for i in ii)
        pairs += choose2(len(ii))
        same_siren += sum(choose2(n) for n in sc.values())
        mixed += len(sc) > 1
        cp = len({population[i] for i in ii}) > 1
        cc = len({partition[i] for i in ii}) > 1
        cross_pop += cp; cross_cv += cc
        rows_cross_pop += len(ii)*cp; rows_cross_cv += len(ii)*cc
        aa = defaultdict(list)
        for i in ii:
            if addresses[i] is not None:
                aa[addresses[i]].append(i)
        for jj in aa.values():
            address_pairs += choose2(len(jj))
            same_siret += sum(choose2(n) for n in Counter(truths[j] for j in jj).values())
    return dict(**size_summary(indices), same_siren_pairs=rate(same_siren, pairs),
                same_siret_same_address_pairs=rate(same_siret, address_pairs), groups_multiple_siren=mixed,
                groups_cross_population_folds=cross_pop, rows_cross_population_folds=rows_cross_pop,
                groups_cross_cv=cross_cv, rows_cross_cv=rows_cross_cv)


def potential(indices, truths, partition, records, predictions, candidates, views, selected, mode):
    lookup = {}
    for ii in indices.values():
        total = Counter(truths[i][:9] for i in ii)
        by_cv, by_record = defaultdict(Counter), defaultdict(Counter)
        for i in ii:
            by_cv[partition[i]][truths[i][:9]] += 1
            by_record[records[i]][truths[i][:9]] += 1
        for i in ii:
            if i not in selected:
                continue
            cc = total.copy()
            if mode == 'cross_cv_only':
                cc.subtract(by_cv[partition[i]])
            elif mode == 'other_record':
                cc.subtract(by_record[records[i]])
            else:
                cc[truths[i][:9]] -= 1
            cc = +cc
            if cc:
                best = max(cc.values())
                winners = [s for s, count in cc.items() if count == best]
                lookup[i] = winners[0] if len(winners) == 1 else None
    counts = Counter()
    decisions = []
    for i in sorted(selected):
        old = int(np.argmax(predictions[i]))
        new = old
        majority = lookup.get(i)
        if i in lookup and majority is None:
            counts['tied_majority'] += 1
        if majority:
            counts['known'] += 1
            counts['correct_siren'] += majority == truths[i][:9]
            valid = np.flatnonzero(np.char.startswith(candidates[i].astype(str), majority))
            if len(valid):
                counts['candidate_siren_available'] += 1
                new = int(valid[np.argmax(predictions[i, valid])])
            else:
                counts['no_candidate_fallback'] += 1
        counts['changed'] += new != old
        counts['exact'] += bool(views[i, new, 0])
        counts['union'] += bool(views[i, new, 1])
        counts['union_gains'] += bool(views[i, new, 1] and not views[i, old, 1])
        counts['union_losses'] += bool(views[i, old, 1] and not views[i, new, 1])
        decisions.append((i, new, bool(majority), bool(views[i, new, 1])))
    return dict(mode=mode, known_siren=rate(counts['known'], len(selected)),
                known_siren_accuracy=rate(counts['correct_siren'], counts['known']), tied_majority=counts['tied_majority'],
                candidate_siren_available=counts['candidate_siren_available'], no_candidate_fallback=counts['no_candidate_fallback'],
                changed=counts['changed'], top1_exact=rate(counts['exact'], len(selected)), top1_union=rate(counts['union'], len(selected)),
                union_gains=counts['union_gains'], union_losses=counts['union_losses'],
                decisions_sha256=hashlib.sha256(json.dumps(decisions, separators=(',', ':')).encode()).hexdigest())


def run(smoke, guard):
    con = connection()
    sources = [ref(CRM), ref(HIST), ref(POP/'queries.parquet'), ref(POP/'manifest.json'),
               ref(OPTION/'cohort.json'), ref(OPTION/'matrix.json'), ref(RISK/'data.json'), ref(BENCH/'data.json'), ref(Path(__file__))]
    triage_path = R/'datasets/crm_new_data_20260817/location_triage_v1/manifest.json'
    sources.append(ref(triage_path))
    triage = read(triage_path)
    assert sources[0]['sha256'] == triage['sources']['input']['sha256']
    assert sources[1]['sha256'] == triage['sources']['existing_crm']['sha256']
    assert sources[2]['sha256'] == read(POP/'manifest.json')['outputs']['queries.parquet']
    cohort = read(OPTION/'cohort.json')
    sources += [verify(cohort['artifacts'][name]) for name in ('training_queries.json', 'label_side.json')]
    queries = read(OPTION/'training_queries.json')
    ids = [q['query_id'] for q in queries]
    assert len(ids) == len(set(ids)) == 17427 and {q['fold'] for q in queries} == {2, 3, 4}
    labels = read(OPTION/'label_side.json')
    assert set(labels) == set(ids)
    truth = [labels[q]['human_ground_truth_siret'] for q in ids]
    del labels
    data = read(RISK/'data.json'); bench = read(BENCH/'data.json')
    assert data['ids'] == bench['ids'] == ids
    assert data['groups'] == bench['groups'] == [t[:9] for t in truth]
    assert data['partition'] == bench['partition'] and data['folds'] == [q['fold'] for q in queries]
    part, pop = data['partition'], data['folds']
    assert len(set(data['groups'])) == 11483
    siren_cv, siren_pop = defaultdict(set), defaultdict(set)
    for s, cv, fold in zip(data['groups'], part, pop):
        siren_cv[s].add(cv); siren_pop[s].add(fold)
    assert max(map(len, siren_cv.values())) == max(map(len, siren_pop.values())) == 1
    # Query-only projection, explicitly limited to the permitted cohort. No other
    # fold/test/control identity or label is used as a sister.
    query_rows = pq.read_table(POP/'queries.parquet', columns=['query_id', 'crm_record_id', 'data_origin', 'oof_fold'],
                              filters=[('oof_fold', 'in', [2, 3, 4]), ('query_id', 'in', ids)], use_threads=False).to_pylist()
    metadata = {q['query_id']: q for q in query_rows}
    assert set(metadata) == set(ids)
    inventory = dict(latest=schema_inventory(con, CRM, 'order_siret'), historical=schema_inventory(con, HIST, 'gt_siret'))
    raw, raw_groups, raw_stats = raw_inventory(con)
    service_refs = defaultdict(set)
    for service, reference, *_ in raw:
        service_refs[service].add(reference)
    groups = {key: defaultdict(list) for key in ('name_norm', 'service', 'reference')}
    addresses, records = [], []
    for i, q in enumerate(queries):
        m = metadata[q['query_id']]
        records.append((m['data_origin'], m['crm_record_id']) if m['crm_record_id'] else ('query', q['query_id']))
        if name := norm(q['crm_name']):
            groups['name_norm'][name].append(i)
        # Historical record IDs are not the service IDs of this new export.
        if m['data_origin'] == 'REAL_CRM_20260817' and m['crm_record_id'] in service_refs:
            service = m['crm_record_id']
            groups['service'][service].append(i)
            if len(service_refs[service]) == 1:
                groups['reference'][next(iter(service_refs[service]))].append(i)
        address = tuple(norm(q[k]) for k in ('crm_insee', 'crm_postcode', 'crm_number', 'crm_suffix', 'crm_street'))
        addresses.append(address if all(address[k] for k in (0, 1, 2, 4)) else None)
    # Conservative reach to missing raw rows: donors must be real permitted G0
    # labels joined by query origin/service, never raw order_siret values.
    reach = {}
    for key in groups:
        donor_keys = set(groups[key])
        missing_reached = sum(not row[-1] and value in donor_keys for row in raw for value in
                              [row[0] if key == 'service' else row[1] if key == 'reference' else norm(row[2])])
        reach[key] = rate(missing_reached, sum(not row[-1] for row in raw))
    matrix = read(OPTION/'matrix.json')
    sources.append(verify(matrix['keys']))
    candidates = np.load(matrix['keys']['path'], mmap_mode='r')
    sources.append(ref(RISK/'result.json'))
    rr = read(RISK/'result.json')['predictions']; sources.append(verify(rr))
    pack = np.load(rr['path']); predictions = pack['predictions']
    sources.append(verify(bench['views']))
    views = np.load(bench['views']['path'], mmap_mode='r')
    sources.append(ref(FT/'keys.npy')); sources.append(ref(FT/'views.npy'))
    assert np.array_equal(candidates, np.load(FT/'keys.npy', mmap_mode='r'))
    assert np.array_equal(views, np.load(FT/'views.npy', mmap_mode='r'))
    assert candidates.shape == predictions.shape == (17427, 100) and views.shape == (17427, 100, 2)
    top = predictions.argmax(axis=1)
    assert np.array_equal(top, pack['top']) and np.array_equal(views[np.arange(len(ids)), top], pack['hits'])
    assert int(pack['hits'][:, 0].sum()) == 12275 and int(pack['hits'][:, 1].sum()) == 13382
    smoke_indices = set(sorted(range(len(ids)), key=lambda i: hashlib.sha256(('crm-structure-smoke|'+ids[i]).encode()).digest())[:100])
    selected = smoke_indices if smoke else set(range(len(ids)))
    potentials = {key: {mode: potential(gg, truth, part, records, predictions, candidates, views, selected, mode)
                        for mode in ('loo_all', 'other_record', 'cross_cv_only')} for key, gg in groups.items()}
    smoke_replay = potentials if smoke else {key: {mode: potential(gg, truth, part, records, predictions, candidates, views, smoke_indices, mode)
                        for mode in ('loo_all', 'other_record', 'cross_cv_only')} for key, gg in groups.items()}
    consistency = {key: coherence(gg, truth, addresses, pop, part) for key, gg in groups.items()}
    closed = dict(con.execute('SELECT oof_fold,count(*) FROM read_parquet(?) GROUP BY 1', [str(POP/'queries.parquet')]).fetchall())
    con.close()
    for source in sources:
        verify(source)
    peak = max(guard.peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return dict(status='SMOKE_READ_ONLY' if smoke else 'MEASURED_DIAGNOSTIC_NOT_PRODUCTION',
                measured_at=datetime.now(timezone.utc).isoformat(), schema=inventory, raw_groups=raw_stats,
                raw_missing_reach_permitted_g0_donors=reach, g0=dict(queries=len(ids), companies=11483,
                by_origin=dict(Counter(metadata[q]['data_origin'] for q in ids)), scored_queries=len(selected),
                baseline_exact=rate(int(pack['hits'][sorted(selected), 0].sum()), len(selected)),
                baseline_union=rate(int(pack['hits'][sorted(selected), 1].sum()), len(selected))),
                coherence=consistency, potential=potentials, smoke_replay=smoke_replay, closed_counts_only=dict(population_fold_counts=closed,
                test4323=cohort['population']['test4323_excluded'], control962=cohort['population']['control962_excluded'],
                other_excluded_pilot_company_queries=cohort['population']['excluded_pilot_company_queries']),
                custody=dict(label_universe='17427 G0 only; folds2/3/4; closed sets and pilot excluded by sealed cohort',
                raw_siret_values_returned=False, raw_presence_not_label_sisters=True, human_siren_cross_cv=0,
                human_siren_cross_population_folds=0, new_fits=0, label_changes=0), sources=sources,
                elapsed_seconds=time.monotonic()-guard.started, peak_rss_bytes=peak)


def fmt(value):
    return f"{value['n']}/{value['denominator']} ({value['percent']:.2f}%)" if value['denominator'] else '0/0 (indéfini)'


def report(r):
    lines = ['# Audit de la structure CRM — lecture seule', '',
        '**Mesuré localement ; aucun entraînement, label corrigé, pool modifié ou promotion.**',
        '## Sources et périmètre',
        f"Export coût : `{CRM}` : **131749 lignes, 102495 entrées brutes distinctes, 88465 sans SIRET** (trois comptes recomptés ici).",
        f"G0 : `{OPTION}/training_queries.json` et `label_side.json`, **17427 dossiers /11483 entreprises**, issus de `queries.parquet` de la population certifiée `73328b7611f37a06`.",
        'Chaîne : export récent → `filter_new_crm_siret_by_location.py` (triage) → `build_crm_gt_v2_population.py` (avec `data/crm_ok_gt.csv` historique) → certification commerciale → entrées retrieval → cohorte autorisée → optionD/G0. Le manifeste de triage cite les deux sources ; les colonnes de validation historiques ne sont pas des informations CRM brutes supplémentaires.',
        f"Répartition G0 : {r['g0']['by_origin']}. Les taux du fichier historique portent sur son inventaire, jamais sur ses valeurs de labels ; il n'est pas concaténé à l'export récent.",
        'Tous les labels utilisés viennent du seul artefact G0 scellé, exact même ordre que les OOF. Les autres labels folds2/3/4 ne sont pas ouverts pour élargir les sœurs : test4323/control962 et les entreprises du pilote restent exclus.',
        f"Comptage population par projection `oof_fold` seule : {r['closed_counts_only']['population_fold_counts']}. test4323=4323/control962=962 : comptes du manifeste de cohorte, pas recomptage de leurs fichiers ; populations non additionnables.",
        '', '## Schéma exhaustif',
        'Remplissage = chaîne non vide après trim ; exemples réels transformés caractère par caractère (A=lettre,9=chiffre), jamais valeur originale. « Utilisée » concerne les entrées G0/content85/S1/S2 et leur préparation, pas les enrichissements officiels candidats.',
    ]
    for key, title in (('latest', 'Export récent'), ('historical', 'Source historique dérivée du banc')):
        lines += ['', f"### {title} — {r['schema'][key]['rows']} lignes", '| Colonne | Remplissage | Exemple masqué | Utilisée ? | Intérêt / risque |', '|---|---:|---|---|---|']
        for c in r['schema'][key]['columns']:
            lines.append(f"| `{c['column']}` | {fmt(c['filled'])} | `{c['example_masked']}` | {c['used']} | {c['interest_risk']} |")
    lines += ['', '**Absents des sources inspectées** : compte/client/contrat explicite, raison sociale de facturation distincte, contact, domaine e-mail, téléphone, dates de création/mise en service, commentaire libre, ancien SIREN/SIRET distinct. `client_final` est le seul nom client ; `order_siret` est un SIRET existant, pas un ancien SIRET daté. Disponibilité dans le CRM vivant : **inconnue**.',
        '', '## Groupes et couverture des lignes sans SIRET',
        'Clés exactes : identifiant service, référence entière, ou nom normalisé NFKD/ASCII/majuscules/ponctuation en espaces. Aucun fuzzy matching/alias ou préfixe de référence supposé compte client. Les noms vides ne forment pas un groupe.',
        '| Clé | Groupes | Groupes ≥2 / lignes concernées | p50 / p95 / max | Sans SIRET avec présence sœur (inventaire seul) | Avec donneur G0 autorisé |', '|---|---:|---:|---|---:|---:|']
    for key in ('service', 'reference', 'name_norm'):
        g = r['raw_groups'][key]; q = g['quantiles']
        lines.append(f"| {key} | {g['groups']} | {g['multi_groups']} / {g['rows_in_multi_groups']} | {q['p50']:g} / {q['p95']:g} / {q['max']:g} | {fmt(g['missing_siret_with_present_sister_inventory_only'])} | {fmt(r['raw_missing_reach_permitted_g0_donors'][key])} |")
    lines += [f"Relation service↔référence : {r['raw_groups']['reference_relation']}. Une bijection observée ne démontre pas une clé compte ; la référence n'apporte alors pas un groupe client plus large. Catégories produit : {r['raw_groups']['distinct_product_categories']} ; états service : {r['raw_groups']['distinct_service_states']}.",
        'La colonne « inventaire » compte uniquement un booléen présence `order_siret`, sans lire/utiliser sa valeur ni sa justesse : ce n’est pas un matching par labels, et ce chiffre ne permet pas d’ouvrir des sœurs des jeux fermés. La dernière colonne est plus conservatrice : existence de donneurs du seul G0, identifiés par nom ou service d’origine récent. Aucun label raw n’est propagé. Ne pas extrapoler sa justesse à 88465 cibles non labellisées.',
        '', '## Cohérence des seuls labels autorisés folds2/3/4',
        'Statistique pondérée par paires non ordonnées de dossiers distincts, pas moyenne des clients. Même adresse = commune+CP+numéro+suffixe+voie normalisés exacts ; commune, CP, numéro, voie obligatoires, aucun rapprochement géographique. Les services/références sont joignables seulement pour la partie récente du banc.',
        '| Clé | Dossiers / groupes | Paires même SIREN | Même SIRET à même adresse | Groupes multi-SIREN | Groupes traversant folds population / CV |', '|---|---:|---:|---:|---:|---:|']
    for key, g in r['coherence'].items():
        lines.append(f"| {key} | {g['rows']} / {g['groups']} | {fmt(g['same_siren_pairs'])} | {fmt(g['same_siret_same_address_pairs'])} | {g['groups_multiple_siren']} | {g['groups_cross_population_folds']} / {g['groups_cross_cv']} |")
    lines += ['', 'Groupes SIREN humains : **0 traversée de CV et de fold population**. Un groupe de nom traversant les CV relie donc plusieurs entreprises humaines. Le JSON donne aussi le nombre de dossiers concernés. Les labels du reste des folds2/3/4 hors cohorte G0 ne sont pas inspectés (restriction volontaire de garde).',
        '', '## Potentiel sur les OOF S1+S2-BASE existants',
        f"Référence actuelle : exact **{fmt(r['g0']['baseline_exact'])}**, UNION **{fmt(r['g0']['baseline_union'])}** ; les 17427 dossiers et les absences restent au dénominateur.",
        'Un vote par dossier sœur, cible exclue ; majorité unique (pluralité maximale), égalité = abstention, jamais départage par identifiant. Filtre sur les 100 candidats inchangés, meilleur score OOF dans le SIREN majoritaire ; si aucun candidat du SIREN, retour top1 original (pas d’injection). Ex æquo score : ordre G0. Labels/UNION scellés inchangés.',
        '- `loo_all` : toutes les autres lignes autorisées du groupe ; **oracle transductif label-side, pas OOF valide pour une entreprise inconnue**.',
        '- `other_record` : exclut aussi toutes les lignes du même identifiant de service/enregistrement et origine ; contrôle des duplications, reste transductif.',
        '- `cross_cv_only` : sœurs hors CV externe cible ; respecte la séparation entreprise pour les donneurs. Ni calibration ni recette de production nouvelle. La vérité du SIREN cible étant entièrement exclue du train entreprise, ce mode ne peut révéler correctement son SIREN : contrôle négatif, pas promesse de gain.',
        '| Groupe / mode | SIREN connu | Justesse SIREN connu | Top1 UNION | Δ succès UNION | Exact | Sans candidat / égalités majorité |', '|---|---:|---:|---:|---:|---:|---:|']
    for key, modes in r['potential'].items():
        for mode, p in modes.items():
            lines.append(f"| {key}/{mode} | {fmt(p['known_siren'])} | {fmt(p['known_siren_accuracy'])} | {fmt(p['top1_union'])} | +{p['union_gains']}/−{p['union_losses']}={p['union_gains']-p['union_losses']:+d} | {fmt(p['top1_exact'])} | {p['no_candidate_fallback']} / {p['tied_majority']} |")
    lines += ['', 'Le gain transductif éventuel est une simulation diagnostique conditionnelle à des labels de sœurs fiables, pas une borne optimale ni un gain certifié du reclasseur. Même sans auto-vote, les sœurs du même SIREN sont dans le même CV externe : leur vérité n’est pas disponible au ranker OOF. En outre une prédiction de sœur produite par un autre modèle OOF pourrait avoir été entraînée sur l’entreprise cible ; elle ne remplace pas naïvement ces vérités. Un futur collectif nécessiterait des composantes client entièrement disjointes et des ancrages antérieurs temporellement vérifiés.',
        '', '## Mesuré / estimé / inconnu et recommandation',
        '**Mesuré** : remplissages, groupes, présence brute agrégée, cohérence et simulations OOF ci-dessus ; sources SHA avant/après et liaison clés/vues/top1 revérifiées. **Estimé** : aucun gain national ni précision de déploiement n’est extrapolé. **Inconnu** : sémantique compte/contrat des références, dates des ancrages, fiabilité des SIRET raw et inventaire du CRM vivant.',
        '**Recommandation** : demander un export métier comportant un vrai compte client stable et les dates de disponibilité des ancrages ; ne pas adopter le filtre SIREN oracle ni changer G0. Tester ensuite un collectif query-only ou des ancrages historiques audités avec composantes client disjointes. À court terme, dédupliquer les services après validation métier et stratifier par catégorie/état ; garder `A_VERIFIER`. Le gain des labels de sœurs n’autorise aucune feature identifiant, ni emploi de labels fermés.',
        '', '## Preuve et reproduction', '```bash',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/crm_structure_audit.py --smoke',
        'nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/crm_structure_audit.py > /tmp/sireto-crm-structure-full.log 2>&1 &',
        '```',
        f"Run complet : {r['elapsed_seconds']:.2f}s ; pic RSS {r['peak_rss_bytes']/1024**3:.3f}Gio, garde 3,85Gio/3500s, DuckDB256Mio sans spill. Aucune lecture de valeur label fermé, entraînement, web, nouveau paquet, action VM/PID8288, modification source/tiers/pending, suppression ou push. Les seuls fichiers écrits par le script sont les deux rapports ; le smoke n’écrit rien."]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    guard = Guard()
    result = run(args.smoke, guard)
    if not args.smoke:
        OUT.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        OUT.with_suffix('.md').write_text(report(result))
    guard.done.set()
    print(json.dumps(dict(status=result['status'], g0=result['g0'], potential=result['potential']['name_norm'],
                          seconds=result['elapsed_seconds'], peak_gib=result['peak_rss_bytes']/1024**3), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
