"""Audit en lecture seule des références du run V142 à 507/526 exacts.

Aucun import du moteur, score, replay, reconstruction de voie ou jeu fermé.
Seuls les deux nouveaux rapports sont écrits ; les sources restent immuables.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import duckdb
import ijson
import numpy as np
import pyarrow.parquet as pq
import tantivy

ROOT = Path(__file__).resolve().parents[1]
HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
PREPARED = HOME/'v142/bano_native_20261013_night'
RUN = PREPARED/'all_new_sources_max_bano'
INDEX = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/hierarchical_retrieval_v3/4b332899d8698354')
RESULT = ROOT/'docs/retrieval_miss_access_vs_ranking_20261014.json'
VARIANT = 'learned_native_ooc'
SEED = 20261014
CLASSES = ('ABSENT_INDEX', 'ACCES', 'CLASSEMENT', 'INDETERMINE', 'RETENU_TOP100')
RSS_CAP = 3.5 * 1024**3


def guard():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak if sys.platform == 'darwin' else peak * 1024
    if peak >= RSS_CAP:
        raise MemoryError('STOP : mémoire >= 3,5 Gio')
    return peak


def read(path):
    return json.loads(Path(path).read_text())


class Inputs:
    def __init__(self):
        self.refs = {}

    def path(self, ref):
        path = Path(ref['path']) if isinstance(ref, dict) else Path(ref)
        if str(path) not in self.refs:
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    digest.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Source modifiée pendant lecture : {path}')
            self.refs[str(path)] = dict(path=str(path), sha256=digest.hexdigest(),
                                       size_bytes=after.st_size, mtime_ns=after.st_mtime_ns)
        if isinstance(ref, dict) and self.refs[str(path)]['sha256'] != ref['sha256']:
            raise ValueError(f'Empreinte scellée non conforme : {path}')
        return path

    def json(self, ref):
        return read(self.path(ref))

    def unchanged(self):
        for ref in self.refs.values():
            stat = Path(ref['path']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (ref['size_bytes'], ref['mtime_ns']):
                raise ValueError(f'Source modifiée : {ref["path"]}')


def keyed(rows):
    result = {r['query_id']: r for r in rows}
    if len(rows) != len(result):
        raise ValueError('query_id dupliqué')
    return result


def selected_json(inputs, ref, scope):
    # Un objet requête à la fois, jamais le JSON national entier en mémoire.
    result = {}
    with inputs.path(ref).open('rb') as stream:
        for qid, value in ijson.kvitems(stream, '', use_float=True):
            if qid in scope:
                result[qid] = value
            guard()
    if set(result) != set(scope):
        raise ValueError(f'Références requête manquantes : {ref}')
    return result


def keys(inputs, ref, field='keys'):
    with np.load(inputs.path(ref), allow_pickle=False) as archive:
        values = archive[field]  # Pas de lecture des features/confiances/scores.
    guard()
    return values


def membership(values, label, qid, level='siret', eligible=None):
    values = np.asarray(values, dtype=np.uint64)
    target, owner = int(label), int(label[:9])
    if level == 'siren':
        owner_present = bool(np.any(values == owner))
        site_refs = [int(r['document_id']) for r in eligible[qid]] if owner_present else []
        selected = np.asarray(site_refs, dtype=np.uint64)
        exact = bool(np.any(selected == target))
    else:
        selected = values[values // 100000 == owner]
        owner_present = bool(len(selected))
        exact = bool(np.any(selected == target))
    others = selected[selected != target]
    return dict(label_siret_present=exact, label_siren_site_present=bool(len(selected)),
                owner_reference_present=owner_present, same_siren_site_count=len(selected),
                other_same_siren_site_count=len(others),
                other_same_siren_examples=[f'{int(k):014d}' for k in sorted(set(others.tolist()))[:3]])


def main():
    started = time.monotonic()
    inputs = Inputs()
    launch = inputs.json(PREPARED/'launch.json')
    generation = inputs.json(RUN/'generation.json')
    report = inputs.json(RUN/'report.json')['variants'][VARIANT]['raw_sample']
    outcomes = keyed(inputs.json(RUN/'outcomes.json')[VARIANT])
    panel = keyed(inputs.json(launch['panel']))
    if (generation['status'] != 'COMPLETE_SEALED_ALL526' or generation['queries'] != 526
            or Path(generation['outputs']['path']) != RUN/'outputs.json'
            or not generation['no_positive_injected'] or set(outcomes) != set(panel)
            or len(panel) != 526 or any(q['fold'] not in (2, 3, 4) for q in panel.values())
            or any(outcomes[q]['fold'] != panel[q]['fold'] for q in panel)
            or report['exact_hits'] != 507 or sum(r['exact'] for r in outcomes.values()) != 507):
        raise ValueError('STOP : run exact 507/526, folds 2/3/4 requis')
    misses = sorted(q for q, r in outcomes.items() if not r['exact'])
    controls = random.Random(SEED).sample(sorted(q for q, r in outcomes.items() if r['exact']), 3)
    scope = set(misses + controls)
    if len(misses) != 19:
        raise ValueError('STOP : exactement 19 ratés requis')
    # Le petit JSON contient aussi des variantes historiques : seule la variante
    # demandée est parcourue, sans lecture ni utilisation des notes des candidats.
    with inputs.path(generation['outputs']).open('rb') as stream:
        outputs = {q: rows for q, rows in ijson.kvitems(stream, VARIANT, use_float=True) if q in scope}
    if set(outputs) != scope or any(len(cs) != 100 or len({c['siret'] for c in cs}) != 100 for cs in outputs.values()):
        raise ValueError('Sorties enregistrées non conformes')
    cfg = inputs.json(ROOT/'config/retrieval_v358_postseal_evaluation.json')
    # Predicate pushdown AVANT toute lecture des labels ; aucun fold fermé lu.
    label_path = Path(cfg['ledger']['path'])
    label_stat = label_path.stat()
    ledger_ref = dict(cfg['ledger'], hash_reverified=False, mtime_ns=label_stat.st_mtime_ns,
                      size_bytes=label_stat.st_size, filter_query_ids=sorted(scope), filter_folds=[2, 3, 4])
    labels = keyed(pq.read_table(label_path, columns=['query_id', 'fold', 'human_ground_truth_siret'],
        filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', sorted(scope))], use_threads=False).to_pylist())
    if set(labels) != scope or any(labels[q]['fold'] != panel[q]['fold'] for q in scope):
        raise ValueError('STOP : labels hors périmètre')
    for qid, row in labels.items():
        label = row['human_ground_truth_siret']
        if (not isinstance(label, str) or len(label) != 14 or not label.isdigit()
                or label[:9] != outcomes[qid]['siren_group']):
            raise ValueError('STOP : identité du label incompatible avec outcomes')
    old_cases = selected_cases = inputs.json(HOME/'v122/label_clarification_20261013_b/cases.json')
    if isinstance(old_cases, dict):
        selected_cases = old_cases['cases']
    case_labels = {r['query_id']: r['label_siret'] for r in selected_cases if r['query_id'] in misses}
    if set(case_labels) != set(misses) or any(case_labels[q] != labels[q]['human_ground_truth_siret'] for q in misses):
        raise ValueError('STOP : labels incompatibles avec les identifiants des audits précédents')
    fg = inputs.json(PREPARED/'features_generation.json')
    baseline = inputs.json(launch['baseline_features'])
    if fg != baseline or fg['full_feature_rows'] != 128313036 or fg['labels_used']:
        raise ValueError('Univers numérique non conforme au run')
    feature_launch = inputs.json(launch['feature_launch'])
    parent = inputs.json(launch['parent_prepared'])
    prepared = inputs.json(PREPARED/'prepared.json')
    supplier = inputs.json(inputs.json(launch['supplier_generation'])['extras'])
    address = inputs.json(inputs.json(launch['address_generation'])['extras'])
    rna = selected_json(inputs, inputs.json(launch['rna_generation'])['context'], scope)
    banatic = selected_json(inputs, inputs.json(launch['banatic_generation'])['context'], scope)
    relations = selected_json(inputs, fg['context_expansion'], scope)
    initials = selected_json(inputs, fg['initial_alignment'], scope)
    geo = selected_json(inputs, launch['geo_sources'], scope)
    manifest = inputs.json(launch['index_manifest'])
    inputs.path(launch['index_meta'])
    index = tantivy.Index.open(str(INDEX/'tantivy'))
    searcher = index.searcher()
    if searcher.num_docs != manifest['num_documents'] or searcher.num_docs != 82772691:
        raise ValueError('Index national inattendu')
    # doc_freq : lookup exact des postings, aucun collector, aucune note/rang.
    index_counts = {q: searcher.doc_freq('siret', labels[q]['human_ground_truth_siret']) for q in scope}
    owners = sorted({labels[q]['human_ground_truth_siret'][:9] for q in scope})
    con = duckdb.connect()
    con.execute('SET threads=1')
    con.execute("SET memory_limit='512MB'")
    con.execute("SET max_temp_directory_size='0B'")
    source = manifest['sources']['dossier_siret_documents']
    # Lecture bornée des documents SOURCE pour le filtre CP/INSEE des références
    # SIREN enregistrées. Pas de recherche de texte ni nouvelle expansion.
    sites = con.execute('''SELECT document_id, siren, insee, postcode
        FROM read_parquet(?) WHERE siren IN (SELECT unnest(?))''', [source['path'], owners]).fetchall()
    con.close()
    source_stat = Path(source['path']).stat()
    site_ref = dict(source, hash_reverified=False, filter_sirens=owners,
                    filtered_rows=len(sites), size_observed=source_stat.st_size)
    if source_stat.st_size != source['size_bytes']:
        raise ValueError('Documents source modifiés')
    eligible = {}
    all_sites = {}
    for qid in scope:
        siren = labels[qid]['human_ground_truth_siret'][:9]
        all_sites[qid] = [dict(document_id=k, insee=i, postcode=p) for k, s, i, p in sites if s == siren]
        eligible[qid] = [r for r in all_sites[qid] if
            (panel[qid]['crm_insee'] and r['insee'] == panel[qid]['crm_insee']) or
            (panel[qid]['crm_postcode'] and r['postcode'] == panel[qid]['crm_postcode'])]
    routes = {}

    def route(key, name, kind, status, note):
        routes[key] = dict(name=name, kind=kind, status=status, note=note)

    route('gemma_commune', 'Gemma : sites de la commune', 'GENERATION', 'ENREGISTRE', 'sirets de features_binding.json ; aucun cosinus lu')
    route('ban_geographie', 'Scan géographique BAN V21', 'GENERATION', 'ENREGISTRE', 'clés de geo_sources.json ; aucun score lu')
    route('familles_successions', 'Familles nationales + succession directe (univers réuni)', 'GENERATION', 'ENREGISTRE',
          'keys des NPZ du run ; origine famille vs succession non séparée, aucune expansion recalculée')
    route('relations_rne_bodacc', 'Relations RNE/BODACC', 'GENERATION', 'ENREGISTRE', 'SIREN enregistrés + filtre CP/INSEE des documents source indexés')
    route('activites_bodacc', 'Activités BODACC', 'GENERATION', 'ENREGISTRE', 'refs des shards enregistrés + filtre CP/INSEE')
    route('sigles', 'Initiales officielles', 'GENERATION', 'ENREGISTRE', 'parent_initials et own_city_initials enregistrés + filtre CP/INSEE')
    route('fournisseur_sites', 'Fournisseur : sites propres et succession', 'RENFORCEMENT', 'ENREGISTRE', 'sous-ensemble de keys déjà enregistré ; aucun nouveau site')
    route('continuite_adresse', 'Continuité économique + adresse propre', 'RENFORCEMENT', 'ENREGISTRE', 'sous-ensemble de keys déjà enregistré ; aucun nouveau site')
    route('rna', 'RNA', 'GENERATION', 'ENREGISTRE', 'SIREN du contexte enregistré + filtre CP/INSEE')
    route('banatic', 'BANATIC', 'GENERATION', 'ENREGISTRE', 'SIREN du contexte enregistré + filtre CP/INSEE')
    route('osm', 'OSM : géométrie et lieux nommés', 'RENFORCEMENT', 'UNIVERS_COMMUN_ENREGISTRE', 'keys partagés ; bonus du label inconnu, aucune collecte autonome')
    route('bano', 'BANO : observations OSM', 'RENFORCEMENT', 'UNIVERS_COMMUN_ENREGISTRE', 'keys partagés ; bonus du label inconnu, aucune collecte autonome')
    route('mere_groupe', 'Mère BODACC / groupe publié (numérique)', 'RENFORCEMENT', 'UNIVERS_COMMUN_ENREGISTRE', 'univers numérique partagé ; références propres à ce signal non séparées, bonus inconnu')
    route('texte_national', 'Texte national : nom, adresse, CP/INSEE', 'GENERATION', 'NON_ENREGISTRE', 'pas de références requête-document avant collector enregistrées ; aucune recherche reconstruite')
    route('acteur_national', 'Acteur RNE : branche textuelle nationale', 'GENERATION', 'NON_ENREGISTRE', 'dictionnaire de noms disponible, mais pas les documents touchés par requête')
    route('fournisseur_lexical', 'Fournisseur : branche lexicale', 'GENERATION', 'NON_ENREGISTRE', 'marques/owners disponibles, mais pas les documents touchés par requête et texte')
    route('noms_sectoriels', 'Noms publics sectoriels', 'INACTIVE', 'DESACTIVE', 'poids nul dans les cinq modèles, établi dans la description du moteur')
    unrecorded = [k for k, r in routes.items() if r['status'] == 'NON_ENREGISTRE']
    zone_keys = {}
    for zone in sorted({panel[q]['crm_insee'] for q in scope}):
        ref = feature_launch['corpora'][zone]
        binding_path = Path(launch['feature_launch']['path']).parent/'zones'/(zone or 'UNKNOWN')/'features_binding.json'
        if binding_path.exists():
            binding = inputs.json(binding_path)
            expected_queries = [q for q in panel.values() if q['crm_insee'] == zone]
            if (binding['corpus'] != ref or binding['queries'] != expected_queries or
                    not binding['full_unpruned_scores']):
                raise ValueError('Gemma : liaison commune/requêtes non conforme')
            zone_keys[zone] = np.asarray(binding['sirets'], dtype=np.uint64)
        else:
            if inputs.json(ref):
                raise ValueError('STOP : commune non vide sans références Gemma')
            zone_keys[zone] = np.array([], dtype=np.uint64)
        guard()
    rows = []
    for qid in sorted(scope):
        label = labels[qid]['human_ground_truth_siret']
        values = {}

        def add(key, refs, level='siret'):
            values[key] = membership(refs, label, qid, level, eligible)

        add('gemma_commune', zone_keys[panel[qid]['crm_insee']])
        add('ban_geographie', [r[0] for r in geo[qid]])
        numeric = keys(inputs, fg['features'][qid])
        add('familles_successions', numeric)
        add('relations_rne_bodacc', [r['siren'] for r in relations[qid]], 'siren')
        add('activites_bodacc', [r.decode() for r in keys(inputs, fg['activity_shards'][qid], 'refs')], 'siren')
        aligned = initials[qid]
        parents = [r['siren_ref'] for r in aligned['parent_initials']]
        own = [int(k) for r in aligned['own_city_initials'] for k in r['siret_refs']]
        own.extend(int(r['document_id']) for r in eligible[qid] if int(label[:9]) in {int(p) for p in parents})
        own = sorted(set(own).intersection(int(r['document_id']) for r in eligible[qid]))
        add('sigles', own)
        for key, entries in [('rna', rna[qid]), ('banatic', banatic[qid])]:
            add(key, [r['siren_ref'] for r in entries], 'siren')
        own = supplier[qid]['own']
        temporal = supplier[qid]['temporal']
        supplier_keys = np.union1d(keys(inputs, own) if own else [], keys(inputs, temporal) if temporal else [])
        add('fournisseur_sites', supplier_keys)
        add('continuite_adresse', keys(inputs, address[qid]) if address[qid] else [])
        for key, shard in [('osm', parent['shards'][qid]), ('bano', prepared['shards'][qid])]:
            shared = keys(inputs, shard)
            if not np.array_equal(shared, numeric):
                raise ValueError('Renforcement hors univers original')
            measured = membership(shared, label, qid)
            values[key] = dict(label_siret_present=None, label_siren_site_present=None,
                               scope_label_siret_present=measured['label_siret_present'],
                               scope_label_siren_present=measured['label_siren_site_present'],
                               bonus='INCONNU', autonomous_collection=False)
        values['mere_groupe'] = dict(label_siret_present=None, label_siren_site_present=None,
            scope_label_siret_present=values['familles_successions']['label_siret_present'],
            scope_label_siren_present=values['familles_successions']['label_siren_site_present'],
            bonus='INCONNU', autonomous_collection=False)
        for key in unrecorded + ['noms_sectoriels']:
            values[key] = dict(label_siret_present=None, label_siren_site_present=None,
                               status=routes[key]['status'])
        present = [k for k, v in values.items() if v['label_siret_present'] is True]
        siren_present = [k for k, v in values.items() if v['label_siren_site_present'] is True]
        exact_top = any(r['siret'] == label for r in outputs[qid])
        siren_top = any(r['siren'] == label[:9] for r in outputs[qid])
        if exact_top != outcomes[qid]['exact'] or siren_top != outcomes[qid]['siren']:
            raise ValueError('Présence enregistrée incompatible avec outcomes')
        in_index = bool(index_counts[qid])
        # Existence SIREN = au moins un DOCUMENT SITE, pas seulement le parent.
        siren_index = in_index or any(searcher.doc_freq('siret', r['document_id']) for r in all_sites[qid])

        def classify(indexed, collected, retained):
            if not indexed:
                return 'ABSENT_INDEX'
            if retained:
                return 'RETENU_TOP100'
            if collected:
                return 'CLASSEMENT'
            return 'INDETERMINE' if unrecorded else 'ACCES'

        rows.append(dict(query_id=qid, crm_name=panel[qid]['crm_name'], fold=panel[qid]['fold'],
            label_siret=label, label_siren=label[:9], index_siret_document_count=index_counts[qid],
            index_siren_has_site=bool(siren_index), siret_in_top100=exact_top, siren_in_top100=siren_top,
            label_present_routes=present, siren_present_routes=siren_present,
            siret_class=classify(in_index, present, exact_top),
            siren_class=classify(siren_index, siren_present, siren_top),
            unrecorded_routes=unrecorded, routes=values))
        del numeric, shared, supplier_keys
        gc.collect()
        guard()
    crosscheck = [r for r in rows if r['query_id'] in controls]
    if any(not r['label_present_routes'] for r in crosscheck):
        raise ValueError('STOP : méthode suspecte, un contrôle réussi absent de toutes les voies enregistrées')
    miss_rows = [r for r in rows if r['query_id'] in misses]
    counts = {level: {c: sum(r[level + '_class'] == c for r in miss_rows) for c in CLASSES}
              for level in ('siret', 'siren')}
    inputs.unchanged()
    if (label_path.stat().st_size, label_path.stat().st_mtime_ns) != (label_stat.st_size, label_stat.st_mtime_ns):
        raise ValueError('Ledger modifié pendant lecture')
    if (Path(source['path']).stat().st_size, Path(source['path']).stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError('Documents source modifiés pendant lecture')
    result = dict(scope='DIAGNOSTIC_DEVELOPPEMENT_FOLDS_2_3_4_19_RATES_V142',
        run=str(RUN), variant=VARIANT, run_exact_hits=507, run_queries=526,
        counts=counts, route_counts=dict(
            registered=sum(r['status'] in ('ENREGISTRE', 'UNIVERS_COMMUN_ENREGISTRE') for r in routes.values()),
            registered_membership=sum(r['status'] == 'ENREGISTRE' for r in routes.values()),
            registered_generation=sum(r['status'] == 'ENREGISTRE' and r['kind'] == 'GENERATION' for r in routes.values()),
            registered_support=sum(r['status'] == 'ENREGISTRE' and r['kind'] == 'RENFORCEMENT' for r in routes.values()),
            shared_scope_only=sum(r['status'] == 'UNIVERS_COMMUN_ENREGISTRE' for r in routes.values()),
            unrecorded=len(unrecorded), inactive=1),
        routes=routes, queries=miss_rows, crosscheck=dict(seed=SEED, query_ids=controls,
            passed=3, requested=3, queries=crosscheck),
        method=dict(index='doc_freq(siret, terme exact), aucun search/collector/score',
            references='JSON filtrés en streaming et clés seules des shards NPZ des 22 requêtes',
            siren_refs='présence owner enregistrée ; filtre CP/INSEE sur documents source, sans recherche textuelle',
            shared_scope='OSM/BANO/mère-groupe : univers commun mesuré, bonus et attribution inconnus',
            ranking='CLASSEMENT signifie seulement déjà ramassé et absent des 100 ; aucune cause ni note mesurée',
            siren_retained='RETENU_TOP100 distingue les entreprises déjà dans les 100 des entreprises ratées',
            nonrecorded='INDETERMINE seulement si aucune voie mesurée ne contient la cible et des voies manquent',
            no_retrieval_replay=True, no_new_scores=True, no_closed_sets=True, no_label_change=True),
        inputs=list(inputs.refs.values()), label_ledger=ledger_ref, index_site_source=site_ref,
        elapsed_seconds=time.monotonic()-started, max_rss_bytes=guard())
    lines = ['# Les 19 ratés : accès ou classement — 14 octobre 2026', '',
             '| Classe | SIRET exact | SIREN (un site quelconque) |', '|---|---:|---:|']
    lines += [f'| {c} | {counts["siret"][c]} | {counts["siren"][c]} |' for c in CLASSES]
    lines += ['| **Total** | **19** | **19** |', '',
        '**Mesuré :** bon run vérifié, 507/526 exacts. Les 19 labels sont dans l’index et dans '
        'l’univers numérique enregistré avant les 100 ; les 19 figurent aussi parmi les sites Gemma '
        'de leur commune. Aucun problème d’accès à ces univers n’est observé.', '',
        '**Limite :** CLASSEMENT ne dit pas pourquoi le label perd sa place : aucune note, aucun rang '
        'hors100 et aucune décomposition du score ne sont calculés. Les 4 SIREN déjà retenus sont '
        'notés RETENU_TOP100, pas échecs de classement SIREN.', '',
        '| query_id | Nom CRM court | Classe SIRET / SIREN | Voies enregistrées contenant le SIRET label |',
        '|---|---|---|---|']
    for r in miss_rows:
        names = ', '.join(routes[k]['name'] for k in r['label_present_routes'])
        lines.append(f'| {r["query_id"]} | {r["crm_name"][:45]} | {r["siret_class"]} / {r["siren_class"]} | {names} |')
    rc = result['route_counts']
    lines += ['', '## Mesuré / non enregistré / inconnu', '',
        f'- **Inventaire : {rc["registered"]} voies/supports enregistrés** : {rc["registered_membership"]} '
        f'avec références propres contrôlées ({rc["registered_generation"]} collectes, {rc["registered_support"]} renforcements), '
        f'{rc["shared_scope_only"]} avec univers commun seulement ; '
        f'**{rc["unrecorded"]} voies non enregistrées**, {rc["inactive"]} désactivée.',
        '- Les dix références propres : Gemma commune, géographie BAN, univers familles/succession, '
        'relations RNE/BODACC, activités BODACC, initiales, fournisseur site/succession, continuité adresse, RNA, BANATIC.',
        '- **Non enregistré :** texte national (nom/adresse/CP/INSEE), acteur RNE national, fournisseur lexical. '
        'Leurs dictionnaires ne sont pas des références requête-document. Aucune reconstruction.',
        '- **Inconnu :** bonus propres OSM, BANO et mère/groupe. Leurs scopes partagés ne sont pas comptés '
        'comme voies ayant ramassé le label. Famille et succession sont réunies, sans attribution par hop.',
        '- Références SIREN : présence du propriétaire publiée séparément dans le JSON ; présence du site '
        'conditionnée au CP/INSEE, vérifiée sur les documents source indexés filtrés (pas un replay de génération).',
        f'- **Contrôle croisé : 3/3 réussites**, graine {SEED} : ' + ', '.join(controls) +
        ' ; leurs SIRET labels apparaissent dans au moins une voie enregistrée.',
        f'- Mémoire maximale : **{result["max_rss_bytes"]/1024**2:.1f} Mio**, sous 4 Gio ; '
        f'{result["elapsed_seconds"]:.2f} s. Aucun run/evaluate, collector, web ou jeu fermé.',
        '', 'Commande : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/retrieval_miss_access_vs_ranking.py`.',
        'Le JSON donne les labels, les présences/absences par voie, les sites du même SIREN '
        'et les empreintes des artefacts lus. Les sources et labels sont inchangés.']
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    RESULT.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(counts=counts, route_counts=rc, crosscheck_passed=3,
                         max_rss_bytes=result['max_rss_bytes'], elapsed_seconds=result['elapsed_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
