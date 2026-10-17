"""Postseal-only evaluation of five site-target models against immutable V142."""
import json
import math
import os
from pathlib import Path
import resource
import time
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.retrieval_site_operational_metric import (
    read, record, verified, fingerprint, index, address_match, candidate_point,
    crm_point, distance, threshold, wilson,
)
from scripts.retrieval_site_target_full import (
    ROOT, OUT, PREPARED, RESULT, GIB, atomic_json, storage, check_inventory,
)

VIEWS = ('exact', 'siren', 'operational', 'site_operational', 'union')
NAMES = dict(exact='SIRET exact', siren='SIREN', operational='Opérationnel existant',
             site_operational='Site opérationnel', union='Union exact ∪ site')


def sealed():
    """This gate is called before even opening the previous metric or label config."""
    plan = read(OUT/'plan.json')
    generation = read(OUT/'generation.json')
    if generation['status'] != 'COMPLETE_SEALED_ALL526_EXIT0_BEFORE_LABELS' or generation['queries'] != 526:
        raise ValueError('STOP: cannot evaluate incomplete generation')
    panel = index(read(verified(generation['panel'])))
    outputs = read(verified(generation['outputs']))
    assignment = read(verified(generation['assignment']))
    if len(panel) != 526 or set(panel) != set(outputs) or set(panel) != set(generation['receipts']):
        raise ValueError('Incomplete unique ALL526')
    for qid, ref in generation['receipts'].items():
        receipt = read(verified(ref))
        if receipt['exit_code'] != 0 or not receipt['owned_child_exit_confirmed']:
            raise ValueError('Missing owned exit0')
        part = read(verified(receipt['artifact']))
        if part['query'] != panel[qid] or part['model_id'] != assignment[qid] or part['candidates'] != outputs[qid]:
            raise ValueError('Sealed output/model binding drift')
        if part['raw_labels_read'] or panel[qid]['fold'] not in (2, 3, 4):
            raise ValueError('Unsafe generation')
        cs = outputs[qid]
        if len(cs) > 100 or len({c['siret'] for c in cs}) != len(cs):
            raise ValueError('Invalid output cap/uniqueness')
        verified(part['model'])
    check_inventory(plan)
    return plan, generation, panel, outputs


def metrics(rows, key, view):
    n = len(rows)
    k = sum(r[key][view] for r in rows)
    weights = sum(r['calibrated_query_weight'] for r in rows)
    if not n or weights <= 0:
        raise ValueError('Empty population or invalid weights')
    return dict(hits=k, queries=n, percent=100*k/n,
        wilson95_percent=[100*x for x in wilson(k, n)],
        weighted_hajek_percent=100*sum(r['calibrated_query_weight']*r[key][view] for r in rows)/weights)


def summarize(rows):
    result = {}
    for view in VIEWS:
        gains = sorted(r['query_id'] for r in rows if not r['v142'][view] and r['new'][view])
        losses = sorted(r['query_id'] for r in rows if r['v142'][view] and not r['new'][view])
        result[view] = dict(v142=metrics(rows, 'v142', view), new=metrics(rows, 'new', view),
            paired=dict(gained=len(gains), lost=len(losses), net=len(gains)-len(losses),
                        gain_query_ids=gains, loss_query_ids=losses))
    return result


def finish_publication(checkpoint):
    payload = read(checkpoint)
    atomic_json(RESULT, payload['result'])
    md = RESULT.with_suffix('.md')
    text = payload['markdown']
    if md.exists():
        if md.read_text() != text:
            raise ValueError('Existing Markdown publication differs')
    else:
        # Immutable complete publication; temporary hardlink is retained on interruption.
        temp = md.with_name(md.name+'.pending.'+uuid.uuid4().hex)
        with temp.open('x') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, md)
    path = OUT/'evaluation_payload.json'
    if path.exists():
        done = read(path)
        verified(done['json'])
        verified(done['markdown'])
    else:
        done = dict(json=record(RESULT), markdown=record(md), generation=record(OUT/'generation.json'),
                    postseal_evaluation=True, completed_unix=time.time())
        atomic_json(path, done)
    atomic_json(OUT/'evaluation_done.json', done)


def evaluate():
    started = time.monotonic()
    plan, generation, panel, outputs = sealed()
    checkpoint = OUT/'evaluation_publication.json'
    if checkpoint.exists():
        finish_publication(checkpoint)
        return
    if RESULT.exists() or RESULT.with_suffix('.md').exists():
        raise FileExistsError('Existing result without owned publication checkpoint; preserve and STOP')
    refs = dict(generation=record(OUT/'generation.json'), outputs=generation['outputs'], panel=generation['panel'],
        implementation=record(Path(__file__)), rule=record(ROOT/'scripts/retrieval_site_operational_metric.py'))
    previous_path = ROOT/'docs/retrieval_site_operational_metric_result_20261014.json'
    refs['previous_metric'] = record(previous_path)
    prior = read(previous_path)
    prior_rows = index(prior['per_query'])
    if set(prior_rows) != set(panel):
        raise ValueError('V142 metric scope drift')
    cfgpath = ROOT/'config/retrieval_v358_postseal_evaluation.json'
    refs['evaluation_config'] = record(cfgpath)
    cfg = read(cfgpath)
    if cfg['ledger'] != prior['source_refs']['ledger']:
        raise ValueError('Evaluation ledger authority differs from frozen V142 metric')
    authority = read(verified(cfg['evaluation']))
    side = read(verified(authority['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger'] != cfg['ledger']:
        raise ValueError('Official label-side seal required')
    verified(side['policy'])
    verified(cfg['ledger'])
    filters = [('fold', 'in', [2, 3, 4]), ('query_id', 'in', sorted(panel))]
    labels = index(pq.read_table(cfg['ledger']['path'], columns=['query_id', 'fold', 'human_ground_truth_siret', 'human_ground_truth_siren'],
        filters=filters, use_threads=False).to_pylist())
    if set(labels) != set(panel) or any(labels[q]['fold'] != panel[q]['fold'] for q in panel):
        raise ValueError('Label fold/scope mismatch')
    accepted = {q: {r['human_ground_truth_siret']} for q, r in labels.items()}
    acceptable_ref = side['files']['acceptable_operational.parquet']
    for r in pq.read_table(verified(acceptable_ref), filters=filters, use_threads=False).to_pylist():
        accepted[r['query_id']].add(r['siret'])
    refs.update(ledger=cfg['ledger'], label_side=authority['label_side'], operational_policy=side['policy'],
                acceptable_operational=acceptable_ref, evaluation=cfg['evaluation'])
    old_generation = read(PREPARED/'all_new_sources_max_bano/generation.json')
    refs['v142_outputs'] = old_generation['outputs']
    old_outputs = read(verified(old_generation['outputs']))['learned_native_ooc']
    if set(old_outputs) != set(panel):
        raise ValueError('V142 output scope mismatch')
    geometry = index(read(verified(prior['source_refs']['geometry'])))
    refs['geometry'] = prior['source_refs']['geometry']
    requested = set()
    for pool in (old_outputs, outputs):
        for qid, cs in pool.items():
            requested.update(c['siret'] for c in cs if c['siren'] == labels[qid]['human_ground_truth_siren'])
    large = {k: fingerprint(prior['source_refs'][k]['path']) for k in ('stock', 'study')}
    for key, sig in large.items():
        if any(prior['source_refs'][key][k] != v for k, v in sig.items()):
            raise ValueError('Frozen site source changed')
        refs[key] = prior['source_refs'][key]
    con = duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '512MB', 'max_temp_directory_size': '0B'})
    con.register('requested', pa.table({'siret': sorted(requested)}))
    cols = ['siret', 'numeroVoieEtablissement', 'indiceRepetitionEtablissement', 'dernierNumeroVoieEtablissement',
        'indiceRepetitionDernierNumeroVoieEtablissement', 'typeVoieEtablissement', 'libelleVoieEtablissement', 'codeCommuneEtablissement']
    def hydrate(path, columns):
        sql = 'SELECT '+','.join('s.'+c for c in columns)+' FROM read_parquet(?) s SEMI JOIN requested r USING(siret)'
        return index(con.execute(sql, [path]).fetch_arrow_table().to_pylist(), 'siret')
    sites = hydrate(large['stock']['path'], cols)
    points = hydrate(large['study']['path'], ['siret', 'x', 'y', 'qualite_xy', 'epsg', 'distance_precision'])
    con.close()
    rows = []
    for qid in sorted(panel):
        label = labels[qid]
        if label['human_ground_truth_siret'][:9] != label['human_ground_truth_siren']:
            raise ValueError('Invalid human identity')
        weight = prior_rows[qid]['calibrated_query_weight']
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Invalid sampling weight')
        row = dict(query_id=qid, fold=panel[qid]['fold'], calibrated_query_weight=weight)
        cp = crm_point(geometry[qid]['geometry'])
        for key, pool in [('v142', old_outputs), ('new', outputs)]:
            cs = pool[qid]
            keys = {c['siret'] for c in cs}
            same = [c for c in cs if c['siren'] == label['human_ground_truth_siren']]
            evidence = []
            for c in same:
                point = candidate_point(points.get(c['siret']))
                d, t = distance(cp, point), threshold(cp, point)
                am = address_match(panel[qid], sites.get(c['siret']))
                dm = d is not None and t is not None and d <= t
                if am or dm:
                    evidence.append(dict(siret=c['siret'], address_match=am, distance_match=dm,
                        distance_m=d, threshold_m=t, candidate_precision=point['precision']))
            exact = label['human_ground_truth_siret'] in keys
            site = bool(evidence)
            row[key] = dict(exact=exact, siren=bool(same), operational=bool(keys & accepted[qid]),
                            site_operational=site, union=exact or site)
            row[key+'_site_witnesses'] = evidence
        for view in VIEWS[:-1]:
            if row['v142'][view] != prior_rows[qid][view]:
                raise ValueError(f'V142 {view} not reproduced on {qid}')
        rows.append(row)
    totals = summarize(rows)
    folds = {str(f): summarize([r for r in rows if r['fold'] == f]) for f in (2, 3, 4)}
    if [totals[v]['v142']['hits'] for v in VIEWS[:4]] != [507, 511, 508, 404]:
        raise ValueError('Historical V142 counts differ')
    for view in VIEWS[:-1]:
        expected = prior['totals'][view]['weighted_hajek_percent']
        if not math.isclose(totals[view]['v142']['weighted_hajek_percent'], expected, abs_tol=1e-10):
            raise ValueError('Historical weighting differs')
    for key, sig in large.items():
        if fingerprint(sig['path']) != sig:
            raise ValueError('Large source mutated during evaluation')
    for ref in refs.values():
        if 'sha256' in ref:
            verified(ref)
    check_inventory(plan)
    receipts = [read(p) for p in (OUT/'attempts').glob('*/receipt.json')]
    peak = max([r['peak_sampled_rss_bytes'] for r in receipts]+[resource.getrusage(resource.RUSAGE_SELF).ru_maxrss])
    disk = storage()
    wall = time.time()-plan['initial']['started_unix']
    resources = dict(wall_seconds=wall, evaluation_seconds=time.monotonic()-started,
        worker_seconds=sum(r['elapsed_seconds'] for r in receipts), peak_rss_bytes=peak,
        namespace_allocated_bytes=disk['namespace_allocated_bytes'], free_disk_bytes=disk['free_bytes'],
        initial_free_bytes=plan['initial']['storage']['free_bytes'],
        volume_free_delta_bytes=plan['initial']['storage']['free_bytes']-disk['free_bytes'],
        minimum_free_bytes=min([r['minimum_free_bytes'] for r in receipts]+[disk['free_bytes']]),
        final_outputs_not_yet_counted_in_namespace_bytes=True, timing_caches_reused_bytes_not_counted=True)
    exact = totals['exact']['paired']
    result = dict(status='COMPLETE_ALL526_SITE_TARGET_PAIRED_DEVELOPMENT_NOT_CERTIFICATION',
        contract_commit=plan['contract_commit'], authorization=plan['authorization'],
        exact_remains_primary=True, queries=526, folds=[2, 3, 4], output_cap=100,
        model0_strict_identity_proved=True, same_v142_assignment=True, full_encoder_oof=False,
        sources_mutated=False, labels_modified=False, closed_sets_opened=False,
        production_promotion_allowed=False, source_refs=refs, generation=record(OUT/'generation.json'),
        totals=totals, by_fold=folds, per_query=rows, resources=resources,
        v142_nineteen_exact_misses_won=exact['gained'], v142_exact_successes_lost=exact['lost'],
        wilson_applies_to_raw_counts_only=True,
        weighted_method='sum(calibrated_query_weight * hit) / sum(calibrated_query_weight)',
        limitations=['Consumed development panel; not independent national validation',
            'Historical human-SIREN-based assignment retained by explicit mandate, not deployable label-free routing',
            'Site rule gives public address/proximity evidence, not certified fiber occupancy',
            'One numeric training target absent from cached universe in internal model0, retained with zero numeric features as before'])
    lines = [
        f'Chiffre principal : SIRET exact **{totals["exact"]["new"]["hits"]}/526 ({totals["exact"]["new"]["percent"]:.2f} %)**, contre {totals["exact"]["v142"]["hits"]}/526 pour V142.',
        f'Gagnés : **{exact["gained"]} des 19 ratés exacts V142**.',
        f'Perdus : **{exact["lost"]} succès exacts V142**, soit un bilan net de {exact["net"]:+d}.',
        f'Coût : **{wall/3600:.2f} h réelles**, pic {peak/GIB:.3f} Gio ; espace propre mesuré {disk["namespace_allocated_bytes"]/GIB:.3f} Gio.',
        f'Conclusion mesurée : exact {exact["net"]:+d}, site {totals["site_operational"]["paired"]["net"]:+d}, union {totals["union"]["paired"]["net"]:+d} ; résultat de développement, pas une certification ni une promotion produit.',
        '', '# Recalcul complet cible site — 16 octobre 2026', '',
        'Cinq V32 puis cinq résiduels V71, mêmes partitions/graines/L2/concurrents et affectation V142. Modèle0 recalculé et identique bit à bit au chronométrage.',
        '526 sorties scellées, toutes à code0, avant ouverture du ledger/sidecar dans un processus séparé.',
        'Aucun label modifié ; folds2/3/4 uniquement ; aucun jeu fermé, web, modèle aval ou positif injecté.', '',
        '| Vue | V142 n/N (%) | Nouveau n/N (%) | Wilson95 V142 | Wilson95 nouveau | Pondéré V142 / nouveau (%) | Gagnés / perdus |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def table_line(view, metric, prefix=''):
        a, b, paired = metric['v142'], metric['new'], metric['paired']
        lo, hi = a['wilson95_percent']
        low, high = b['wilson95_percent']
        return (f'| {prefix}{NAMES[view]} | {a["hits"]}/{a["queries"]} ({a["percent"]:.2f}) | '
            f'{b["hits"]}/{b["queries"]} ({b["percent"]:.2f}) | [{lo:.2f} ; {hi:.2f}] | [{low:.2f} ; {high:.2f}] | '
            f'{a["weighted_hajek_percent"]:.2f} / {b["weighted_hajek_percent"]:.2f} | +{paired["gained"]} / −{paired["lost"]} |')
    lines += [table_line(v, totals[v]) for v in VIEWS]
    lines += ['', 'Wilson bilatéral95 sur le brut uniquement ; pondération Hájek historique, sans IC pondéré inventé.',
              'Exact n’est pas un succès site automatique ; union = exact OU site, pas une nouvelle vérité humaine.', '', '## Par fold', '',
              '| Fold / vue | V142 n/N (%) | Nouveau n/N (%) | Wilson95 V142 | Wilson95 nouveau | Pondéré V142 / nouveau (%) | Gagnés / perdus |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for fold, ms in folds.items():
        lines += [table_line(v, ms[v], prefix=fold+' / ') for v in VIEWS]
    lines += ['', '## Gains et pertes appariés : query_id', '']
    for scope, ms in [('Total', totals), *[('Fold '+f, m) for f, m in folds.items()]]:
        lines.append('### '+scope)
        for v in VIEWS:
            p = ms[v]['paired']
            lines.append(f'- **{NAMES[v]}** — gains : `{json.dumps(p["gain_query_ids"], ensure_ascii=False)}` ; pertes : `{json.dumps(p["loss_query_ids"], ensure_ascii=False)}`.')
        lines.append('')
    lines += ['## Coût, intégrité et limites', '',
        f'- Temps réel : {wall:.2f} s ; somme des workers terminés : {resources["worker_seconds"]:.2f} s (hors évaluation encore en cours de publication).',
        f'- Pic RSS : {peak/GIB:.3f} Gio ; garde8Gio/processus ; calcul séquentiel un thread, yield/attente si hydraulic démarre, aucune action VM.',
        f'- Disque : espace propre alloué {disk["namespace_allocated_bytes"]/GIB:.3f} Gio ; libre actuel {disk["free_bytes"]/GIB:.2f} Gio ; '
        f'minimum observé {resources["minimum_free_bytes"]/GIB:.2f} Gio ; variation libre du volume {resources["volume_free_delta_bytes"]/GIB:.3f} Gio (inclut les autres activités).',
        '- Les fichiers finaux de publication et le reçu de fin ne sont pas inclus dans ce relevé ; réutilisation en lecture seule des caches du chronométrage, sans duplication.',
        '- Sources grandes : taille/mtime/inode et hashes hérités, inventaire352 fichiers index vérifié avant/après ; pas de nouveau hash complet des gros segments.',
        '- Les métadonnées points/précision/adresses sont celles de la métrique site préexistante ; son code est réutilisé sans réécriture.',
        '- Modèle choisi avec l’affectation historique fondée sur le SIREN humain, conformément au brief pour appariement ; ce défaut de routage production n’est pas corrigé ici.',
        '- Le panel a déjà servi au développement et Gemma n’est pas entièrement OOF ; aucun nouveau gate national ou SLO établi.',
        '- L’interdit de lecture des labels pré-génération concerne le ledger d’évaluation : apprentissage sur le mapping de cibles déjà scellé et partitions préexistantes, indispensables à la supervision.',
        '', 'Commande : `.venv/bin/python -u scripts/retrieval_site_target_full.py --commit-on-complete`.',
        'Reprise : même commande, mêmes artefacts immuables ; checkpoints et reçus par requête, aucune échéance09h.',
        f'Namespace : `{OUT}` ; suivi `--status` et journal `launch.log`.']
    atomic_json(checkpoint, dict(result=result, markdown='\n'.join(lines)+'\n'))
    finish_publication(checkpoint)
    print(json.dumps(dict(exact=totals['exact'], site=totals['site_operational'], union=totals['union'], resources=resources)), flush=True)
