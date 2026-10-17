"""Read-only top1/rank baseline on the sealed learned_native_ooc V142 lists."""
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_site_operational_metric import (
    RUN, VARIANT, read, record, verified, fingerprint, index, address_match,
    candidate_point, crm_point, distance, threshold, wilson,
)

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

RESULT = ROOT/'docs/step2_baseline_v142.json'
VIEWS = ('exact', 'siren', 'operational', 'site_operational', 'union')
NAMES = dict(exact='SIRET exact', siren='SIREN', operational='Opérationnel historique',
             site_operational='Site opérationnel', union='Union exact ∪ site')
BINS = ('1', '2-3', '4-10', '11-30', '31-100', 'absent')


def rank_bin(rank):
    if rank is None:
        return 'absent'
    return next(name for name, end in zip(BINS, (1, 3, 10, 30, 100)) if rank <= end)


def metric(hits, n):
    return dict(hits=hits, queries=n, percent=100*hits/n,
                wilson95_percent=[100*x for x in wilson(hits, n)])


def summarize(rows):
    n = len(rows)
    bins = {name: sum(rank_bin(r['first_rank']['exact']) == name for r in rows) for name in BINS}
    if sum(bins.values()) != n:
        raise ValueError('Rank histogram denominator mismatch')
    return dict(queries=n, exact_rank_distribution=bins,
        top1={v: metric(sum(r['first_rank'][v] == 1 for r in rows), n) for v in VIEWS},
        recall100={v: metric(sum(r['first_rank'][v] is not None for r in rows), n) for v in VIEWS})


def markdown(result):
    lines = ['# Étape 2 — baseline V142, sans reclassement ni apprentissage', '',
        '`outputs.json["learned_native_ooc"]` ; ordre émis conservé, rangs indexés à1.',
        '**526 requêtes / 52 600 candidats**, folds2/3/4 seulement ; 100 uniques par requête.',
        '`baseline` est une autre variante (481 exacts@100), pas la référence507.', '',
        '| Périmètre | Vue | Top1 n/N (%) | Wilson95 top1 (%) | Présent@100 n/N (%) | Wilson95 @100 (%) |',
        '|---|---|---:|---:|---:|---:|']
    for scope, summary in [('Total', result['totals']), *[(f'Fold {f}', m) for f, m in result['by_fold'].items()]]:
        for view in VIEWS:
            a, b = summary['top1'][view], summary['recall100'][view]
            lo, hi = a['wilson95_percent']
            low, high = b['wilson95_percent']
            lines.append(f'| {scope} | {NAMES[view]} | {a["hits"]}/{a["queries"]} ({a["percent"]:.2f}) | '
                         f'[{lo:.2f} ; {hi:.2f}] | {b["hits"]}/{b["queries"]} ({b["percent"]:.2f}) | [{low:.2f} ; {high:.2f}] |')
    lines += ['', '## Rang du SIRET humain exact', '',
        '| Périmètre | N | 1 | 2–3 | 4–10 | 11–30 | 31–100 | Absent |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope, summary in [('Total', result['totals']), *[(f'Fold {f}', m) for f, m in result['by_fold'].items()]]:
        bins = summary['exact_rank_distribution']
        lines.append(f'| {scope} | {summary["queries"]} | '+' | '.join(str(bins[b]) for b in BINS)+' |')
    lines += ['', '## Définitions et limites', '',
        '- Exact/SIREN : identité humaine ; opérationnel historique : exact ou SIRET accepté dans le sidecar officiel scellé.',
        '- Site opérationnel : même SIREN ET (adresse concordante OU distance admissible), fonctions du contrat `fb604665` réutilisées sans changement.',
        '- Distance200 m si les deux précisions sont au numéro, sinon500 m pour une précision de voie ; commune/inconnue/absente exclue de cette preuve.',
        '- Union = exact OU site. Exact ne vaut pas automatiquement succès site ; absence de preuve site ne prouve pas une erreur fibre.',
        '- Top1 global = succès/N, y compris les19 absences exactes ; aucun statut/label ne filtre le dénominateur.',
        '- Aucun accepteur appliqué : taux et précision des `SUR` **non mesurés**, pas de seuil inventé.',
        '- Wilson bilatéral95 descriptif sur les comptes bruts ; aucune précision nationale indépendante ni borne pondérée revendiquée.',
        '- Pilote déjà consommé, encodeur non entièrement OOF, affectation amont fondée sur le SIREN humain : défaut de serving non corrigé.',
        '- Sources stock/étude : signatures taille/mtime/inode contrôlées avant/après, SHA hérités non recalculés ; petites autorités hashées avant/après.',
        '- Hydratation label-side des seuls candidats émis du bon SIREN pour la métrique, jamais une feature ni une modification des listes.',
        '- Aucun apprentissage, génération, web, jeu fermé, label modifié ou positif injecté.', '',
        'Inventaire et première expérience recommandée : `docs/step2_downstream_inventory.md`.',
        'Commande : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_baseline_v142.py`.',
        'JSON voisin : sources/hashes, totaux, folds et526 lignes query_id avec rangs des cinq vues (aucun nom/adresse CRM brut).']
    return '\n'.join(lines)+'\n'


def main():
    refs = dict(implementation=record(Path(__file__)),
        contract=record(ROOT/'docs/step2_choose_one_contract.md'),
        metric_rule=record(ROOT/'scripts/retrieval_site_operational_metric.py'),
        comparison_rule=record(ROOT/'scripts/retrieval_site_target_full_evaluate.py'),
        full_result=record(ROOT/'docs/retrieval_site_target_full_result_20261016.json'))
    full = read(refs['full_result']['path'])
    refs['previous_metric'] = full['source_refs']['previous_metric']
    prior = read(verified(refs['previous_metric']))
    previous = index(prior['per_query'])
    paired = index(full['per_query'])
    for key in ('generation', 'outputs', 'outcomes', 'report', 'panel', 'evaluation_config',
                'evaluation', 'label_side', 'ledger', 'operational_policy', 'acceptable_operational', 'geometry', 'normalizer'):
        refs[key] = prior['source_refs'][key]
        verified(refs[key])
    generation = read(refs['generation']['path'])
    if (generation['status'] != 'COMPLETE_SEALED_ALL526' or generation['queries'] != 526
            or generation['no_positive_injected'] is not True
            or generation['outputs']['sha256'] != refs['outputs']['sha256']
            or Path(generation['outputs']['path']) != RUN/'outputs.json'):
        raise ValueError('Incomplete or mismatched V142 seal')
    panel = index(read(refs['panel']['path']))
    if len(panel) != 526 or any(r['fold'] not in (2, 3, 4) for r in panel.values()):
        raise ValueError('Only ALL526 folds2/3/4 are allowed')
    outputs = read(refs['outputs']['path'])[VARIANT]
    old = index(read(refs['outcomes']['path'])[VARIANT])
    geometry = index(read(refs['geometry']['path']))
    if any(set(x) != set(panel) for x in (outputs, old, previous, geometry, paired)):
        raise ValueError('Query scope mismatch')
    for cs in outputs.values():
        if len(cs) != 100 or len({c['siret'] for c in cs}) != 100:
            raise ValueError('Expected exactly100 unique emitted candidates')
        if any(not re.fullmatch(r'\d{14}', c['siret']) or c['siren'] != c['siret'][:9]
               or not math.isfinite(c['score']) for c in cs):
            raise ValueError('Malformed candidate')

    cfg = read(refs['evaluation_config']['path'])
    authority = read(refs['evaluation']['path'])
    side = read(refs['label_side']['path'])
    if (cfg['ledger'] != prior['source_refs']['ledger'] or authority['label_side'] != refs['label_side']
            or cfg['evaluation'] != refs['evaluation'] or side['ledger'] != cfg['ledger']
            or side['never_used_in_retrieval'] is not True or side['policy'] != refs['operational_policy']
            or side['files']['acceptable_operational.parquet'] != refs['acceptable_operational']):
        raise ValueError('Label-side authority mismatch')
    filters = [('fold', 'in', [2, 3, 4]), ('query_id', 'in', sorted(panel))]
    labels = index(pq.read_table(cfg['ledger']['path'],
        columns=['query_id', 'fold', 'human_ground_truth_siret', 'human_ground_truth_siren'],
        filters=filters, use_threads=False).to_pylist())
    if set(labels) != set(panel) or any(labels[q]['fold'] != panel[q]['fold'] for q in panel):
        raise ValueError('Filtered label scope/fold mismatch')
    accepted = {q: {r['human_ground_truth_siret']} for q, r in labels.items()}
    for r in pq.read_table(refs['acceptable_operational']['path'], filters=filters, use_threads=False).to_pylist():
        accepted[r['query_id']].add(r['siret'])

    # Evaluation only: hydrate emitted same-SIREN candidates, never regenerate a pool.
    requested = {c['siret'] for q, cs in outputs.items() for c in cs
                 if c['siren'] == labels[q]['human_ground_truth_siren']}
    for key in ('stock', 'study'):
        refs[key] = prior['source_refs'][key]
        signature = fingerprint(refs[key]['path'])
        if any(refs[key][k] != v for k, v in signature.items()):
            raise ValueError('Frozen site source changed')
    con = duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '512MB', 'max_temp_directory_size': '0B'})
    con.register('requested', pa.table({'siret': sorted(requested)}))

    def hydrate(key, columns):
        sql = 'SELECT '+','.join('s.'+c for c in columns)+' FROM read_parquet(?) s SEMI JOIN requested r USING(siret)'
        return index(con.execute(sql, [refs[key]['path']]).fetch_arrow_table().to_pylist(), 'siret')

    sites = hydrate('stock', ['siret', 'numeroVoieEtablissement', 'indiceRepetitionEtablissement',
        'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement',
        'typeVoieEtablissement', 'libelleVoieEtablissement', 'codeCommuneEtablissement'])
    points = hydrate('study', ['siret', 'x', 'y', 'qualite_xy', 'epsg', 'distance_precision'])
    con.close()
    rows = []
    for q in sorted(panel):
        label = labels[q]
        gt, siren = label['human_ground_truth_siret'], label['human_ground_truth_siren']
        if not re.fullmatch(r'\d{14}', gt) or gt[:9] != siren or siren != old[q]['siren_group']:
            raise ValueError('Invalid human identity')
        cp = crm_point(geometry[q]['geometry'])
        ranks = {v: None for v in VIEWS}
        for rank, c in enumerate(outputs[q], 1):
            same = c['siren'] == siren
            site = False
            if same:
                point = candidate_point(points.get(c['siret']))
                d, t = distance(cp, point), threshold(cp, point)
                site = address_match(panel[q], sites.get(c['siret'])) or (d is not None and t is not None and d <= t)
            exact = c['siret'] == gt
            hits = dict(exact=exact, siren=same, operational=c['siret'] in accepted[q],
                        site_operational=site, union=exact or site)
            for v in VIEWS:
                if hits[v] and ranks[v] is None:
                    ranks[v] = rank
        for v in VIEWS[:-1]:
            if (ranks[v] is not None) != previous[q][v]:
                raise ValueError(f'Historical V142 {v} not reproduced: {q}')
        if any((ranks[v] is not None) != old[q][v] for v in VIEWS[:3]):
            raise ValueError('Outcomes mismatch')
        if any((ranks[v] is not None) != paired[q]['v142'][v] for v in VIEWS):
            raise ValueError('Full paired evaluation mismatch')
        rows.append(dict(query_id=q, fold=panel[q]['fold'], top1_siret=outputs[q][0]['siret'], first_rank=ranks))
    totals = summarize(rows)
    if [totals['recall100'][v]['hits'] for v in VIEWS] != [507, 511, 508, 404, 511]:
        raise ValueError('Historical recall100 must be507/511/508/404/511')
    report = read(refs['report']['path'])['variants'][VARIANT]['raw_sample']
    if report['queries'] != 526 or report['exact_hits'] != 507:
        raise ValueError('Incorrect report variant')
    for ref in refs.values():
        if 'sha256' in ref:
            verified(ref)
        else:
            if any(ref[k] != v for k, v in fingerprint(ref['path']).items()):
                raise ValueError('Large source changed during evaluation')
    result = dict(status='MEASURED_STEP2_BASELINE_DEVELOPMENT_NOT_CERTIFICATION',
        variant=VARIANT, queries=526, candidate_count=52600, output_cap=100,
        source_refs=refs, folds=[2, 3, 4], no_regeneration=True, no_training=True,
        sources_mutated=False, labels_modified=False, closed_sets_opened=False,
        sur_policy_evaluated=False, sur_metrics=None, production_promotion_allowed=False,
        totals=totals, by_fold={str(f): summarize([r for r in rows if r['fold'] == f]) for f in (2, 3, 4)},
        per_query=rows)
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    RESULT.with_suffix('.md').write_text(markdown(result))
    print(json.dumps(dict(top1=totals['top1'], exact_rank_distribution=totals['exact_rank_distribution'],
        exact_present100=totals['recall100']['exact']['hits'], queries=526), ensure_ascii=False))


if __name__ == '__main__':
    main()
