"""Read-only additional CRM site metric on the sealed v142 ALL526 outputs."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
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
from scripts.embeddinggemma2_v17_structural_retrieval import normalize

HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
RUN = HOME/'v142/bano_native_20261013_night/all_new_sources_max_bano'
VARIANT = 'learned_native_ooc'
CONTRACT = ROOT/'docs/retrieval_site_operational_metric_contract_20261014.md'
RESULT = ROOT/'docs/retrieval_site_operational_metric_result_20261014.json'
STUDY_RECEIPT = Path('/Volumes/CATNAT_DATA/oubusiness/raw/receipts/sirene_geolocalisation.json')
VIEWS = ('exact', 'siren', 'operational', 'site_operational', 'site_100m', 'site_200m', 'site_500m')
NAMES = ('SIRET exact', 'SIREN', 'Opérationnel existant', 'Site opérationnel',
         'Site seuil plat 100 m', 'Site seuil plat 200 m', 'Site seuil plat 500 m')
MAX_RSS = int(3.5 * 1024**3)


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def record(path, expected=None):
    ref = fingerprint(path)
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    if fingerprint(path) != ref:
        raise ValueError(f'Input changed: {path}')
    ref['sha256'] = digest.hexdigest()
    if expected is not None and ref['sha256'] != expected:
        raise ValueError(f'Sealed SHA mismatch: {path}')
    return ref


def verified(ref):
    record(ref['path'], ref['sha256'])
    return Path(ref['path'])


def index(rows, key='query_id'):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return result


def number(value, suffix=''):
    text = normalize(f'{value or ""} {suffix or ""}')
    match = re.fullmatch(r'(\d+)\s*(BIS|TER|QUATER|[A-Z])?', text)
    return (int(match[1]), match[2] or '') if match and int(match[1]) > 0 else None


def address_match(query, site):
    if not site or site.get('dernierNumeroVoieEtablissement') or site.get('indiceRepetitionDernierNumeroVoieEtablissement'):
        return False
    qnum = number(query['crm_number'], query['crm_suffix'])
    snum = number(site['numeroVoieEtablissement'], site['indiceRepetitionEtablissement'])
    qstreet = normalize(query['crm_street'], street=True)
    sstreet = normalize(f'{site["typeVoieEtablissement"] or ""} {site["libelleVoieEtablissement"] or ""}', street=True)
    return bool(query['crm_insee'] and query['crm_insee'] == site['codeCommuneEtablissement']
                and qnum and qnum == snum and qstreet and qstreet == sstreet)


def supported(x, y):
    try:
        x, y = float(x), float(y)
    except (ValueError, TypeError):
        return None
    return [x, y] if math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1300000 and 6000000 <= y <= 7300000 else None


def crm_point(g):
    point = supported(g.get('ban_x_l93'), g.get('ban_y_l93'))
    if not g.get('ban_resolved') or not g.get('spatial_crs_compatible'):
        point = None
    kind = str(g.get('ban_type_position', '')).lower()
    if point is None:
        precision = 'missing_coordinates'
    elif kind in ('municipality', 'commune', 'city'):
        precision = 'commune'
    elif g.get('precision') == 'STREET_FOOTPRINT_NOT_HOUSE' or g.get('ban_declared_weak'):
        precision = 'street_imprecise'
    elif kind in ('entrée', 'bâtiment', 'logement'):
        precision = 'housenumber_equivalent'
    else:
        precision = 'street_imprecise'
    return dict(source='BAN_CRM_v17', point_l93=point, precision=precision,
                confidence_score=None, raw_type_position=g.get('ban_type_position'),
                raw_precision=g.get('precision'), declared_weak=g.get('ban_declared_weak'),
                uncertainty_m=g.get('ban_uncertainty_m'), source_position=g.get('ban_source_position'))


def candidate_point(row):
    row = row or {}
    point = supported(row.get('x'), row.get('y')) if row.get('epsg') == '2154' else None
    quality = row.get('qualite_xy')
    precision = {'11': 'housenumber_equivalent', '21': 'housenumber_equivalent',
                 '12': 'street_imprecise', '22': 'street_imprecise', '33': 'commune'}.get(quality, 'unknown')
    if point is None:
        precision = 'missing_coordinates'
    dp = row.get('distance_precision')
    return dict(source='SIRENE_STUDY_2026_07_01', point_l93=point, precision=precision,
                confidence_score=None, raw_quality=quality, epsg=row.get('epsg'),
                distance_precision_m=float(dp) if dp is not None and math.isfinite(dp) else None)


def distance(a, b):
    pa_, pb = a['point_l93'], b['point_l93']
    return math.hypot(pa_[0] - pb[0], pa_[1] - pb[1]) if pa_ and pb else None


def threshold(a, b):
    kinds = (a['precision'], b['precision'])
    if any(k not in ('housenumber_equivalent', 'street_imprecise') for k in kinds):
        return None
    return 200 if all(k == 'housenumber_equivalent' for k in kinds) else 500


def wilson(hits, n):
    z = 1.959963984540054
    p = hits / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def summarize(rows):
    weight = sum(r['calibrated_query_weight'] for r in rows)
    return {view: dict(hits=sum(r[view] for r in rows), queries=len(rows),
                       percent=100*sum(r[view] for r in rows)/len(rows),
                       wilson95_percent=[100*x for x in wilson(sum(r[view] for r in rows), len(rows))],
                       weighted_hajek_percent=100*sum(r['calibrated_query_weight']*r[view] for r in rows)/weight)
            for view in VIEWS}


def main():
    started = time.monotonic()
    process = psutil.Process()
    peak = 0

    def guard():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        if peak >= MAX_RSS:
            raise MemoryError(f'STOP RSS >=3.5 GiB: {peak}')

    # Require the premeasurement contract to be committed and unchanged.
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(CONTRACT.relative_to(ROOT))], cwd=ROOT, text=True).strip()
    committed = subprocess.check_output(['git', 'show', f'{commit}:{CONTRACT.relative_to(ROOT)}'], cwd=ROOT)
    if not commit or committed != CONTRACT.read_bytes():
        raise ValueError('STOP: contract must be committed before calculation')
    refs = dict(contract=record(CONTRACT), implementation=record(Path(__file__)),
                launch=record(RUN.parent/'launch.json'), outcomes=record(RUN/'outcomes.json'),
                generation=record(RUN/'generation.json'), report=record(RUN/'report.json'),
                normalizer=record(ROOT/'scripts/embeddinggemma2_v17_structural_retrieval.py'))
    launch = read(RUN.parent/'launch.json')
    generation = read(RUN/'generation.json')
    if generation['status'] != 'COMPLETE_SEALED_ALL526' or generation['queries'] != 526 or not generation['no_positive_injected']:
        raise ValueError('STOP: incomplete or unsealed ALL526')
    if Path(generation['outputs']['path']) != RUN/'outputs.json':
        raise ValueError('Unexpected output binding')
    refs['outputs'] = record(verified(generation['outputs']))
    panel = index(read(verified(launch['panel'])))
    scope = set(panel)
    old = index(read(RUN/'outcomes.json')[VARIANT])
    outputs = read(RUN/'outputs.json')[VARIANT]
    if len(panel) != 526 or set(old) != scope or set(outputs) != scope or any(q['fold'] not in (2, 3, 4) for q in panel.values()):
        raise ValueError('STOP: exact unique ALL526 scope and folds 2/3/4 required')
    for qid, cs in outputs.items():
        if len(cs) > 100 or len({c['siret'] for c in cs}) != len(cs):
            raise ValueError('STOP: output cap or duplicate candidates')
        if any(not re.fullmatch(r'\d{14}', c['siret']) or c['siren'] != c['siret'][:9] for c in cs):
            raise ValueError('Malformed candidate identity')
    geometry = index(read(verified(launch['geometry'])))
    if set(geometry) != scope:
        raise ValueError('Geometry scope mismatch')
    refs.update(panel=launch['panel'], geometry=launch['geometry'])
    cfgpath = ROOT/'config/retrieval_v358_postseal_evaluation.json'
    refs['evaluation_config'] = record(cfgpath)
    cfg = read(cfgpath)
    evaluation = read(verified(cfg['evaluation']))
    side = read(verified(evaluation['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger'] != cfg['ledger']:
        raise ValueError('Unsealed official label-side authority')
    verified(side['policy'])
    verified(cfg['ledger'])
    filters = [('fold', 'in', [2, 3, 4]), ('query_id', 'in', sorted(scope))]
    labels = index(pq.read_table(cfg['ledger']['path'], filters=filters, use_threads=False).to_pylist())
    if set(labels) != scope or any(r['fold'] != panel[q]['fold'] for q, r in labels.items()):
        raise ValueError('Label scope mismatch')
    accepted = {q: {r['human_ground_truth_siret']} for q, r in labels.items()}
    accepted_ref = side['files']['acceptable_operational.parquet']
    for r in pq.read_table(verified(accepted_ref), filters=filters, use_threads=False).to_pylist():
        accepted[r['query_id']].add(r['siret'])
    refs.update(ledger=cfg['ledger'], evaluation=cfg['evaluation'], label_side=evaluation['label_side'],
                acceptable_operational=accepted_ref, operational_policy=side['policy'])
    rows = []
    for qid in sorted(scope):
        gt = labels[qid]
        siret, siren = gt['human_ground_truth_siret'], gt['human_ground_truth_siren']
        if not re.fullmatch(r'\d{14}', siret) or siret[:9] != siren or siren != old[qid]['siren_group']:
            raise ValueError('Invalid human identity')
        keys = {c['siret'] for c in outputs[qid]}
        row = dict(query_id=qid, fold=panel[qid]['fold'], stratum=old[qid]['stratum'],
                   calibrated_query_weight=old[qid]['calibrated_query_weight'],
                   exact=siret in keys, siren=any(c['siren'] == siren for c in outputs[qid]),
                   operational=bool(keys & accepted[qid]))
        if not math.isfinite(row['calibrated_query_weight']) or row['calibrated_query_weight'] <= 0:
            raise ValueError('Invalid sampling weight')
        if any(row[k] != old[qid][k] for k in ('exact', 'siren', 'operational')):
            raise ValueError(f'STOP: existing views disagree for query_id {qid}')
        rows.append(row)
    counts = [sum(r[k] for r in rows) for k in VIEWS[:3]]
    if counts != [507, 511, 508]:
        raise ValueError(f'STOP: existing views must be 507/511/508, observed {counts}')
    report = read(RUN/'report.json')['variants'][VARIANT]
    total_weight = sum(r['calibrated_query_weight'] for r in rows)
    for k in VIEWS[:3]:
        weighted = sum(r['calibrated_query_weight']*r[k] for r in rows)/total_weight
        if not math.isclose(weighted, report['weighted'][k]['poststratified_hajek_estimate'], abs_tol=1e-12):
            raise ValueError('STOP: weighted method differs from existing report')
    print('VERIFIED_EXISTING', counts, 'queries', len(rows), flush=True)
    guard()

    # Only emitted same-SIREN candidates and exact labels are hydrated; never a new pool.
    requested = {r['human_ground_truth_siret'] for r in labels.values()}
    for qid, cs in outputs.items():
        requested.update(c['siret'] for c in cs if c['siren'] == labels[qid]['human_ground_truth_siren'])
    receipt = read(STUDY_RECEIPT)
    refs['study_receipt'] = record(STUDY_RECEIPT)
    stock = fingerprint(launch['stock_source']['path'])
    study = fingerprint(receipt['file'])
    if stock['size_bytes'] != launch['stock_source']['size_bytes'] or study['size_bytes'] != receipt['bytes']:
        raise ValueError('Large source size mismatch')
    refs['stock'] = dict(stock, inherited_sha256=launch['stock_source']['sha256'], sha256_recomputed=False)
    refs['study'] = dict(study, inherited_sha256=receipt['sha256'], sha256_recomputed=False)
    cols = ['siret', 'numeroVoieEtablissement', 'indiceRepetitionEtablissement',
            'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement',
            'typeVoieEtablissement', 'libelleVoieEtablissement', 'codeCommuneEtablissement']
    con = duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '512MB', 'max_temp_directory_size': '0B'})
    con.register('requested', pa.table({'siret': sorted(requested)}))

    def filtered_source(path, columns):
        query = 'SELECT ' + ','.join('s.'+c for c in columns) + ' FROM read_parquet(?) s SEMI JOIN requested r USING(siret)'
        cursor = con.execute(query, [str(path)])
        result = {}
        while batch := cursor.fetchmany(4096):
            for values in batch:
                row = dict(zip(columns, values))
                if row['siret'] in result:
                    raise ValueError('Duplicate source SIRET')
                result[row['siret']] = row
            guard()
        return result

    sites = filtered_source(stock['path'], cols)
    print('FILTERED_STOCK', len(sites), 'requested', len(requested), flush=True)
    points = filtered_source(study['path'], ['siret', 'x', 'y', 'qualite_xy', 'epsg', 'distance_precision'])
    con.close()
    print('FILTERED_STUDY', len(points), flush=True)
    for ref in (stock, study):
        if fingerprint(ref['path']) != ref:
            raise ValueError('Large source changed during analysis')
    for row in rows:
        qid = row['query_id']
        cp = crm_point(geometry[qid]['geometry'])
        evidence = []
        for rank, c in enumerate(outputs[qid], 1):
            if c['siren'] != labels[qid]['human_ground_truth_siren']:
                continue
            point = candidate_point(points.get(c['siret']))
            d, t = distance(cp, point), threshold(cp, point)
            address = address_match(panel[qid], sites.get(c['siret']))
            evidence.append(dict(siret=c['siret'], emitted_rank=rank, address_match=address,
                                 candidate_point=point, distance_m=d, threshold_m=t,
                                 main_distance_match=bool(d is not None and t is not None and d <= t)))
        measured = [e for e in evidence if e['distance_m'] is not None]
        nearest = min(measured, key=lambda e: (e['distance_m'], e['emitted_rank'])) if measured else None
        addresses = [e for e in evidence if e['address_match']]
        close = [e for e in evidence if e['main_distance_match']]
        row['site_operational'] = bool(addresses or close)
        for flat in (100, 200, 500):
            row[f'site_{flat}m'] = bool(addresses or any(e['distance_m'] <= flat for e in measured))
        witness = addresses[0] if addresses else min(close, key=lambda e: (e['distance_m'], e['emitted_rank'])) if close else nearest
        row.update(validation_condition='adresse' if addresses else 'distance' if close else 'aucune',
                   crm_point=cp, witness_siret=witness['siret'] if witness else None,
                   witness_candidate_precision=witness['candidate_point']['precision'] if witness else None,
                   witness_distance_m=witness['distance_m'] if witness else None,
                   witness_threshold_m=witness['threshold_m'] if witness else None,
                   minimum_same_siren_distance_m=nearest['distance_m'] if nearest else None,
                   nearest_same_siren_siret=nearest['siret'] if nearest else None,
                   nearest_same_siren_precision=nearest['candidate_point']['precision'] if nearest else None,
                   same_siren_candidates=evidence)
        label_point = candidate_point(points.get(labels[qid]['human_ground_truth_siret']))
        ld, lt = distance(cp, label_point), threshold(cp, label_point)
        row['label_site_diagnostic'] = dict(point=label_point, distance_m=ld, threshold_m=lt,
                                            comparable=ld is not None and lt is not None,
                                            label_far_from_crm=ld > lt if ld is not None and lt is not None else None)
        guard()
    totals = summarize(rows)
    folds = {str(fold): summarize([r for r in rows if r['fold'] == fold]) for fold in (2, 3, 4)}
    diagnostics = dict(exact_misses=19, exact_misses_becoming_site_success=sum(not r['exact'] and r['site_operational'] for r in rows),
                       exact_hits=507, exact_hits_label_far_from_crm=sum(r['exact'] and r['label_site_diagnostic']['label_far_from_crm'] is True for r in rows),
                       exact_hits_label_distance_comparable=sum(r['exact'] and r['label_site_diagnostic']['comparable'] for r in rows),
                       exact_hits_not_site_success=sum(r['exact'] and not r['site_operational'] for r in rows),
                       validation_conditions=dict(Counter(r['validation_condition'] for r in rows)),
                       crm_precision_counts=dict(Counter(r['crm_point']['precision'] for r in rows)))
    diagnostics['exact_hits_label_distance_not_comparable'] = 507-diagnostics['exact_hits_label_distance_comparable']
    # Recheck hashed small authorities and source signatures before publishing.
    for ref in refs.values():
        if 'sha256' in ref:
            record(ref['path'], ref['sha256'])
    for ref in (stock, study):
        if fingerprint(ref['path']) != ref:
            raise ValueError('Large input mutated before publication')
    guard()
    result = dict(status='MEASURED_SECONDARY_CRM_SITE_ALL526_NOT_CERTIFICATION', contract_commit=commit,
                  exact_remains_primary=True, closed_sets_opened=False, sources_mutated=False,
                  full_oof=False, production_promotion_allowed=False, source_refs=refs,
                  large_source_hashes_recomputed=False, source_points_semantic_date='2026-07-01',
                  source_address_date='2026-08-01', queries=526, output_cap=max(map(len, outputs.values())),
                  totals=totals, by_fold=folds, diagnostics=diagnostics,
                  weighted_method='sum(calibrated_query_weight * hit) / sum(calibrated_query_weight)',
                  weighted_wilson_not_claimed=True, hydrated_sirets=len(requested),
                  resources=dict(peak_guarded_rss_bytes=peak, elapsed_seconds=time.monotonic()-started,
                                 duckdb_memory_limit_bytes=512*1024**2, threads=1), per_query=rows)
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    lines = ['# Mesure supplémentaire « site opérationnel » — 14 octobre 2026', '',
             f'Contrat préenregistré : `{commit}`. Run v142 `all_new_sources_max_bano`,',
             '`learned_native_ooc` ; **526/526**, folds 2/3/4, sorties scellées ≤100.',
             'Analyse secondaire, exacte toujours principale ; pas de certification OOF/nationale ni promotion.', '',
             '| Vue | Succès | % brut | Wilson 95 % brut | % pondéré Hájek |',
             '|---|---:|---:|---:|---:|']
    for view, name in zip(VIEWS, NAMES):
        m = totals[view]
        lo, hi = m['wilson95_percent']
        lines.append(f'| {name} | {m["hits"]}/526 | {m["percent"]:.2f} | [{lo:.2f} ; {hi:.2f}] | {m["weighted_hajek_percent"]:.2f} |')
    lines += ['', 'Pondération et trois vues historiques reproduites (507/511/508 et chaque booléen).',
              'Wilson seulement sur le brut, pas un IC pondéré.', '', '## Folds', '',
              '| Fold | Vue | Succès | % brut | Wilson 95 % brut | % pondéré Hájek |',
              '|---|---|---:|---:|---:|---:|']
    for fold, metrics in folds.items():
        for view, name in zip(VIEWS, NAMES):
            m = metrics[view]
            lo, hi = m['wilson95_percent']
            lines.append(f'| {fold} | {name} | {m["hits"]}/{m["queries"]} | {m["percent"]:.2f} | [{lo:.2f} ; {hi:.2f}] | {m["weighted_hajek_percent"]:.2f} |')
    lines += ['', '## Compteurs et limites', '',
              f'- Ratés exacts devenant succès site : **{diagnostics["exact_misses_becoming_site_success"]}/19**.',
              f'- Succès exacts dont le point label est loin du CRM (>T) : **{diagnostics["exact_hits_label_far_from_crm"]}/507** ; '
              f'{diagnostics["exact_hits_label_distance_comparable"]} comparables, {diagnostics["exact_hits_label_distance_not_comparable"]} non mesurables.',
              f'- Succès exacts sans succès site : **{diagnostics["exact_hits_not_site_success"]}/507** (absence de preuve ≠ site incorrect).',
              f'- Preuves principales : `{json.dumps(diagnostics["validation_conditions"], ensure_ascii=False)}`.',
              '- Coordonnées candidat : étude statistique SIRENE juillet ; adresses courantes août ; CRM BAN v17.',
              '- 11/21 = numéro identifié, pas précision actuelle certifiée ; pas de transfert de qualité vers XY v142.',
              '- Absences/qualités inconnues conservées, règle principale adresse seule si distance inadmissible.',
              '- Sensibilités plates ignorent les niveaux de précision, y compris commune, mais exigent un XY compatible.',
              '- Sources grandes : SHA hérités, taille/mtime/inode vérifiés avant/après, pas de rehash complet.',
              f'- RSS gardée maximale : {peak/1024**2:.1f} Mio ; durée : {result["resources"]["elapsed_seconds"]:.2f} s.',
              '', 'Commande : `.venv/bin/python scripts/retrieval_site_operational_metric.py`.',
              'JSON voisin : totaux, folds, sources et **526 query_id uniques**, preuves sans noms/adresses CRM bruts.']
    RESULT.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(totals=totals, diagnostics=diagnostics, resources=result['resources']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
