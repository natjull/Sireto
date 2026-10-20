"""Read-only CRM inventory and bounded real G0 cost replay (no training/labels).

Only docs/production_cost_study.json is written. Run phases separately to
release national-index mappings. Existing BASE vectors are matched by text,
not by SIRET alone. No closed evaluation labels are opened.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
H = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
R = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100')
SOURCE = H/'step2_option_d_20261017/full'
RISK = H/'step2_risk_model_v1_20261020_c/full'
FT = H/'step2_gemma_finetune_10h_20261019/full'
OUTPUT = ROOT/'docs/production_cost_study.json'
CRM = R/'datasets/crm_new_data_20260817/source/plateforme_new_data_to_cure.csv'


def read(path):
    return json.loads(Path(path).read_text())


def save(key, value):
    data = read(OUTPUT) if OUTPUT.exists() else {}
    data[key] = value
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def ref(path):
    path = Path(path)
    return dict(path=str(path), size_bytes=path.stat().st_size,
                sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest())


def verify(value):
    actual = ref(value['path'])
    if actual['sha256'] != value['sha256']:
        raise ValueError('SHA changed: '+value['path'])
    return Path(value['path'])


def connection():
    import duckdb
    return duckdb.connect(':memory:', config={'threads':'1', 'memory_limit':'512MB', 'max_temp_directory_size':'0B'})


class Guard:
    def __init__(self):
        import psutil
        self.proc = psutil.Process()
        self.peak = 0
        self.started = time.monotonic()
        self.done = threading.Event()
        threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        while not self.done.wait(.05):
            rss = self.proc.memory_info().rss
            self.peak = max(self.peak, rss)
            if rss > 3.85*1024**3 or time.monotonic()-self.started > 3500:
                print('STOP own resource cap', rss, flush=True)
                os._exit(75)


def inventory():
    import numpy as np
    con = connection()
    # Source order_siret is an existing CRM field, NOT a ground-truth table.
    # Return only presence aggregates; no SIRET value is materialized in Python.
    sql = """SELECT count(*) AS n_rows, count(DISTINCT stc_service_id) services,
        count(*) FILTER(WHERE trim(coalesce(order_siret,''))<>'') siret_present,
        count(*) FILTER(WHERE regexp_full_match(trim(coalesce(order_siret,'')), '[0-9]{14}')) siret_14digits,
        count(*) FILTER(WHERE trim(coalesce(stc_insee_commune,''))<>'') insee_present,
        count(*) FILTER(WHERE regexp_full_match(trim(coalesce(stc_insee_commune,'')), '[0-9ABab]{5}')) insee_5chars
        FROM read_csv(?,delim=';',all_varchar=true)"""
    columns = ['rows','distinct_services','siret_present','siret_14digits','insee_present','insee_5chars']
    counts = dict(zip(columns, con.execute(sql,[str(CRM)]).fetchone()))
    cities = dict(con.execute("SELECT trim(stc_insee_commune),count(*) FROM read_csv(?,delim=';',all_varchar=true) WHERE trim(coalesce(stc_insee_commune,''))<>'' GROUP BY 1", [str(CRM)]).fetchall())
    counts['distinct_raw_query_inputs'] = con.execute("SELECT count(DISTINCT (client_final,stc_street_number,stc_street_name,stc_zipcode,stc_city,stc_insee_commune)) FROM read_csv(?,delim=';',all_varchar=true)",[str(CRM)]).fetchone()[0]
    counts.update(siret_absent=counts['rows']-counts['siret_present'], insee_absent=counts['rows']-counts['insee_present'], distinct_communes=len(cities))
    stock = read(SOURCE/'plan.json')['official']['stock']['path']
    stocks = con.execute("SELECT codeCommuneEtablissement,etatAdministratifEtablissement,statutDiffusionEtablissement,count(*) FROM read_parquet(?) GROUP BY 1,2,3",[stock]).fetchall()
    national = Counter(); bycity = Counter()
    for city, state, public, n in stocks:
        national['total'] += n
        national['active' if state=='A' else 'closed' if state=='F' else 'other_state'] += n
        if public=='O':
            national['public'] += n
            national['public_active' if state=='A' else 'public_closed' if state=='F' else 'public_other'] += n
        bycity[city] += n
    sizes = np.array([bycity.get(c,0) for c in cities])
    boundaries = [0,1000,10000,50000,100000,1000000,float('inf')]
    distribution = []
    for a,z in zip(boundaries[:-1],boundaries[1:]):
        included = [c for c in cities if a<=bycity.get(c,0)<z]
        distribution.append(dict(min_sites=a,max_sites=None if not math.isfinite(z) else z,communes=len(included),crm_rows=sum(cities[c] for c in included)))
    closed = {}
    # Projection/count only; never read labels or any label-side ledger.
    for name in ('13978e04af810d8b','35e5038d1e7e492c','05e73a179a6bbbe0'):
        p = R/'datasets/crm_gt_v3_retrieval_input_all_human'/name/'manifest.json'
        m = read(p)
        closed[str(p)] = {k:m[k] for k in ('query_count','split_counts')}
    ledger = R/'datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet'
    import pyarrow.parquet as pq
    schema = pq.read_schema(ledger).names
    foldcol = 'fold' if 'fold' in schema else 'oof_fold'
    closed['population_fold_counts_only'] = dict(con.execute('SELECT '+foldcol+',count(*) FROM read_parquet(?) GROUP BY 1',[str(ledger)]).fetchall())
    population = R/'datasets/crm_gt_v2_commercial_certified_population/73328b7611f37a06/queries.parquet'
    closed['population_all_folds_counts_only'] = dict(con.execute('SELECT oof_fold,count(*) FROM read_parquet(?) GROUP BY 1',[str(population)]).fetchall())
    control = H/'v130/generalization_plan_20261013_night/report.json'
    documented = read(control)['populations']
    closed['test4323_documented_count'] = documented['reserved_embedding_test']['queries']
    closed['control962_documented_count'] = documented['independent_new_pipeline_control_previously_planned']['queries']
    closed['test_control_count_authority'] = str(control)+' (metadata only, not independently recounted)'
    con.close()
    save('inventory',dict(source=ref(CRM), measured_at=datetime.now(timezone.utc).isoformat(), counts=counts,
        crm_rows_per_commune_quantiles=dict(zip(['min','p50','p90','p95','p99','max'],np.quantile(list(cities.values()),[0,.5,.9,.95,.99,1]).tolist())),
        official_sites_per_crm_commune_quantiles=dict(zip(['min','p50','p90','p95','p99','max'],np.quantile(sizes,[0,.5,.9,.95,.99,1]).tolist())),
        commune_distribution_by_official_sites=distribution, national_stock=dict(national),
        crm_communes_sites=sum(bycity.get(c,0) for c in cities), unmatched_communes=sum(c not in bycity for c in cities),
        pairs_crm_times_commune_sites=sum(n*bycity.get(c,0) for c,n in cities.items()), closed_counts_only=closed,
        scope='Latest available raw CRM export; service duplicates retained. No assertion that this is the current live CRM; historical training exports not concatenated.'))
    print('INVENTORY',counts,dict(national),flush=True)


def bench():
    import numpy as np
    import pyarrow as pa
    from scripts import step2_option_d as g
    from scripts import step2_colocated_signals as s
    from scripts import step2_risk_model_v1 as risk
    from src.xgb_matcher.hierarchical_retrieval import normalize_text
    import xgboost as xgb
    start = time.monotonic(); stages = {}; sources = []
    queries = read(SOURCE/'training_queries.json')
    # Uniform random records, chosen before any timing or prediction, no labels.
    assert len(queries)==17427 and all(q['fold'] in (2,3,4) for q in queries)
    origin = np.sort(np.random.default_rng(20261021).choice(len(queries),200,replace=False))
    panel = [queries[i] for i in origin]; ids = [q['query_id'] for q in panel]
    positions = {q['query_id']:i for i,q in enumerate(queries)}
    geom = read(SOURCE/'geometry.json'); plan = read(SOURCE/'plan.json'); official = plan['official']
    began = time.monotonic(); engine = g.G0(plan); stages['generation_init'] = time.monotonic()-began
    rows = []; timings = []
    for q in panel:
        began = time.monotonic(); candidates,meta = engine.retrieve(q,geom[q['query_id']]['geometry'])
        timings.append(time.monotonic()-began); rows.append(dict(query_id=q['query_id'],candidates=candidates))
    stages['generation'] = sum(timings)
    del engine; gc.collect()
    outputs = {r['query_id']:r['candidates'] for r in rows}; pmap = {q['query_id']:q for q in panel}
    keys = np.array([[c['siret'] for c in r['candidates']] for r in rows],dtype='S14')
    stored = np.load(SOURCE/'candidate_keys.npy',mmap_mode='r') if (SOURCE/'candidate_keys.npy').exists() else np.load(read(SOURCE/'matrix.json')['keys']['path'],mmap_mode='r')
    if not np.array_equal(keys,stored[origin]): raise ValueError('Regenerated candidate order changed')
    began = time.monotonic(); sites,units,points = g.b.hydrate(official,outputs)
    con = connection(); con.register('requested',pa.table({'siret':sorted(sites)}))
    naf_rows = con.execute('SELECT siret,activitePrincipaleEtablissement,nomenclatureActivitePrincipaleEtablissement FROM read_parquet(?) s SEMI JOIN requested r USING(siret)',[official['stock']['path']]).fetch_arrow_table().to_pylist()
    for row in naf_rows:
        sites[row['siret']].update(row)
    closures = {r['siret']:r['closure_date'] for r in con.execute("""SELECT siret,MAX(try_cast(dateDebut AS DATE)) closure_date FROM read_parquet(?) h SEMI JOIN requested r USING(siret) WHERE etatAdministratifEtablissement='F' AND try_cast(changementEtatAdministratifEtablissement AS BOOLEAN)=true AND try_cast(dateDebut AS DATE)<=DATE '2026-08-01' GROUP BY siret""",[official['history']['path']]).fetch_arrow_table().to_pylist()}
    pairs = sorted({(c['siren'],pmap[q]['crm_insee']) for q,cs in outputs.items() for c in cs if pmap[q]['crm_insee']})
    con.register('requested_pairs',pa.table({'siren':[p[0] for p in pairs],'commune':[p[1] for p in pairs]}))
    counts = {(r['siren'],r['commune']):r['active_count'] for r in con.execute("""SELECT s.siren,s.codeCommuneEtablissement commune,count(*) active_count FROM read_parquet(?) s JOIN requested_pairs r ON s.siren=r.siren AND s.codeCommuneEtablissement=r.commune WHERE s.etatAdministratifEtablissement='A' AND s.statutDiffusionEtablissement='O' GROUP BY s.siren,s.codeCommuneEtablissement""",[official['stock']['path']]).fetch_arrow_table().to_pylist()}
    old = g.b.feature_cube(pmap,outputs,geom,sites,units,points)
    new = g.v2.feature_cube(old,pmap,outputs,geom,sites,points,closures,counts)
    X = np.concatenate([old,new],axis=2)[:,:,g.CONTENT_COLS]
    stages['content85_hydration_features'] = time.monotonic()-began
    expected = np.load(SOURCE/'features.npy',mmap_mode='r')[origin]
    if not np.array_equal(X,expected,equal_nan=True):raise ValueError('85 features replay differs')
    # Fresh candidate-only metadata, identical official S1/S2 text construction.
    began = time.monotonic(); labels,nafrefs = s.nomenclature(); classes=[sorted({k[:2] for k in labels}),sorted({k[:3] for k in labels})]
    maps=[{v:i for i,v in enumerate(cs)} for cs in classes]
    codes=np.full((200,100,2),-1,dtype=np.int16); addresses=np.full((200,100),-1,dtype=np.int32)
    textkeys=[]; addrmap={}; neighbours=np.zeros((200,100,3),dtype=np.int16)
    for i,q in enumerate(panel):
        ct=[]; xy=np.full((100,2),np.nan)
        for j,c in enumerate(outputs[q['query_id']]):
            site=sites.get(c['siret'],{}); unit=units.get(c['siren'],{}); public=site.get('statutDiffusionEtablissement')=='O'
            code=g.b.text(site.get('activitePrincipaleEtablissement')).replace('.','').upper()
            if not (public and site.get('nomenclatureActivitePrincipaleEtablissement')=='NAFRev2' and code in labels):code=''
            if code:codes[i,j]=[maps[0][code[:2]],maps[1][code[:3]]]
            addr=s.address_key(site)
            if addr is not None:addresses[i,j]=addrmap.setdefault(addr,len(addrmap))
            names=[g.b.text(site.get(k)) for k in ('enseigne1Etablissement','enseigne2Etablissement','enseigne3Etablissement','denominationUsuelleEtablissement')] if public else []
            if unit.get('statutDiffusionUniteLegale')=='O':names=[g.b.text(unit.get(k)) for k in ('denominationUniteLegale','denominationUsuelle1UniteLegale','sigleUniteLegale','nomUniteLegale','prenomUsuelUniteLegale')]+names
            names=list(dict.fromkeys(n.strip() for n in names if n.strip()));value='; '.join(names+([labels[code]] if code else []));ct.append(s.DOC_PROMPT+value if value else None)
            point=g.b.candidate_point(points.get(c['siret'],{}))
            if public and point['precision'] in ('housenumber_equivalent','street_imprecise'):xy[j]=point['point_l93']
        same=(addresses[i,:,None]==addresses[i,None,:])&(addresses[i,:,None]>=0);near=np.square(xy[:,None]-xy[None,:]).sum(2)<=2500
        neighbours[i]=np.column_stack([same.sum(1),near.sum(1),(same|near).sum(1)])
        textkeys.append(ct)
    cities=sorted({q['crm_insee'] for q in panel if q['crm_insee']});con.register('communes',pa.table({'codeCommuneEtablissement':cities}))
    density_counts=dict(con.execute("SELECT s.codeCommuneEtablissement,count(*) FROM read_parquet(?) s SEMI JOIN communes r USING(codeCommuneEtablissement) WHERE s.statutDiffusionEtablissement='O' GROUP BY 1",[official['stock']['path']]).fetchall());con.close()
    density=np.array([math.log1p(density_counts.get(q['crm_insee'],0)) if q['crm_insee'] else np.nan for q in panel],dtype=np.float32)
    stages['candidate_metadata_neighbours_density'] = time.monotonic()-began
    # Hydrate() projects STOCK_COLUMNS, which must include NAF for the above.
    metadata=Path(read(FT/'data.json')['metadata'])
    if not np.array_equal(codes,np.load(metadata/'naf.npy',mmap_mode='r')[origin]):raise ValueError('NAF metadata replay')
    began=time.monotonic();base=read(FT/'base_encode/receipt.json');binding=read(verify(base['binding']));assert binding['adapter'] is None
    texts=read(verify(binding['texts']));lookup={t:i for i,t in enumerate(texts)}
    qi=np.array([lookup.get(s.QUERY_PROMPT+g.b.text(q['crm_name']).strip(),-1) for q in panel]);ci=np.array([[lookup.get(t,-1) for t in row] for row in textkeys])
    missing=[t for row in textkeys for t in row if t is not None and t not in lookup]
    if missing:raise ValueError('New Gemma texts need separate authorized worker: '+str(len(set(missing))))
    used=np.unique(np.r_[qi,ci.ravel()]);used=used[used>=0];shards={int(i//base['shard_size']) for i in used};arrays={}
    for k in shards:arrays[k]=np.load(verify(base['shards'][k]),mmap_mode='r')
    def vectors(indices):return np.array([arrays[int(k//base['shard_size'])][int(k%base['shard_size'])] for k in indices])
    cosine=np.full((200,100),np.nan,dtype=np.float32)
    for i in range(200):
        ix=np.flatnonzero(ci[i]>=0)
        if qi[i]>=0:cosine[i,ix]=vectors(ci[i,ix])@vectors([qi[i]])[0]
    semantic=np.stack([cosine,s.relative(cosine,addresses)],axis=2)
    if not np.array_equal(semantic,np.load(RISK/'semantic.npy',mmap_mode='r')[origin],equal_nan=True):raise ValueError('BASE semantic replay')
    stages['s2_cache_verify_read_cosines']=time.monotonic()-began
    del texts,lookup,arrays;gc.collect()
    partition=np.array(read(RISK/'data.json')['partition'])[origin];nb=np.empty((200,100,7),dtype=np.float32);pred=np.empty((200,100),dtype=np.float32);riskprob=np.empty(200,dtype=np.float32);stages.update(s1_load=0.,s1_inference=0.,ranker_load=0.,ranker_inference=0.,risk_load=0.,risk_scene_inference=0.)
    for cv in range(5):
        loc=np.flatnonzero(partition==cv);m=read(RISK/f'rank{cv}_outer/attempt0/models.json')[0];u=m['uses'][0];assert set(origin[loc]).issubset(u['scored_indices']) and not set(origin[loc])&set(m['train_indices'])
        began=time.monotonic()
        with gzip.open(verify(u['nb_scoring']['model']),'rb') as stream:vectorizer,models=pickle.load(stream)
        stages['s1_load']+=time.monotonic()-began;began=time.monotonic();values=[]
        for k,model in enumerate(models):
            probs=model.predict_log_proba(vectorizer.transform([g.b.normalize_name(g.b.text(panel[i]['crm_name'])) for i in loc]));cc=codes[loc,:,k];v=np.full(cc.shape,np.nan,dtype=np.float32);a,z=np.where(cc>=0);v[a,z]=probs[a,cc[a,z]];values.append(v)
        rd,dd=s.relative(values[0],addresses[loc],ranks=True);rg,dg=s.relative(values[1],addresses[loc],ranks=True)
        nb[loc]=np.stack([values[0],values[1],rd,rg,dd,dg,(codes[loc,:,0]<0).astype(np.float32)],axis=2);stages['s1_inference']+=time.monotonic()-began
        began=time.monotonic();ranker=xgb.Booster();ranker.load_model(verify(m['model']));stages['ranker_load']+=time.monotonic()-began
        began=time.monotonic();xx=np.concatenate([X[loc],nb[loc],semantic[loc]],axis=2);pred[loc]=ranker.predict(xgb.DMatrix(xx.reshape(-1,94))).reshape(-1,100);stages['ranker_inference']+=time.monotonic()-began
        rp=read(RISK/f'rank{cv}_outer/receipt.json')['predictions'];stored_pred=np.load(verify(rp));imap={v:i for i,v in enumerate(stored_pred['indices'])};expected=stored_pred['predictions'][[imap[v] for v in origin[loc]]]
        if not np.array_equal(pred[loc],expected):raise ValueError('Rank prediction replay')
        began=time.monotonic();rr=read(RISK/f'risk{cv}/receipt.json');model=xgb.Booster();model.load_model(verify(rr['model']));stages['risk_load']+=time.monotonic()-began
        began=time.monotonic();order=np.argsort(-pred[loc],axis=1,kind='stable');a,z,t=order[:,:3].T;i=np.arange(len(loc));f={n:j for j,n in enumerate(g.CONTENT)};top=X[loc][i,a];signals=np.concatenate([nb[loc],semantic[loc]],axis=2);p=s.d.softmax(np.take_along_axis(pred[loc],order[:,:10],axis=1))
        vals=[pred[loc][i,a],pred[loc][i,a]-pred[loc][i,z],pred[loc][i,a]-pred[loc][i,t],-(p*np.log(np.maximum(p,1e-30))).sum(1),a+1,top[:,f['candidate_same_siren_count']]]
        vals.extend(neighbours[loc,a,k] for k in range(3));vals.extend([top[:,f['geo_log_distance']],top[:,f['geo_missing']]])
        for k in range(9):vals.extend([signals[i,a,k],signals[i,z,k],signals[i,a,k]-signals[i,z,k]])
        vals.extend(top[:,f[n]] for n in ('name_jaro_max','number_exact','street_jaro','postcode_match','v2_is_active','v2_is_closed','v2_is_siege'));vals.append(density[loc]);rx=np.column_stack(vals).astype(np.float32)
        riskprob[loc]=model.predict(xgb.DMatrix(rx,feature_names=risk.FEATURES));stages['risk_scene_inference']+=time.monotonic()-began
        if not np.array_equal(rx,stored_pred['risk'][[imap[v] for v in origin[loc]]],equal_nan=True):raise ValueError('Risk feature replay')
        decisions=np.load(verify(rr['decisions']));assert np.array_equal(riskprob[loc],decisions['risk_score'][[imap[v] for v in origin[loc]]])
    save('bench',dict(status='COMPLETE_HOT_CACHE_NOT_COLD_END_TO_END',seed=20261021,queries=200,query_ids=ids,origin=origin.tolist(),population_folds=Counter(q['fold'] for q in panel),
        stages_seconds=stages,total_wall_seconds=time.monotonic()-start,generation_quantiles_seconds=dict(zip(['p50','p95','p99'],np.quantile(timings,[.5,.95,.99]).tolist())),
        unique_candidate_sirets=len(np.unique(keys)),candidate_exposures=20000,unique_base_keys=len(used),base_shards_verified=len(shards),new_gemma_texts=0,
        exact_replay=dict(candidate_keys=20000,content_features=1700000,ranking_vectors=200,risk_vectors=200),
        limitations=['Geometry is already cached: fresh BAN/CRM geocoding not measured.','All BASE texts already cached: cold Gemma runtime estimated from existing measurements, not re-encoded.','Five company-OOF models replayed: no deployable full-fit/validated production policy.'],
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('BENCH',stages,flush=True)


def disk():
    # Disk use: allocated bytes, not apparent size; do not follow symlinks.
    roots=sorted(Path('/Volumes/CATNAT_DATA').glob('SIRETO_*'))
    def sizes(paths):
        lines=subprocess.check_output(['du','-sk']+[str(p) for p in paths],text=True).splitlines()
        return {line.split('\t',1)[1]:int(line.split('\t',1)[0])*1024 for line in lines}
    top=sizes(roots);children=[]
    for name in ('SIRETO_RECALL100','SIRETO_EMBEDDINGGEMMA2'):
        children += [p for p in (Path('/Volumes/CATNAT_DATA')/name).iterdir() if p.is_dir() and not p.is_symlink() and '.pending.' not in p.name]
    sub=sizes(children);recent=subprocess.check_output(['git','log','-100','--format=%H','--name-only','--','docs','scripts','handover.md'],cwd=ROOT,text=True)
    recent_paths=set(line for line in recent.splitlines() if '/' in line or line=='handover.md')
    frozen = (ROOT/'docs/retrieval_frozen_v142.json').read_text()
    records=[]
    for path,n in sorted(sub.items(),key=lambda x:-x[1])[:10]:
        p=Path(path);token=p.name;hits=[]
        for doc in sorted((ROOT/'docs').glob('*.md')):
            if '.pending.' in doc.name:continue
            content=doc.read_text(errors='replace')
            import re
            pattern=re.escape(token)+(r'(?![0-9])' if token.startswith('v') and token[1:].isdigit() else '')
            if re.search(pattern,content):hits.append(str(doc.relative_to(ROOT)))
        records.append(dict(path=path,allocated_bytes=n,directory_mtime=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),
            docs_referencing=hits[:25],recent100_commit_docs=[d for d in hits if d in recent_paths],
            frozen_v142_manifest_path_reference=(path+'/' in frozen),
            retention='Reference found is NOT deletion clearance; descendants and SHA manifests may remain dependencies.'))
    save('disk',dict(measured_at=datetime.now(timezone.utc).isoformat(),free_bytes=os.statvfs('/Volumes/CATNAT_DATA').f_bavail*os.statvfs('/Volumes/CATNAT_DATA').f_frsize,
        roots=[dict(path=p,allocated_bytes=n) for p,n in sorted(top.items(),key=lambda x:-x[1])],top10_nonoverlapping_children=records,
        date_definition='Directory mtime, not recursive last write; docs references are literal folder-name matches, not exhaustive dependency reachability.'))
    print('DISK',top,flush=True)


def finalize():
    data=read(OUTPUT);inv=data['inventory'];b=data['bench'];n=inv['counts']['rows'];sites=inv['national_stock']['total']
    # Count of legal-unit rows from Parquet footer only (not all are necessarily
    # needed, and official PARENT text deduplication is not measured nationally).
    import pyarrow.parquet as pq
    parents=pq.ParquetFile(read(SOURCE/'plan.json')['official']['legal_units']['path']).metadata.num_rows
    index_manifest=read(R/'indexes/hierarchical_retrieval_v3/4b332899d8698354/manifest.json')
    parent_views=read(H/'v18/identity_site_20261010_a/identity_counts.json')
    assert index_manifest['num_establishment_documents']==sites
    frozen=(ROOT/'docs/retrieval_frozen_v142.json').read_text()
    for row in data['disk']['roots']:
        row['directory_mtime']=datetime.fromtimestamp(Path(row['path']).stat().st_mtime,timezone.utc).isoformat()
    for row in data['disk']['top10_nonoverlapping_children']:
        row['frozen_v142_manifest_path_reference']=row['path']+'/' in frozen
    stages=b['stages_seconds'];warm=sum(stages.values())/200
    fast=6154.344204917084/958595;slow=.022821
    exposures=100*n
    scenarios=[]
    for reuse in (0.,.5,.8):
        u=math.ceil(exposures*(1-reuse))
        scenarios.append(dict(reused_fraction_of_candidate_exposures=reuse,new_unique_candidate_texts_assumed=u,
            new_candidate_encoding_hours=[u*fast/3600,u*slow/3600],
            total_hours_with_warm_and_query_encoding=[n*warm/3600+(u+n)*fast/3600,n*warm/3600+(u+n)*slow/3600],
            new_candidate_vectors_float32_gib=u*768*4/2**30,new_candidate_vectors_float16_gib=u*768*2/2**30))
    v=dict(national_sites=sites,active=inv['national_stock']['active'],closed=inv['national_stock']['closed'],
        public_sites=inv['national_stock']['public'],parent_one_vector_per_legal_unit_scenario_rows=parents,
        index_establishment_documents=index_manifest['num_establishment_documents'],index_parent_documents=index_manifest['num_siren_documents'],
        historical_parent_view_counts=parent_views,parent_views_national_is_unknown=True,
        site_float32_gib=sites*768*4/2**30,site_float16_gib=sites*768*2/2**30,
        parent_float32_capacity_gib=parents*768*4/2**30,parent_float16_capacity_gib=parents*768*2/2**30,
        site_encoding_hours=[sites*.020/3600,sites*.025/3600],parent_preparation_proxy_hours=[sites*.009/3600,sites*.012/3600],
        native_generation_hours=[n*17.067/3600,n*19.34101177356206/3600],
        commune_scoring_proxy_hours=inv['pairs_crm_times_commune_sites']*(23.732/319488)/3600,
        crm_communes_sites=inv['crm_communes_sites'],crm_communes_site_encoding_hours=[inv['crm_communes_sites']*.020/3600,inv['crm_communes_sites']*.025/3600])
    v['bank_preparation_proxy_hours']=[v['site_encoding_hours'][i]+v['parent_preparation_proxy_hours'][i] for i in (0,1)]
    v['bank_amortized_seconds_per_crm_row']=[x*3600/n for x in v['bank_preparation_proxy_hours']]
    projections=dict(g0=dict(warm_seconds_per_row=warm,warm_hours_all_rows=n*warm/3600,
        warm_stage_hours={k:t*n/200/3600 for k,t in stages.items()},
        candidate_text_scenarios=scenarios,crm_name_encoding_hours=[n*fast/3600,n*slow/3600],
        base_seconds_per_new_text_measured_reference=fast,slow_site_reference_seconds_per_text=slow,
        features94_all_rows_float32_gib=n*100*94*4/2**30,
        note='Scenarios, not confidence intervals or bounds on actual runtime. Unique new text count/reuse on full CRM unknown; existing caches are not subtracted without measuring overlap.'),v142=v)
    data['projections']=projections
    data['verification']=dict(status='COMPLETE_COST_STUDY_NOT_PRODUCTION',script=ref(__file__),
        no_training=True,no_new_gemma_encoding=True,closed_labels_opened=False,
        calculation_budget_seconds=3600,memory_cap_gib=3.85,handover_initial_line_concurrent_commit='be26eab2',
        limitations=['The exported raw order_siret was counted only by nonempty/14-digit predicates: no value returned or used as a label.',
                    'One aborted preliminary bench (missing NAF projection) and one bounded 512MB DISTINCT SIREN diagnostic failed before publication; no partial result used.',
                    'National unique parents needed was not measured (bounded DISTINCT diagnostic out of memory); legal-unit footer count is capacity only.',
                    'Times are local wall-clock, not exclusive CPU or actual GPU busy time; concurrent agent/I/O can affect them.',
                    'No live CRM connector: export scope/refresh and duplicate service semantics need business confirmation.'])
    OUTPUT.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fmt=lambda value:f'{value:,.2f}'.replace(',',' ')
    c=inv['counts'];d=data['disk'];lines=[
        '# Resiretisation de la base CRM — étude de coût',
        '',f'Étude du {datetime.now(timezone.utc).isoformat()}. **Mesures locales, estimations et inconnues séparées ; aucune promotion production.**',
        '', '## 1. Périmètre et inventaire mesuré',
        f"Export : `{CRM}` (SHA dans le JSON). **{c['rows']} lignes / {c['distinct_services']} services distincts / {c['distinct_raw_query_inputs']} entrées CRM brutes distinctes**, {c['distinct_communes']} communes renseignées.",
        f"SIRET existant présent/absent : **{c['siret_present']} / {c['siret_absent']}** ; présents tous de 14 chiffres (ni validité Luhn ni rattachement vérifiés). INSEE présent/absent : **{c['insee_present']} / {c['insee_absent']}** ; format cinq caractères, pas preuve de validité : {inv['unmatched_communes']} codes absents du snapshot officiel.",
        'Les 26 398 répétitions de service ne sont pas supprimées : plusieurs lignes peuvent porter des entrées différentes. Les exports historiques d’apprentissage ne sont pas additionnés à cet export, sous peine de doubles comptes. Une déduplication des seules entrées strictement identiques serait possible après validation métier ; projections conservatrices sur toutes les lignes.',
        'Le champ source `order_siret` est compté par présence uniquement, jamais utilisé comme vérité. Aucun label fermé ouvert. Comptage par projection `oof_fold` seulement : fold0 **7009**, fold1 **6622** ; métadonnées existantes seulement : test4323 **4323**, control962 **962**, non recomptés individuellement. Ce ne sont pas quatre populations disjointes à additionner. Ledger développement : 7767/7967/7837 lignes folds2/3/4.',
        '', '| Taille de commune : établissements officiels tous états | Communes | Lignes CRM |','|---|---:|---:|']
    for row in inv['commune_distribution_by_official_sites']:
        if row['communes']:lines.append(f"| {row['min_sites']}–<{row['max_sites']} | {row['communes']} | {row['crm_rows']} |")
    lines += [f"Fiches par commune : médiane2, p95=45, maximum1616. Sites officiels par commune CRM : médiane777, p95={fmt(inv['official_sites_per_crm_commune_quantiles']['p95'])}, maximum503476. Union : **{inv['crm_communes_sites']} sites** ; **{inv['pairs_crm_times_commune_sites']} couples dossier×site** (hors163 lignes sans commune).",'',
        '## 2. Rejeu G0 réel sur 200 dossiers',
        f"Tirage uniforme sans remise seed20261021 dans les17427 requêtes des folds2/3/4, avant scores ; identifiants/indices et plis dans le JSON. **20000 candidats régénérés** ; {b['unique_candidate_sirets']} SIRET uniques ; candidats ordonnés,85 features,200 vecteurs ranking et200 risques identiques aux artefacts scellés. Aucun fit ni évaluation de précision nouvelle.",
        '| Étage mesuré | Secondes / 200 | Secondes / fiche | Projection131749 (h) |','|---|---:|---:|---:|']
    for k,t in stages.items():lines.append(f'| {k} | {fmt(t)} | {t/200:.6f} | {fmt(t*n/200/3600)} |')
    lines += [f"Somme des étages : **{warm:.4f}s/fiche**, soit **{fmt(n*warm/3600)}h** extrapolées. Mur du rejeu hors imports : {fmt(b['total_wall_seconds'])}s ; pic RSS **{b['peak_rss_bytes']/2**30:.3f}Gio**, garde3,85Gio. Génération p50/p95/p99 : {b['generation_quantiles_seconds']}. Démarrages/chargements des cinq modèles OOF compris ; pas cinq inférences par fiche.",
        f"**Chemin avec caches, pas temps froid complet** : géométrie CRM déjà disponible ; {b['unique_base_keys']} clés BASE exactes couvertes, {b['base_shards_verified']} shards SHA vérifiés, **0 nouveau texte Gemma**. La phase S2 chronométrée comprend lecture/vérification/cosines, pas un encodage fictif. Les textes candidats officiels nom+NAF ont été reconstruits ; correspondance par texte/encodeur, pas seul SIRET. S1 utilise le nom normalisé et ses modèles NB sérialisés, sans entraînement.",
        'Les mêmes candidats SIRET répétés partagent un vecteur seulement si le texte, checkpoint, prompt, troncature et binding de batching sont identiques ; le vecteur requête reste propre au nom. Les cosines et classements restent par dossier. Vérifier les SHA des shards une fois par banque et utiliser des lookups nationaux persistants éviterait les scans répétés par lot ; gain non mesuré/non implémenté.',
        '', '### G0 à froid : estimations conditionnelles',
        f"Débit BASE existant : {fast:.6f}s/nouveau texte (958595 nouveaux en6154,344s, `step2_gemma_finetune_10h_20261019/full/base_encode/receipt.json`). Référence SITE plus lente :0,022821s/texte (`docs/step2_error_families_g0.md`). Pas le même mélange de longueurs ; intervalle de scénarios, pas une garantie. Requêtes CRM à encoder : environ {fmt(n*fast/3600)}–{fmt(n*slow/3600)}h supplémentaires si aucune réutilisation.",
        '| Réutilisation/dédup hypothétique des13 174 900 expositions | Nouveaux textes candidats | Encodage (h) | Vecteurs f32/f16 (Gio) |','|---|---:|---:|---:|']
    for row in scenarios:
        lines.append(f"| {row['reused_fraction_of_candidate_exposures']*100:.0f}% | {row['new_unique_candidate_texts_assumed']} | {fmt(row['new_candidate_encoding_hours'][0])}–{fmt(row['new_candidate_encoding_hours'][1])} | {fmt(row['new_candidate_vectors_float32_gib'])}/{fmt(row['new_candidate_vectors_float16_gib'])} |")
    lines += [f"Ajouter le chemin chaud ({fmt(n*warm/3600)}h), l’encodage des noms CRM et les nouveaux textes candidats : **pas de total unique honnête sans mesurer l’union des candidats sur toute la base**. Le scénario sans partage reste une capacité maximale de100 textes/fiche, pas une mesure de l’union. Banque BASE existante1266170 textes≈3,62Gio, à réutiliser si compatible ; aucun gain national de recouvrement supposé. Features94 persistées sur toutes les lignes≈{fmt(projections['g0']['features94_all_rows_float32_gib'])}Gio ; préférer lots bornés et export final seulement.",
        '', '## 3. V142 : estimation sans nouvel encodage',
        f"Snapshot officiel : **{sites} sites = {v['active']} actifs + {v['closed']} fermés**, dont {v['public_sites']} publics. Les82 772 691 documents Tantivy ne sont PAS autant de sites. Aucun filtre de fermeture validé pour retirer des sites : tous les fermés conservés, ni label ni règle de cas. Pertinence temporelle des fermés inconnue ; la restreindre changerait la recette gelée.",
        f"Banque SITE nationale768 : **{fmt(v['site_float32_gib'])}Gio float32**, **{fmt(v['site_float16_gib'])}Gio float16**, hors clés/textes/index/temp. Déduplication des textes peut réduire, réduction nationale non mesurée. PARENT : scénario d’un vecteur par chacune des{parents} unités légales (footer parquet), soit **{fmt(v['parent_float32_capacity_gib'])}/{fmt(v['parent_float16_capacity_gib'])}Gio** supplémentaires ; ce n’est PAS une borne : un parent peut avoir plusieurs vues et des textes identiques peuvent partager un vecteur. Le pilote officiel compte{parent_views['parents']} parents/{parent_views['views']} vues/{parent_views['unique_texts']} textes uniques ; ratio non extrapolé nationalement. Les{index_manifest['num_siren_documents']} documents parent Tantivy ne sont pas un décompte de vecteurs. SITE+PARENT à un vecteur/unité≈{fmt(v['site_float32_gib']+v['parent_float32_capacity_gib'])}Gio f32. Besoin réel PARENT inconnu. Les non-publics ne permettent aucune reconstruction de nom masqué.",
        f"À0,020–0,025s/site mesurées auparavant : **{fmt(v['site_encoding_hours'][0])}–{fmt(v['site_encoding_hours'][1])}h SITE**. Proxy PARENT0,009–0,012s/site : **{fmt(v['parent_preparation_proxy_hours'][0])}–{fmt(v['parent_preparation_proxy_hours'][1])}h**, soit **{fmt(v['bank_preparation_proxy_hours'][0])}–{fmt(v['bank_preparation_proxy_hours'][1])}h banques**. Ce sont des heures équivalentes de préparation incluantCPU/packing/premier score, pas des heures GPUbusy mesurées nationalement. Temps partageable une fois par binding ; ne pas le multiplier par les fiches.",
        f"Banques amorties : **{fmt(v['bank_amortized_seconds_per_crm_row'][0])}–{fmt(v['bank_amortized_seconds_per_crm_row'][1])}s/fiche** sur cet export. Génération native seule après préparation, proxy mesuré17,067–19,341s/fiche : **{fmt(v['native_generation_hours'][0])}–{fmt(v['native_generation_hours'][1])}h** de plus. Les banques+native seules totalisent≈{fmt(sum([v['bank_preparation_proxy_hours'][0],v['native_generation_hours'][0]]))}–{fmt(sum([v['bank_preparation_proxy_hours'][1],v['native_generation_hours'][1]]))}h ; autres preuves/scoring/préparation requêtes/reclassement/attentes en plus. Pas de coût end-to-end V142 national validé.",
        f"Ordre de grandeur isolé V19 : les8,67milliards de couples ×23,732s/319488 donnent≈{fmt(v['commune_scoring_proxy_hours'])}h ; ce proxy de boucle seule ne couvre pas les autres preuves. Si on ne prépare que les communes CRM :{inv['crm_communes_sites']} sites, SITE≈{fmt(v['crm_communes_site_encoding_hours'][0])}–{fmt(v['crm_communes_site_encoding_hours'][1])}h, encore hors PARENT. Point de comparaison historique :3000 dossiers160–250h avec partage, pas une règle de trois linéaire.",
        'Float16 est une estimation de stockage uniquement : conversion/changement de précision non autorisé par le gel et aucune fidélité des sorties démontrée. V142 n’a pas de politique de serving sans vérité : routage OOF dépendant du SIREN humain, full-fit distinct à autoriser (45–75min estimées), sources/caches en ligne et validation indépendante manquants. Pic natif historique9,14Gio ; réserver12Gio à cette phase (non garanti nationalement), séquentiellement avec l’autre agent, pas une banque entière en RAM.',
        '', '## 4. Disque : inventaire sans suppression',
        f"Libre au relevé : **{d['free_bytes']/2**30:.2f}Gio**. `du -sk` mesure les octets alloués, sans suivre les liens. Parents ci-dessous et leurs enfants ne s’additionnent pas :"]
    for row in d['roots']:lines.append(f"- `{row['path']}` : {fmt(row['allocated_bytes']/2**30)}Gio ; mtime répertoire {row['directory_mtime'][:10]}.")
    lines += ['', 'Dix plus gros sous-dossiers non chevauchants des deux racines principales :',
        '| Dossier | Gio | Date mtime répertoire UTC | Doc récent100commits | Chemin dans sceauV142 |','|---|---:|---|---|---|']
    for row in d['top10_nonoverlapping_children']:
        lines.append(f"| `{row['path']}` | {fmt(row['allocated_bytes']/2**30)} | {row['directory_mtime'][:10]} | {'oui' if row['recent100_commit_docs'] else 'non trouvé'} | {'oui' if row['frozen_v142_manifest_path_reference'] else 'non trouvé'} |")
    lines += ['Dates=mtime du répertoire, PAS dernière écriture récursive. Références docs au nom du dossier et ensemble des100 derniers commits : heuristique, pas preuve de non-utilisation. Les détails et exemples de docs sont dans le JSON ; un ancêtre peut rester nécessaire même sans référence récente. Ne pas supprimer index/sources/caches scellés ; demander une analyse des dépendances par fichier avant tout ménage. Aucun fichier pending ouvert ; aucun fichier étranger ni cache modifié, aucune action sur la VM.',
        '', '## 5. Recommandation et conditions',
        f"**Choisir G0 + S1/S2-BASE pour une chaîne de propositions à relire (`A_VERIFIER`), pas pour écraser automatiquement les SIRET.** Moindre coût de recherche, encodage limité aux candidats réellement nouveaux, banque partagée et traitement par lots≤200 sous4Gio ; aucun nouveau produit déployé ici. Projection du chemin chaud :{fmt(n*warm/3600)}h ; sans aucun partage des candidats, **{fmt(scenarios[0]['total_hours_with_warm_and_query_encoding'][0])}–{fmt(scenarios[0]['total_hours_with_warm_and_query_encoding'][1])}h** avec encodage des candidats et noms CRM ; avec80% de partage,{fmt(scenarios[2]['total_hours_with_warm_and_query_encoding'][0])}–{fmt(scenarios[2]['total_hours_with_warm_and_query_encoding'][1])}h. Ce sont des scénarios, hors nouveau géocodage, pas une promesse.",
        'Avant exécution globale : confirmer que cet export est bien toute la base à traiter, valider la sémantique des doublons et la politique de conservation des SIRET, autoriser une recette de serving/full-fit et une validation indépendante ; mesurer la génération de tous les candidats sans labels pour connaître les vrais textes nouveaux, puis réserver leur disque+temporaires+10Gio de sécurité. Exécution séquentielle CPU/Metal avec l’autre agent ; un seul Gemma, jamais de contrôle dehydraulic/PID8288.',
        'Réserve conseillée : pour le scénario G0 sans partage,≈38Gio de vecteurs f32 + marge/exports/10Gio de sécurité, donc **au moins50–60Gio libres** ; c’est une capacité, pas le besoin réellement mesuré. Pour une banque V142 SITE+PARENT f32,≈211Gio hors textes/index/temp, donc **≥250Gio libres** de capacité initiale à confirmer. Les19Gio libres sont insuffisants pour promettre le full : même f16 candidats sans partage≈18,85Gio avant marge. RAM G0 d’inférence mesurée ci-dessus<4Gio par lots ; GemmaBASE historique≈2,2–4,3Gio selon phase, donc worker strictement borné et pas chargement complet de banque. Entraînement/full-fit non budgété ici.',
        'Qualité : G0 top100478/526 vsV142507/526 (pilote consommé). G0+S1S2BASE top1 UNION13382/17427=76,79%, exact12275/17427=70,44%. Risque calibration97% :4360/4459=97,78% UNION, couverture25,59%, mais4151/4459=93,09% exact. Calibration97,5% :98,02% UNION, borneWilson97,37%, couverture13,59%. Calibration98% :0SUR. **Aucun statut SUR98% certifié ; conserver A_VERIFIER.** Ces chiffres de développement ne prédisent ni la couverture ni les erreurs nationales.',
        '', '## Preuve et reproduction',
        '```bash',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/production_cost_study.py inventory',
        'nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/production_cost_study.py bench > /tmp/sireto-production-cost-bench.log 2>&1 &',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/production_cost_study.py disk',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/production_cost_study.py finalize',
        '```',
        'La première ligne de suivi a été intégrée par la publication concurrente `be26eab2` ; son attestation finale est mise à jour dans le commit isolé des quatre livrables de cette étude. Aucun commit tiers réécrit. Exécuter les phases séquentiellement pour éviter une mise à jour concurrente du JSON. Suivi du run livré : `/tmp/sireto-production-cost-bench-corrected.log`, `/tmp/sireto-production-cost-disk.log`. Petit correctif de projection NAF et normalisationNB avant le rejeu final ; première tentative conservée dans le journal, sans résultat retenu. Un diagnostic DISTINCT SIREN arrêté par DuckDB512Mio ; remplacé par le compte de capacité du footer, pas par un faux nombre mesuré. Aucun test unitaire, fit, nouvel encodage, web, suppression, actionVM/PID8288 ou ouverture de label fermé.']
    (ROOT/'docs/production_cost_study.md').write_text('\n'.join(lines)+'\n')
    print('FINALIZED',projections['g0']['warm_hours_all_rows'],v['bank_preparation_proxy_hours'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['inventory','bench','disk','finalize']);args=parser.parse_args();guard=Guard()
    globals()[args.phase]()
    guard.done.set();print('DONE',args.phase,'peak_bytes',max(guard.peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),flush=True)


if __name__=='__main__':main()
