"""Preregistered company-crossfit activity and BASE semantic signals on frozen G0.

Sequential sealed workers; no retrieval, adapted encoder, label edits or promotion.
"""
import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import threading
import time
import warnings

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
OWN = HOME/'step2_colocated_signals_20261019'
DOC = ROOT/'docs/step2_colocated_signals.md'
NAF = Path('/Volumes/CATNAT_DATA/oubusiness/gold/naf_activity_reference_2026-07-24.parquet')
NAF_MANIFEST = NAF.with_suffix('.manifest.json')
VARIANTS = ('base', 's1', 's2', 's1_s2')
S1 = ['naf_div_logp', 'naf_group_logp', 'coloc_div_rank', 'coloc_group_rank', 'coloc_div_delta_best', 'coloc_group_delta_best', 'naf_missing']
S2 = ['gemma_base_cosine', 'coloc_gemma_delta_best']
SEED = 20261017
QUERY_PROMPT, DOC_PROMPT = 'task: search result | query: ', 'title: none | text: '


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def record(path):
    path = Path(path); before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Source changed during hashing')
    return dict(path=str(path), size_bytes=after.st_size, mtime_ns=after.st_mtime_ns, inode=after.st_ino, sha256=digest.hexdigest())


def verify(ref):
    actual = record(ref['path'])
    if actual['sha256'] != ref['sha256'] or actual['size_bytes'] != ref['size_bytes']:
        raise ValueError('Source seal mismatch: '+ref['path'])
    return Path(ref['path'])


def wait_resources():
    import psutil
    while True:
        vm = psutil.virtual_memory(); processes = []
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                name = (p.info['name'] or '').lower(); args = p.info['cmdline'] or []
                if any(Path(a).name == 'train3000_frozen_v142_supervisor.py' for a in args):
                    processes += [(c.pid, 'v142') for c in [p]+p.children(recursive=True) if c.memory_info().rss > 4*1024**3]
                # Inspect metadata only. Unknown virtualization process is conservative.
                vm_process = any(x in name for x in ('qemu', 'vfkit', 'virtualization', 'prl_vm', 'vmware-vmx'))
                hydraulic = any('hydraulic' in a.lower() for a in args) and name not in ('bash', 'zsh', 'python', 'python3', 'rg', 'grep')
                if (vm_process or hydraulic) and vm.free+vm.inactive < 8*1024**3:
                    processes.append((p.pid, 'vm_low_free_inactive'))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if not processes: return
        print(json.dumps(dict(stage='WAIT_RESOURCE_NO_VM_CONTROL', processes=processes, free_inactive=vm.free+vm.inactive)), flush=True)
        time.sleep(60)


class Guard:
    def __init__(self, directory, metal=False):
        import psutil
        self.directory, self.metal, self.peak, self.rss_peak, self.driver_peak = directory, metal, 0, 0, 0
        self.process = psutil.Process(); self.stop = threading.Event()
        self.rss_peak = self.process.memory_info().rss
        if metal:
            import torch
            self.driver_peak = torch.mps.driver_allocated_memory()
        self.peak = self.rss_peak+self.driver_peak
        threading.Thread(target=self.monitor, daemon=True).start()

    def monitor(self):
        while not self.stop.wait(.1):
            rss = self.process.memory_info().rss
            if self.metal:
                import torch
                driver = torch.mps.driver_allocated_memory()
            else: driver = 0
            self.rss_peak = max(self.rss_peak, rss); self.driver_peak = max(self.driver_peak, driver)
            self.peak = max(self.peak, rss+driver)
            if self.peak >= int(5.75*1024**3):
                write(self.directory/'resource_stop.json', dict(status='STOP_RSS_PLUS_DRIVER_5_75GIB', peak=self.peak))
                os._exit(75)


def check(out):
    plan = read(out/'plan.json')
    for ref in plan['implementations']+[plan['preregistration']]:
        if ref['path']==str(Path(__file__).resolve()) and (out/'resume.json').exists():
            resume=read(out/'resume.json');verify(resume['original_plan'])
            snapshot=verify(plan['implementation_snapshot'])
            if record(snapshot)['sha256']!=ref['sha256']:raise ValueError('Original implementation custody lost')
            verify(resume['implementation'])
        else:verify(ref)
    return plan


def encode_worker(out, directory, probe):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    check(out); cfg = read(out/'encoding.json')
    for ref in cfg['base_files']: verify(ref)
    texts = read(verify(cfg['texts']))
    guard = Guard(directory, metal=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(SEED)
    if not torch.backends.mps.is_available(): raise ValueError('Metal unavailable')
    torch.mps.set_per_process_memory_fraction(.12)
    began = time.monotonic(); config = AutoConfig.from_pretrained(HOME/'base', local_files_only=True)
    config.audio_config = None; config.vision_config = None
    model = AutoModel.from_pretrained(HOME/'base', config=config, dtype=torch.bfloat16, attn_implementation='eager', local_files_only=True).to('mps').eval()
    for p in model.parameters(): p.requires_grad_(False)
    if any('lora' in n.lower() for n, _ in model.named_modules()): raise ValueError('Adapted encoder forbidden')
    tokenizer = AutoTokenizer.from_pretrained(HOME/'base', local_files_only=True)
    load_seconds = time.monotonic()-began; cropped = 0; max_tokens = 0

    def encode(batch):
        nonlocal cropped, max_tokens
        raw = tokenizer(batch, padding=False, truncation=False)['input_ids']
        cropped += sum(len(v)>512 for v in raw); max_tokens = max(max_tokens, max(map(len, raw), default=0))
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        tokens = {k: v.to('mps') for k, v in tokens.items() if k in ('input_ids', 'attention_mask')}
        hidden = model(**tokens).last_hidden_state.float(); mask = tokens['attention_mask'].unsqueeze(-1)
        value = torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1), dim=-1).cpu().numpy()
        if value.shape != (len(batch), 768) or not np.isfinite(value).all(): raise ValueError('Bad BASE vectors')
        return value

    with torch.inference_mode():
        warm = time.monotonic(); encode(texts[:4]); torch.mps.synchronize(); warm_seconds = time.monotonic()-warm
        if probe:
            q = sorted([i for i, s in enumerate(texts) if s.startswith(QUERY_PROMPT)], key=lambda i: hashlib.sha256(texts[i].encode()).digest())[:500]
            c = sorted([i for i, s in enumerate(texts) if s.startswith(DOC_PROMPT)], key=lambda i: hashlib.sha256(texts[i].encode()).digest())[:500]
            indices = np.array(q+c); vectors = np.empty((len(indices), 768), dtype=np.float32)
            start = time.monotonic()
            for at in range(0, len(indices), 4): vectors[at:at+4] = encode([texts[i] for i in indices[at:at+4]])
            torch.mps.synchronize(); elapsed = time.monotonic()-start
            np.save(directory/'vectors.npy', vectors); write(directory/'texts.json', [texts[i] for i in indices])
            result = dict(status='BASE_NO_ADAPTER_PROBE', items=len(indices), crm=len(q), sites=len(c), seconds=elapsed,
                seconds_per1000=1000*elapsed/len(indices), load_seconds=load_seconds, warmup_seconds=warm_seconds,
                vectors=record(directory/'vectors.npy'), texts=record(directory/'texts.json'), total_texts=len(texts),
                projected_seconds=load_seconds+1.5*elapsed*len(texts)/len(indices), truncated=cropped, max_raw_tokens=max_tokens,
                runtime=dict(python=sys.executable, torch=torch.__version__, transformers=transformers.__version__), adapter_loaded=False)
        else:
            probe_info = read(out/'probe/result.json'); pv = np.load(verify(probe_info['vectors']))
            reuse = dict(zip(read(verify(probe_info['texts'])), pv))
            matrix = np.lib.format.open_memmap(directory/'vectors.npy', mode='w+', dtype=np.float32, shape=(len(texts), 768))
            remaining = []
            for i, text in enumerate(texts):
                if text in reuse: matrix[i] = reuse[text]
                else: remaining.append(i)
            start = time.monotonic(); paused = 0.
            for at in range(0, len(remaining), 4):
                if at % 4000 == 0:
                    waiting = time.monotonic(); wait_resources(); paused += time.monotonic()-waiting
                if time.monotonic()-began-paused+probe_info['seconds']+probe_info['load_seconds'] > 10800:
                    raise TimeoutError('S2 three-hour encoding budget exceeded; no result promotion')
                ix = remaining[at:at+4]; matrix[ix] = encode([texts[i] for i in ix])
                if at % 4000 == 0:
                    matrix.flush(); write(directory/'progress.json', dict(encoded=at+len(ix), remaining_total=len(remaining), seconds=time.monotonic()-start))
            torch.mps.synchronize(); matrix.flush(); del matrix
            result = dict(status='BASE_NO_ADAPTER_ENCODED', items=len(texts), new_items=len(remaining), reused_probe=len(texts)-len(remaining),
                seconds=time.monotonic()-start-paused, resource_wait_seconds=paused, load_seconds=load_seconds, warmup_seconds=warm_seconds,
                vectors=record(directory/'vectors.npy'), texts=cfg['texts'], truncated_new=cropped, max_raw_tokens=max_tokens,
                adapter_loaded=False, gradients=False)
    result.update(peak_rss_bytes=guard.rss_peak, peak_driver_bytes=guard.driver_peak, peak_rss_plus_driver_bytes=guard.peak)
    write(directory/'result.json', result); guard.stop.set()
    return result


# Runtime Metal environment need not install XGBoost/Arrow/scikit-learn.
if len(sys.argv)>1 and sys.argv[1] in ('_probe', '_encode'):
    parser = argparse.ArgumentParser(); parser.add_argument('phase'); parser.add_argument('--output-dir', type=Path, required=True); parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(OWN) or args.directory.parent != args.output_dir: raise ValueError('Owned runtime namespace required')
    wait_resources(); encode_worker(args.output_dir, args.directory, args.phase=='_probe'); sys.exit(0)

from scripts import step2_g0_bench as bench
b, d, old, np, pa, pq, duckdb, xgb = bench.b, bench.d, bench.old, bench.np, bench.pa, bench.pq, bench.duckdb, bench.xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression


def pinned(commit, path):
    if json.loads(subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT)) != read(ROOT/path):
        raise ValueError('Pinned authority changed: '+path)


def connection(directory):
    return duckdb.connect(':memory:', config={'threads':'1', 'memory_limit':'512MB', 'temp_directory':str(directory/'ducktmp'), 'max_temp_directory_size':'4GB'})


def address_key(site):
    if site.get('statutDiffusionEtablissement')!='O' or site.get('dernierNumeroVoieEtablissement') or site.get('indiceRepetitionDernierNumeroVoieEtablissement'): return None
    num = b.number(site.get('numeroVoieEtablissement'), site.get('indiceRepetitionEtablissement'))
    street = b.normalize(' '.join(str(site.get(k) or '') for k in ('typeVoieEtablissement','libelleVoieEtablissement')), street=True)
    city = site.get('codeCommuneEtablissement')
    return (city, num, street) if city and num and street else None


def nomenclature():
    refs = [record(NAF), record(NAF_MANIFEST)]; manifest = read(NAF_MANIFEST)
    for ref in manifest['receipts']: verify(dict(ref, size_bytes=Path(ref['path']).stat().st_size))
    rows = pq.read_table(NAF, columns=['code','label','version','source_url']).to_pylist()
    labels = {r['code'].replace('.',''): b.text(r['label']) for r in rows if r['version']=='NAF_REV2' and r['source_url'].startswith('https://www.insee.fr/')}
    if len(labels)!=732: raise ValueError('Official NAF vocabulary drift')
    return labels, refs


def metadata(directory, queries, keys, official, want_text_rows, truth=None):
    """Candidate-only streams first, separate training/evaluation target join last."""
    labels, nafrefs = nomenclature(); divisions = sorted({k[:2] for k in labels}); groups = sorted({k[:3] for k in labels})
    maps = [{v:i for i,v in enumerate(x)} for x in (divisions, groups)]
    unique, inverse = np.unique(keys, return_inverse=True); inverse = inverse.reshape(keys.shape)
    con = connection(directory); con.register('requested', pa.table({'siret':unique.astype(str), 'i':np.arange(len(unique), dtype=np.int32)}))
    columns = list(dict.fromkeys(b.STOCK_COLUMNS+['activitePrincipaleEtablissement','nomenclatureActivitePrincipaleEtablissement']))
    sql = 'SELECT r.i,'+','.join('s.'+k for k in columns)+' FROM read_parquet(?) s RIGHT JOIN requested r USING(siret)'
    codes = np.full((len(unique),2), -1, dtype=np.int16); addresses = np.full(len(unique), -1, dtype=np.int32)
    addr_lookup, text_sites = {}, {}
    text_unique = set(np.unique(inverse[want_text_rows]).tolist()) if len(want_text_rows) else set()
    for batch in con.execute(sql, [official['stock']['path']]).fetch_record_batch(16384):
        for site in batch.to_pylist():
            i = site['i']; public = site.get('statutDiffusionEtablissement')=='O'
            code = b.text(site.get('activitePrincipaleEtablissement')).replace('.','').upper()
            if public and site.get('nomenclatureActivitePrincipaleEtablissement')=='NAFRev2' and code in labels:
                codes[i] = [maps[0][code[:2]], maps[1][code[:3]]]
            else: code = ''
            addr = address_key(site)
            if addr is not None: addresses[i] = addr_lookup.setdefault(addr, len(addr_lookup))
            if i in text_unique:
                names = [b.text(site.get(k)) for k in ('enseigne1Etablissement','enseigne2Etablissement','enseigne3Etablissement','denominationUsuelleEtablissement')] if public else []
                text_sites[i] = dict(siren=bytes(unique[i]).decode()[:9], names=names, naf=labels.get(code,''))
    units_needed = sorted({s['siren'] for s in text_sites.values()})
    con.register('requested_ul', pa.table({'siren':units_needed}))
    units = b.index(con.execute('SELECT '+','.join('u.'+k for k in b.UL_COLUMNS)+' FROM read_parquet(?) u SEMI JOIN requested_ul r USING(siren)', [official['legal_units']['path']]).fetch_arrow_table().to_pylist(), 'siren') if units_needed else {}
    texts, text_lookup = [], {}
    def text_index(value):
        if not value: return -1
        if value not in text_lookup: text_lookup[value] = len(texts); texts.append(value)
        return text_lookup[value]
    qi = np.full(len(queries), -1, dtype=np.int32); ti_unique = np.full(len(unique), -1, dtype=np.int32)
    for i in want_text_rows:
        name = b.text(queries[i]['crm_name']).strip(); qi[i] = text_index(QUERY_PROMPT+name) if name else -1
    for i in sorted(text_sites):
        site = text_sites[i]
        unit = units.get(site['siren'], {}); names = site['names'][:]
        if unit.get('statutDiffusionUniteLegale')=='O':
            names = [b.text(unit.get(k)) for k in ('denominationUniteLegale','denominationUsuelle1UniteLegale','sigleUniteLegale','nomUniteLegale','prenomUsuelUniteLegale')]+names
        names = list(dict.fromkeys(n.strip() for n in names if n.strip()))
        value = '; '.join(names+([site['naf']] if site['naf'] else []))
        ti_unique[i] = text_index(DOC_PROMPT+value) if value else -1
    np.save(directory/'naf.npy', codes[inverse]); np.save(directory/'addresses.npy', addresses[inverse]); np.save(directory/'qtext.npy', qi); np.save(directory/'ctext.npy', ti_unique[inverse])
    write(directory/'texts.json', texts)
    seal = dict(files={p:record(directory/p) for p in ('naf.npy','addresses.npy','qtext.npy','ctext.npy','texts.json')}, naf_refs=nafrefs,
        code_classes=[divisions,groups], candidate_keys_sha256=bench.sha(keys), status='LABEL_FREE_METADATA_SEALED')
    write(directory/'feature_seal.json', seal)
    # Only now build S1 supervised targets, never candidate generation or text.
    targets = np.full((len(queries),2), -1, dtype=np.int16)
    if truth is not None:
        con.register('truth', pa.table({'siret':truth}))
        rows = con.execute('SELECT s.siret,s.activitePrincipaleEtablissement,s.nomenclatureActivitePrincipaleEtablissement FROM read_parquet(?) s SEMI JOIN truth t USING(siret)', [official['stock']['path']]).fetch_arrow_table().to_pylist()
        lookup = {r['siret']:r for r in rows}
        for i,s in enumerate(truth):
            raw = lookup.get(s,{}); code = b.text(raw.get('activitePrincipaleEtablissement')).replace('.','').upper()
            if raw.get('nomenclatureActivitePrincipaleEtablissement')=='NAFRev2' and code in labels: targets[i] = [maps[0][code[:2]],maps[1][code[:3]]]
    np.save(directory/'naf_targets.npy', targets); con.close()
    return seal


def prepare(out, directory):
    plan = check(out); pinned('9a8ecdd6','docs/step2_g0_bench.json'); pinned('c37a1a7b','docs/step2_error_families_g0.json'); pinned('d54cec2c','docs/train3000_frozen_v142_sample.json')
    result, matrix, queries, labels = bench.source(); parent=read(ROOT/'docs/step2_g0_bench.json'); data=read(verify(parent['data']))
    if data['ids']!=[q['query_id'] for q in queries] or data['groups']!=[labels[q['query_id']]['human_ground_truth_siren'] for q in queries]:raise ValueError('Parent G0 row/group drift')
    verify(data['views'])
    for ref in data['feature_files'].values():verify(ref)
    sample = set(read(ROOT/'docs/train3000_frozen_v142_sample.json')['selected_query_ids'])
    selected = np.array([i for i,q in enumerate(queries) if q['query_id'] in sample])
    if len(selected)!=3000: raise ValueError('Sealed3000 intersection drift')
    original = np.arange(len(queries)); allgroups = np.array(data['groups'])
    if plan['smoke']:
        companies = sorted(set(allgroups[selected]), key=lambda g:hashlib.sha256(('20261019|colocated-smoke|'+g).encode()).digest())[:100]
        original = selected[np.isin(allgroups[selected],companies)]; selected = np.arange(len(original))
    queries = [queries[i] for i in original]; keys = np.load(matrix['keys']['path'],mmap_mode='r')[original]
    official = data['official_refs']
    for ref in official.values(): verify(ref)
    seal = metadata(directory, queries, keys, official, selected, [labels[q['query_id']]['human_ground_truth_siret'] for q in queries])
    ids = [q['query_id'] for q in queries]; group = allgroups[original]; partition = np.array(data['partition'])[original]
    np.save(out/'source_indices.npy', original); np.save(out/'sub_indices.npy', selected)
    families = {a['query_id'] for a in read(ROOT/'docs/step2_error_families_g0.json')['assignments'] if a['family']=='colocation'}
    np.save(out/'family.npy', np.array([i in families for i in ids])); np.save(out/'views.npy', np.load(bench.OWN/'full/views.npy',mmap_mode='r')[original])
    basefiles = [record(p) for p in sorted((HOME/'base').glob('*')) if p.is_file() and p.suffix in ('.json','.safetensors','.model')]
    write(out/'encoding.json', dict(texts=seal['files']['texts.json'], base_files=basefiles))
    sources = [record(ROOT/'docs/step2_g0_bench.json'),record(ROOT/'docs/step2_error_families_g0.json'),record(ROOT/'docs/train3000_frozen_v142_sample.json'), matrix['features'],matrix['keys'],record(bench.OWN/'full/data.json'),record(bench.OWN/'full/views.npy')]+list(data['feature_files'].values())+list(result['cohort']['artifacts'].values())+list(official.values())+seal['naf_refs']
    payload = dict(ids=ids, groups=group.tolist(), partition=partition.tolist(), folds=[q['fold'] for q in queries],
        names=[b.normalize_name(b.text(q['crm_name'])) for q in queries], classes=seal['code_classes'], metadata=str(directory),
        source_matrix=matrix['features'], official=official, sources=sources, feature_seal=record(directory/'feature_seal.json'),
        queries=len(ids), companies=len(set(group)), initial_subset=len(selected), labels_changed=False,
        supervised_naf_targets=record(directory/'naf_targets.npy'), views=record(out/'views.npy'), family=record(out/'family.npy'),
        source_indices=record(out/'source_indices.npy'), naf_available_candidate_pairs=int((np.load(directory/'naf.npy')[:,:,0]>=0).sum()),
        naf_available_supervised_queries=int((np.load(directory/'naf_targets.npy')[:,0]>=0).sum()))
    write(out/'data.json', payload)
    return dict(queries=len(ids), initial_subset=len(selected), unique_texts=len(read(seal['files']['texts.json']['path'])), data=record(out/'data.json'))


def select_subset(out, directory):
    check(out); data = read(out/'data.json'); probe = read(out/'probe/result.json'); encoding = read(out/'encoding.json')
    sub = np.load(out/'sub_indices.npy'); meta = Path(data['metadata']); qi = np.load(meta/'qtext.npy'); ci = np.load(meta/'ctext.npy')
    original_sub = sub.copy(); budget = 10800.; rate = probe['seconds']/probe['items']
    if probe['projected_seconds']>budget:
        company = np.array(data['groups']); ordered = sorted(set(company[sub]),key=lambda g:hashlib.sha256(('20261019|colocated-budget|'+g).encode()).digest())
        limit = math.floor((budget-probe['load_seconds'])/(1.5*rate)); chosen, used = [], set()
        for g in ordered:
            rows = sub[company[sub]==g]; needed = set(ci[rows].reshape(-1).tolist())|set(qi[rows].tolist()); needed.discard(-1)
            if len(used|needed)>limit: break
            chosen.extend(rows.tolist()); used |= needed
        sub = np.array(sorted(chosen),dtype=int)
        if len(set(company[sub]))<50 or set(np.array(data['partition'])[sub])!=set(range(5)):
            raise ValueError('S2 budget leaves insufficient companies/CV coverage')
    needed = np.unique(np.r_[qi[sub],ci[sub].reshape(-1)]); needed = needed[needed>=0]
    texts = read(verify(encoding['texts'])); write(directory/'texts.json',[texts[i] for i in needed])
    remap = np.full(len(texts),-1,dtype=np.int32); remap[needed]=np.arange(len(needed))
    qremap = np.full(len(qi),-1,dtype=np.int32); crem = np.full_like(ci,-1)
    qremap[sub] = np.where(qi[sub]>=0,remap[np.maximum(qi[sub],0)],-1)
    crem[sub] = np.where(ci[sub]>=0,remap[np.maximum(ci[sub],0)],-1)
    np.save(directory/'qtext.npy',qremap); np.save(directory/'ctext.npy',crem); np.save(out/'sub_indices.npy',sub)
    encoding['texts'] = record(directory/'texts.json'); write(out/'encoding.json',encoding)
    result = dict(original_queries=len(original_sub), queries=len(sub), companies=len({data['groups'][i] for i in sub}), reduced=len(sub)!=len(original_sub),
        unique_texts=len(needed), projected_seconds=probe['load_seconds']+1.5*rate*len(needed), budget_seconds=budget,
        indices=record(out/'sub_indices.npy'), text_files={f:record(directory/f) for f in ('texts.json','qtext.npy','ctext.npy')})
    write(out/'selection.json',result); return result


def relative(values, addresses, ranks=False):
    result = np.full_like(values,np.nan,dtype=np.float32); rank = np.full_like(result,np.nan)
    for i in range(len(values)):
        for key in np.unique(addresses[i]):
            if key<0: continue
            ix = np.flatnonzero((addresses[i]==key)&np.isfinite(values[i]))
            if len(ix):
                result[i,ix] = values[i,ix]-values[i,ix].max()
                if ranks:
                    unique = np.unique(values[i,ix])[::-1]
                    rank[i,ix] = [1+int(np.flatnonzero(unique==v)[0]) for v in values[i,ix]]
    return (rank,result) if ranks else result


def semantic_features(out, directory):
    data=read(out/'data.json'); selection=read(out/'selection.json'); meta=Path(data['metadata']); sub=np.load(out/'sub_indices.npy')
    vectors=np.load(verify(read(out/'encode/result.json')['vectors']),mmap_mode='r')
    qi=np.load(out/'selection/qtext.npy'); ci=np.load(out/'selection/ctext.npy'); values=np.full(ci.shape,np.nan,dtype=np.float32)
    for i in sub:
        valid=np.flatnonzero(ci[i]>=0)
        if qi[i]>=0: values[i,valid]=vectors[ci[i,valid]]@vectors[qi[i]]
    addresses=np.load(meta/'addresses.npy'); rel=relative(values[sub],addresses[sub]); features=np.full((*ci.shape,2),np.nan,dtype=np.float32)
    features[sub,:,0]=values[sub]; features[sub,:,1]=rel
    np.save(out/'semantic.npy',features); return dict(features=record(out/'semantic.npy'), finite_cosines=int(np.isfinite(values[sub]).sum()), selection=selection)


class Dataset:
    def __init__(self,out,indices):
        self.out=out; self.data=read(verify(read(out/'preparation/receipt.json')['data'])); self.indices=np.asarray(indices); self.meta=Path(self.data['metadata'])
        seal=read(verify(self.data['feature_seal']))
        for ref in list(seal['files'].values())+[self.data[k] for k in ('supervised_naf_targets','views','family','source_indices','source_matrix')]:verify(ref)
        source=np.load(self.data['source_matrix']['path'],mmap_mode='r'); origin=np.load(out/'source_indices.npy')[self.indices]
        self.X=source[origin]; self.scene=np.load(bench.OWN/'full/scene.npy',mmap_mode='r')[origin]
        self.views=np.load(out/'views.npy',mmap_mode='r')[self.indices]; self.groups=np.array(self.data['groups'])[self.indices]
        self.partition=np.array(self.data['partition'])[self.indices]; self.folds=np.array(self.data['folds'])[self.indices]
        self.ids=np.array(self.data['ids'])[self.indices]; self.names=np.array(self.data['names'])[self.indices]
        self.codes=np.load(self.meta/'naf.npy',mmap_mode='r')[self.indices]; self.targets=np.load(self.meta/'naf_targets.npy')[self.indices]
        self.addresses=np.load(self.meta/'addresses.npy',mmap_mode='r')[self.indices]; self.semantic=None
        self.family=np.load(out/'family.npy')[self.indices]; self.classes=self.data['classes']


def nb_features(ds,train,test,directory,purpose):
    if set(ds.groups[train])&set(ds.groups[test]): raise ValueError('S1 company leakage')
    vectorizer=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[a-z]{2,}\b',min_df=2,max_features=20000)
    try: a=vectorizer.fit_transform(ds.names[train]); z=vectorizer.transform(ds.names[test])
    except ValueError as e:
        if 'vocabulary' not in str(e) and 'terms remain' not in str(e): raise
        vectorizer=CountVectorizer(vocabulary={'__empty_training_vocabulary__':0}); a=vectorizer.transform(ds.names[train]); z=vectorizer.transform(ds.names[test])
    models=[]; probabilities=[]
    for k,classes in enumerate(ds.classes):
        valid=ds.targets[train,k]>=0
        model=MultinomialNB(alpha=1.,fit_prior=False)
        # partial_fit admits every public class, even unseen in this fit.
        if valid.any(): model.partial_fit(a[valid],ds.targets[train[valid],k],classes=np.arange(len(classes)))
        else:
            # Empty supervised set: analytic Laplace prior, no fabricated row.
            model.classes_=np.arange(len(classes)); model.class_count_=np.zeros(len(classes))
            model.feature_count_=np.zeros((len(classes),a.shape[1])); model.n_features_in_=a.shape[1]
            model._update_feature_log_prob(1.)
        model.class_log_prior_=np.log((model.class_count_+1)/(model.class_count_.sum()+len(classes)))
        probabilities.append(model.predict_log_proba(z)); models.append(model)
    path=directory/(purpose+'.pkl.gz')
    with gzip.open(path,'wb',compresslevel=1) as stream: pickle.dump((vectorizer,models),stream,protocol=5)
    with gzip.open(path,'rb') as stream: vr,mr=pickle.load(stream)
    replay=[model.predict_log_proba(vr.transform(ds.names[test])) for model in mr]
    if any(not np.array_equal(a,z) for a,z in zip(probabilities,replay)): raise ValueError('NB serialized replay mismatch')
    values=[]
    for k,p in enumerate(probabilities):
        codes=ds.codes[test,:,k]; value=np.full(codes.shape,np.nan,dtype=np.float32)
        row,col=np.where(codes>=0); value[row,col]=p[row,codes[row,col]]
        values.append(value)
    rd,dd=relative(values[0],ds.addresses[test],ranks=True); rg,dg=relative(values[1],ds.addresses[test],ranks=True)
    features=np.stack([values[0],values[1],rd,rg,dd,dg,(ds.codes[test,:,0]<0).astype(np.float32)],axis=2)
    receipt=dict(model=record(path),train_indices=train.tolist(),scored_indices=test.tolist(),company_overlap=0,
        target_sha256=bench.sha(ds.targets[train]),feature_sha256=bench.sha(features),serialized_replay_equal=True)
    return features,receipt


class Runner:
    def __init__(self,ds,name,directory):
        self.ds,self.name,self.directory=ds,name,directory; directory.mkdir(); self.models={};self.receipts=[]
        if 's2' in name:
            ref=read(ds.out/'semantic/receipt.json')['features'];ds.semantic=np.load(verify(ref),mmap_mode='r')[ds.indices]

    def score(self,train,test,purpose):
        wait_resources(); ds=self.ds;train=np.sort(train);test=np.asarray(test)
        if set(ds.groups[train])&set(ds.groups[test]):raise ValueError('Ranker company overlap')
        key=d.digest(train.tolist()); use_s1='s1' in self.name;use_s2='s2' in self.name
        if key not in self.models:
            folder=self.directory/key;folder.mkdir(); features=[ds.X[train]];nb=[]
            if use_s1:
                s1=np.full((len(train),100,len(S1)),np.nan,dtype=np.float32)
                splitter=GroupKFold(n_splits=3,shuffle=True,random_state=SEED)
                for k,(a,z) in enumerate(splitter.split(train,groups=ds.groups[train])):
                    value,receipt=nb_features(ds,train[a],train[z],folder,f'oof{k}');s1[z]=value;nb.append(receipt)
                features.append(s1)
            if use_s2:features.append(ds.semantic[train])
            X=np.concatenate(features,axis=2) if len(features)>1 else features[0];target=ds.views[train,:,0].astype(np.float32)
            # The concatenation owns its data; do not retain its large inputs during fit.
            del features
            if use_s1:del s1,value
            gc.collect()
            model=xgb.XGBRanker(**old.PARAMS);began=time.monotonic()
            model.fit(X.reshape(-1,X.shape[2]),target.reshape(-1),group=np.full(len(train),100))
            path=folder/'model.json';model.save_model(path)
            receipt=dict(model=record(path),train_indices=train.tolist(),target_sha256=bench.sha(target),features_sha256=bench.sha(X),
                seconds=time.monotonic()-began,zero_positive_queries=int((target.sum(1)==0).sum()),nb_training=nb,uses=[])
            self.models[key]=(model,receipt);self.receipts.append(receipt);del X,target;gc.collect()
        model,receipt=self.models[key];features=[ds.X[test]]
        if use_s1:
            folder=self.directory/key;value,nbr=nb_features(ds,train,test,folder,'score_'+str(len(receipt['uses'])));features.append(value)
        else:nbr=None
        if use_s2:features.append(ds.semantic[test])
        X=np.concatenate(features,axis=2) if len(features)>1 else features[0]
        pred=model.predict(X.reshape(-1,X.shape[2])).reshape(len(test),100)
        replay=xgb.XGBRanker(n_jobs=1);replay.load_model(verify(receipt['model']))
        actual=replay.predict(X.reshape(-1,X.shape[2])).reshape(len(test),100)
        if not np.isfinite(pred).all() or not np.array_equal(pred,actual):raise ValueError('Ranker serialized replay mismatch')
        receipt['uses'].append(dict(purpose=purpose,scored_indices=test.tolist(),company_overlap=0,nb_scoring=nbr,
            feature_sha256=bench.sha(X),score_sha256=bench.sha(pred),replay_equal=True))
        return pred


def evaluate(ds,name,directory,cv_only=None):
    runner=Runner(ds,name,directory/'models');n=len(ds.ids);pred=np.full((n,100),np.nan,dtype=np.float32);common=np.full_like(pred,np.nan)
    selected={k:np.zeros(n,dtype=bool) for k in ('margin','scene_isotonic')};receipts=[];metarefs=[]
    for cv in (range(5) if cv_only is None else [cv_only]):
        train=np.flatnonzero(ds.partition!=cv);test=np.flatnonzero(ds.partition==cv)
        pred[test]=runner.score(train,test,f'outer{cv}')
        companies=np.random.default_rng(SEED+100+cv).permutation(np.array(sorted(set(ds.groups[train]))));count=max(1,math.ceil(.2*len(companies)))
        limit=train[np.isin(ds.groups[train],companies[:count])];iso_rows=train[np.isin(ds.groups[train],companies[count:2*count])];proper=train[~np.isin(ds.groups[train],companies[:2*count])]
        roles=[proper,iso_rows,limit,test]
        if any(set(ds.groups[a])&set(ds.groups[z]) for i,a in enumerate(roles) for z in roles[i+1:]):raise ValueError('Confidence overlap')
        meta=np.full((len(proper),100),np.nan,dtype=np.float32)
        for k,(a,z) in enumerate(GroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(proper,groups=ds.groups[proper])):
            meta[z]=runner.score(proper[a],proper[z],f'meta{cv}_{k}')
        raw=runner.score(proper,np.r_[iso_rows,limit,test],f'proper{cv}');pi,pl,pt=np.split(raw,[len(iso_rows),len(iso_rows)+len(limit)]);common[test]=pt
        yy=ds.views[proper,b.select_top(meta)[0],1].astype(int);classifier=d.logistic(yy);classifier.fit(d.scene(meta,ds.scene[proper]),yy)
        iso=IsotonicRegression(out_of_bounds='clip',y_min=0.,y_max=1.);iso.fit(d.probability(classifier,d.scene(pi,ds.scene[iso_rows])),ds.views[iso_rows,b.select_top(pi)[0],1])
        path=directory/f'confidence{cv}.pkl'
        with path.open('wb') as stream:pickle.dump((classifier,iso),stream,protocol=5)
        with path.open('rb') as stream:rc,ri=pickle.load(stream)
        lt,lm=b.select_top(pl);tt,tm=b.select_top(pt);lp=iso.predict(d.probability(classifier,d.scene(pl,ds.scene[limit])));tp=iso.predict(d.probability(classifier,d.scene(pt,ds.scene[test])))
        if not np.array_equal(tp,ri.predict(d.probability(rc,d.scene(pt,ds.scene[test])))):raise ValueError('Confidence pickle replay mismatch')
        limits={}
        for policy,cal,value in [('margin',lm,tm),('scene_isotonic',lp,tp)]:
            threshold,stats=b.threshold_choice(cal,ds.views[limit,lt,1]);selected[policy][test]=value>=threshold if threshold is not None else False
            limits[policy]=dict(threshold=threshold,calibration_union=stats)
        receipts.append(dict(cv=cv,proper=proper.tolist(),isotonic=iso_rows.tolist(),threshold=limit.tolist(),test=test.tolist(),company_overlap=0,limits=limits));metarefs.append(record(path))
        print(json.dumps(dict(stage='CV_COMPLETE',recipe=name,cv=cv,queries=len(test))),flush=True)
    if cv_only is not None:
        np.savez_compressed(directory/'predictions.npz',predictions=pred,common=common,**{k+'_sur':v for k,v in selected.items()})
        result=dict(cv=cv_only,params=old.PARAMS,ids_sha256=bench.sha(ds.ids),partition_sha256=bench.sha(ds.partition),models=runner.receipts,
            confidence=dict(receipts=receipts,models=metarefs),predictions=record(directory/'predictions.npz'))
        write(directory/'result.json',result);return dict(result=record(directory/'result.json'),fit_count=len(runner.receipts))
    return evaluation_result(ds,name,directory,pred,common,selected,receipts,metarefs,runner.receipts)


def evaluation_result(ds,name,directory,pred,common,selected,receipts,metarefs,models):
    n=len(ds.ids)
    if not np.isfinite(pred).all() or not np.isfinite(common).all():raise ValueError('Incomplete fold coverage')
    top=b.select_top(pred)[0];hits=ds.views[np.arange(n),top];commonhits=ds.views[np.arange(n),b.select_top(common)[0]]
    np.savez_compressed(directory/'predictions.npz',predictions=pred,hits=hits,common=common,common_hits=commonhits,**{k+'_sur':v for k,v in selected.items()})
    result=dict(recipe=name,queries=n,companies=len(set(ds.groups)),features=old.CONTENT+(S1 if 's1' in name else [])+(S2 if 's2' in name else []),params=old.PARAMS,
        top1=bench.metrics(hits),by_cv={str(cv):bench.metrics(hits[ds.partition==cv]) for cv in range(5)},by_population_fold={str(f):bench.metrics(hits[ds.folds==f]) for f in (2,3,4)},
        availability=dict(exact=int(ds.views[:,:,0].any(1).sum()),union=int(ds.views[:,:,1].any(1).sum())),
        confidence=dict(common_top1=bench.metrics(commonhits),policies={p:bench.metrics(commonhits,mask) for p,mask in selected.items()},receipts=receipts,models=metarefs),
        models=models,fit_count=len(models),predictions=record(directory/'predictions.npz'),all_vectors_replayed=sum(len(u['scored_indices']) for r in models for u in r['uses']))
    write(directory/'result.json',result);return dict(result=record(directory/'result.json'),fit_count=len(models))


def full_fold(out,directory,cv):
    check(out);ds=Dataset(out,np.arange(read(out/'data.json')['queries']))
    return evaluate(ds,'s1',directory,cv_only=cv)


def combine_full(out,directory):
    check(out);ds=Dataset(out,np.arange(read(out/'data.json')['queries']));n=len(ds.ids)
    pred=np.full((n,100),np.nan,dtype=np.float32);common=np.full_like(pred,np.nan)
    selected={k:np.zeros(n,dtype=bool) for k in ('margin','scene_isotonic')};receipts=[];metarefs=[];models=[]
    for cv in range(5):
        receipt=read(out/f'full_s1_cv{cv}/receipt.json');r=read(verify(receipt['result']));pack=np.load(verify(r['predictions']));mask=ds.partition==cv
        if r['cv']!=cv or r['params']!=old.PARAMS or r['ids_sha256']!=bench.sha(ds.ids) or r['partition_sha256']!=bench.sha(ds.partition):raise ValueError('Fold custody drift')
        if len(r['models'])!=5 or not np.isnan(pack['predictions'][~mask]).all():raise ValueError('Unexpected isolated fold fit/score scope')
        pred[mask]=pack['predictions'][mask];common[mask]=pack['common'][mask]
        for k in selected:selected[k][mask]=pack[k+'_sur'][mask]
        receipts.extend(r['confidence']['receipts']);metarefs.extend(r['confidence']['models']);models.extend(r['models'])
    return evaluation_result(ds,'s1',directory,pred,common,selected,receipts,metarefs,models)


def variant_worker(out,directory,name,full=False):
    check(out);data=read(out/'data.json');indices=np.arange(data['queries']) if full else np.load(out/'sub_indices.npy')
    ds=Dataset(out,indices);return evaluate(ds,name,directory)


def boot(hits,groups,mask=None):
    if mask is None:mask=np.ones(len(groups),dtype=bool)
    names=[n for n in hits if n!='base'];_,codes=np.unique(groups,return_inverse=True);count=np.bincount(codes,weights=mask)
    good=count>0;count=count[good];diff=np.array([np.bincount(codes,weights=mask*(hits[n][:,j].astype(int)-hits['base'][:,j].astype(int)),minlength=len(good))[good] for n in names for j in (0,1)])
    if not len(count):return None
    rng=np.random.default_rng(SEED);samples=[]
    for _ in range(100):
        weights=rng.multinomial(len(count),np.full(len(count),1/len(count)),size=100);samples.append(100*(weights@diff.T)/(weights@count)[:,None])
    samples=np.concatenate(samples);result={};comparisons=max(1,2*len(names))
    for k,n in enumerate(names):
        result[n]={view:dict(delta_pp=float(100*(hits[n][mask,j].mean()-hits['base'][mask,j].mean())),paired_company_bootstrap95_pp=np.quantile(samples[:,2*k+j],[.025,.975]).tolist(),
            simultaneous95_pp=np.quantile(samples[:,2*k+j],[.05/(2*comparisons),1-.05/(2*comparisons)]).tolist(),gains=int((mask&hits[n][:,j]&~hits['base'][:,j]).sum()),losses=int((mask&~hits[n][:,j]&hits['base'][:,j]).sum())) for j,view in enumerate(('exact','union'))}
    return result


def summarize(out,directory):
    check(out);data=read(out/'data.json');indices=np.load(out/'sub_indices.npy');groups=np.array(data['groups'])[indices];family=np.load(out/'family.npy')[indices]
    variants={n:read(out/n/'result.json') for n in VARIANTS};hits={n:np.load(verify(v['predictions']))['hits'] for n,v in variants.items()}
    deltas=boot(hits,groups);familyd=boot(hits,groups,family);full=None
    if not read(out/'plan.json')['smoke']:
        vf=read(out/'full_s1/result.json');bh=np.load(verify(read(ROOT/'docs/step2_g0_bench.json')['variants']['base']['predictions']))['hits'];fh=np.load(verify(vf['predictions']))['hits'];mask=np.load(out/'family.npy')
        full=dict(variant=vf,base=bench.metrics(bh),deltas=boot({'base':bh,'s1':fh},np.array(data['groups'])),family_count=int(mask.sum()),family_deltas=boot({'base':bh,'s1':fh},np.array(data['groups']),mask))
    eligible=[n for n in VARIANTS[1:] if deltas[n]['union']['paired_company_bootstrap95_pp'][0]>0]
    winner=max(eligible,key=lambda n:deltas[n]['union']['delta_pp']) if eligible else None;scope='sub'
    if winner is None and full and full['deltas']['s1']['union']['paired_company_bootstrap95_pp'][0]>0:winner='s1';scope='full'
    r=dict(status='G0_MEASURED_CONFIRMATION_PENDING' if winner else 'G0_COMPLETE_NO_POSITIVE_UNION_CI',smoke=read(out/'plan.json')['smoke'],queries=len(indices),companies=len(set(groups)),
        selection=read(out/'selection.json'),variants=variants,deltas=deltas,bootstrap_replicates=10000,bootstrap_seed=SEED,
        family=dict(reference='c37a1a7b',count=int(family.sum()),base=bench.metrics(hits['base'][family]),deltas=familyd,
            variants={n:bench.metrics(h[family]) for n,h in hits.items()}),full_s1=full,winner=winner,winner_scope=scope,
        encoding=dict(probe=read(out/'probe/result.json'),full=read(out/'encode/result.json')),confirmation=None,labels_changed=False,candidates_changed=False,closed_sets_opened=False,production_allowed=False,
        source_refs=data['sources'],plan=record(out/'plan.json'),data=record(out/'data.json'),
        naf_availability=dict(known_candidate_pairs=data['naf_available_candidate_pairs'],known_supervised_queries=data['naf_available_supervised_queries']))
    write(out/'result.json',r);return dict(result=record(out/'result.json'),winner=winner)


def confirm_prepare(out,directory):
    r=read(out/'result.json')
    if not r['winner']:return dict(skipped=True)
    Xold,y,views,groups,folds,ids,native,refs,outer,*_=d.load(False)
    pack=np.load(verify(read(ROOT/'docs/step2_features_v2.json')['matrix']));order=[pack['query_ids'].tolist().index(q) for q in ids]
    X=pack['features'][order][:,:,old.CONTENT_COLS];np.save(directory/'pilot_base.npy',X);np.save(directory/'pilot_scene.npy',Xold);np.save(directory/'pilot_views.npy',views)
    queries=b.index(read(old.OWN/'full/pilot_queries.json'));queries=[queries[q] for q in ids]
    keys=np.array([[c['siret'].encode() for c in native[q]] for q in ids],dtype='S14')
    seal=metadata(directory,queries,keys,read(out/'data.json')['official'],np.arange(len(ids)),None)
    write(directory/'pilot.json',dict(ids=ids.tolist(),groups=groups.tolist(),folds=folds.tolist(),outer=[(int(f),tr.tolist(),te.tolist()) for f,tr,te in outer],seal=record(directory/'feature_seal.json')))
    if 's2' in r['winner']:
        encoding=read(out/'encoding.json');encoding['texts']=seal['files']['texts.json'];write(out/'encoding.json',encoding)
    return dict(pilot_queries=len(ids),metadata=record(directory/'pilot.json'))


def confirm(out,directory):
    r=read(out/'result.json');prep=out/'confirmation_prepare';p=read(prep/'pilot.json');data=read(out/'data.json')
    indices=np.arange(data['queries']) if r['winner_scope']=='full' else np.load(out/'sub_indices.npy');ds=Dataset(out,indices);n=len(ds.ids);m=len(p['ids'])
    pilotX=np.load(prep/'pilot_base.npy');pilotviews=np.load(prep/'pilot_views.npy')
    ds.X=np.concatenate([ds.X,pilotX]);ds.scene=np.concatenate([ds.scene,np.load(prep/'pilot_scene.npy')]);ds.views=np.concatenate([ds.views,pilotviews[:,:,[b.VIEWS.index('exact'),b.VIEWS.index('union')]]])
    pilot_queries=read(old.OWN/'full/pilot_queries.json')
    if [q['query_id'] for q in pilot_queries]!=p['ids']:raise ValueError('Pilot name row mapping changed')
    ds.groups=np.r_[ds.groups,p['groups']];ds.ids=np.r_[ds.ids,p['ids']];ds.names=np.r_[ds.names,[b.normalize_name(b.text(q['crm_name'])) for q in pilot_queries]]
    ds.codes=np.concatenate([ds.codes,np.load(prep/'naf.npy')]);ds.targets=np.concatenate([ds.targets,np.full((m,2),-1,dtype=np.int16)]);ds.addresses=np.concatenate([ds.addresses,np.load(prep/'addresses.npy')])
    # Ranking context has its own metadata and scoring target remains inaccessible by train indices.
    runner=Runner(ds,r['winner'],directory/'content_models')
    if 's2' in r['winner']:
        train_sem=np.load(out/'semantic.npy',mmap_mode='r')[indices];e=read(out/'confirmation_encode/result.json');vectors=np.load(verify(e['vectors']),mmap_mode='r');qi=np.load(prep/'qtext.npy');ci=np.load(prep/'ctext.npy');cos=np.full((m,100),np.nan,dtype=np.float32)
        for i in range(m):
            valid=np.flatnonzero(ci[i]>=0)
            if qi[i]>=0:cos[i,valid]=vectors[ci[i,valid]]@vectors[qi[i]]
        sem=np.stack([cos,relative(cos,np.load(prep/'addresses.npy'))],axis=2);ds.semantic=np.concatenate([train_sem,sem])
    pred=runner.score(np.arange(n),np.arange(n,n+m),'external_content526');SX=old.stack_cube(np.load(prep/'pilot_scene.npy'),pred)
    y=pilotviews[:,:,b.VIEWS.index('exact')].astype(np.float32);groups=np.array(p['groups']);ids=np.array(p['ids']);folds=np.array(p['folds']);outer=[(f,np.array(a),np.array(z)) for f,a,z in p['outer']]
    class ReplayedStack(old.StackRunner):
        def score(self,train,test,columns,purpose):
            value=super().score(train,test,columns,purpose)
            if self.fixed is None:
                model,receipt=self.models[d.digest(np.sort(train).tolist())];replay=xgb.XGBRanker(n_jobs=1);replay.load_model(verify(receipt['model']))
                actual=replay.predict(self.X[test].reshape(-1,len(old.STACK))).reshape(len(test),100)
                if not np.array_equal(value,actual):raise ValueError('Stack serialized replay mismatch')
                receipt['uses'][-1]['replay_equal']=True
            return value
    stack=ReplayedStack(SX,y,groups,ids,directory/'stack_models');stacked=np.full_like(pred,np.nan)
    for f,a,z in outer:stacked[z]=stack.score(a,z,list(range(len(old.STACK))),f'outer{f}')
    fixed=ReplayedStack(SX,y,groups,ids,directory/'fixed_content',fixed=pred);confidence={}
    for name,scorer in [('content',fixed),('stack',stack)]:
        confidence[name],common=old.v2.confidence(scorer,outer,np.load(prep/'pilot_scene.npy'),pilotviews,groups,folds,ids,list(range(len(old.STACK))),directory,name)
        published=b.index(confidence[name]['per_query'])
        for modelref,role in zip(confidence[name]['model_refs'],confidence[name]['receipts']):
            with verify(modelref).open('rb') as stream:classifier,iso=pickle.load(stream)
            test=np.array([p['ids'].index(q) for q in role['test_queries']])
            replay=iso.predict(d.probability(classifier,d.scene(common[test].astype(np.float32),np.load(prep/'pilot_scene.npy')[test])))
            expected=np.array([published[str(ids[i])]['scores']['scene_isotonic'] for i in test])
            if not np.array_equal(replay,expected):raise ValueError('Pilot confidence serialized replay mismatch')
            role['serialized_replay_equal']=True
    result=dict(status='DESCRIPTIVE_ALREADY_CONSUMED526',queries=m,recipe=r['winner'],training_queries=n,
        content={v:b.metric(int(pilotviews[np.arange(m),b.select_top(pred)[0],j].sum()),m) for j,v in enumerate(b.VIEWS)},
        stack={v:b.metric(int(pilotviews[np.arange(m),b.select_top(stacked)[0],j].sum()),m) for j,v in enumerate(b.VIEWS)},
        confidence=confidence,content_models=runner.receipts,stack_models=stack.receipts,production_allowed=False)
    np.savez_compressed(directory/'predictions.npz',content=pred,stack=stacked);result['predictions']=record(directory/'predictions.npz');write(directory/'result.json',result)
    r['confirmation']=result;r['status']='COMPLETE_WITH_DESCRIPTIVE526_CONFIRMATION';write(out/'result.json',r);return dict(result=record(directory/'result.json'))


def report(r):
    def pct(m):
        if m is None:return 'non définie'
        a,z=m['wilson95_percent'];return f'{m["hits"]}/{m["queries"]}={m["percent"]:.2f}% [{a:.2f};{z:.2f}]'
    lines=['## Résultats','',f'**{r["status"]}** — développement, aucune promotion.',f'Sous-banc{r["queries"]} dossiers/{r["companies"]} entreprises ; sélection budget :{r["selection"]}.','',
        '| Recette | Exact, Wilson95 | UNION, Wilson95 | Δ UNION, IC95 entreprise |','|---|---|---|---|']
    for n,v in r['variants'].items():
        dd=r['deltas'].get(n);delta='—' if dd is None else f'{dd["union"]["delta_pp"]:+.3f}pt {dd["union"]["paired_company_bootstrap95_pp"]}'
        lines.append(f'| {n} | {pct(v["top1"]["exact"])} | {pct(v["top1"]["union"])} | {delta} |')
    lines+=['','### SUR : cible interne UNION≥98%, pas une garantie test','| Recette / méthode | Précision UNION, Wilson95 | Couverture, Wilson95 |','|---|---|---|']
    for n,v in r['variants'].items():
        for policy,s in v['confidence']['policies'].items():lines.append(f'| {n}/{policy} | {pct(s["precision"]["union"])} | {pct(s["coverage"])} |')
    lines+=['',f'### Famille figée `c37a1a7b` :{r["family"]["count"]} cas',f'Base :{r["family"]["base"]}. Deltas/gains ET pertes :{r["family"]["deltas"]}.',
        'Aucune sélection des lignes de fit sur cette famille ; sa définition historique n’est pas recomputée pour favoriser les variantes.','',
        f'### S1 complet17427 (secondaire)\n\n{r["full_s1"]["base"] if r["full_s1"] else "Smoke seulement"}',]
    if r['full_s1']:
        ff=r['full_s1'];lines += [f'Top1 :{ff["variant"]["top1"]}. Deltas :{ff["deltas"]}. Famille :{ff["family_deltas"]}.',f'SUR :{ff["variant"]["confidence"]["policies"]}.']
    lines += ['', '### Encodage BASE local',json.dumps(r['encoding'],ensure_ascii=False), '', '### Confirmation conditionnelle526',
        'Non déclenchée : aucun IC95 UNION positif.' if not r['winner'] else f'Recette{r["winner"]}, scope{r["winner_scope"]}. '+('En attente.' if r['confirmation'] is None else f'Content{r["confirmation"]["content"]}, stack{r["confirmation"]["stack"]} ; précision et couverture SUR dans JSON.'),
        '', '### Preuve et limites',
        'SHA sources/modèle/code/protocole avant/après ; chaque vecteur ranking, NB et scène/isotonique rejoué depuis modèle sérialisé ; zéro entreprise commune entre fit/scoring. S1 est recrossfitté DANS chaque fit ranking, y compris proper/meta, pas seulement dans le CV externe.',
        'Labels exacts et union contractuelle inchangés, absences conservées ; ID/groupes uniquement pour jointures/plis. Aucun score sur jeu fermé, alias ajouté, retrieval, web, VM contrôlée ou push. Wilson descriptif ; bootstrap entreprise10000, IC simultanés dans JSON ; décision de confirmation sur IC ordinaire exploratoire, pas certification.',
        'Libellés NAF officiels INSEE locaux retrouvés, sans web ; anciennes nomenclatures/missing marquées et aucune ligne éliminée. BASE sans adapter, aucun ancien vecteur adapté réutilisé. Éventuelles troncatures512 et budget réellement observé publiés.',
        'Reproduction : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_colocated_signals.py run --smoke --output-dir <namespace_neuf>` puis sans`--smoke`, `--commit` pour publication isolée. Namespace sous`H/step2_colocated_signals_20261019/` ; logs `*/worker.log`, `result.json`, `publication.json`.']
    return '\n'.join(lines)+'\n'


def final_report(r):
    def delta(d):
        lo,hi=d['paired_company_bootstrap95_pp']
        return f'{d["delta_pp"]:+.2f} [{lo:+.2f};{hi:+.2f}]'
    def sur(s):
        p=s['precision']['union'];return ('indéf.' if p is None else f'{p["percent"]:.2f}')+f' / {s["coverage"]["percent"]:.2f}'
    lines=['## Fait',f'Banc terminé :{r["queries"]} dossiers/{r["companies"]} entreprises ; code et docs`step2_colocated_signals`, journal`handover.md`.',
        '## Preuve',f'Deltas en points, IC95 bootstrap10000 entreprises ; famille même adresse figée n={r["family"]["count"]}.',
        '| Recette | Exact % | UNION % | Δ exact [IC] | Δ UNION [IC] | Δ famille UNION [IC] |', '|---|---:|---:|---|---|---|']
    for n,v in r['variants'].items():
        dlt=r['deltas'].get(n);fd=r['family']['deltas'].get(n) if r['family']['deltas'] else None
        lines.append(f'| {n} | {v["top1"]["exact"]["percent"]:.2f} | {v["top1"]["union"]["percent"]:.2f} | '+
            ('— | — | — |' if dlt is None else f'{delta(dlt["exact"])} | {delta(dlt["union"])} | {delta(fd["union"]) if fd else "indéf."} |'))
    lines+=['| SUR | Marge : précision UNION / couverture % | Scène+iso : précision UNION / couverture % |','|---|---:|---:|']
    for n,v in r['variants'].items():
        s=v['confidence']['policies'];lines.append(f'| {n} | {sur(s["margin"])} | {sur(s["scene_isotonic"])} |')
    f=r['full_s1']
    if f:
        lines += [f'S1 sur17427 : exact{f["base"]["exact"]["hits"]}→{f["variant"]["top1"]["exact"]["hits"]}, UNION{f["base"]["union"]["hits"]}→{f["variant"]["top1"]["union"]["hits"]}.',
            f'Δ exact{delta(f["deltas"]["s1"]["exact"])}, UNION{delta(f["deltas"]["s1"]["union"])} ; famille1046 :{delta(f["family_deltas"]["s1"]["union"])}.',
            'S1 complet SUR P/C : marge'+sur(f['variant']['confidence']['policies']['margin'])+' ; scène+iso'+sur(f['variant']['confidence']['policies']['scene_isotonic'])+'.']
    c=r['confirmation']
    lines += [f'Confirmation526 descriptive ({r["winner"]}) : content exact/UNION{c["content"]["exact"]["hits"]}/{c["content"]["union"]["hits"]}, stack{c["stack"]["exact"]["hits"]}/{c["stack"]["union"]["hits"]}.' if c else 'Confirmation526 non déclenchée : aucun IC95 UNION positif.',
        f'S2 :{r["encoding"]["full"]["items"]} textes déjà encodés, non réencodés lors de la reprise ; reçus des4 recettes réutilisés.',
        '## Écarts / choix','Correction mémoire seulement : copies temporaires libérées, plis en processus neufs ; protocole, paramètres, données et labels inchangés.',
        '## Reste / risques','Développement sur jeux consommés ;98% est la cible interne du seuil, pas la précision test garantie. Wilson/IC simultanés dans JSON ; aucune promotion.',
        '**Recommandation unique :** '+('Conserver '+str(r['winner'])+' pour audit indépendant, sans promotion sur ces jeux consommés.' if r['winner'] else 'Ne pas promouvoir ces signaux sans gain UNION confirmé.')]
    if len(lines)>30:raise ValueError('Final report over30 lines')
    return '\n'.join(lines)+'\n'


def publish(out,directory):
    check(out);r=read(out/'result.json')
    if (out/'resume.json').exists():
        r['resumption']=read(out/'resume.json');r['resumption_verification']=read(out/'resume_verification.json')
        r['failed_attempt_peak_bytes']=read(verify(r['resumption']['failure']))['peak']
        r['elapsed_wall_includes_interruption']=True
    if r['smoke']:raise ValueError('Do not publish smoke')
    for ref in r['source_refs']:verify(ref)
    for ref in read(out/'data.json')['sources']:verify(ref)
    r['peak_rss_plus_driver_bytes']=max([read(p).get('peak_rss_bytes',0) for p in out.glob('*/receipt.json')]+[r['encoding']['full']['peak_rss_plus_driver_bytes'],r['encoding']['probe']['peak_rss_plus_driver_bytes']])
    r['seconds']=time.time()-read(out/'plan.json')['started_unix'];write(out/'result.json',r)
    (out/'result.md').write_text(report(r));(out/'final_report.md').write_text(final_report(r))
    if subprocess.check_output(['git','diff','HEAD','--','handover.md'],cwd=ROOT).strip():raise ValueError('Foreign dirty journal; preserve and stop automatic publication')
    for path in ('docs/step2_colocated_signals.json','docs/step2_colocated_signals.md'):
        if subprocess.check_output(['git','diff','HEAD','--',path],cwd=ROOT).strip():raise ValueError('Foreign dirty owned doc; preserve and stop')
    journal=ROOT/'handover.md';before=journal.read_text();write(ROOT/'docs/step2_colocated_signals.json',r)
    DOC.write_text((out/'preregistered.md').read_text()+'\n'+report(r))
    values=';'.join(f'{n} exact{v["top1"]["exact"]["hits"]}/union{v["top1"]["union"]["hits"]}' for n,v in r['variants'].items())
    line=f'\n- 2026-10-19 — Étape2-8 COMPLETE :{r["queries"]} G0,{values}, S1 complet17427 dans JSON ; NAF NB crossfit dans chaque ranking fit, BASE sans adapter, sous-banc3000/partitionG0 conservée (budget réduction{r["selection"]["reduced"]}), bootstrap entreprise10000/famille figée/SUR union98 interne. Confirmation526{bool(r["confirmation"])} ; aucune promotion/label/pool/jeux fermés/v174/web/VM/push. SHA/rejeux/modèles et rapport≤30 lignes `{out}/`. Pic RSS+driver{r["peak_rss_plus_driver_bytes"]/1024**3:.2f}Gio. Commit résultats auto-référencé par `git log -1 --format=%H -- docs/step2_colocated_signals.json`.\n'
    with journal.open('a') as stream:stream.write(line)
    if journal.read_text()!=before+line:raise ValueError('Concurrent journal edit; preserve and stop')
    paths=['docs/step2_colocated_signals.json','docs/step2_colocated_signals.md','handover.md']
    subprocess.run(['git','add','--',paths[0]],cwd=ROOT,check=True);subprocess.run(['git','diff','--check','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','feat: measure crossfit activity and base semantic colocated signals','--',*paths],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',paths[0]],cwd=ROOT,text=True).strip();write(out/'publication.json',dict(status='VERIFIED_COMMITTED',commit=commit))
    return dict(commit=commit)


def execute(out,phase,directory,extra=(),metal=False):
    wait_resources();directory.mkdir()
    python=HOME/'runtime/bin/python' if metal else Path(sys.executable)
    with (directory/'worker.log').open('x') as log:
        subprocess.run([str(python),'-u',str(Path(__file__).resolve()),phase,'--output-dir',str(out),'--directory',str(directory),*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(stage=phase,directory=directory.name,receipt=record(directory/('result.json' if metal else 'receipt.json')))),flush=True)


def verify_smoke(out):
    original=OWN/'smoke'
    if original==out or not (original/'result.json').exists():return
    a=read(original/'result.json');z=read(out/'result.json');data=read(out/'data.json');groups=np.array(data['groups'])[np.load(out/'sub_indices.npy')]
    if (a['queries'],a['companies'])!=(z['queries'],z['companies']):raise ValueError('Smoke cohort drift')
    rows=[];uses=nbuses=0
    for n in VARIANTS:
        av=np.load(verify(a['variants'][n]['predictions']));zv=np.load(verify(z['variants'][n]['predictions']))
        identical={k:bool(np.array_equal(av[k],zv[k])) for k in av.files}
        if n in ('base','s1') and not all(identical.values()):raise ValueError('Baseline/S1 real smoke reproduction mismatch')
        for model in z['variants'][n]['models']:
            verify(model['model']);train=np.array(model['train_indices'])
            for u in model['uses']:
                if set(groups[train])&set(groups[u['scored_indices']]):raise ValueError('Smoke ranking company overlap')
                if not u['replay_equal']:raise ValueError('Replay absent')
                uses+=len(u['scored_indices'])
            for u in model['nb_training']+[u['nb_scoring'] for u in model['uses'] if u['nb_scoring']]:
                verify(u['model'])
                if set(groups[u['train_indices']])&set(groups[u['scored_indices']]):raise ValueError('Smoke NB company overlap')
                if not u['serialized_replay_equal']:raise ValueError('NB replay absent')
                nbuses+=1
        rows.append(dict(recipe=n,arrays_equal=identical,fits=z['variants'][n]['fit_count']))
    proof=dict(passed=True,queries=z['queries'],companies=z['companies'],fit_count=sum(v['fit_count'] for v in z['variants'].values()),ranking_vectors_replayed=uses,
        nb_serialized_company_disjoint_uses=nbuses,recipes=rows,semantic_note='Candidate texts now explicitly sorted; any S2 difference vs initial smoke is disclosed, no tuning.',
        probe_texts=z['encoding']['probe']['items'],encoded_texts=z['encoding']['full']['items'],sources_unchanged=True)
    write(out/'verification.json',proof);print(json.dumps(proof),flush=True)


def resume_run(out,commit):
    if (out/'resume.json').exists():raise ValueError('Resume already established; inspect existing receipt before another launch')
    plan=read(out/'plan.json');snapshot=verify(plan['implementation_snapshot'])
    for ref in plan['implementations']+[plan['preregistration']]:
        if ref['path']==str(Path(__file__).resolve()):
            if record(snapshot)['sha256']!=ref['sha256']:raise ValueError('Original script snapshot mismatch')
        else:verify(ref)
    if plan['smoke']:raise ValueError('Full resume requires full population')
    logged={}
    for line in (out.parent/'full.log').read_text().splitlines():
        if line.startswith('{'):
            item=json.loads(line)
            if 'receipt' in item:logged[item['directory']]=item['receipt']
    reused=[]
    for name in ('preparation','probe','selection','encode','semantic')+VARIANTS:
        ref=logged[name];payload=read(verify(ref));reused.append(ref)
        if name in VARIANTS:
            result=read(verify(payload['result']));verify(result['predictions'])
            if result['params']!=plan['params'] or result['fit_count']!=25:raise ValueError('Existing variant protocol drift')
            for model in result['models']:
                verify(model['model'])
                for use in model['uses']:
                    if not use['replay_equal']:raise ValueError('Existing ranker replay missing')
                for nb in model['nb_training']+[u['nb_scoring'] for u in model['uses'] if u['nb_scoring']]:verify(nb['model'])
            for model in result['confidence']['models']:verify(model)
    encoded=read(out/'encode/result.json');reused.extend([encoded['vectors'],encoded['texts'],record(out/'semantic.npy'),record(out/'data.json'),record(out/'selection.json')])
    for ref in reused:verify(ref)
    failed=out/'full_s1';stop=read(failed/'resource_stop.json')
    if stop['status']!='STOP_RSS_PLUS_DRIVER_5_75GIB' or (failed/'receipt.json').exists():raise ValueError('Unexpected failure; preserve and stop')
    archive=out/'full_s1.failed_memory';failed.rename(archive)
    (out/'resume_implementation.py').write_text(Path(__file__).read_text())
    manifest=dict(status='RESUMING_MEMORY_PLUMBING_ONLY',started_unix=time.time(),original_plan=record(out/'plan.json'),implementation=record(Path(__file__)),
        implementation_snapshot=record(out/'resume_implementation.py'),reused=reused,failure=record(archive/'resource_stop.json'),
        original_log=record(out.parent/'full.log'),encoding_repeated=False,protocol_changed=False,fold_process_isolation=True)
    write(out/'resume.json',manifest);check(out)
    for cv in range(5):execute(out,'_full_fold',out/f'full_s1_cv{cv}',['--cv',str(cv)])
    execute(out,'_combine_full',out/'full_s1');execute(out,'_summarize',out/'summary');r=read(out/'result.json')
    if r['winner']:
        execute(out,'_confirm_prepare',out/'confirmation_prepare')
        if 's2' in r['winner']:execute(out,'_encode',out/'confirmation_encode',metal=True)
        execute(out,'_confirm',out/'confirmation')
    for ref in reused:verify(ref)
    write(out/'resume_verification.json',dict(passed=True,reused_files_unchanged=len(reused),encoding_repeated=False,full_fits=25,
        folds=5,guard_gib=5.75,protocol_changed=False,resume_manifest=record(out/'resume.json')))
    if commit:execute(out,'_publish',out/'publication')
    print(json.dumps(dict(stage='RESUME_COMPLETE',output=str(out))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','resume','_prepare','_select','_semantic','_variant','_full_fold','_combine_full','_summarize','_confirm_prepare','_confirm','_publish']);p.add_argument('--output-dir',type=Path);p.add_argument('--directory',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--variant',choices=VARIANTS);p.add_argument('--full',action='store_true');p.add_argument('--cv',type=int,choices=range(5));p.add_argument('--commit',action='store_true')
    args=p.parse_args();out=args.output_dir or OWN/('smoke' if args.smoke else 'full')
    if not out.resolve().is_relative_to(OWN) or out.resolve()==OWN:raise ValueError('Fresh owned namespace required')
    if args.phase=='resume':
        resume_run(out,args.commit);return
    if args.phase=='run':
        out.mkdir(parents=True,exist_ok=False);prereg=DOC.read_text().split('\n## Résultats\n')[0]+'\n';(out/'preregistered.md').write_text(prereg)
        (out/'implementation.py').write_text(Path(__file__).read_text())
        write(out/'plan.json',dict(smoke=args.smoke,started_unix=time.time(),preregistration=record(out/'preregistered.md'),params=old.PARAMS,
            implementation_snapshot=record(out/'implementation.py'),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            versions=dict(python=sys.version,xgboost=xgb.__version__,sklearn=b.sklearn.__version__,numpy=np.__version__),
            implementations=[record(Path(__file__).resolve()),record(Path(bench.__file__)),record(Path(old.__file__)),record(Path(b.__file__)),record(Path(d.__file__)),record(Path(old.v2.__file__))]))
        execute(out,'_prepare',out/'preparation');execute(out,'_probe',out/'probe',metal=True);execute(out,'_select',out/'selection');execute(out,'_encode',out/'encode',metal=True);execute(out,'_semantic',out/'semantic')
        for n in VARIANTS:execute(out,'_variant',out/n,['--variant',n])
        if not args.smoke:
            for cv in range(5):execute(out,'_full_fold',out/f'full_s1_cv{cv}',['--cv',str(cv)])
            execute(out,'_combine_full',out/'full_s1')
        execute(out,'_summarize',out/'summary');r=read(out/'result.json')
        if r['winner'] and not args.smoke:
            execute(out,'_confirm_prepare',out/'confirmation_prepare')
            if 's2' in r['winner']:execute(out,'_encode',out/'confirmation_encode',metal=True)
            execute(out,'_confirm',out/'confirmation')
        if args.commit and not args.smoke:execute(out,'_publish',out/'publication')
        else:(out/'result.md').write_text(report(read(out/'result.json')));(out/'final_report.md').write_text(final_report(read(out/'result.json')))
        if args.smoke:verify_smoke(out)
        print(json.dumps(dict(stage='COMPLETE',output=str(out))),flush=True);return
    if args.directory is None or args.directory.parent!=out:raise ValueError('Owned worker namespace required')
    wait_resources();guard=Guard(args.directory);began=time.monotonic()
    if args.phase=='_prepare':value=prepare(out,args.directory)
    elif args.phase=='_select':value=select_subset(out,args.directory)
    elif args.phase=='_semantic':value=semantic_features(out,args.directory)
    elif args.phase=='_variant':value=variant_worker(out,args.directory,args.variant,args.full)
    elif args.phase=='_full_fold':value=full_fold(out,args.directory,args.cv)
    elif args.phase=='_combine_full':value=combine_full(out,args.directory)
    elif args.phase=='_summarize':value=summarize(out,args.directory)
    elif args.phase=='_confirm_prepare':value=confirm_prepare(out,args.directory)
    elif args.phase=='_confirm':value=confirm(out,args.directory)
    else:value=publish(out,args.directory)
    value.update(seconds=time.monotonic()-began,peak_rss_bytes=guard.peak);write(args.directory/'receipt.json',value);guard.stop.set()


if __name__=='__main__':main()
