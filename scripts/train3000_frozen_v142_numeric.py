"""Query-only V14/V18/V19 -> V23/V32/V57/V68/V71/V75 reconstruction.

Frozen query-independent official document/vector banks may be read, never
query score matrices. All new scores and stage files live below owned v174.
"""
import ast
from collections import defaultdict, OrderedDict
import gc
import hashlib
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import textwrap
import time

import numpy as np
from scripts import train3000_frozen_v142_fidelity as mini
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v18_identity_retrieval as identity
from scripts import embeddinggemma2_v19_proof_retrieval as own
from scripts import embeddinggemma2_v21_relation_views as relations
from scripts import embeddinggemma2_v23_native_features as proof
from scripts import embeddinggemma2_v32_native_features as base
from scripts import embeddinggemma2_v56_corporate_site as corporate
from scripts import embeddinggemma2_v57_corporate_site_nullsafe as nullsafe
from scripts import embeddinggemma2_v68_filled_site_geo as filled
from scripts import embeddinggemma2_v71_frozen_residual as residual
from scripts import embeddinggemma2_v75_published_supplier_context as supplier

ATOL = 1e-7  # Original V19 replay bound; fixed before reconstruction.
FROZEN = {r['path']:r for r in mini.read(mini.bano.cohort.preflight.FROZEN)['artifacts']}
BODIES = {}
AUTHORITIES = {}
SOURCE_ANCHOR_COMMIT = '36c7a476'


def pinned(module):
    path = str(Path(module.__file__).resolve())
    if path in FROZEN:
        ref=FROZEN[path]
    else:
        # Upstream V18/V19/V57 code was not enumerated by the file inventory;
        # bind its exact bytes to the frozen V142 Git snapshot, not current HEAD.
        relative=Path(path).relative_to(mini.ROOT)
        raw=subprocess.check_output(['git','show',SOURCE_ANCHOR_COMMIT+':'+str(relative)],cwd=mini.ROOT)
        ref=dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),git_authority=SOURCE_ANCHOR_COMMIT)
    mini.verified(ref);AUTHORITIES[path]=ref
    return Path(path).read_text()


def slice_body(module, start, end, signature, ending, overrides=None):
    text = pinned(module)
    if text.count(start)!=1 or text.count(end)!=1:
        raise ValueError('STOP_NUMERIC_CONSTRUCTOR_NOT_FOUND: '+str(module.__file__))
    body = text[text.index(start):text.index(end)]
    namespace = dict(vars(module));namespace.update(overrides or {})
    exec(compile('def build('+signature+'):\n'+textwrap.indent(textwrap.dedent(body),'    ')+ending,
        str(module.__file__)+':unchanged_scientific_body','exec'),namespace)
    BODIES[str(module.__file__)+':'+start.strip()] = hashlib.sha256(body.encode()).hexdigest()
    return namespace['build']


def install_io(directory):
    original_write = engine.io.write
    def write(path,value):
        path = Path(path)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('STOP_NUMERIC_WRITE_OUTSIDE_OWNED_V174: '+str(path))
        if path.name in ('worker_state.json','state.json'):
            path = path.with_name(path.stem+'_'+str(time.time_ns())+'.json')
        mini.save(path,value)
    engine.io.write = write
    engine.guard = mini.guard
    for module in (proof,corporate,filled,residual,supplier):
        module.guard = lambda *args: mini.guard()
    return original_write


def source_geometry(panel, directory, source, launch):
    import duckdb
    from src.xgb_matcher import retrieval_query_ban_v327 as ban
    import inspect
    pinned(ban);pinned(inspect.getmodule(ban._resolve_ban));pinned(proof.structural)
    class Files:
        def file(self, ref):
            if ref != source['geographic']['ban']:
                raise ValueError('Only the original BAN source')
            return mini.verified(ref)
    con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='384MB'")
    tmp=directory/'ban_spill';tmp.mkdir();con.execute('SET temp_directory=?',[str(tmp)])
    try: resolved=ban.resolve(con,panel,Files())
    finally: con.close()
    original={q['query_id']:dict(geometry=g) for q,g in zip(panel,resolved)}
    geometries=proof.structural.fallback_geometries(panel,original,source,directory)
    path=directory/'geometries.json'
    if not path.exists(): mini.save(path,geometries)
    # V21 consumes V17's geometry envelope, not geographic top100 candidates.
    geo_path=directory/'geometry_rows.json'
    mini.save(geo_path,[dict(query_id=q['query_id'],geometry=geometries[q['query_id']]) for q in panel])
    adapted=dict(launch,geo=mini.record(geo_path))
    geo=relations.geometry_sources(panel,adapted,source,directory)
    return geometries,geo,mini.record(geo_path)


def corpora(panel, directory, source, cache):
    """Original V14 complete commune extraction and original SITE banks."""
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
    backend=TantivyBackend(engine.io.INDEX)
    if backend.searcher.num_docs!=82772691:
        raise ValueError('Frozen national index document count')
    zones=[]
    for zone in sorted({q['crm_insee'] for q in panel}):
        hits=backend.searcher.search(engine.old.zone_query(backend,zone),limit=1,count=True)
        zones.append(dict(zone=zone,indexed_sites=hits.count))
    del backend;gc.collect()
    mini.save(directory/'census.json',dict(zones=zones))
    panel_path=directory/'panel.json';mini.save(panel_path,panel)
    names=mini.verified(cache['names'])
    nd=directory/'names';nd.mkdir()
    os.link(names,nd/'owners.sqlite')
    engine.seal(nd/'owners.sqlite')
    mini.save(nd/'generation.json',dict(database=mini.record(names),mtime_ns=names.stat().st_mtime_ns))
    adapted=dict(source,panel=mini.record(panel_path),census=mini.record(directory/'census.json'),
        prepared_corpora=cache['corpora'])
    mini.save(directory/'launch.json',adapted)
    original_authority=engine.authority
    engine.authority=lambda out:adapted if Path(out)==directory else original_authority(out)
    refs={};banks={}
    for z in zones:
        mini.guard();zone=z['zone'];engine.prepare_zone(directory,zone)
        dest=engine.zone_path(directory,zone);refs[zone]=mini.record(dest/'documents.json')
        old_zone=engine.zone_path(own.SOURCE,zone)
        historical=old_zone/'v3_official512'
        if not historical.exists() and (old_zone/'v3_official512.seal.json').exists():
            historical=Path(mini.read(old_zone/'v3_official512.seal.json')['path']).parent/'v3_official512'
        if historical.exists():
            # No original per-query outputs, only model/text-bound SITE vectors.
            banks[zone]=historical
        else:
            engine.dense_zone(directory,zone,'v3_official512')
            banks[zone]=dest/'v3_official512'
    return refs,banks


class IdentityBank:
    """Original identities are immutable documents, not frozen query scores."""
    def __init__(self, launch, directory, inference, tokenizer, parents):
        self.old=sqlite3.connect(f"file:{mini.verified(launch['identity_database'])}?mode=ro&immutable=1",uri=True)
        self.cache=identity.VectorCache(identity.OUTPUT/'identity_vectors')
        self.offset=self.old.execute('SELECT max(id) FROM texts').fetchone()[0]
        self.new=sqlite3.connect(directory/'new_identities.sqlite')
        self.new.executescript('''CREATE TABLE texts(id INTEGER PRIMARY KEY,text TEXT UNIQUE NOT NULL);
            CREATE TABLE views(parent INTEGER,ordinal INTEGER,text_id INTEGER,PRIMARY KEY(parent,ordinal));
            CREATE TABLE parents(parent INTEGER PRIMARY KEY);''')
        names=sqlite3.connect(f"file:{mini.verified(launch['names'])}?mode=ro&immutable=1",uri=True)
        missing=[p for p in sorted(parents) if not self.old.execute('SELECT 1 FROM views WHERE parent=? LIMIT 1',(p,)).fetchone()]
        # Exact original per-parent V18 packing, outside historical fixed295 scope.
        pack=slice_body(identity,"                legal=[r[0] for r in names.execute(",
            "                if done%256==0:", 'p,names,db,tokenizer,done=0', '\n    return done\n',
            dict(data=own.data))
        for p in missing: mini.guard();pack(p,names,self.new,tokenizer)
        self.new.commit();names.close();self.fresh={};self.global_ids={};self.loaded=OrderedDict()
        chunk=[];encoded=0
        def encode_chunk():
            nonlocal encoded
            mini.guard();values=inference([text for _,text in chunk])
            path=directory/f'new_identity_vectors_{encoded:08d}.npy'
            np.save(path,values,allow_pickle=False)
            for row,(idx,_) in enumerate(chunk):self.fresh[self.offset+idx-1]=(path,row)
            encoded+=len(chunk);chunk.clear()
        for idx,text in self.new.execute('SELECT id,text FROM texts ORDER BY id'):
            existing=self.old.execute('SELECT id FROM texts WHERE text=?',(text,)).fetchone()
            # Exact text/model-bound reuse within the PARENT bank, never SITE.
            self.global_ids[idx]=existing[0] if existing else self.offset+idx
            if not existing:chunk.append((idx,text))
            if len(chunk)==128:encode_chunk()
        if chunk:encode_chunk()
    def execute(self,sql,args):
        rows=self.old.execute(sql,args).fetchall()
        if rows:return rows
        return [(self.global_ids[r[0]],) for r in self.new.execute(sql,args)]
    def get(self,idx):
        if idx<self.offset:return self.cache.get(idx)
        path,row=self.fresh[idx]
        if path not in self.loaded:
            self.loaded[path]=np.load(path,mmap_mode='r',allow_pickle=False)
            if len(self.loaded)>32:
                _,matrix=self.loaded.popitem(last=False);matrix._mmap.close()
        self.loaded.move_to_end(path)
        return self.loaded[path][row].copy()
    def close(self):
        self.old.close();self.new.close();self.cache.close()
        for matrix in self.loaded.values():matrix._mmap.close()
        self.loaded.clear()


def own_scores(panel, directory, source, cache_launch, refs, banks, geometries, encoder_panel):
    import torch,transformers
    from scripts import embeddinggemma2_v3_fit as fit
    if torch.__version__!='2.14.1' or transformers.__version__!='5.19.0':
        raise ValueError('Original Gemma runtime required')
    cfg=mini.read(mini.verified(source['night_config']))
    mini.guard()
    model,params,encode,inference,tokenizer=fit.load_model(cfg,mini.verified(cache_launch['adapter']))
    keys=np.load(mini.verified(source['geographic']['keys']),mmap_mode='r',allow_pickle=False)
    xy=np.load(mini.verified(source['geographic']['xy']),mmap_mode='r',allow_pickle=False)
    names_db=sqlite3.connect(f"file:{mini.verified(cache_launch['names'])}?mode=ro&immutable=1",uri=True)
    build=slice_body(own,'                views=[]\n','                for q,a,b,c in zip(qs,best,site_replay,joint_replay):',
        'docs,qs,site_cache,dest,inference,tokenizer,db,cache,names_db,keys,xy,geometries,out,zone,count=0',
        '\n    return features,matrices\n',dict(np=np,old=engine.old))
    output={}
    try:
        for zone,ref in sorted(refs.items()):
            mini.guard();docs=mini.read(mini.verified(ref));qs=[q for q in encoder_panel if q['crm_insee']==zone]
            dest=directory/'zones'/(zone or 'UNKNOWN');dest.mkdir(parents=True)
            if not docs:
                for q in qs:output[q['query_id']]=(np.array([],dtype=np.uint64),np.zeros((0,5),dtype=np.float32))
                continue
            bank=IdentityBank(cache_launch,dest,inference,tokenizer,{int(d['siren']) for d in docs})
            try:
                features,matrices=build(docs,qs,banks[zone],dest,inference,tokenizer,bank,bank,names_db,
                    keys,xy,geometries,directory,zone)
                own_keys=np.array([d['key'] for d in docs],dtype=np.uint64)
                for i,q in enumerate(qs):
                    if q['query_id'] not in {r['query_id'] for r in panel}:continue
                    output[q['query_id']]=(own_keys,np.column_stack([features[f][i] for f in ('site','identity','name','address','street')]))
                for value in features.values():value.flush();value._mmap.close()
                for value in matrices:value._mmap.close()
            finally:bank.close()
            del docs;gc.collect()
    finally:names_db.close();keys._mmap.close();xy._mmap.close()
    del model,params,encode,inference,tokenizer
    gc.collect();torch.mps.empty_cache()
    return output


def reconstruct(panel, directory, encoder_panel=None):
    if encoder_panel is None:encoder_panel=panel
    if not panel or any(set(q)!=set(mini.bano.cohort.old.CRM) or q['fold'] not in (2,3,4) for q in panel):
        raise ValueError('Query-only CRM8/folds234 required')
    directory.mkdir(parents=True)
    for module in (engine,identity,own,relations,proof,base,corporate,nullsafe,filled,residual,supplier):pinned(module)
    launch=mini.read(supplier.OUTPUT/'launch.json')
    cache=mini.read(mini.verified(launch['feature_launch']))
    source=mini.read(mini.verified(cache['source_launch']))
    original_io=install_io(directory)
    from scripts import embeddinggemma2_v3_fit as fit
    for module in (fit,own.data,engine.old):pinned(module)
    original_loader=fit.load_model;model_cache={}
    def load_once(cfg,initial):
        key=(str(Path(initial).resolve()),mini.canonical(cfg))
        if model_cache and key not in model_cache:raise ValueError('Only the one original Gemma model')
        if key not in model_cache:model_cache[key]=original_loader(cfg,initial)
        return model_cache[key]
    fit.load_model=load_once
    mini.save(directory/'panel.json',panel)
    if any(set(q)!=set(mini.bano.cohort.old.CRM) or q['fold'] not in (2,3,4) for q in encoder_panel):
        raise ValueError('Encoder padding companions must also be query-only/folds234')
    launch=dict(launch,panel=mini.record(directory/'panel.json'),deadline_unix=time.time()+8*3600)
    try:
        geo_dir=directory/'geometry';geo_dir.mkdir()
        geometries,geo,geometry_ref=source_geometry(encoder_panel,geo_dir,source,cache)
        geo={q['query_id']:geo[q['query_id']] for q in panel}
        base_dir=directory/'v14';base_dir.mkdir()
        refs,banks=corpora(panel,base_dir,source,cache)
        own_dir=directory/'v19';own_dir.mkdir()
        scores=own_scores(panel,own_dir,source,cache,refs,banks,geometries,encoder_panel)
        model_cache.clear();gc.collect()
        import torch
        torch.mps.empty_cache()
        proof_dir=directory/'v23';proof_dir.mkdir();(proof_dir/'features').mkdir()
        # Original family-link compiler consumes bindings of all full own leaves.
        for zone,ref in refs.items():
            z=own_dir/'zones'/(zone or 'UNKNOWN');z.mkdir(parents=True,exist_ok=True)
            qs=[q for q in panel if q['crm_insee']==zone]
            ks=scores[qs[0]['query_id']][0]
            mini.save(z/'features_binding.json',dict(sirets=[str(int(k)).zfill(14) for k in ks]))
            engine.seal(z/'features_binding.json')
        old_cache=proof.CACHE;proof.CACHE=own_dir
        try: links=proof.compile_family_links(proof_dir,launch,dict(corpora=refs),geo)
        finally:proof.CACHE=old_cache
        edges=proof.EdgeViews(launch,links)
        names=sqlite3.connect(f"file:{mini.verified(launch['names'])}?mode=ro&immutable=1",uri=True)
        stage_refs={}
        try:
            for q in panel:
                mini.guard();qid=q['query_id'];ks,values=scores.pop(qid)
                possible=set(map(int,ks//100000));possible.update(int(k)//100000 for k,s in geo[qid])
                family=proof.expand_families(edges.keys,possible);edge=edges.select(family)
                possible.update(map(int,edges.targets[edge]//100000))
                edges.portfolios,edges.legal_names,edges.own_names=proof.load_name_roles(names,sorted(possible))
                keys,x=proof.make_features(ks,values,geo[qid],geometries[qid],edges,q['crm_name'])
                path=proof.feature_path(proof_dir,qid);np.savez(path,keys=keys,features=x);engine.seal(path)
                stage_refs[qid]=mini.record(path)
                del keys,x,ks,values,family,edge;gc.collect()
        finally:names.close();edges.close_files()
        gen=dict(features=stage_refs,queries=len(panel),labels_used=False,
            family_sources=mini.record(proof_dir/'family_sources.json'),raw_successions=launch['raw_successions'],
            commune_sites=sum(len(mini.read(mini.verified(ref))) for ref in refs.values()))
        mini.save(proof_dir/'features_generation.json',gen)
        # Execute whole original prepare functions: only authority/input roots,
        # execution guards and output I/O are rebound, not their scientific body.
        source_gen=mini.record(proof_dir/'features_generation.json')
        stages={}
        for module,label in ((base,'v32'),(corporate,'v57'),(filled,'v68'),(residual,'v71'),(supplier,'v75')):
            root=directory/label;root.mkdir();(root/'features').mkdir()
            inputs=dict(launch,prepared_generation=source_gen,geometry=geometry_ref)
            if module is corporate:
                # The exact null-safe V57 continuation, not the old None->NONE rule.
                proof.structural.normalize=nullsafe.normalize
            if module is filled:
                authority=mini.read(filled.OUTPUT/'launch.json')
                inputs.update(atlas_geometry=authority['atlas_geometry'],atlas_generation=authority['atlas_generation'])
            if module is residual:
                inputs.update(corp_features=mini.record(stages['v57']/'features_generation.json'),
                    geo_features=mini.record(stages['v68']/'features_generation.json'))
            if module is supplier:inputs['previous_features']=mini.record(stages['v71']/'features_generation.json')
            module.authority=lambda out,config=inputs:config
            module.prepare(root)
            stages[label]=root
            if module is base:source_gen=mini.record(root/'features_generation.json')
        final=mini.read(stages['v75']/'features_generation.json')
        mini.save(directory/'construction.json',dict(status='COMPLETE_QUERY_ONLY_NUMERIC51',queries=len(panel),
            stage_roots={k:str(v) for k,v in stages.items()},bodies=BODIES,constructor_authorities=AUTHORITIES,
            frozen_query_score_matrices_used_as_inputs=False,frozen_query_independent_document_banks_reused=True,
            encoder_query_ids=[q['query_id'] for q in encoder_panel],
            original_padding_batches_preserved_for_fidelity=encoder_panel!=panel,
            raw_labels_read=False,new_fitting=False,final_features=final['features']))
        return final['features']
    finally:
        fit.load_model=original_loader
        engine.io.write=original_io


def fidelity(directory):
    start=time.monotonic()
    run=mini.bano.cohort.preflight.RUN
    panel=mini.read(mini.bano.cohort.OUT/'pilot_fidelity_queries.json')
    original=mini.read(mini.verified(mini.read(run/'launch.json')['panel']))
    ids=set(mini.read(mini.bano.cohort.OUT/'sample.json')['fidelity_query_ids'])
    if len(panel)!=20 or {q['query_id']:q for q in panel}!={q['query_id']:q for q in original if q['query_id'] in ids}:
        raise ValueError('STOP_NUMERIC_SEALED_PILOT_SCOPE')
    # Gemma uses padding=True and batches8. Replaying only selected rows would
    # change the historical batch/padding, not just I/O. Keep the original safe
    # query-only commune companions, emit/compare only the sealed20 rows.
    zones={q['crm_insee'] for q in panel}
    companions=[q for q in original if q['crm_insee'] in zones]
    refs=reconstruct(panel,directory/'reconstruction',encoder_panel=companions)
    expected=mini.read(run/'features_generation.json')['features'];rows=[]
    for q in panel:
        mini.guard();qid=q['query_id']
        with np.load(mini.verified(refs[qid]),allow_pickle=False) as f:keys=f['keys'];x=f['features']
        with np.load(mini.verified(expected[qid]),allow_pickle=False) as f:
            same=np.array_equal(keys,f['keys']) and x.shape==f['features'].shape
            error=float(np.max(np.abs(x-f['features']),initial=0.)) if same else None
            bitwise=same and np.array_equal(x,f['features'])
        row=dict(query_id=qid,keys_and_shape_identical=same,bitwise=bitwise,max_absolute_error=error,
            passed=same and error is not None and error<=ATOL,rows=len(keys))
        mini.save(directory/(hashlib.sha256(qid.encode()).hexdigest()+'.json'),row);rows.append(row)
        if not row['passed']:
            mini.save(directory/'failure.json',dict(status='STOP_NUMERIC51_NONFIDELITY',query=row))
            raise ValueError('STOP_NUMERIC51_NONFIDELITY')
    mini.save(directory/'result.json',dict(status='PASS',family='numeric51',queries=20,
        matches=20,max_absolute_error=max(r['max_absolute_error'] for r in rows),absolute_tolerance=ATOL,
        bitwise_matches=sum(r['bitwise'] for r in rows),elapsed_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,raw_labels_read=False,new_fitting=False))
