"""V36: unchanged sealed V23 full proofs; only training objective/mining changes."""
from pathlib import Path
import argparse
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from src.xgb_matcher.retrieval_v20_workbench import record, verify

PREPARED=io.HOME/'v23/hierarchical_native_20261012_a'
OUTPUT=io.HOME/'v36/global_cached_parent_dense_forder_20261012_a'
IDENTITY_SOURCE=io.HOME/'v18/identity_site_20261010_a'
LEXICAL_FIELDS=list(source.LEXICAL_FIELDS)+['global_cached_parent_identity_cosine']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
RECIPE=dict(source.RECIPE,feature_fields=FEATURE_FIELDS,feature_values_changed=False,
    global_identity_scope='ALL2126387_CACHED_PARENTS_OF295_COMMUNES_NOT_ALL_FRENCH_PARENTS',
    identity_fields='ORIGINAL_LEGAL_AND_NATIONAL_SIBLING_NAMES_ROLE_SEPARATED_VIEWS',
    identity_reference='BROADCAST_COMPUTED_SCORE_TO_ALL_NATIONAL_MEMBER_LEAVES',
    no_parent_beam_or_ann=True,
    score_bank_layout='FORTRAN_Q_BY_PARENT_PARENT_MAJOR_SAME_FLOAT32_MAX',
    original_query_encoding_reused_without_reexecution=True,no_new_document_encoding=True,no_new_encoder_fit=True,
    unavailable_name_placeholder_views_have_zero_name_evidence=True)

def initialize(out,deadline):
    source.initialize(out,deadline)
    previous=source.authority(PREPARED)
    generation=io.read(PREPARED/'features_generation.json')
    ex=io.read(PREPARED/'prepare_exit.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed'] or generation['queries']!=526:
        raise ValueError('Owned complete original features only')
    launch=io.read(out/'launch.json')
    if previous['panel']!=launch['panel']:raise ValueError('Same complete query panel')
    io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_launch=record(PREPARED/'launch.json'),prepared_generation=record(PREPARED/'features_generation.json'),
        prepared_exit=record(PREPARED/'prepare_exit.json'))
    from scripts import embeddinggemma2_v18_identity_retrieval as original
    binding=io.read(IDENTITY_SOURCE/'identity_vectors/binding.json')
    if binding['adapter']!=launch['adapter'] or binding['database']!=record(engine.sealed(IDENTITY_SOURCE/'identities.sqlite')):raise ValueError('Same original complete vector bank')
    for ref in (binding['database'],binding['adapter'],binding['recipe'],binding['implementation']):verify(ref)
    launch.update(identity_database=binding['database'],identity_counts=record(engine.sealed(IDENTITY_SOURCE/'identity_counts.json')),
        vector_binding=record(IDENTITY_SOURCE/'identity_vectors/binding.json'))
    launch['helpers'].extend([record(Path(source.__file__)),record(Path(original.__file__)),record(io.ROOT/'scripts/embeddinggemma2_v3_fit.py')])
    io.write(out/'launch.json',launch)

def authority(out):
    launch=source.authority(out)
    for name in ('prepared_launch','prepared_generation','prepared_exit','identity_database','identity_counts','vector_binding'):verify(launch[name])
    return launch


ACTIVE_OUTPUT=OUTPUT
BANK=None
LAST_PART=None
def bind_runtime(out):
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT=out

def encode_queries(out):
    import shutil
    launch=authority(out)
    previous=io.HOME/'v33/global_cached_parent_dense_20261012_b'
    receipt=io.read(previous/'encode_queries_exit.json')
    if receipt['exit_code']!=0 or not receipt['own_child_exit_confirmed']:raise ValueError('Owned original query encoding required')
    vectors=engine.sealed(previous/'query_identity_vectors.npy');binding=engine.sealed(previous/'query_vector_binding.json')
    if not vectors or not binding:raise ValueError('Both original query artifacts sealed')
    prior=io.read(binding)
    if prior['panel']!=launch['panel'] or prior['adapter']!=launch['adapter']:raise ValueError('Exact original query encoding scientific binding')
    target=out/'query_identity_vectors.npy'
    if not engine.sealed(target):shutil.copyfile(vectors,target);engine.seal(target)
    meta=dict(prior,query_vectors=record(target),reused_binding=record(binding),original_owned_exit=record(previous/'encode_queries_exit.json'),same_bytes=True)
    io.write(out/'query_vector_binding.json',meta);engine.seal(out/'query_vector_binding.json')
    return


def unused_original_encoder(out):
    import torch,transformers
    from scripts import embeddinggemma2_v3_fit as fit
    if torch.__version__!='2.14.1' or transformers.__version__!='5.19.0':raise ValueError('Original encoder numeric runtime')
    launch=authority(out);path=out/'query_identity_vectors.npy'
    if engine.sealed(path):return
    source_launch=io.read(verify(launch['source_launch']));cfg=io.read(verify(source_launch['night_config']))
    model,params,encode,inference,tokenizer=fit.load_model(cfg,verify(launch['adapter']))
    panel=io.read(verify(launch['panel']));vectors=np.zeros((len(panel),768),dtype=np.float32)
    actual=[i for i,q in enumerate(panel) if data.safe(q['crm_name'])]
    for start in range(0,len(actual),8):
        guard(launch['deadline_unix']);indices=actual[start:start+8]
        values=inference([io.QUERY_PROMPT+'noms: '+data.safe(panel[i]['crm_name']) for i in indices])
        if values.shape!=(len(indices),768) or values.dtype!=np.float32 or not np.isfinite(values).all():raise ValueError('Original query vector shape')
        vectors[indices]=values
        engine.progress(out,'ENCODING_QUERY_NAMES_ONLY_NO_DOCUMENTS',start+len(indices),total=len(actual))
    temp=path.with_suffix('.pending.npy');np.save(temp,vectors,allow_pickle=False);os.replace(temp,path);engine.seal(path)
    io.write(out/'query_vector_binding.json',dict(panel=launch['panel'],adapter=launch['adapter'],query_vectors=record(path),
        query='ORIGINAL_NAMES_ONLY_QUERY_PROMPT',batch8=True,document_vectors_changed=False,
        blank_query_names_zeroed=True,no_numerical_equivalence_to_old_query_padding_claim=True));engine.seal(out/'query_vector_binding.json')

def prepare(out):
    import sqlite3
    launch=authority(out);ex=io.read(out/'encode_queries_exit.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete query-only encoding first')
    panel=io.read(verify(launch['panel']));qv=np.load(engine.sealed(out/'query_identity_vectors.npy'),allow_pickle=False)
    counts=io.read(verify(launch['identity_counts']))
    db=sqlite3.connect(f"file:{verify(launch['identity_database'])}?mode=ro&immutable=1",uri=True)
    parents=np.asarray([r[0] for r in db.execute('SELECT parent FROM parents ORDER BY parent')],dtype=np.uint64)
    if len(parents)!=counts['parents'] or len(parents)!=2126387:raise ValueError('Complete original label-free parent bank')
    placeholder=io.DOC_PROMPT+'nom public indisponible'
    rows=np.asarray(db.execute('SELECT v.parent,v.text_id FROM views v JOIN texts t ON t.id=v.text_id WHERE t.text<>? ORDER BY v.text_id,v.parent',(placeholder,)).fetchall(),dtype=np.int64)
    db.close();text=rows[:,1]-1;parent_at=np.searchsorted(parents,rows[:,0].astype(np.uint64))
    if (parent_at>=len(parents)).any() or not np.array_equal(parents[parent_at],rows[:,0]):raise ValueError('Exact owner lookup')
    del rows
    path=out/'global_parent_cosine.npy';cursor=out/'bank_cursor.json';shape=(len(panel),len(parents))
    if not path.exists():
        mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape,fortran_order=True);mm[:]=0;mm.flush()
        io.write(cursor,dict(next_text=0,binding=record(out/'query_vector_binding.json')))
    else:mm=np.lib.format.open_memmap(path,mode='r+')
    state=io.read(cursor)
    if state['binding']!=record(out/'query_vector_binding.json') or mm.shape!=shape or mm.dtype!=np.float32 or not mm.flags.f_contiguous:raise ValueError('Same resumed complete bank binding')
    start=int(state['next_text']);begun=time.monotonic()
    for lo in range(start,counts['unique_texts'],1024):
        guard(launch['deadline_unix'])
        shard=IDENTITY_SOURCE/'identity_vectors'/f'vectors_{lo:08d}.npy';sealed=engine.sealed(shard)
        if not sealed:raise ValueError('Every original vector shard sealed')
        matrix=np.load(sealed,mmap_mode='r',allow_pickle=False)
        if matrix.shape!=(min(1024,counts['unique_texts']-lo),768) or matrix.dtype!=np.float32 or not np.isfinite(matrix).all():raise ValueError('Original immutable source shard')
        scores=np.clip(qv@matrix.T,0,1);matrix._mmap.close()
        left=np.searchsorted(text,lo);right=np.searchsorted(text,lo+1024)
        pp=parent_at[left:right];tt=text[left:right]-lo
        for i in range(len(panel)):np.maximum.at(mm[i],pp,scores[i,tt])
        if lo%16384==0 or lo+1024>=counts['unique_texts']:
            mm.flush()
            with path.open('rb') as durable:os.fsync(durable.fileno())
            io.write(cursor,dict(next_text=min(lo+1024,counts['unique_texts']),binding=record(out/'query_vector_binding.json')))
        engine.progress(out,'ALL_CACHED_PARENTS_GLOBAL_NAME_SCORE_NO_ANN',lo+len(scores[0]),total=counts['unique_texts'],
            parents=len(parents),queries=len(panel),elapsed_s=time.monotonic()-begun)
    if not np.isfinite(mm).all() or (mm<0).any() or (mm>1).any():raise ValueError('Finite complete global bank')
    mm.flush();mm._mmap.close();engine.seal(path)
    ppath=out/'parent_keys.npy';np.save(ppath,parents,allow_pickle=False);engine.seal(ppath)
    original=io.read(verify(launch['prepared_generation']))
    io.write(out/'features_generation.json',dict(original,
        immutable_shared_source=launch['prepared_generation'],global_parent_scores=record(path),
        parent_keys=record(ppath),bank_source=launch['identity_database'],identity_counts=launch['identity_counts'],
        queries=526,global_parent_count=len(parents),actual_name_views_used=len(text),
        no_name_placeholder_views_not_identity_evidence=True,
        scope='ALL_PARENTS_OF_295_CACHED_COMMUNES_NOT_ALL_FRENCH_LEGAL_ENTITIES',
        all_national_member_leaves_broadcast_by_reference=True,labels_used=False,no_new_document_encoding=True))

def parent_component(backend,q):
    global BANK,LAST_PART
    if LAST_PART and LAST_PART[0] is backend and LAST_PART[1]==q['query_id']:return LAST_PART[2]
    if BANK is None:
        launch=authority(ACTIVE_OUTPUT);gen=io.read(ACTIVE_OUTPUT/'features_generation.json')
        panel=io.read(verify(launch['panel']))
        values=np.load(verify(gen['global_parent_scores']),mmap_mode='r',allow_pickle=False)
        parents=np.load(verify(gen['parent_keys']),mmap_mode='r',allow_pickle=False)
        BANK=(values,[f'{int(p):09d}' for p in parents],{r['query_id']:i for i,r in enumerate(panel)})
        parents._mmap.close()
    values,parents,positions=BANK
    scores=values[positions[q['query_id']]];maximum=(1<<RECIPE['quantization_bits'])-1
    integers=np.floor(np.clip(scores,0,1)*maximum).astype(np.uint32);tv=backend._tantivy;parts=[]
    for bit in range(RECIPE['quantization_bits']):
        selected=np.flatnonzero(integers & (1<<bit))
        if len(selected):
            lookup=tv.Query.term_set_query(backend.index.schema,'siren',[parents[i] for i in selected])
            parts.append(tv.Query.const_score_query(lookup,float((1<<bit)/maximum)))
    result=tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]) if parts else None
    LAST_PART=(backend,q['query_id'],result)
    return result

def lexical_parts(backend,q):
    return source.lexical_parts(backend,q)+[parent_component(backend,q)]


def native_query(backend,q,keys,features,coefficients):
    """ONE native national query, learned lexical + ALL numeric proof views.

    Bit planes attach a *computed proof score* to its official reference. IDs
    have no learned coefficient and are never numerical model inputs. No top-k
    prepool, no sampled subset, no capped channel union is materialized here.
    """
    coefficients=np.asarray(coefficients,dtype=float)
    if coefficients.shape!=(len(FEATURE_FIELDS),) or not np.isfinite(coefficients).all() or (coefficients<0).any():
        raise ValueError('Finite learned nonnegative typed coefficients')
    tv=backend._tantivy;parts=[]
    for component,weight in zip(lexical_parts(backend,q),coefficients[:len(LEXICAL_FIELDS)]):
        if component is not None and weight>0:parts.append(tv.Query.boost_query(component,float(weight)))
    weights=coefficients[len(LEXICAL_FIELDS):];bound=float(weights.sum());bits=RECIPE['quantization_bits'];maximum=(1<<bits)-1
    if bound>0 and len(keys):
        scores=features.astype(np.float64)@weights
        if not np.isfinite(scores).all() or (scores<0).any() or (scores>bound+1e-5).any():raise ValueError('Full numeric proof score bound')
        integers=np.floor(np.clip(scores/bound,0,1)*maximum).astype(np.uint32)
        refs=[f'{int(k):014d}' for k in keys]
        for bit in range(bits):
            selected=np.flatnonzero(integers & (1<<bit))
            if len(selected):
                mask=tv.Query.term_set_query(backend.index.schema,'siret',[refs[i] for i in selected])
                parts.append(tv.Query.const_score_query(mask,float(bound*(1<<bit)/maximum)))
        del scores,integers,refs
    if not parts:return tv.Query.empty_query(),bound/maximum
    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    return tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))]),bound/maximum


def native_search(backend,q,keys,features,coefficients):
    started=time.monotonic();query,error=native_query(backend,q,keys,features,coefficients)
    built=time.monotonic();rows=[]
    for score,loc in backend.searcher.search(query,limit=100,count=False).hits:
        raw=backend.searcher.doc(loc).to_dict();siret=backend._first(raw,'siret');siren=backend._first(raw,'siren')
        if not siret.isdigit() or len(siret)!=14 or len(siren)!=9 or not siret.startswith(siren) or not np.isfinite(score):
            raise ValueError('Valid official native leaf and score')
        rows.append(dict(siret=siret,siren=siren,score=float(score)))
    if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows):raise ValueError('Native unique absolute100')
    rows.sort(key=lambda r:(-r['score'],r['siret']))
    return rows,dict(query_build_seconds=built-started,search_seconds=time.monotonic()-built,
        total_seconds=time.monotonic()-started,quantization_abs_error_bound=error,
        full_nonlexical_leaves=len(keys),national_lexical_sites=43896818,no_prepool=True,
        global_parent_quantization_error_bound=float(coefficients[5])/((1<<RECIPE['quantization_bits'])-1),
        no_parent_beam=True,global_identity_parent_count=2126387)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','encode_queries','prepare']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--deadline-unix',type=float);a=p.parse_args()
    if a.action=='initialize':initialize(a.output,a.deadline_unix)
    else:globals()[a.action](a.output)



