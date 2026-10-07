"""Immutable official multi-view data; never opens reserved test or eval sidecar.

IDs are joins/masks only. Sibling aliases stay parent-context, NOT target-site
aliases. The native corpus is used for TRAIN mining, never a retrieval eval.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
from pathlib import Path
import re
import time
import unicodedata

from scripts import embeddinggemma2_night as io
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record,verify

CONFIG=io.ROOT/'config/embeddinggemma2_v2.json'
SOURCE=io.HOME/'night_20261006_a'
OUTPUT=io.HOME/'v2'
DICTIONARY=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v130_variant_dictionary/compile_20261003_a/manifest.json',
    sha256='a69477e35c53f9f6d702b9e870580e377a765fb1e92e990782c909fed29241f4')
MASKED={'ND','NON DIFFUSIBLE','NON DIFFUSIBLE PARTIEL','NON RENSEIGNE','NONE','NULL','NAN'}


def safe(value):
    # Normalize compatibility digits BEFORE stripping identifier-shaped strings.
    text=unicodedata.normalize('NFKC',str(value or ''))
    text=re.sub(r'(?<!\d)(?:\d{14}|\d{9})(?!\d)',' ',text)
    text=' '.join(text.split())
    return '' if text.upper() in MASKED else text


def canonical(value):
    value=unicodedata.normalize('NFKD',safe(value))
    value=''.join(c for c in value if not unicodedata.combining(c))
    value=re.sub(r'[^A-Za-z0-9]+',' ',value).upper().split()
    replacements={'AV':'AVENUE','BD':'BOULEVARD','BLVD':'BOULEVARD','RTE':'ROUTE','CRS':'COURS','CHE':'CHEMIN','IMP':'IMPASSE','PL':'PLACE','R':'RUE','ST':'SAINT','STE':'SAINTE'}
    return ' '.join(replacements.get(t,t) for t in value)


def street_key(value):
    normalized=canonical(value)
    m=re.fullmatch(r'(\d+) (.*)',normalized)
    if not m or int(m[1])==0 or not m[2].strip(): return None
    street=m[2]
    suffix=''
    if street.split()[0] in ('BIS','TER','QUATER'):
        suffix,_,street=street.partition(' ')
    if not street or len(street.split())<2: return None
    return str(int(m[1])),suffix,street


def query_site_keys(row):
    key=street_key(' '.join(row[k] for k in ('crm_number','crm_suffix','crm_street')))
    return {key} if key else set()


def eligible_other_site(query, positive, negative):
    """Conservative TRAIN-only proof: query confirms positive; NO shared history.

    Unknown/unnumbered sites, suffix ambiguity and conflicting historical
    addresses remain masked. These are not automatic label corrections.
    """
    if positive['siren']!=negative['siren'] or positive['key']==negative['key']: return False
    if not (positive['insee'] and positive['insee']==negative['insee']==query['crm_insee']): return False
    qkeys=query_site_keys(query); pkeys={street_key(a) for a in positive['all_addresses']}; nkeys={street_key(a) for a in negative['all_addresses']}
    if None in pkeys or None in nkeys: return False
    if not qkeys or not qkeys<=pkeys or not nkeys or pkeys&nkeys: return False
    # Same number/street with an unspecified suffix is not a proven different site.
    for p in pkeys:
        for n in nkeys:
            if p[0]==n[0] and p[2]==n[2] and (not p[1] or not n[1]): return False
    return bool(nkeys)


def pack_atoms(atoms,tokenizer,max_tokens=512,anchor=''):
    """Every safe source atom represented, no tokenizer truncation ever enabled.

    Oversized individual atoms get overlapping token chunks, explicitly counted.
    Their complete original strings remain in the document metadata.
    """
    anchor=safe(anchor); atoms=list(dict.fromkeys(safe(a) for a in atoms if safe(a)))
    if not atoms: atoms=['nom public indisponible']
    def size(body): return len(tokenizer(io.DOC_PROMPT+body,add_special_tokens=True)['input_ids'])
    if size(anchor)>max_tokens//2: anchor=''
    chunks=[]; current=anchor; oversized=0
    for atom in atoms:
        candidate=' | '.join(x for x in (current,atom) if x)
        if size(candidate)<=max_tokens: current=candidate; continue
        if current and current!=anchor: chunks.append(current)
        current=anchor
        if size(' | '.join(x for x in (anchor,atom) if x))<=max_tokens:
            current=' | '.join(x for x in (anchor,atom) if x); continue
        oversized+=1
        tokens=tokenizer(atom,add_special_tokens=False)['input_ids']; offset=0
        while offset<len(tokens):
            end=min(offset+max_tokens-32,len(tokens))
            body=tokenizer.decode(tokens[offset:end],skip_special_tokens=True)
            while size(body)>max_tokens and end>offset+1:
                end-=1; body=tokenizer.decode(tokens[offset:end],skip_special_tokens=True)
            if size(body)>max_tokens: raise ValueError('Single token cannot fit official view')
            chunks.append(body)
            if end==len(tokens): break
            offset=max(offset+1,end-16)
    if current and (current!=anchor or not chunks): chunks.append(current)
    chunks=list(dict.fromkeys(chunks))
    if not chunks or any(size(c)>max_tokens for c in chunks): raise ValueError('Unchecked/truncated view')
    return chunks,oversized


def identity_atoms(names):
    # Endpoint names are explicitly CONTEXT OF PARENT, not names of each sibling.
    return ['nom entreprise: '+n for n in names['legal']]+['nom de site appartenant a cette entreprise: '+n for n in names['sites']]


def site_atoms(doc,extra):
    return (['nom entreprise: '+n for n in extra['legal']]+['nom etablissement: '+n for n in doc['own_names']]+
        ['ancien nom etablissement: '+n for n in extra['own_site']]+['adresse etablissement: '+a for a in doc['all_addresses']]+
        ['code postal: '+doc['postcode']])


def native_document(backend,d):
    first=backend._first
    own=[v for f in ('legal_names','trade_names','site_names','historical_names','supporting_names') for v in d.get(f,[])]
    addresses=[v for f in ('addresses','historical_addresses','supporting_addresses') for v in d.get(f,[])]
    return dict(key=first(d,'siret'),siren=first(d,'siren'),insee=first(d,'insee'),postcode=first(d,'postcode'),
        own_names=list(dict.fromkeys(safe(v) for v in own if safe(v))),all_addresses=list(dict.fromkeys(safe(a) for a in addresses if safe(a))),
        state=first(d,'state'),related_parents=str(first(d,'linked_sirens')).split(),
        source_kind='NATIVE_OFFICIAL_SIRET',source_text=io.clean_text(own,addresses,first(d,'postcode')))


def guard(start,seconds=1800):
    import psutil
    free,level=io.pressure()
    if free<50 or level!=1 or psutil.Process().memory_info().rss>5*1024**3 or time.monotonic()-start>seconds:
        raise RuntimeError('STOP_V2_DATA_RESOURCE_OR_TIME')


def prepare(out):
    import pyarrow.dataset as ds
    from transformers import AutoTokenizer
    cfg=io.read(CONFIG); out.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    previous=io.read(SOURCE/'manifest.json'); inputs={'manifest':record(SOURCE/'manifest.json')}
    memberships={}; legacy={}
    for split in ('train','validation'):
        membership=SOURCE/(split+'_membership.json'); dataset=SOURCE/(split+'.json')
        if io.digest(membership)!=previous['split_hashes'][split] or io.digest(dataset)!=previous['training_hashes'][split]: raise ValueError('Immutable original split/data drift')
        memberships[split]=io.read(membership); legacy[split]=io.read(dataset)
        inputs[split+'_membership']=record(membership); inputs[split]=record(dataset)
        if any(r['fold'] not in (2,3,4) for r in memberships[split]): raise ValueError('Closed fold')
    train_parents={r['siren'] for r in memberships['train']}; val_parents={r['siren'] for r in memberships['validation']}
    if train_parents&val_parents: raise ValueError('Entity leakage')
    # Approved old negatives already excluded held-out groups in the sealed v1 prepare.
    approved={split:{n['siren'] for r in legacy[split] for n in r['negatives']} for split in legacy}
    if approved['train']&val_parents: raise ValueError('Held-out SIREN in original train negatives')
    parents=train_parents|val_parents|approved['train']|approved['validation']
    backend=TantivyBackend(io.INDEX); tv=backend._tantivy
    if backend.searcher.num_docs!=82772691: raise ValueError('Native index drift')
    inventory_before=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted((io.INDEX/'tantivy').iterdir()) if p.is_file()]
    native={}; by_parent=defaultdict(list)
    for i in range(0,len(train_parents|val_parents),64):
        part=sorted(train_parents|val_parents)[i:i+64]
        q=tv.Query.boolean_query([(tv.Occur.Must,tv.Query.term_set_query(backend.index.schema,'siren',part)),
            (tv.Occur.Must,tv.Query.term_query(backend.index.schema,'document_type','siret','basic'))])
        n=backend.searcher.search(q,limit=1,count=True).count
        if len(native)+n>cfg['max_training_native_sites']: raise RuntimeError('STOP_NATIVE_TRAIN_BANK_SIZE_NOT_SILENT_CAP')
        hits=backend.searcher.search(q,limit=max(1,n+1),count=True)
        if len(hits.hits)!=n or hits.count!=n: raise ValueError('Incomplete official parent inventory')
        for _,loc in hits.hits:
            d=native_document(backend,backend.searcher.doc(loc).to_dict()); native[d['key']]=d; by_parent[d['siren']].append(d['key'])
        guard(start); io.write(out/'state.json',dict(status='ALL_OFFICIAL_SITES_OF_TRAIN_VALIDATION_PARENTS',parent_batches=i//64+1,native_sites=len(native)))
    # Fully pinned official history/name dictionary. Never hydrate test CRM/labels.
    dictionary=io.read(verify(DICTIONARY)); parent_names=defaultdict(lambda:dict(legal=set(),sites=set())); own_names=defaultdict(set); refs=[]; epochs={}
    if dictionary['status']!='SEALED_V130_COMPLETE_ALL_OFFICIAL_NAME_OWNER_DICTIONARY': raise ValueError('Unsealed name authority')
    for i,bucket in enumerate(dictionary['buckets']):
        f=bucket['artifacts']['facts.parquet']; names_ref=bucket['artifacts']['names.parquet']
        fp=verify(f); np=verify(names_ref); refs.extend([f,names_ref])
        for path in (fp,np): epochs[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns)
        facts=ds.dataset(fp).to_table(filter=ds.field('parent_key').isin([int(p) for p in parents]),
            columns=['name_id','parent_key','owner_key','subject_code','scope_code']).to_pylist()
        if facts:
            names={r['name_id']:safe(r['name_text']) for r in ds.dataset(np).to_table(
                filter=ds.field('name_id').isin(sorted({r['name_id'] for r in facts})),columns=['name_id','name_text']).to_pylist()}
            for r in facts:
                text=names[r['name_id']]
                if not text: continue
                parent=f"{r['parent_key']:09d}"
                if r['subject_code']==1 and r['scope_code'] in (1,2): parent_names[parent]['legal'].add(text)
                elif r['subject_code']==2 and r['scope_code']==3:
                    parent_names[parent]['sites'].add(text); own_names[f"{r['owner_key']:014d}"].add(text)
                else: raise ValueError('Name owner/scope drift')
        guard(start); io.write(out/'state.json',dict(status='ALL64_OFFICIAL_NAME_BUCKETS',buckets=i+1))
    for parent in parents: parent_names[parent]  # include explicitly missing-name parents
    tokenizer=AutoTokenizer.from_pretrained(io.HOME/'base',local_files_only=True)
    oversized=0; parent_views={}
    for parent,names in parent_names.items():
        for field in names: names[field]=sorted(names[field])
        views,over=pack_atoms(identity_atoms(names),tokenizer,cfg['max_tokens']); oversized+=over
        parent_views[parent]=views
    for key,d in native.items():
        names=parent_names[d['siren']]
        extra=dict(legal=names['legal'],own_site=sorted(own_names.get(key,())))
        anchor=io.clean_text(d['own_names'][:1],d['all_addresses'][:1],d['postcode'])
        d['site_views'],over=pack_atoms(site_atoms(d,extra),tokenizer,cfg['max_tokens'],anchor); oversized+=over
    datasets={}; stats={}
    for split in ('train','validation'):
        old_rows={r['query_id']:r for r in legacy[split]}; rows=[]; absent=[]; siblings=0
        for q in memberships[split]:
            if q['siret'] not in native or q['query_id'] not in old_rows:
                absent.append(q['query_id']); continue
            old=old_rows[q['query_id']]; positive=native[q['siret']]; negatives=[]
            for n in old['negatives']:
                if n['siren'] in positive['related_parents']: continue  # uncertain relation != identity/negative
                key='OFFICIAL_LEGACY:'+hashlib.sha256((n['siren']+'|'+n['text']).encode()).hexdigest()
                if key not in native:
                    site_views,over=pack_atoms([safe(n['text'])]+['nom entreprise: '+x for x in parent_names[n['siren']]['legal']],tokenizer,cfg['max_tokens']); oversized+=over
                    native[key]=dict(key=key,siren=n['siren'],insee='',postcode='',own_names=[],all_addresses=[],site_views=site_views,
                        related_parents=[],state='UNKNOWN',source_kind='SEALED_OFFICIAL_BM25_NEGATIVE',source_text=safe(n['text']))
                negatives.append(dict(key=key,kind='LEXICAL_DIFFERENT_PARENT',proven_other_site=False))
            other=[native[k] for k in by_parent[q['siren']] if eligible_other_site(q,positive,native[k])]
            other.sort(key=lambda d:hashlib.sha256((q['query_id']+'|'+d['key']).encode()).digest())
            if other: negatives.append(dict(key=other[0]['key'],kind='SAME_PARENT_PROVEN_OTHER_SITE',proven_other_site=True)); siblings+=1
            site_query=io.clean_text([q['crm_name']],[' '.join(q[k] for k in ('crm_number','crm_suffix','crm_street'))],q['crm_postcode'])
            identity_query='noms: '+safe(q['crm_name'])
            # Never silently cut CRM inputs either: fail for audit rather than drop a label.
            if any(len(tokenizer(io.QUERY_PROMPT+x)['input_ids'])>cfg['max_tokens'] for x in (site_query,identity_query)):
                raise RuntimeError('STOP_OVERSIZED_QUERY_REQUIRES_EXPLICIT_HANDLING')
            rows.append(dict(query_id=q['query_id'],fold=q['fold'],siren=q['siren'],positive=q['siret'],
                site_query=site_query,identity_query=identity_query,negatives=negatives,crm=q,
                query_confirmed_positive_site=bool(query_site_keys(q)&{street_key(a) for a in positive['all_addresses']})))
        datasets[split]=rows; stats[split]=dict(all_membership_rows=len(memberships[split]),usable_rows=len(rows),
            official_positive_absent_query_ids=absent,rows_with_proven_other_site_negative=siblings,
            excluded_from_retrieval_denominator=False)
    if [(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted((io.INDEX/'tantivy').iterdir()) if p.is_file()]!=inventory_before: raise ValueError('Index inventory drift')
    for ref in refs:
        p=Path(ref['path'])
        if (p.stat().st_size,p.stat().st_mtime_ns)!=epochs[str(p)]: raise ValueError('Name source epoch drift')
    io.write(out/'documents.json',native); io.write(out/'parent_views.json',parent_views)
    for split,rows in datasets.items(): io.write(out/(split+'.json'),rows)
    # Dense mining only in TRAIN native sites and approved old TRAIN negatives.
    train_keys={d['key'] for d in native.values() if d['source_kind']=='NATIVE_OFFICIAL_SIRET' and d['siren'] in train_parents}
    train_keys|={n['key'] for r in datasets['train'] for n in r['negatives']}
    if any(native[k]['siren'] in val_parents for k in train_keys): raise ValueError('Training bank includes held-out entity')
    io.write(out/'train_bank.json',sorted(train_keys))
    io.write(out/'manifest.json',dict(status='SEALED_V2_MULTIVIEW_OFFICIAL_TRAIN_VALIDATION_NO_TEST',config=record(CONFIG),
        implementation=record(Path(__file__)),inputs=inputs,dictionary=DICTIONARY,consumed_name_refs=refs,
        index_manifest=record(io.INDEX/'manifest.json'),index_meta=record(io.INDEX/'tantivy/meta.json'),index_inventory=inventory_before,
        index_segments_full_hashed=False,stats=stats,native_official_sites=sum(d['source_kind']=='NATIVE_OFFICIAL_SIRET' for d in native.values()),
        documents=len(native),parent_views=len(parent_views),site_views=sum(len(d['site_views']) for d in native.values()),
        identity_views=sum(len(v) for v in parent_views.values()),oversized_atoms_explicitly_chunked=oversized,
        document_token_budget=cfg['max_tokens'],silent_token_truncations=0,sibling_site_aliases_not_copied_to_target=True,
        artifacts={name:record(out/name) for name in ('documents.json','parent_views.json','train.json','validation.json','train_bank.json')},
        test_membership_read=False,fold0_opened=False,fold1_opened=False,eval_sidecar_read=False,geometry_in_embedding_text=False,
        successor_relations_not_positive_identity_labels=True,downstream_models_changed=False,elapsed_seconds=time.monotonic()-start))
    io.write(out/'state.json',dict(status='DATA_PREPARED',stats=stats,test_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not a.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated immutable V2 namespace')
    prepare(a.output)
