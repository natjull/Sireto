"""Frozen Gemma passage-similarity probe, NOT a retrieval recall evaluation.

This bank is explicitly assembled for label-side source diagnosis. It must
never be exported as a candidate pool, served, or called national recall.
No training, identifiers as text, cropping, coefficient fitting or GT repair.
"""
from pathlib import Path
import json
import re
import time
import unicodedata

import numpy as np

from scripts.embeddinggemma2_v49_postseal_bodacc_roles import BASE,read,write,reference
from scripts import embeddinggemma2_v4_fit as fit
from scripts import embeddinggemma2_v2_data as data
from scripts import embeddinggemma2_compare as formatting
from scripts import embeddinggemma2_night as io

OUT=BASE/'v54/postseal_bodacc_embedding_probe_20261012_a'
SOURCE=BASE/'v53/postseal_bodacc_reverse_20261012_a'
ADAPTER=BASE/'v6/deadline_recovery_20261008_a/fit/best_adapter.pt'
CONFIG=BASE/'v4/night_20261008_b/config.json'


def sanitize(text):
    text=unicodedata.normalize('NFKC',text)
    text=re.sub(r'(?<![0-9])[0-9]{3}\s+[0-9]{3}\s+[0-9]{3}\s+[0-9]{5}(?![0-9])',' ',text)
    text=re.sub(r'(?<![0-9])[0-9]{3}\s+[0-9]{3}\s+[0-9]{3}(?![0-9])',' ',text)
    text=data.safe(text)
    if re.search(r'(?<![0-9])(?:[0-9]{9}|[0-9]{14})(?![0-9])',text):raise ValueError('Identifier in encoder text')
    return text


def main():
    import torch
    if OUT.exists():raise ValueError('Fresh passage diagnostic namespace')
    summary=read(SOURCE/'summary.json')
    if summary['status']!='COMPLETE_POSTSEAL_REVERSE_BODACC_TEXT_NOT_RETRIEVAL':raise ValueError('Full source audit first')
    cfg=read(CONFIG);checkpoint=torch.load(ADAPTER,map_location='cpu',weights_only=True)
    if checkpoint['step']!=1000:raise ValueError('Same frozen best1000')
    if not torch.backends.mps.is_available():raise ValueError('Mac only MPS required')
    # Source boundary: genuine resource pressure, not an arbitrary long run.
    fit.resource_guard(time.time()+600,cfg['max_rss_bytes'])
    OUT.mkdir(parents=True)
    write(OUT/'launch.json',dict(implementation=reference(Path(__file__)),source_summary=reference(SOURCE/'summary.json'),
        adapter=reference(ADAPTER),config=reference(CONFIG),step=1000,
        scope='LABEL_SIDE_DOCUMENT_SIMILARITY_ONLY_NEVER_RETRIEVAL_CANDIDATES_OR_RECALL',
        no_training=True,no_gradient=True,no_hyperparameter_search=True,max_tokens=512,batch_size=8))
    model,params,encode,inference,tokenizer=fit.load_model(cfg,ADAPTER)
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    bank={};source_paths=[];oversized=0
    for path in sorted((SOURCE/'subjects').glob('*.json')):
        source=read(path);source_paths.append(reference(path))
        for ann in source['records']:
            for part in ann['texts']:
                field=part['field']
                if not (field=='administration' or field=='establishment.activite' or field.startswith(('acte.descriptif','modificationsgenerales.descriptif','divers'))):continue
                text=sanitize(part['text'])
                if not text or len(text)<20:continue
                # Cover all tokens of oversized single atoms with overlap;
                # no tokenizer truncation and no source field dropped by length.
                chunks,count=data.pack_atoms([text],tokenizer,512)
                oversized+=count
                for chunk in chunks:
                    entry=bank.setdefault(chunk,dict(text=chunk,source_parents=set(),refs=[]))
                    entry['source_parents'].add(source['audited_parent'])
                    ref=dict(id=ann['id'],date=ann['date'],field=field,url=ann['url'])
                    if ref not in entry['refs']:entry['refs'].append(ref)
    docs=[]
    for entry in bank.values():
        entry['source_parents']=sorted(entry['source_parents']);docs.append(entry)
    write(OUT/'passages.json',docs)
    cases=read(SOURCE/'cases.json')
    if len(cases)!=31:raise ValueError('All31 diagnosis cases kept')
    qtexts=[io.QUERY_PROMPT+formatting.query_text(r['crm']) for r in cases]
    qvectors=inference(qtexts);vectors=[];start=time.monotonic();deadline=time.time()+600
    for offset in range(0,len(docs),64):
        fit.resource_guard(deadline,cfg['max_rss_bytes'])
        vectors.extend(inference([io.DOC_PROMPT+r['text'] for r in docs[offset:offset+64]]))
        print(json.dumps(dict(encoded=min(offset+64,len(docs)),total=len(docs),seconds=round(time.monotonic()-start,1))),flush=True)
    dv=np.stack(vectors);scores=qvectors@dv.T
    result=[]
    for q,row in enumerate(cases):
        parent=row['label']['siren'];own=[i for i,d in enumerate(docs) if parent in d['source_parents']]
        other=[i for i,d in enumerate(docs) if parent not in d['source_parents']]
        def best(indices):
            if not indices:return None
            i=max(indices,key=lambda i:float(scores[q,i]));d=docs[i]
            return dict(cosine=float(scores[q,i]),passage_index=i,text=d['text'],source_parents=d['source_parents'],refs=d['refs'])
        own_free=[i for i in own if any(r['field']!='administration' for r in docs[i]['refs'])]
        result.append(dict(query_id=row['query_id'],crm=row['crm'],label=row['label'],
            with_any_own_passage=bool(own),best_own_any=best(own),best_own_without_administration=best(own_free),
            best_other_diagnostic_passage=best(other),
            diagnostic_source_selection_uses_labels=True,no_candidates_or_recall_produced=True,
            score_not_probability=True,source_parent_context_not_target_site_identity=True))
    write(OUT/'cases.json',result)
    summary=dict(status='COMPLETE_FROZEN_GEMMA_BODACC_PASSAGE_DIAGNOSIS_NOT_RETRIEVAL',queries=31,passages=len(docs),
        oversized_atoms_fully_split=oversized,step=1000,batch=8,max_tokens=512,
        cases=reference(OUT/'cases.json'),passage_bank=reference(OUT/'passages.json'),sources=source_paths,
        launch=reference(OUT/'launch.json'),elapsed_seconds=time.monotonic()-start,
        no_training=True,no_positive_candidate_injected=True,no_retrieval_evaluation=True,
        no_prediction_changed=True,no_closed_fold_opened=True,no_coefficient_fitted=True,
        no_truncation=True,identifiers_removed_before_encoding=True,
        no_national_precision_recall_or_rank_claim=True)
    write(OUT/'summary.json',summary);print(json.dumps({k:summary[k] for k in ('status','queries','passages','oversized_atoms_fully_split','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
