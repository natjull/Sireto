"""Isolated TRAIN-only query-role fit; serving never reopens TRAIN scenes."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
from scripts import retrieval_v279_municipal_structure as source
from src.xgb_matcher.retrieval_public_role_v280 import ROLES,RIDGE,fit,controls,validate
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,POLICY,quiet,validate_report,parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_postscore_audit_v187 import canonical_epochs,check_epochs
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v280_public_role_training')
INPUTS=dict(
 municipal=dict(path=str(source.OUTPUT/'compile_20261005_a/manifest.json'),sha256='2c0f9b6c924ea2833f1a372a329d05e082f780fd55f88246ae9f5be0ed06a8f1',size_bytes=1283),
 training=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v52_contrastive_training/probe_20261002_a/manifest.json',sha256='95af09fa0448cbc60f0b2b21600162426aa44c9912628d74845022f39b3e02db'),
 query_launch=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v78_safe_sparse_initial/pilot_20261003_a/launch.json',sha256='6fc2cb89b489e8038e6f211e0fac0a36954874e066e8a727558cfa01d9d90956',size_bytes=85950))
FALSE={k:False for k in ('pilot_truth_opened','recall_measured','fold0_opened','fold1_opened','identifiers_as_features',
 'candidates_emitted','ready_for_fold1','downstream_models_changed','crm_alias_dictionary','probabilities_calibrated')}
STATUS='SEALED_V280_TRAIN_ONLY_OOF_QUERY_PUBLIC_ROLE'


def implementation():
 paths=(Path(__file__),ROOT/'docs/retrieval_v280_public_role_contract.md',
  ROOT/'src/xgb_matcher/retrieval_public_role_v280.py',ROOT/'src/xgb_matcher/retrieval_activity_decoder_v111.py',
  Path(source.__file__),ROOT/'docs/retrieval_v279_municipal_structure_contract.md',
  ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py',
  ROOT/'src/xgb_matcher/retrieval_structural_channels_v210.py',ROOT/'src/xgb_matcher/retrieval_postscore_audit_v187.py')
 return {str(p):record(p) for p in paths}


def load(root):
 root=Path(root)
 if not root.resolve().is_relative_to(OUTPUT.resolve()) or (root/'status.json').exists() or any(root.glob('phase/*stop.json')):
  raise ValueError('Own fresh TRAIN-only root, no STOP reuse')
 l=json.loads((root/'launch.json').read_text())
 if l['inputs']!=INPUTS or l['limits']!=LIMITS or l['ridge']!=RIDGE or l['roles']!=list(ROLES) or any(l.get(k) is not False for k in FALSE):
  raise ValueError('Frozen TRAIN-only query role policy required')
 for ref in l['implementation'].values():verify(ref)
 return l


def training(files):
 m=json.loads(files.file(INPUTS['training']).read_text())
 if m['status']!='SEALED_TRAINING_ONLY_STRUCTURED_CONTRASTIVE_PROBE' or any(m[k] is not False for k in ('recall_measured','fold0_opened','fold1_opened')):
  raise ValueError('Original sealed TRAIN-only source required')
 l=json.loads(files.file(m['launch']).read_text());w=json.loads(files.file(m['worker']).read_text())
 for ref in l['implementation'].values():files.file(ref)
 if w['training_panel_only'] is not True or w['output_candidates'] is not None or w['identifiers_as_features'] is not False or w['dimension']!=768:
  raise ValueError('Original masked fixed NOM+address cache required')
 d=json.loads(files.file(l['inputs']).read_text());validate(d['rows'],d['registry'])
 if (d['truth_scope']!='TRAINER_ONLY' or d['all_positive_documents_preserved'] is not True or d['training_queries']!=512
  or not isinstance(d['masked_identifier_spans'],int) or d['masked_identifier_spans']<1
  or w['query_ids']!=[r['query_id'] for r in d['rows']]):raise ValueError('Every original TRAIN row and vector aligned required')
 x=np.load(files.file(w['query_vectors']),allow_pickle=False)
 if x.shape!=(512,768) or not np.isfinite(x).all():raise ValueError('Full unchanged fixed TRAIN vector cache required')
 old=json.loads(files.file(INPUTS['query_launch']).read_text())
 bundle=json.loads(files.file(old['bundle']).read_text());encoded=json.loads(files.file(old['expansions']).read_text())
 if (old['truth_opened'] is not False or encoded['truth_opened'] is not False or bundle['encoder_prefix']!='nom'
  or bundle['model_refs']!=l['model_refs'] or bundle['model_path']!=l['config']['model_path']
  or bundle['max_length']!=l['config']['max_length']):raise ValueError('Original NOM serving encoder compatibility required')
 for ref in l['model_refs'].values():files.file(ref)
 pilot_ids={r['query_id'] for r in encoded['rows']}
 if len(pilot_ids)!=128 or pilot_ids&set(w['query_ids']) or any(r['fold'] not in (2,3,4) for r in encoded['rows']):
  raise ValueError('TRAIN512 must remain disjoint from original query-only128')
 heads=json.loads(files.file(old['models']).read_text())
 if set(heads)!={'2','3','4'}:raise ValueError('Three original protected safe heads required')
 return d,x[:,:384],heads,dict(model_refs=l['model_refs'],model_path=l['config']['model_path'],
  max_length=l['config']['max_length'],encoder_prefix='nom',training_vectors=w['query_vectors'],
  training_inputs=l['inputs'],safe_heads=old['models'],query_launch=INPUTS['query_launch'],pilot_ids=sorted(pilot_ids))


def worker(root):
 l=load(root);dest=root/'phase'
 try:
  with Budget(LIMITS) as budget:
   def guard():
    if budget.failure:raise RuntimeError(budget.failure)
   files=SourceFiles(guard);d,x,heads,provenance=training(files)
   municipal=source.MunicipalIndex(files,INPUTS['municipal'])
   try:roles=municipal.all_roles()
   finally:municipal.close()
   models={};fitted={}
   for held in (2,3,4):
    guard();p,metadata=fit(x,d['rows'],d['registry'],heads[str(held)],roles,held)
    if metadata['query_ids']!=heads[str(held)]['query_ids']:raise ValueError('No extra TRAIN family authority')
    path=dest/f'public_role_fold{held}.npy';np.save(path,p,allow_pickle=False)
    models[str(held)]=dict(**metadata,parameter=record(path),source=INPUTS['municipal'],query_encoder='NOM_ONLY_ORIGINAL_384')
    fitted[str(held)]=(p,metadata)
   reports=controls(x,d['rows'],roles,fitted)
   files.verify_after();load(root)
   write_json(dest/'models.json',models)
   write_json(dest/'report.json',dict(controls=reports,role_parent_counts={r:len(roles[r]) for r in ROLES},
    training_queries=512,training_truth_scope='TRAINER_ONLY_FOLDS234',original_queries_disjoint=True,
    source_epochs=list(files.entries.values()),provenance=provenance,model_fit=True,**FALSE))
  validate_report(budget.report())
  write_json(dest/'manifest.json',dict(status='QUARANTINED_V280_TRAIN',launch=record(root/'launch.json'),
   preflight=record(dest/'preflight.json'),models=record(dest/'models.json'),report=record(dest/'report.json'),resources=budget.report(),**FALSE))
 except BaseException as exc:write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**FALSE));raise


def admission(root):
 root=Path(root);load(root);d=root/'phase';parent_exit(d)
 m=json.loads((d/'manifest.json').read_text());validate_report(m['resources'])
 gate=json.loads(verify(m['preflight']).read_text());r=json.loads(verify(m['report']).read_text())
 if (m['status']!='QUARANTINED_V280_TRAIN' or m['launch']!=record(root/'launch.json') or m['preflight']!=record(d/'preflight.json')
  or m['models']!=record(d/'models.json') or m['report']!=record(d/'report.json')
  or gate['policy_id']!=POLICY or gate['quiet_seconds']<30 or gate['pass_system_quiet_policy'] is not True
  or r['training_truth_scope']!='TRAINER_ONLY_FOLDS234' or r['training_queries']!=512 or r['model_fit'] is not True
  or r['original_queries_disjoint'] is not True or any(x.get(k) is not False for x in (m,r) for k in FALSE)):
  raise ValueError('Actual successful own TRAIN-only complete producer required')
 # Epoch verification is metadata-only: serving must NOT parse/open TRAIN scenes.
 check_epochs(canonical_epochs(r['source_epochs']))
 models=json.loads(verify(m['models']).read_text())
 if set(models)!={'2','3','4'}:raise ValueError('Three OOF role heads required')
 for f,p in models.items():
  a=np.load(verify(p['parameter']),allow_pickle=False)
  if (p['heldout_fold']!=int(f) or set(p['training_folds'])!={2,3,4}-{int(f)} or p['roles']!=list(ROLES)
   or p['ridge']!=RIDGE or p['feature_dimension']!=384 or p['heldout_entity_overlap']!=0
   or p['source']!=INPUTS['municipal'] or p['identifiers_as_features'] is not False or p['crm_alias_dictionary'] is not False
   or p['probabilities_calibrated'] is not False or p['query_encoder']!='NOM_ONLY_ORIGINAL_384'
   or len(set(p['query_ids']))!=p['training_queries'] or set(p['query_ids'])&set(r['provenance']['pilot_ids'])
   or a.shape!=(385,2) or not np.isfinite(a).all()):raise ValueError('Protected original fixed query-role heads required')
 return m,r,models


def sealed(root):
 root=Path(root);w,r,models=admission(root);m=json.loads((root/'manifest.json').read_text())
 if (m['status']!=STATUS or m['producer']!=record(root/'phase/manifest.json') or m['models']!=w['models']
  or m['report']!=w['report'] or m['launch']!=record(root/'launch.json') or any(m.get(k) is not False for k in FALSE)):
  raise ValueError('Same actual complete TRAIN-only role producer required')
 return m,r,models


def run(root):
 if root.exists() or not root.resolve().is_relative_to(OUTPUT.resolve()):raise ValueError('Fresh isolated TRAIN-only root required')
 root.mkdir(parents=True,exist_ok=False);d=root/'phase';d.mkdir()
 try:
  write_json(root/'launch.json',dict(inputs=INPUTS,limits=LIMITS,ridge=RIDGE,roles=list(ROLES),implementation=implementation(),**FALSE))
  write_json(d/'preflight.json',quiet(30,50))
  with (d/'worker.log').open('x') as log:
   s=supervise([sys.executable,'-u','-m','scripts.retrieval_v280_public_role_training','worker','--output',str(root)],log,
    {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},d/'supervisor_stop.json')
  write_json(d/'supervision.json',s);w,r,models=admission(root)
  write_json(root/'manifest.json',dict(status=STATUS,launch=record(root/'launch.json'),producer=record(d/'manifest.json'),
   models=w['models'],report=w['report'],training_truth_scope='TRAINER_ONLY_FOLDS234',**FALSE))
  sealed(root);print(json.dumps(dict(status=STATUS,controls=r['controls'],role_parent_counts=r['role_parent_counts'])),flush=True)
 except BaseException as exc:write_json(root/'status.json',dict(status='STOP',reason=str(exc),**FALSE));raise


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker'))
 p.add_argument('--output',type=Path,required=True);a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
