"""Postseal-only initial100 parent/site audit; never a retrieval feature."""
import argparse,importlib,json,os,sys
from pathlib import Path
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,POLICY,quiet,validate_report,parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_postscore_audit_v187 import check_epochs
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v299_parent_site_diagnostic')
VERSIONS=('288','291','294','298')
STATUS='SEALED_V299_POSTSEAL_ONLY_INITIAL100_PARENT_SITE_DIAGNOSTIC'
FALSE={k:False for k in ('fold0_opened','fold1_opened','downstream_models_changed','ready_for_fold1',
 'full_oof','production_slo_validated','generation_modified','aliases_as_features','positive_injected',
 'target_used_during_generation','exhaustive_fine_loss_attribution_claimed')}


def module(version):
 if version not in VERSIONS:raise ValueError('Explicit closed-fold complete postseal evaluator required')
 return importlib.import_module('scripts.evaluate_retrieval_v'+version+'_postseal')


def source(evaluation,version):
 n=module(version);admitted=n.admission(Path(evaluation)) # FIRST, before target table/target refs opened.
 p,ref=n.permitted(Path(evaluation));r=json.loads(verify(admitted['artifacts']['report']).read_text())
 if r['primary']['queries']!=128 or r['generation']!=p['generation']:raise ValueError('Same original complete allhuman128 metric required')
 return n,p,r


def score(outputs,controls,labels):
 if len(controls)!=128 or len(outputs)!=128 or len(labels)!=128:raise ValueError('Allhuman128 denominator required')
 scope={c['query_id']:c['fold'] for c in controls}
 if len(scope)!=128 or any(f not in (2,3,4) for f in scope.values()):raise ValueError('Only original safe-fold128 controls')
 byid={r['query_id']:r for r in labels}
 if set(byid)!=set(scope) or len(byid)!=len(labels):raise ValueError('All original targets required, no filtering')
 rows=[]
 for o,c in zip(outputs,controls):
  r=byid[c['query_id']];s,p=r['human_ground_truth_siret'],r['human_ground_truth_siren']
  if (o['query_id']!=c['query_id'] or o['fold']!=c['fold'] or o['position']!=c['position'] or r['fold']!=c['fold']
   or not isinstance(s,str) or not isinstance(p,str) or len(s)!=14 or len(p)!=9
   or not s.isascii() or not p.isascii() or not s.isdigit() or not p.isdigit() or not s.startswith(p)):
   raise ValueError('Exact original query/entity controls, IDs only postseal metric joins')
  candidates=o['results']['primary']
  if len(candidates)>100 or len({x['siret'] for x in candidates})!=len(candidates):raise ValueError('Original emitted100 unique cap required')
  exact=any(x['siret']==s for x in candidates);ranks=[i for i,x in enumerate(candidates,1) if x['siren']==p]
  if exact and not ranks:raise ValueError('Exact site cannot conflict with its official parent')
  rows.append(dict(position=c['position'],fold=c['fold'],exact_hit=exact,parent_hit=bool(ranks),
   parent_sites_returned=len(ranks),first_parent_rank=min(ranks) if ranks else None,
   loss='HIT' if exact else ('PARENT_RETURNED_EXACT_SITE_MISSING' if ranks else 'PARENT_MISSING')))
 exact=sum(r['exact_hit'] for r in rows);parents=sum(r['parent_hit'] for r in rows)
 return dict(queries=128,exact_hits=exact,exact_recall=exact/128,parent_hits=parents,parent_recall=parents/128,
  exact_given_parent=exact/parents if parents else None,counts={k:sum(r['loss']==k for r in rows)
   for k in ('HIT','PARENT_RETURNED_EXACT_SITE_MISSING','PARENT_MISSING')},outcomes=rows)


def implementation(version):
 refs=module(version).implementation()
 for p in (Path(__file__),ROOT/'docs/retrieval_v299_parent_site_diagnostic_contract.md',
  ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py'):
  refs[str(p)]=record(p)
 return refs


def load(root):
 root=Path(root)
 if not root.resolve().is_relative_to(OUTPUT.resolve()) or (root/'status.json').exists() or any(root.glob('phase/*stop.json')):
  raise ValueError('Own fresh postseal root, no STOP reuse')
 l=json.loads((root/'launch.json').read_text())
 if l['version'] not in VERSIONS or l['limits']!=LIMITS or any(l.get(k) is not False for k in FALSE):
  raise ValueError('Frozen diagnostic only authorized scope required')
 for ref in l['implementation'].values():verify(ref)
 verify(l['evaluation']);source(Path(l['evaluation']['path']).parent,l['version'])
 return l


def worker(root):
 l=load(root);d=root/'phase'
 try:
  with Budget(LIMITS) as budget:
   n,p,measured=source(Path(l['evaluation']['path']).parent,l['version'])
   labels,official,acceptable,refs,epochs,excluded=n.targets(n.config(),p)
   outputs=json.loads(verify(p['outputs']).read_text())['rows'];result=score(outputs,p['controls'],labels)
   if result['exact_hits']!=measured['primary']['exact_hits']:raise ValueError('Same exact allhuman metric required')
   check_epochs(epochs);source(Path(l['evaluation']['path']).parent,l['version']);load(root)
   write_json(d/'report.json',dict(**result,source_evaluation=l['evaluation'],generation=p['generation'],
    initial100_output=p['outputs'],target_references=refs,target_epochs=epochs,excluded_missing_human_labels=excluded,
    truth_opened=True,recall_measured=True,scope='ADAPTIVE_DEVELOPMENT128_POSTSEAL_NOT_FULL_OOF',**FALSE))
  validate_report(budget.report())
  write_json(d/'manifest.json',dict(status='QUARANTINED_V299_POSTSEAL',launch=record(root/'launch.json'),
   preflight=record(d/'preflight.json'),report=record(d/'report.json'),resources=budget.report(),**FALSE))
 except BaseException as exc:write_json(d/'worker_stop.json',dict(status='STOP',reason=str(exc),**FALSE));raise


def admission(root):
 root=Path(root);l=load(root);d=root/'phase';parent_exit(d)
 m=json.loads((d/'manifest.json').read_text());validate_report(m['resources'])
 gate=json.loads(verify(m['preflight']).read_text());r=json.loads(verify(m['report']).read_text())
 if (m['status']!='QUARANTINED_V299_POSTSEAL' or m['launch']!=record(root/'launch.json') or m['preflight']!=record(d/'preflight.json')
  or m['report']!=record(d/'report.json') or gate['policy_id']!=POLICY or gate['quiet_seconds']<30 or gate['pass_system_quiet_policy'] is not True
  or r['queries']!=128 or r['truth_opened'] is not True or r['source_evaluation']!=l['evaluation']
  or any(x.get(k) is not False for x in (m,r) for k in FALSE)):
  raise ValueError('Actual own successful postseal complete diagnostic required')
 check_epochs(r['target_epochs'])
 for ref in r['target_references']:verify(ref)
 return m,r


def sealed(root):
 root=Path(root);w,r=admission(root);m=json.loads((root/'manifest.json').read_text())
 if m['status']!=STATUS or m['producer']!=record(root/'phase/manifest.json') or m['report']!=w['report']:
  raise ValueError('Same actual complete postseal producer required')
 return m,r


def run(root,evaluation,version):
 n,p,r=source(evaluation,version) # FIRST: no targets until all actual generation/evaluation exits PASS.
 if root.exists() or not root.resolve().is_relative_to(OUTPUT.resolve()):raise ValueError('Fresh own diagnostic root required')
 root.mkdir(parents=True,exist_ok=False);d=root/'phase';d.mkdir()
 try:
  write_json(root/'launch.json',dict(evaluation=record(evaluation/'manifest.json'),version=version,limits=LIMITS,implementation=implementation(version),**FALSE))
  write_json(d/'preflight.json',quiet(30,50))
  with (d/'worker.log').open('x') as log:
   s=supervise([sys.executable,'-u','-m','scripts.retrieval_v299_parent_site_diagnostic','worker','--output',str(root)],log,
    {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},d/'supervisor_stop.json')
  write_json(d/'supervision.json',s);w,r=admission(root)
  write_json(root/'manifest.json',dict(status=STATUS,producer=record(d/'manifest.json'),report=w['report'],**FALSE))
  sealed(root);print(json.dumps({k:r[k] for k in ('queries','exact_hits','parent_hits','counts')}),flush=True)
 except BaseException as exc:write_json(root/'status.json',dict(status='STOP',reason=str(exc),**FALSE));raise


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker'));p.add_argument('--output',type=Path,required=True)
 p.add_argument('--evaluation',type=Path);p.add_argument('--version',choices=VERSIONS);a=p.parse_args()
 if a.mode=='run' and (a.evaluation is None or a.version is None):p.error('Explicit complete evaluated generation required')
 run(a.output,a.evaluation,a.version) if a.mode=='run' else worker(a.output)
