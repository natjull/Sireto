"""Compile ALL official municipality/parent/CP facts, source-only readonly joins."""
import argparse,json,os,sqlite3,sys
from pathlib import Path
from urllib.parse import quote
from scripts import retrieval_v267_official_communes_snapshot as source
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,POLICY,quiet,validate_report,parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_postscore_audit_v187 import canonical_epochs,check_epochs
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v279_municipal_structure')
INPUT=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v267_official_communes_snapshot/snapshot_20261005_a/manifest.json',
 sha256='68427d6357b2fb9324ad26603df712183bb6a9807a02811e11d7a26cb7410177',size_bytes=984)
FALSE={k:False for k in ('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened',
 'identifiers_as_features','candidates_emitted','ready_for_fold1','downstream_models_changed')}
STATUS='SEALED_V279_ALL_RETURNED_OFFICIAL_MUNICIPAL_PARENT_JOINS'


def source_rows(files):
 root=files.file(INPUT).parent;m,s=source.sealed(root)
 for path in ('launch.json','phase/manifest.json','phase/supervision.json','phase/preflight.json'):
  files.file(record(root/path))
 files.file(m['source']);files.file(m['producer'])
 for ref in json.loads(files.file(record(root/'launch.json')).read_text())['implementation'].values():files.file(ref)
 rows,stats=source.parse(files.file(s['response']).read_bytes())
 if stats!=m['stats']:raise ValueError('Exact whole source required')
 return rows,stats


def readonly(path):
 c=sqlite3.connect('file:'+quote(str(Path(path).resolve()),safe='/')+'?mode=ro&immutable=1',uri=True)
 c.execute('PRAGMA query_only=ON');c.execute('PRAGMA cache_size=-2048')
 return c


def canonical(row):
 return (row['code'],row['nom'],row.get('siren') or None,row.get('codeEpci') or None)


def compile_db(path,rows,guard=lambda:None):
 if Path(path).exists():raise ValueError('Fresh own source database only')
 c=sqlite3.connect(path)
 try:
  c.execute('CREATE TABLE communes(code TEXT PRIMARY KEY, nom TEXT NOT NULL, siren TEXT, epci TEXT) WITHOUT ROWID')
  c.execute('CREATE TABLE postcodes(code TEXT NOT NULL, ordinal INTEGER NOT NULL, postcode TEXT NOT NULL, PRIMARY KEY(code,ordinal)) WITHOUT ROWID')
  for lo in range(0,len(rows),512):
   guard();block=rows[lo:lo+512]
   c.executemany('INSERT INTO communes VALUES(?,?,?,?)',[canonical(r) for r in block])
   c.executemany('INSERT INTO postcodes VALUES(?,?,?)',[(r['code'],i,cp) for r in block for i,cp in enumerate(r['codesPostaux'])])
  c.execute('CREATE INDEX postal_join ON postcodes(postcode,code)')
  c.commit()
 finally:c.close()
 return validate_db(path,rows)


def validate_db(path,rows=None):
 c=readonly(path)
 try:
  if c.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('Complete database integrity required')
  records=c.execute('SELECT code,nom,siren,epci FROM communes ORDER BY code').fetchall()
  postal=c.execute('SELECT code,ordinal,postcode FROM postcodes ORDER BY code,ordinal').fetchall()
  if rows is not None:
   if records!=sorted(canonical(r) for r in rows) or postal!=sorted((r['code'],i,cp) for r in rows for i,cp in enumerate(r['codesPostaux'])):
    raise ValueError('Every official source field and CP membership must roundtrip, no drops')
  return dict(rows=len(records),missing_siren=sum(r[2] is None for r in records),
   missing_epci=sum(r[3] is None for r in records),postal_memberships=len(postal))
 finally:c.close()


def implementation():
 paths=(Path(__file__),ROOT/'docs/retrieval_v279_municipal_structure_contract.md',
  ROOT/'scripts/retrieval_v267_official_communes_snapshot.py',ROOT/'docs/retrieval_v267_official_communes_contract.md',
  ROOT/'src/xgb_matcher/retrieval_structural_channels_v210.py',ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',
  ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py',ROOT/'src/xgb_matcher/retrieval_postscore_audit_v187.py')
 return {str(p):record(p) for p in paths}


def load(root):
 root=Path(root)
 if not root.resolve().is_relative_to(OUTPUT.resolve()) or (root/'status.json').exists() or any(root.glob('phase/*stop.json')):
  raise ValueError('Own fresh source, no STOP reuse')
 l=json.loads((root/'launch.json').read_text())
 if l['input']!=INPUT or l['limits']!=LIMITS or any(l.get(k) is not False for k in FALSE):raise ValueError('Frozen source-only launch required')
 for ref in l['implementation'].values():verify(ref)
 return l


def worker(root):
 l=load(root);dest=root/'phase';budget=None
 try:
  with Budget(LIMITS) as budget:
   def guard():
    if budget.failure:raise RuntimeError(budget.failure)
   files=SourceFiles(guard);rows,stats=source_rows(files)
   observed=compile_db(dest/'municipal.sqlite',rows,guard)
   if observed!=stats:raise ValueError('All source rows and published UNKNOWNs required')
   files.verify_after();load(root)
   write_json(dest/'payload.json',dict(database=record(dest/'municipal.sqlite'),stats=stats,
    source_epochs=list(files.entries.values()),all_records_roundtrip=True,source_only=True,**FALSE))
  validate_report(budget.report())
  write_json(dest/'manifest.json',dict(status='QUARANTINED_V279_SOURCE',launch=record(root/'launch.json'),
   preflight=record(dest/'preflight.json'),payload=record(dest/'payload.json'),resources=budget.report(),**FALSE))
 except BaseException as exc:write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**FALSE));raise


def admission(root):
 load(root);d=root/'phase';parent_exit(d);m=json.loads((d/'manifest.json').read_text());validate_report(m['resources'])
 gate=json.loads(verify(m['preflight']).read_text());p=json.loads(verify(m['payload']).read_text())
 if (m['status']!='QUARANTINED_V279_SOURCE' or m['launch']!=record(root/'launch.json')
  or m['preflight']!=record(d/'preflight.json') or gate['policy_id']!=POLICY
  or gate['quiet_seconds']<30 or gate['pass_system_quiet_policy'] is not True
  or p['all_records_roundtrip'] is not True or p['source_only'] is not True
  or any(x.get(k) is not False for x in (m,p) for k in FALSE)):raise ValueError('Actual own complete successful source producer required')
 check_epochs(canonical_epochs(p['source_epochs']))
 if validate_db(verify(p['database']))!=p['stats']:raise ValueError('Same whole source database required')
 return m,p


def sealed(root):
 root=Path(root);m,p=admission(root);s=json.loads((root/'manifest.json').read_text())
 if (s['status']!=STATUS or s['producer']!=record(root/'phase/manifest.json') or s['database']!=p['database']
  or s['stats']!=p['stats'] or s['input']!=INPUT or any(s.get(k) is not False for k in FALSE)):
  raise ValueError('Same actual full source producer required')
 return s,p


class MunicipalIndex:
 def __init__(self,files,reference):
  root=files.file(reference).parent;m,p=sealed(root)
  for path in ('launch.json','phase/manifest.json','phase/supervision.json','phase/preflight.json','phase/payload.json'):
   files.file(record(root/path))
  for ref in json.loads(files.file(record(root/'launch.json')).read_text())['implementation'].values():files.file(ref)
  # Used original source binding is propagated, not replaced by metadata.
  for e in p['source_epochs']:files.file(e['reference'])
  self.c=readonly(files.file(m['database']));self.files=files
 def parents(self,q):
  self.files.guard()
  rows=self.c.execute('SELECT siren,epci FROM communes WHERE code=? UNION SELECT c.siren,c.epci FROM communes c JOIN postcodes p ON p.code=c.code WHERE p.postcode=?',
   (q.insee,q.postcode)).fetchall()
  return {role:{r[j] for r in rows if r[j] is not None} for j,role in enumerate(('commune','epci'))}
 def all_roles(self):
  return {role:{r[0] for r in self.c.execute('SELECT DISTINCT '+column+' FROM communes WHERE '+column+' IS NOT NULL')}
   for role,column in (('commune','siren'),('epci','epci'))}
 def close(self):self.c.close();self.files.check_epochs()


def run(root):
 if root.exists() or not root.resolve().is_relative_to(OUTPUT.resolve()):raise ValueError('Fresh own output required')
 root.mkdir(parents=True,exist_ok=False);d=root/'phase';d.mkdir()
 try:
  write_json(root/'launch.json',dict(input=INPUT,limits=LIMITS,implementation=implementation(),**FALSE))
  write_json(d/'preflight.json',quiet(30,50))
  with (d/'worker.log').open('x') as log:
   s=supervise([sys.executable,'-u','-m','scripts.retrieval_v279_municipal_structure','worker','--output',str(root)],log,
    {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},d/'supervisor_stop.json')
  write_json(d/'supervision.json',s);m,p=admission(root)
  write_json(root/'manifest.json',dict(status=STATUS,input=INPUT,producer=record(d/'manifest.json'),
   database=p['database'],stats=p['stats'],source_only=True,**FALSE))
  sealed(root);print(json.dumps(dict(status=STATUS,stats=p['stats'])),flush=True)
 except BaseException as exc:write_json(root/'status.json',dict(status='STOP',reason=str(exc),**FALSE));raise


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker'))
 p.add_argument('--output',type=Path,required=True);a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
