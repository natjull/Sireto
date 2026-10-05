"""Explicit whole INSEE COG region snapshots, SOURCE-ONLY, no CRM requests."""
import argparse,csv,io,json,os,re,sys,zipfile
from pathlib import Path
from urllib.request import Request,urlopen
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,POLICY,quiet,validate_report,parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v281_official_region_snapshots')
SOURCES={
 '2015':dict(url='https://www.insee.fr/fr/statistiques/fichier/2560698/reg2015-txt.zip',filename='reg2015-txt.zip',member='reg2015.txt',encoding='latin-1',delimiter='\t',fields=['REGION','CHEFLIEU','TNCC','NCC','NCCENR'],rows=27),
 '2016':dict(url='https://www.insee.fr/fr/statistiques/fichier/2114819/reg2016-txt.zip',filename='reg2016-txt.zip',member='reg2016.txt',encoding='latin-1',delimiter='\t',fields=['REGION','CHEFLIEU','TNCC','NCC','NCCENR'],rows=18),
 '2026':dict(url='https://www.insee.fr/fr/statistiques/fichier/8740222/v_region_2026.csv',filename='v_region_2026.csv',member=None,encoding='utf-8-sig',delimiter=',',fields=['REG','CHEFLIEU','TNCC','NCC','NCCENR','LIBELLE'],rows=18)}
MAX_BYTES=2*1024*1024
FALSE={k:False for k in ('truth_opened','crm_opened','fold0_opened','fold1_opened','model_fit','recall_measured',
 'ready_for_fold1','downstream_models_changed','identifiers_as_features','candidates_emitted')}
STATUS='SEALED_V281_THREE_EXPLICIT_COMPLETE_OFFICIAL_REGION_SNAPSHOTS'


def parse(payload,year):
 cfg=SOURCES[year]
 if len(payload)>MAX_BYTES:raise ValueError('STOP_WHOLE_SOURCE_PAYLOAD_NO_TRUNCATION')
 if cfg['member'] is not None:
  with zipfile.ZipFile(io.BytesIO(payload)) as z:
   entries=z.infolist()
   if (len(entries)!=1 or entries[0].filename!=cfg['member'] or entries[0].is_dir()
    or entries[0].file_size>MAX_BYTES or entries[0].flag_bits&1):
    raise ValueError('Exact single explicit ZIP member within byte guard, no filesystem extraction')
   content=z.read(entries[0])
 else:content=payload
 reader=csv.DictReader(io.StringIO(content.decode(cfg['encoding']),newline=''),delimiter=cfg['delimiter'])
 if reader.fieldnames!=cfg['fields']:raise ValueError('Exact official per-year schema required, no guessed source')
 rows=list(reader);codefield='REG' if year=='2026' else 'REGION';seen=set()
 if len(rows)!=cfg['rows']:raise ValueError('Complete explicit annual table cardinality required')
 for r in rows:
  if (set(r)!=set(cfg['fields']) or any(not isinstance(v,str) or not v.strip() for v in r.values())
   or not re.fullmatch('[0-9]{2}',r[codefield]) or r[codefield] in seen
   or not re.fullmatch('[0-9A-Z]{5}',r['CHEFLIEU']) or not re.fullmatch('[0-9]',r['TNCC'])):
   raise ValueError('Every original official record retained/typed, no silent drops')
  seen.add(r[codefield])
 return rows


def implementation():
 return {str(p):record(p) for p in (Path(__file__),ROOT/'docs/retrieval_v281_official_regions_contract.md',
  ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py')}


def load(root):
 root=Path(root)
 if not root.resolve().is_relative_to(OUTPUT.resolve()) or (root/'status.json').exists() or any(root.glob('phase/*stop.json')):
  raise ValueError('Own fresh source-only root, no STOP reuse')
 l=json.loads((root/'launch.json').read_text())
 if l['sources']!=SOURCES or l['limits']!=LIMITS or l['max_bytes']!=MAX_BYTES or any(l.get(k) is not False for k in FALSE):
  raise ValueError('Frozen whole official source endpoints and resource guards required')
 for ref in l['implementation'].values():verify(ref)
 return l


def worker(root):
 load(root);d=root/'phase'
 try:
  with Budget(LIMITS) as budget:
   def guard():
    if budget.failure:raise RuntimeError(budget.failure)
   snapshots={}
   for year,cfg in SOURCES.items():
    guard();request=Request(cfg['url'],headers={'User-Agent':'SIRETO-OfficialSourceSnapshot/281'})
    with urlopen(request,timeout=20) as response:
     if response.status!=200 or response.geturl()!=cfg['url']:raise ValueError('Exact successful official HTTPS endpoint, no source drift redirect')
     headers=dict(content_type=response.headers.get('Content-Type'),date=response.headers.get('Date'),
      last_modified=response.headers.get('Last-Modified'),etag=response.headers.get('ETag'))
     payload=response.read(MAX_BYTES+1)
    guard();rows=parse(payload,year);path=d/cfg['filename'];path.write_bytes(payload)
    if parse(path.read_bytes(),year)!=rows:raise ValueError('All original fields must roundtrip')
    snapshots[year]=dict(url=cfg['url'],headers=headers,response=record(path),rows=rows,all_returned_rows_preserved=True)
   write_json(d/'source.json',dict(snapshots=snapshots,rows=sum(len(s['rows']) for s in snapshots.values()),
    complete_history_claimed=False,crm_request_sent=False,independent_territory_lineage_claimed=False,source_only=True,**FALSE))
   load(root)
  validate_report(budget.report())
  write_json(d/'manifest.json',dict(status='QUARANTINED_V281_SOURCE',launch=record(root/'launch.json'),
   preflight=record(d/'preflight.json'),source=record(d/'source.json'),resources=budget.report(),**FALSE))
 except BaseException as exc:write_json(d/'worker_stop.json',dict(status='STOP',reason=str(exc),**FALSE));raise


def admission(root):
 root=Path(root);load(root);d=root/'phase';parent_exit(d)
 m=json.loads((d/'manifest.json').read_text());validate_report(m['resources']);gate=json.loads(verify(m['preflight']).read_text())
 s=json.loads(verify(m['source']).read_text())
 if (m['status']!='QUARANTINED_V281_SOURCE' or m['launch']!=record(root/'launch.json') or m['preflight']!=record(d/'preflight.json')
  or gate['policy_id']!=POLICY or gate['quiet_seconds']<30 or gate['pass_system_quiet_policy'] is not True
  or set(s['snapshots'])!=set(SOURCES) or s['rows']!=63 or s['source_only'] is not True
  or s['complete_history_claimed'] is not False or s['crm_request_sent'] is not False or s['independent_territory_lineage_claimed'] is not False
  or any(x.get(k) is not False for x in (m,s) for k in FALSE)):
  raise ValueError('Actual own complete source producer and declared history limitations required')
 for year,cfg in SOURCES.items():
  snapshot=s['snapshots'][year]
  if snapshot['url']!=cfg['url'] or snapshot['all_returned_rows_preserved'] is not True or parse(verify(snapshot['response']).read_bytes(),year)!=snapshot['rows']:
   raise ValueError('Whole original official annual source required')
 return m,s


def sealed(root):
 root=Path(root);m,s=admission(root);out=json.loads((root/'manifest.json').read_text())
 if (out['status']!=STATUS or out['producer']!=record(root/'phase/manifest.json') or out['source']!=m['source']
  or out['rows']!=63 or any(out.get(k) is not False for k in FALSE)):raise ValueError('Same actual source-only complete producer required')
 return out,s


def run(root):
 if root.exists() or not root.resolve().is_relative_to(OUTPUT.resolve()):raise ValueError('Fresh own source root required')
 root.mkdir(parents=True,exist_ok=False);d=root/'phase';d.mkdir()
 try:
  write_json(root/'launch.json',dict(sources=SOURCES,limits=LIMITS,max_bytes=MAX_BYTES,implementation=implementation(),**FALSE))
  write_json(d/'preflight.json',quiet(30,50))
  with (d/'worker.log').open('x') as log:
   s=supervise([sys.executable,'-u','-m','scripts.retrieval_v281_official_region_snapshots','worker','--output',str(root)],log,
    {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},d/'supervisor_stop.json')
  write_json(d/'supervision.json',s);m,s=admission(root)
  write_json(root/'manifest.json',dict(status=STATUS,producer=record(d/'manifest.json'),source=m['source'],rows=63,**FALSE))
  sealed(root);print(json.dumps(dict(status=STATUS,rows={y:len(v['rows']) for y,v in s['snapshots'].items()})),flush=True)
 except BaseException as exc:write_json(root/'status.json',dict(status='STOP',reason=str(exc),**FALSE));raise


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker'))
 p.add_argument('--output',type=Path,required=True);a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
