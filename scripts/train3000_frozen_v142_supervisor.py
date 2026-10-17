"""Unified v174 controller. --prepare-ready NEVER starts full3000.

The explicit --full command is gated by sealed four-family fidelity and20
complete new native smokes. Frozen8/12GiB stopped-VM full-run policy retained.
"""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train3000_frozen_v142_brief10 as control
mini=control.mini
GIB=control.GIB
OUT=control.OUT
SOURCE_NAMES=('supervisor','pipeline','contexts','numeric','osm','activity','brief10','fidelity','bano')


def sources():
    return [mini.record(ROOT/'scripts'/('train3000_frozen_v142_'+s+'.py')) for s in SOURCE_NAMES]


def task(phase,inputs,full=False):
    code={Path(ref['path']).name:ref['sha256'] for ref in sources()}
    key=phase+'_'+mini.hashlib.sha256(mini.canonical(dict(inputs=inputs,code=code))).hexdigest()
    completed=OUT/'completed'/(key+'.json')
    if completed.exists():
        saved=mini.read(completed)
        for ref in saved['sources']:mini.record(ref['path'],ref['sha256'])
        receipt=mini.read(mini.verified(saved['receipt']))
        if receipt['exit_code'] or not receipt['owned_child_exit_confirmed']:raise ValueError('Owned exit0 required')
        mini.verified(saved['result'])
        if phase=='audit':
            for path,expected in mini.read(saved['result']['path'])['signatures'].items():
                stat=Path(path).stat()
                if [stat.st_size,stat.st_mtime_ns,stat.st_ino]!=expected:raise ValueError('Frozen authority changed after SHA audit')
        return saved
    required=8*GIB if full else 6*GIB
    for old in OUT.glob(key+'_*'+'/receipt.json'):
        r=mini.read(old)
        if r['stop_reason']=='WAIT_BRIEF10_RESOURCE_CAP':
            if r['cap_bytes']>=12*GIB or (r['cap_bytes']>=8*GIB and (not full or phase!='generate')):
                raise RuntimeError('Terminal memory stop; preserve all attempts, no skipped query')
            required=max(required,12*GIB if full and phase=='generate' else 8*GIB)
    while True:
        d=OUT/(key+'_'+uuid.uuid4().hex[:10]);d.mkdir(parents=True)
        if required<=8*GIB:
            cap,waited,evidence=control.admission(d,required)
        else:
            waited=0
            while True:
                vm=control.resources.hydraulic();mem=control.resources.free_inactive_memory()
                evidence=dict(hydraulic=vm,memory=mem,required_free_inactive_bytes=9*GIB)
                with (d/'admission.jsonl').open('a') as f:f.write(json.dumps(dict(evidence,wait_seconds=waited))+'\n')
                if vm['status']=='Stopped' and not vm.get('uncertain') and (mem['free_inactive_bytes'] or 0)>=9*GIB:break
                if control.STOP:raise InterruptedError('No worker; interruption during escalation wait')
                time.sleep(60);waited+=60
            cap=12*GIB
        refs=sources()
        for ref in refs:
            with (d/Path(ref['path']).name).open('xb') as f:f.write(Path(ref['path']).read_bytes())
        mini.save(d/'input.json',inputs)
        tmp=d/'tmp';tmp.mkdir()
        interpreter=mini.bano.cohort.preflight.H/'runtime/bin/python' if phase=='prepare' else ROOT/'.venv/bin/python'
        command=[str(interpreter),'-u',str(Path(__file__).resolve()),'--worker',phase,'--directory',str(d)]
        env=dict(os.environ,SIRETO_BRIEF10_CAP_BYTES=str(cap),SIRETO_BRIEF10_TELEMETRY=str(d/'memory.jsonl'),
            TMPDIR=str(tmp),SQLITE_TMPDIR=str(tmp),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OSMIUM_POOL_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        peak=0;reason=None;started=time.monotonic()
        with (d/'worker.log').open('xb') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            observed=[evidence['hydraulic']]
            def watch(owned_child=child,observed_state=observed):
                while owned_child.poll() is None:observed_state[0]=control.resources.hydraulic();time.sleep(2)
            threading.Thread(target=watch,daemon=True).start()
            mini.save(d/'controller.json',dict(command=command,child_pid=child.pid,sources=refs,cap_bytes=cap,
                admission=evidence,full_mode=full,phase=phase,wait_seconds=waited))
            try:
                while child.poll() is None:
                    try:
                        p=control.psutil.Process(child.pid)
                        rss=sum(p.memory_info().rss for p in [p,*p.children(recursive=True)] if p.is_running());peak=max(peak,rss)
                    except control.psutil.NoSuchProcess:pass
                    if control.STOP:reason='USER_INTERRUPT'
                    elif peak>=cap-control.MARGIN:reason='WAIT_BRIEF10_RESOURCE_CAP'
                    elif (full or cap>=8*GIB) and (observed[0]['status']!='Stopped' or observed[0].get('uncertain')):reason='WAIT_VM_RE_ADMISSION'
                    elif control.psutil.disk_usage(mini.bano.cohort.preflight.H).free<10*GIB:reason='STOP_DISK_BELOW_10_GIB'
                    if reason:child.terminate();break
                    time.sleep(.05)
                try:rc=child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();rc=child.wait()
            finally:
                if child.poll() is None:child.terminate();child.wait()
        if (d/'resource_yield.json').exists():reason='WAIT_BRIEF10_RESOURCE_CAP'
        mini.save(d/'receipt.json',dict(exit_code=rc,owned_child_exit_confirmed=True,stop_reason=reason,
            peak_sampled_child_rss_bytes=peak,cap_bytes=cap,wait_seconds=waited,elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(phase=phase,directory=str(d),exit_code=rc,reason=reason)),flush=True)
        if reason=='WAIT_VM_RE_ADMISSION':continue
        if reason=='WAIT_BRIEF10_RESOURCE_CAP':
            if cap<8*GIB:required=8*GIB;continue
            if full and phase=='generate' and cap==8*GIB:required=12*GIB;continue
        if reason or rc:raise RuntimeError('STOP: '+str(d/'worker.log'))
        saved=dict(result=mini.record(d/'result.json'),receipt=mini.record(d/'receipt.json'),sources=refs)
        mini.save(completed,saved);return saved


def automatic_commit(report,full):
    # Never stage any foreign path, nor absorb a pre-existing handover edit.
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Foreign staged changes; final commit refused')
    protected=['handover.md']
    if full:protected+=['docs/train3000_frozen_v142_result.json','docs/train3000_frozen_v142_result.md']
    if subprocess.run(['git','diff','--quiet','--',*protected],cwd=ROOT).returncode:
        raise RuntimeError('Protected report/handover modified externally; commit refused')
    stem='train3000_frozen_v142_result' if full else 'train3000_frozen_v142_brief10_result'
    path=ROOT/'docs'/(stem+'.json');markdown=path.with_suffix('.md')
    # These are the authorized final reports, not immutable generation receipts.
    with path.open('w') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with markdown.open('w') as f:
        if report['status']=='STOP_BEFORE_COMPLETION':
            f.write('# STOP train3000 V142\n'+report['error']+'\n')
            f.write('Artefacts et reçus conservés ; aucun succès non mesuré.\n')
            f.write('Aucun lancement complet automatique ni modification de labels.\n')
            f.write('Détails : `'+path.name+'` ; examiner le dernier worker.log.\n')
        else:
            smoke=report['smoke'];error=report['families']['numeric51']['result']['max_absolute_error']
            f.write('# Train3000 V142 — '+report['status']+'\n\n')
            f.write('Fidélité : activité20/20, OSM20/20, numérique51 :20/20 ; BANO20/20 hérité.\n')
            f.write('Écart maximal des vecteurs numérique51 : '+str(error)+'.\n')
            f.write('Smoke natif :20 nouveaux dossiers, aucun label évalué pendant le smoke.\n')
            f.write('Préparation : '+str(smoke['seconds_per_query_preparation'])+'s/requête ; génération : '+str(smoke['seconds_per_query_generation'])+'s/requête.\n')
            f.write('Pics RSS préparation/génération : '+str(smoke['preparation_peak_rss_bytes'])+'/'+str(smoke['generation_peak_rss_bytes'])+'octets.\n')
            f.write('ETA3000 préparation/génération : '+str(smoke['preparation_eta3000_seconds'])+'/'+str(smoke['generation_eta3000_seconds'])+'s, hors attentes ressources illimitées.\n')
            f.write('Attentes documentées (pas de60s achevés) : '+str(report['completed_60s_wait_seconds'])+'s.\n')
            f.write('Full lancé : '+str(report['full3000_started'])+' ; coefficient-OOF seulement, aucun apprentissage.\n')
            if 'evaluation' in report:f.write('Mesures exact/SIREN/site_union : '+str(report['evaluation']['counts'])+' sur3000, aucun dénominateur filtré.\n')
            f.write('\nCommande complète, UNIQUEMENT après autorisation explicite :\n```bash\n'+report['launch_command']+'\n```\n')
            f.write('\nDétails et sources : `'+path.name+'`.\n')
    with (ROOT/'handover.md').open('a') as f:
        f.write('\n- 2026-10-17 — Brief10 '+report['status']+' ; détails `docs/'+stem+'.json`. '
            'Commit isolé auto-référencé : `git log -1 --format=%H -- docs/'+stem+'.json`. '
            'Aucune action VM/web/push/fit ni modification de vérité ; reçus exit0 possédés et gates avant évaluation.\n')
    paths=[str(path.relative_to(ROOT)),str(markdown.relative_to(ROOT)),'handover.md']
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())
    if staged!=set(paths):raise RuntimeError('Unexpected staging preserved; no commit')
    subprocess.run(['git','commit','--only','-m','docs: record '+report['status']+' frozen V142 train3000','--',*paths],cwd=ROOT,check=True)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def run(full):
    OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,control.interrupted);signal.signal(signal.SIGINT,control.interrupted)
    with (OUT/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cohort_binding=OUT/'cohort_binding.json'
        if not cohort_binding.exists():
            mini.save(cohort_binding,dict(refs={name:mini.record(mini.bano.cohort.OUT/name) for name in
                ('sample.json','queries.json','assignment.json','pilot_fidelity_queries.json','label_side_sample.json')},
                private_sidecar_hash_only_no_labels_decoded=True))
        for ref in mini.read(cohort_binding)['refs'].values():mini.verified(ref)
        completion=OUT/('full_complete.json' if full else 'ready_complete.json')
        if completion.exists():
            saved=mini.read(completion)
            for ref in saved['sources']:mini.record(ref['path'],ref['sha256'])
            mini.verified(saved['report'])
            print(json.dumps(dict(status='ALREADY_COMPLETE',completion=str(completion))),flush=True);return
        if full:
            if not (OUT/'ready.json').exists():raise ValueError('Four-family+new20 smoke ready seal REQUIRED before --full')
            for ref in mini.read(OUT/'ready.json')['sources']:mini.record(ref['path'],ref['sha256'])
        results={family:control.stage(family) for family in ('activity','osm','numeric51')}
        bano=mini.read(ROOT/'docs/train3000_frozen_v142_bano_fidelity_brief9.json')
        if bano['result']['context_matches']!=20 or bano['result']['weighted_delta_matches']!=20:raise ValueError('BANO20 gate')
        mini.verified(bano['result_ref']);receipt=mini.read(mini.verified(bano['receipt_ref']))
        if receipt['exit_code'] or not receipt['owned_child_exit_confirmed']:raise ValueError('BANO owned exit0')
        audit=task('audit',dict(frozen_manifest=mini.record(mini.bano.cohort.preflight.FROZEN),mode='full' if full else 'ready'),full=full)
        sample=mini.read(mini.bano.cohort.OUT/'sample.json')
        smoke=sample['smoke_query_ids']
        if len(smoke)!=20:raise ValueError('Sealed20 new smoke scope')
        prep=task('prepare',dict(ids=smoke),full=full)
        pr=mini.read(mini.verified(prep['result']));prepared=Path(prep['result']['path']).parent
        smoke_outputs={qid:task('generate',dict(binding=pr['bindings'][qid],prepared=str(prepared)),full=full) for qid in smoke}
        generated=[mini.read(mini.verified(r['result'])) for r in smoke_outputs.values()]
        exits=[mini.read(mini.verified(r['receipt'])) for r in smoke_outputs.values()]
        preparation_exit=mini.read(mini.verified(prep['receipt']))
        preparation=preparation_exit['elapsed_seconds'];generation=sum(r['elapsed_seconds'] for r in exits)
        prep_peak=max(pr['peak_rss_bytes'],preparation_exit['peak_sampled_child_rss_bytes'])
        gen_peak=max([r['peak_rss_bytes'] for r in generated]+[r['peak_sampled_child_rss_bytes'] for r in exits])
        smoke_report=dict(queries=20,preparation_seconds=preparation,generation_seconds=generation,
            preparation_worker_seconds=pr['elapsed_seconds'],generation_worker_seconds=sum(r['elapsed_seconds'] for r in generated),
            seconds_per_query_preparation=preparation/20,seconds_per_query_generation=generation/20,
            preparation_peak_rss_bytes=prep_peak,generation_peak_rss_bytes=gen_peak,peak_rss_bytes=max(prep_peak,gen_peak),
            preparation_eta3000_seconds=preparation*150,generation_eta3000_seconds=generation*150,
            eta_excludes_unbounded_resource_waits=True,eta_method='Owned exit0 process wall times including startup,20-query batch times150; descriptive extrapolation, not SLO')
        launch='nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/train3000_frozen_v142_supervisor.py --full >> '+str(OUT/'full3000.log')+' 2>&1 < /dev/null &'
        waited=0
        for path in OUT.glob('**/admission.jsonl'):
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            waited+=max((r.get('previous_wait_seconds',r.get('wait_seconds',0)) for r in rows),default=0)
        report=dict(status='READY_FOR_EXPLICIT_FULL_LAUNCH',families=results,bano=bano['result'],smoke=smoke_report,
            completed_60s_wait_seconds=waited,
            full3000_started=False,launch_command=launch,full_hash_audit=audit,sources=sources())
        if not full:
            mini.save(OUT/'ready.json',report)
            commit=automatic_commit(report,False)
            mini.save(completion,dict(commit=commit,report=mini.record(OUT/'ready.json'),sources=sources()))
            return
        ready=mini.read(OUT/'ready.json')
        for ref in ready['sources']:mini.record(ref['path'],ref['sha256'])
        outputs=dict(smoke_outputs)
        remaining=[q for q in sample['selected_query_ids'] if q not in outputs]
        for lo in range(0,len(remaining),20):
            ids=remaining[lo:lo+20];p=task('prepare',dict(ids=ids),full=True)
            r=mini.read(mini.verified(p['result']));root=Path(p['result']['path']).parent
            for qid in ids:outputs[qid]=task('generate',dict(binding=r['bindings'][qid],prepared=str(root)),full=True)
        if len(outputs)!=3000 or set(outputs)!=set(sample['selected_query_ids']):raise ValueError('No partial population seal')
        mini.save(OUT/'generation_seal.json',dict(outputs=outputs,queries=3000,no_labels_opened=True))
        evaluation=task('evaluate',dict(seal=mini.record(OUT/'generation_seal.json')),full=True)
        report.update(status='COMPLETE_TRAIN3000_FROZEN_V142',full3000_started=True,
            evaluation=mini.read(mini.verified(evaluation['result'])),evaluation_receipt=evaluation,
            generation_seal=mini.record(OUT/'generation_seal.json'))
        commit=automatic_commit(report,True)
        mini.save(completion,dict(commit=commit,report=mini.record(ROOT/'docs/train3000_frozen_v142_result.json'),sources=sources()))


def worker(phase,d):
    if d is None or not d.resolve().is_relative_to(OUT.resolve()):raise ValueError('Owned v174 directory')
    control.guard()
    import duckdb
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    real_connect=duckdb.connect
    class Connection:
        def __init__(self,*args,**kwargs):
            self.c=real_connect(*args,**kwargs);self.c.execute('SET threads=1')
        def __getattr__(self,name):return getattr(self.c,name)
        def __enter__(self):self.c.__enter__();return self
        def __exit__(self,*args):return self.c.__exit__(*args)
        def execute(self,sql,*args,**kwargs):
            return self.c.execute(re.sub(r'(?i)\bSET\s+threads\s*=\s*\d+\b','SET threads=1',sql),*args,**kwargs)
    duckdb.connect=Connection
    from scripts import train3000_frozen_v142_pipeline as pipeline
    args=mini.read(d/'input.json')
    try:
        if phase=='prepare':pipeline.prepare(d,args['ids'])
        elif phase=='generate':pipeline.generate(d,args['binding'],Path(args['prepared']))
        elif phase=='audit':pipeline.audit(d)
        else:pipeline.evaluate(d,args['seal'])
    except MemoryError as error:
        mini.save(d/'resource_yield.json',dict(reason=str(error),cap_bytes=control.CAP));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-ready',action='store_true')
    mode.add_argument('--full',action='store_true')
    mode.add_argument('--worker',choices=['prepare','generate','audit','evaluate'])
    parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.worker,args.directory)
    else:
        try:run(args.full)
        except Exception as error:
            ready=mini.read(OUT/'ready.json') if (OUT/'ready.json').exists() else None
            failures=list(OUT.glob('*/failure.json'))
            detail=mini.read(max(failures,key=lambda p:p.stat().st_mtime_ns)) if failures else None
            report=dict(status='STOP_BEFORE_COMPLETION',error=repr(error),full3000_started=bool(args.full),
                fidelity_failure=detail,native_smoke_complete=ready is not None,smoke=ready['smoke'] if ready else None,
                eta3000_seconds=None,launch_command=None,
                explanation='No unexecuted fidelity or smoke is declared successful; all prior attempts retained')
            mini.save(OUT/('terminal_error_'+uuid.uuid4().hex+'.json'),report)
            print('STOP — '+repr(error)+'\nLes artefacts et reçus antérieurs sont conservés.\n'
                'Aucune fidélité ou smoke non exécuté n’est annoncé.\n'
                'Aucune modification de labels, action VM ou lancement full automatique.\n'
                'Examiner le dernier worker.log dans '+str(OUT),flush=True)
            if not isinstance(error,InterruptedError):automatic_commit(report,args.full)
            raise
