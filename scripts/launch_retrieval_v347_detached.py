"""One detached, authorized child session; no retry or target lookup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTROL_BASE = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v347_persistent_launcher')
GENERATION_BASE = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v344_corroborated_population')
PANEL_MANIFEST = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v328_serving_panel/package_20261005_a/manifest.json',
    sha256='36dbf3c4367a57847ac4534e217967ed121d085cadb4b0ddc18984853bbb2e04', size_bytes=5432)
THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1',
              VECLIB_MAXIMUM_THREADS='1', PYTHONUNBUFFERED='1')


def reference(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return dict(path=str(path), sha256=h.hexdigest(), size_bytes=path.stat().st_size)


def checked(ref):
    if reference(ref['path']) != ref:
        raise ValueError('Frozen launcher/code/package authority drift')
    return Path(ref['path'])


def write(path, value, exclusive=False):
    content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if exclusive:
        with Path(path).open('x') as f:
            f.write(content)
    else:
        temporary = Path(str(path)+'.own_tmp')
        temporary.write_text(content)
        os.replace(temporary, path)


def scoped(path, base):
    path = Path(path).resolve()
    if path.parent != base.resolve() or not re.fullmatch(r'[a-z0-9._-]+', path.name):
        raise ValueError('One named direct child of the authorized audit directory required')
    return path


def command(control):
    return [str(ROOT/'.venv/bin/python'), '-u', str(Path(__file__).resolve()),
            'job', '--control', str(control)]


def prepare(control, generation):
    from scripts import retrieval_v344_corroborated_population as driver
    control = scoped(control, CONTROL_BASE)
    generation = scoped(generation, GENERATION_BASE)
    if control.exists() or generation.exists():
        raise ValueError('Fresh control and generation roots only, no resume/retry')
    checked(PANEL_MANIFEST)
    driver.population.sealed(Path(PANEL_MANIFEST['path']).parent)
    refs = driver.implementation()
    refs['detached-launcher'] = reference(Path(__file__))
    refs['detached-contract'] = reference(ROOT/'docs/retrieval_v347_detached_launch_contract.md')
    refs['launcher-runtime'] = reference(ROOT/'.venv/bin/python')
    control.mkdir(parents=True, exist_ok=False)
    launch = dict(status='PREPARED_V347_ONE_SHOT_NOT_STARTED', implementation=refs,
        generation_root=str(generation), panel_manifest=PANEL_MANIFEST,
        transport='POSIX_NEW_SESSION_NO_TTY_INHERITED_ACCESS', job_command=command(control),
        environment=THREADS, start_new_session=True, stdin_null=True, close_fds=True,
        automatic_retry=False, partial_outputs_reused=False, truth_opened=False,
        evaluation_only_after_all2000=True)
    write(control/'launch.json', launch, exclusive=True)
    print(json.dumps(dict(control=str(control), status=launch['status'],
                          next_mode='start')), flush=True)
    return launch


def load(control):
    control = scoped(control, CONTROL_BASE)
    launch = json.loads((control/'launch.json').read_text())
    scoped(launch['generation_root'], GENERATION_BASE)
    if (launch['status'] != 'PREPARED_V347_ONE_SHOT_NOT_STARTED'
            or launch['automatic_retry'] is not False or launch['partial_outputs_reused'] is not False
            or launch['truth_opened'] is not False or launch['evaluation_only_after_all2000'] is not True
            or launch['panel_manifest'] != PANEL_MANIFEST
            or launch['transport'] != 'POSIX_NEW_SESSION_NO_TTY_INHERITED_ACCESS'
            or launch['job_command'] != command(control) or launch['environment'] != THREADS
            or any(launch.get(k) is not True for k in ('start_new_session','stdin_null','close_fds'))):
        raise ValueError('Original nonrecurring detached launch with closed labels required')
    checked(launch['panel_manifest'])
    for ref in launch['implementation'].values():
        checked(ref)
    return launch


def start(control):
    control = Path(control)
    launch = load(control)
    if Path(launch['generation_root']).exists() or (control/'one_shot_claim.json').exists():
        raise ValueError('No existing generation or begun job reuse')
    write(control/'start_claim.json', dict(pid=os.getpid(), time=time.time()), exclusive=True)
    with (control/'service_stdout.log').open('x') as out, (control/'service_stderr.log').open('x') as err:
        child = subprocess.Popen(launch['job_command'], cwd=ROOT,
            env={**os.environ, **THREADS}, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            start_new_session=True, close_fds=True)
    began = time.monotonic()
    while time.monotonic()-began < 10:
        if child.poll() is not None:
            code = child.wait(timeout=5)
            write(control/'start_supervision.json', dict(status='STOP_START', child_pid=child.pid,
                exit_code=code, own_child_exit_confirmed=True, automatic_retry=False))
            raise RuntimeError('Detached job exited before initial state; no retry')
        state_path = control/'state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get('job_pid') != child.pid or state.get('status') != 'RUNNING_V347_NOT_COMPLETE':
                raise ValueError('Exact owned detached child initial state required')
            result = dict(status='STARTED_V347_NOT_COMPLETE', child_pid=child.pid,
                workflow_pid=state['workflow_pid'], start_new_session=True, automatic_retry=False,
                own_child_exit_confirmed=False, recall_admitted=False)
            write(control/'start_supervision.json', result)
            print(json.dumps(result), flush=True)
            return result
        time.sleep(.2)
    terminate_owned_group(child)
    raise TimeoutError('No owned detached job initial state within10seconds')


def claim(control):
    # Even manual kickstart/rebootstrap cannot silently retry a begun job.
    write(Path(control)/'one_shot_claim.json', dict(pid=os.getpid(), time=time.time()), exclusive=True)


def progress(path, previous):
    if not path.exists():
        return previous
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size-8192))
        lines = f.read().decode('utf8', errors='replace').splitlines()
    for line in lines:
        try:
            value = json.loads(line).get('generated')
        except (ValueError, AttributeError):
            continue
        if type(value) is int and 0 <= value <= 2000:
            previous = max(previous, value)
    return previous


def terminate_owned_group(child):
    # The workflow was started in a NEW session; this PGID contains only it
    # and its own workers, never another user's process or a workspace service.
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def job(control):
    control = Path(control)
    launch = load(control)
    claim(control)
    if Path(launch['generation_root']).exists():
        raise ValueError('No generation root reuse after claim')
    child = None
    awake = None
    began = time.monotonic()
    original_handlers = {}
    done = 0
    def interrupted(sig, frame):
        raise RuntimeError('STOP_MANAGED_JOB_SIGNAL_'+str(sig))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        original_handlers[sig] = signal.signal(sig, interrupted)
    try:
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with (control/'workflow.log').open('x') as log:
            child = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u', '-m',
                'scripts.retrieval_v344_corroborated_population', 'workflow', '--output',
                launch['generation_root']], cwd=ROOT, env={**os.environ, **THREADS},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            while child.poll() is None:
                done = progress(control/'workflow.log', done)
                write(control/'state.json', dict(status='RUNNING_V347_NOT_COMPLETE',
                    job_pid=os.getpid(), job_ppid=os.getppid(), workflow_pid=child.pid,
                    admitted_generation_queries=done, required_queries=2000,
                    generation_root=launch['generation_root'], updated_at_unix=time.time(),
                    elapsed_seconds=time.monotonic()-began, automatic_retry=False,
                    own_workflow_exit_confirmed=False, recall_admitted=False))
                time.sleep(5)
            code = child.wait(timeout=5)
        write(control/'workflow_supervision.json', dict(child_pid=child.pid,
            exit_code=code, own_child_exit_confirmed=True, wall_seconds=time.monotonic()-began,
            automatic_retry=False))
        if code:
            raise subprocess.CalledProcessError(code, child.args)
        load(control)
        gen_root = Path(launch['generation_root'])
        measured_root = GENERATION_BASE.parent/'retrieval_v346_postseal_evaluation'/gen_root.name
        generation = json.loads((gen_root/'manifest.json').read_text())
        measured = json.loads((measured_root/'manifest.json').read_text())
        if (generation.get('status') != 'SEALED_V344_ALL2000_CORROBORATED_INITIAL100_PILOT'
                or generation.get('all2000_outputs_sealed') is not True or generation.get('queries') != 2000
                or measured.get('status') != 'MEASURED_V346_EXACT_INITIAL100_ALL2000_NEW_DEV_PANEL'
                or measured.get('recall_measured') is not True or measured.get('queries') != 2000):
            raise ValueError('Actual complete generation and postseal measurement required')
        write(control/'state.json', dict(status='COMPLETE_V347_ONE_SHOT_WORKFLOW',
            job_pid=os.getpid(), workflow_pid=child.pid, own_workflow_exit_confirmed=True,
            exit_code=0, admitted_generation_queries=2000, generation=reference(gen_root/'manifest.json'),
            evaluation=reference(measured_root/'manifest.json'), report=measured['report'],
            elapsed_seconds=time.monotonic()-began, updated_at_unix=time.time(),
            automatic_retry=False, recall_admitted=True))
    except BaseException as exc:
        terminate_owned_group(child)
        write(control/'state.json', dict(status='STOP_V347_MANAGED_WORKFLOW', reason=str(exc),
            job_pid=os.getpid(), workflow_pid=child.pid if child is not None else None,
            exit_code=child.returncode if child is not None else None,
            own_workflow_exit_confirmed=child is not None and child.returncode is not None,
            admitted_generation_queries=done, required_queries=2000,
            elapsed_seconds=time.monotonic()-began, updated_at_unix=time.time(),
            automatic_retry=False, recall_admitted=False))
        raise
    finally:
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
        if awake is not None:
            if awake.poll() is None:
                awake.terminate()
            awake.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'job', 'start'))
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--generation', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.generation is None:
            parser.error('Fresh explicitly authorized generation root required')
        prepare(args.control, args.generation)
    elif args.mode == 'start':
        start(args.control)
    else:
        job(args.control)
