"""All canonical relations audit and complete BODACC projection; no CRM/GT."""
import argparse
from collections import Counter
from datetime import date
import json
import os
from pathlib import Path
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from src.xgb_matcher.official_evidence import OfficialRelation
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget, LIMITS, POLICY, quiet, validate_report, parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_postscore_audit_v187 import canonical_epochs, check_epochs
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

OUTPUT = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v320_bodacc_relations')
INPUT = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/siren_dossier_v3/01eb624cca97c6f3/relations.parquet',
    sha256='3516e4080fbf4332fd8d289c982903ba858219cec4b891c105d1cfb7c9059ab5', size_bytes=35147267)
DOSSIER = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/siren_dossier_v3/01eb624cca97c6f3/manifest.json',
    sha256='6d8ed0053b78862adafd76d0b2ac46e06ac5fc735e81723bb21229528fcea198', size_bytes=3417)
EXPECTED_ROWS = 692010
MAX_OUTPUT_BYTES = 256*1024**2
FIELDS = ('relation_id','source','source_record_id','relation_type','from_kind','from_identifier',
          'to_kind','to_identifier','effective_date','observed_at','source_priority')
STATUS = 'SEALED_V320_ALL_CANONICAL_RELATIONS_AND_COMPLETE_BODACC_PROJECTION'
FALSE = {k: False for k in ('crm_opened','truth_opened','recall_measured','candidates_emitted','model_fit',
    'downstream_models_changed','fold0_opened','fold1_opened','ready_for_fold1','identifiers_as_features')}


def raw(row):
    if set(row) != set(FIELDS):
        raise ValueError('All original11 fields required')
    return {k: value.isoformat() if isinstance(value, date) else value for k,value in row.items()}


def typed(row):
    original = raw(row)
    canonical = OfficialRelation.from_dict(row).to_dict()
    if {k: canonical[k] for k in FIELDS} != original:
        raise ValueError('Every canonical value must reconstruct exactly, no repairing input')
    return original


def rows(path):
    f = pq.ParquetFile(path)
    if tuple(f.schema_arrow.names) != FIELDS:
        raise ValueError('Same complete canonical relation schema required')
    for batch in f.iter_batches(batch_size=4096, use_threads=False):
        yield from batch.to_pylist()


def project(path, guard, expected_rows=EXPECTED_ROWS):
    counts, selected, total = Counter(), [], 0
    for ordinal, row in enumerate(rows(path)):
        if ordinal % 1024 == 0: guard()
        value = typed(row)
        counts[(value['source'],value['relation_type'],value['from_kind'],value['to_kind'])] += 1
        if value['source'] == 'BODACC':
            selected.append(dict(ordinal=ordinal, fields=value))
        total += 1
    if total != expected_rows or total <= 0:
        raise ValueError('ALL original source rows required')
    return selected, dict(records=total, bodacc_records=len(selected),
        relation_counts=[dict(source=k[0],relation_type=k[1],from_kind=k[2],to_kind=k[3],records=n)
            for k,n in sorted(counts.items())])


def roundtrip(path, selected, guard, expected_rows=EXPECTED_ROWS):
    cursor, total = 0, 0
    for ordinal, row in enumerate(rows(path)):
        if ordinal % 1024 == 0: guard()
        value = typed(row)
        if value['source'] == 'BODACC':
            if cursor >= len(selected) or selected[cursor] != dict(ordinal=ordinal, fields=value):
                raise ValueError('Every original BODACC row and ordinal must roundtrip')
            cursor += 1
        total += 1
    if cursor != len(selected) or total != expected_rows:
        raise ValueError('Full original source and projection required')


def implementation():
    paths = [Path(__file__), ROOT/'docs/retrieval_v320_bodacc_relations_contract.md',
        ROOT/'src/xgb_matcher/official_evidence.py', ROOT/'src/xgb_matcher/hierarchical_retrieval.py',
        ROOT/'src/xgb_matcher/retrieval_structural_channels_v210.py',
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py', ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py',
        ROOT/'src/xgb_matcher/retrieval_postscore_audit_v187.py', ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
        ROOT/'src/xgb_matcher/retrieval_name_pages_v151.py']
    return {str(p):record(p) for p in paths}


def load(root):
    if not root.resolve().is_relative_to(OUTPUT.resolve()) or (root/'status.json').exists() or any(root.glob('phase/*stop.json')):
        raise ValueError('Own fresh source root only')
    launch = json.loads((root/'launch.json').read_text())
    if (launch['input'] != INPUT or launch['dossier'] != DOSSIER or launch['limits'] != LIMITS
        or launch['arrow_version'] != pa.__version__ or any(launch.get(k) is not False for k in FALSE)):
        raise ValueError('Frozen whole source and guards required')
    for ref in launch['implementation'].values(): verify(ref)
    return launch


def worker(root):
    load(root)
    d = root/'phase'
    with Budget(LIMITS) as budget:
        def guard():
            if budget.failure: raise RuntimeError(budget.failure)
        files = SourceFiles(guard)
        dossier = json.loads(files.file(DOSSIER).read_text())
        if (dossier['policy']['crm_labels_present'] is not False or dossier['policy']['model_scores_present'] is not False
            or dossier['counts']['relations'] != EXPECTED_ROWS
            or dossier['files']['relations.parquet'] != {k: INPUT[k] for k in ('sha256','size_bytes')}):
            raise ValueError('Same label-free complete official dossier source required')
        path = files.file(INPUT)
        selected, stats = project(path, guard)
        write_json(d/'payload.json', dict(rows=selected, stats=stats, all_source_rows_typed=True,
            all_bodacc_rows_and_fields_preserved=True, source_only=True, **FALSE))
        if (d/'payload.json').stat().st_size > MAX_OUTPUT_BYTES: raise ValueError('STOP_OUTPUT_SIZE')
        written = json.loads((d/'payload.json').read_text())
        roundtrip(path, written['rows'], guard)
        files.verify_after()
        written['source_epochs'] = list(files.entries.values())
        write_json(d/'payload.json', written)
        if (d/'payload.json').stat().st_size > MAX_OUTPUT_BYTES: raise ValueError('STOP_OUTPUT_SIZE')
        load(root)
    validate_report(budget.report())
    write_json(d/'manifest.json', dict(status='QUARANTINED_V320_SOURCE',launch=record(root/'launch.json'),
        preflight=record(d/'preflight.json'),payload=record(d/'payload.json'),resources=budget.report(), **FALSE))


def admission(root):
    root = Path(root); load(root); d = root/'phase'; parent_exit(d)
    m = json.loads((d/'manifest.json').read_text()); validate_report(m['resources'])
    p = json.loads(verify(m['payload']).read_text()); gate = json.loads(verify(m['preflight']).read_text())
    if (m['status'] != 'QUARANTINED_V320_SOURCE' or m['launch'] != record(root/'launch.json')
        or m['preflight'] != record(d/'preflight.json') or m['payload'] != record(d/'payload.json')
        or gate['policy_id'] != POLICY or gate['pass_system_quiet_policy'] is not True or gate['quiet_seconds'] < 30
        or p['stats']['records'] != EXPECTED_ROWS or p['stats']['bodacc_records'] != len(p['rows'])
        or p['all_source_rows_typed'] is not True or p['all_bodacc_rows_and_fields_preserved'] is not True
        or p['source_only'] is not True or any(x.get(k) is not False for x in (m,p) for k in FALSE)):
        raise ValueError('Actual own complete source producer required')
    epochs = canonical_epochs(p['source_epochs']); check_epochs(epochs)
    for ref in (INPUT,DOSSIER):
        if not any(e['reference']['path'] == ref['path'] and e['reference']['sha256'] == ref['sha256'] for e in epochs):
            raise ValueError('Original full official source authority required')
    return m,p


def sealed(root):
    root = Path(root); m,p = admission(root); s = json.loads((root/'manifest.json').read_text())
    if (s['status'] != STATUS or s['producer'] != record(root/'phase/manifest.json') or s['payload'] != m['payload']
        or s['stats'] != p['stats'] or any(s.get(k) is not False for k in FALSE)):
        raise ValueError('Same actual complete source required')
    return s,p


def run(root):
    if root.exists() or not root.resolve().is_relative_to(OUTPUT.resolve()): raise ValueError('Fresh own source output required')
    root.mkdir(parents=True,exist_ok=False); d=root/'phase'; d.mkdir()
    try:
        write_json(root/'launch.json',dict(input=INPUT,dossier=DOSSIER,limits=LIMITS,arrow_version=pa.__version__,
            implementation=implementation(), **FALSE))
        write_json(d/'preflight.json',quiet(30,50))
        with (d/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.retrieval_v320_bodacc_relations','worker','--output',str(root)],
                log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},d/'supervisor_stop.json')
        write_json(d/'supervision.json',s); m,p=admission(root)
        write_json(root/'manifest.json',dict(status=STATUS,producer=record(d/'manifest.json'),payload=m['payload'],stats=p['stats'],**FALSE))
        sealed(root); print(json.dumps(dict(status=STATUS,stats=p['stats'])),flush=True)
    except BaseException as exc:
        write_json(root/'status.json',dict(status='STOP',reason=str(exc),**FALSE)); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('run','worker')); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); (run if a.mode=='run' else worker)(a.output)
