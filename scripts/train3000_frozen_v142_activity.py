"""Exact V112 query-context body and lossless V115 projection, private v174 I/O."""
import hashlib
from pathlib import Path
import textwrap
import resource
import time
from types import SimpleNamespace
import gc

import duckdb
import ijson
import numpy as np
from scripts import train3000_frozen_v142_fidelity as mini
from scripts import embeddinggemma2_v112_bodacc_activity_context as source
from scripts import embeddinggemma2_v115_streamed_activity_ablation as projection

ATOL = 1e-7  # Historical V19 vector replay tolerance, fixed before this run.


class SingleThreadConnection:
    def __init__(self):
        self.connection = duckdb.connect()

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, sql, *args, **kwargs):
        # Execution resource setting only; the historical scientific SQL is intact.
        if sql == 'SET threads=2':
            sql = 'SET threads=1'
        return self.connection.execute(sql, *args, **kwargs)


def historical_slice(path, expected, start, end, signature, ending, namespace):
    mini.verified(dict(path=str(path), sha256=expected))
    text = Path(path).read_text()
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('STOP_HISTORICAL_ACTIVITY_CONSTRUCTOR_NOT_FOUND')
    body = text[text.index(start):text.index(end)]
    program = 'def build('+signature+'):\n'+textwrap.indent(textwrap.dedent(body), '    ')+ending
    exec(compile(program, str(path)+':unchanged_body', 'exec'), namespace)
    return namespace['build'], hashlib.sha256(body.encode()).hexdigest()


def context(panel, directory):
    if not panel or any(set(q) != set(mini.bano.cohort.old.CRM) or q['fold'] not in (2,3,4) for q in panel):
        raise ValueError('Query-only CRM8, folds234 required')
    generation = mini.read(source.OUTPUT/'generation.json')
    launch = mini.read(mini.verified(generation['launch']))
    mini.verified(launch['public_units'])
    current = sorted(str(p) for p in source.words.SOURCES.glob('*/manifest.json'))
    expected = sorted(r['manifest']['path'] for r in launch['sources'])
    if current != expected:
        raise ValueError('STOP_ACTIVITY_SOURCE_MANIFEST_SET_DRIFT')
    for ref in launch['sources']:
        mini.verified(ref['manifest'])
        mini.verified(ref['source'])
    directory.mkdir(parents=True)
    panel_path = directory/'panel.json'
    mini.save(panel_path, panel)
    root = directory/'source'

    def write(path, value):
        path = Path(path)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Refuse any write outside owned v174 context namespace')
        if path.name == 'state.json':
            path = path.with_name('state_'+str(time.time_ns())+'.json')
        mini.save(path, value)

    namespace = dict(vars(source))
    namespace.update(OUTPUT=root, io=SimpleNamespace(**dict(vars(source.io), write=write)),
        engine=SimpleNamespace(**dict(vars(source.engine), guard=mini.guard)),
        duckdb=SimpleNamespace(connect=SingleThreadConnection))
    build, body_sha = historical_slice(Path(source.__file__), generation['implementation']['sha256'],
        "    qs={r['query_id']:words.tokens(r['crm_name']) for r in panel}",
        "    io.write(OUTPUT/'activity_context.json',output)", 'panel,base',
        '\n    return output,dict(source_owners=n,source_owner_token_pairs=sum(len(v) for v in by_token.values()))\n', namespace)
    value, counts = build(panel, dict(panel=mini.record(panel_path)))
    mini.guard()
    mini.save(root/'activity_context.json', value)
    mini.save(directory/'construction.json', dict(source=generation['implementation'], body_sha256=body_sha,
        queries=len(panel), counts=counts, raw_labels_read=False, historical_source_launch=generation['launch'],
        execution_only_changes=['v174 output root','supplied query-only panel','one DuckDB thread','under4GiB guard']))
    return value, body_sha


def project(entries):
    # Exact value/provenance checks and dtypes from V115, not a new representation.
    mini.verified(dict(path=str(Path(projection.__file__)), sha256='71d68822067668bc01a7714ff98dea424eb637bbd5a62d8ef7a27bf90a11a2af'))
    text = Path(projection.__file__).read_text()
    start = "            refs=np.array([e['siren_ref'] for e in entries],dtype='S9')"
    end = "            path=target/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('STOP_V115_PROJECTION_CONSTRUCTOR_NOT_FOUND')
    body = text[text.index(start):text.index(end)]
    namespace = dict(np=np)
    exec(compile('def project(entries):\n'+textwrap.indent(textwrap.dedent(body), '    ')+
        '\n    return refs,confidence\n', str(Path(projection.__file__)), 'exec'), namespace)
    return namespace['project'](entries), hashlib.sha256(body.encode()).hexdigest()


def fidelity(directory):
    started = time.monotonic()
    run = mini.bano.cohort.preflight.RUN
    sample = mini.read(mini.bano.cohort.OUT/'sample.json')
    panel = mini.read(mini.bano.cohort.OUT/'pilot_fidelity_queries.json')
    ids = set(sample['fidelity_query_ids'])
    original = mini.read(mini.verified(mini.read(run/'launch.json')['panel']))
    if len(panel) != 20 or {q['query_id']:q for q in panel} != {q['query_id']:q for q in original if q['query_id'] in ids}:
        raise ValueError('STOP_SEALED_ACTIVITY_PILOT_SCOPE')
    values, context_sha = context(panel, directory/'reconstruction')
    fg = mini.read(run/'features_generation.json')
    generation = mini.read(source.OUTPUT/'generation.json')
    rows = []
    projection_sha = None
    with mini.verified(generation['context']).open('rb') as stream:
        for qid, frozen in ijson.kvitems(stream, '', use_float=True):
            mini.guard()
            if qid not in ids:
                continue
            fresh = values.pop(qid)
            aligned = len(fresh) == len(frozen)
            error = 0.
            if aligned:
                for a,b in zip(fresh,frozen):
                    error = max(error, abs(a['confidence']-b['confidence']))
                    if {k:v for k,v in a.items() if k!='confidence'} != {k:v for k,v in b.items() if k!='confidence'}:
                        aligned = False
                        break
            (refs, confidence), projection_sha = project(fresh)
            with np.load(mini.verified(fg['activity_shards'][qid]), allow_pickle=False) as archive:
                same_refs = np.array_equal(refs, archive['refs'])
                vector_error = float(np.max(np.abs(confidence-archive['confidence']), initial=0.)) if same_refs else None
                bitwise = same_refs and np.array_equal(confidence, archive['confidence'])
            passed = aligned and same_refs and error<=ATOL and vector_error is not None and vector_error<=ATOL
            row = dict(query_id=qid, passed=passed, context_provenance_identical=aligned, refs_identical=same_refs,
                bitwise_confidence=bitwise, max_context_confidence_error=error,
                max_vector_absolute_error=vector_error, owner_refs=len(refs))
            mini.save(directory/(hashlib.sha256(qid.encode()).hexdigest()+'.json'), row)
            print(row, flush=True)
            rows.append(row)
            if not passed:
                mini.save(directory/'failure.json', dict(status='STOP_ACTIVITY_NONFIDELITY', completed=len(rows), query=row))
                raise ValueError('STOP_ACTIVITY_NONFIDELITY')
            np.savez_compressed(directory/(hashlib.sha256(qid.encode()).hexdigest()+'.npz'), refs=refs, confidence=confidence)
            del fresh,refs,confidence
            gc.collect()
    if len(rows)!=20 or values:
        raise ValueError('STOP_ACTIVITY_INCOMPLETE20')
    mini.save(directory/'result.json', dict(status='PASS', family='activity', queries=20,
        context_matches=20, weighted_projection_matches=20,
        bitwise_confidence_matches=sum(r['bitwise_confidence'] for r in rows),
        max_absolute_error=max(r['max_vector_absolute_error'] for r in rows), absolute_tolerance=ATOL,
        context_body_sha256=context_sha, projection_body_sha256=projection_sha,
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        raw_labels_read=False, native_queries=0))
