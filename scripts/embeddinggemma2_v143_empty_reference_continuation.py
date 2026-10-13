"""Frozen V132 recovery: empty numeric keys yield zero new evidence, not loss.

Only the empty reduction and execution namespace change. All nonempty score
inputs, models, sources and seven predeclared variants remain identical.
This module also exports the original interfaces for unlaunched downstreams.
"""
import hashlib as _hashlib
import json as _json
from pathlib import Path as _Path
import shutil as _shutil

_SOURCE = _Path(__file__).with_name('embeddinggemma2_v132_local_evidence_ablation.py')
_EXPECTED = '58bd94842b7f2565ef20bce85023bb5f082d7bb86b7cdb0571aa78874717eee8'
_OLD = _Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v132/local_evidence_20261013_night')
_text = _SOURCE.read_text()
if _hashlib.sha256(_text.encode()).hexdigest() != _EXPECTED:
    raise ValueError('Historical frozen V132 implementation drift')
if _text.count("\nif __name__=='__main__':") != 1:
    raise ValueError('Exact historical entry point required')
_body = _text.rsplit("\nif __name__=='__main__':", 1)[0]
_changes = {
    "ROOT=io.HOME/'v132/local_evidence_20261013_night'":
        "ROOT=io.HOME/'v143/local_evidence_empty_recovery_20261013_night'",
    'int(group.max())+1': 'int(group.max(initial=-1))+1',
    "'scripts.embeddinggemma2_v132_local_evidence_ablation'":
        "'scripts.embeddinggemma2_v143_empty_reference_continuation'",
}
for _before, _after in _changes.items():
    if _body.count(_before) != 1:
        raise ValueError('Bound namespace/empty-only transformation drift')
    _body = _body.replace(_before, _after)
exec(compile(_body, str(_SOURCE), 'exec'), globals())


def migrate():
    guard()
    ex = io.read(_OLD/'prepare_owned_tool_exit.json')
    if ex['exit_code'] != 1 or not ex['own_tool_exit_confirmed']:
        raise ValueError('Real observed failed prepare before continuation')
    old = io.read(_OLD/'launch.json')
    if verify(old['implementation']) != _SOURCE:
        raise ValueError('Original bound source')
    if io.read(verify(old['recipe'])) != _json.loads(_json.dumps(RECIPE)):
        raise ValueError('Scientific recipe and resource policy unchanged')
    if ROOT.exists():
        raise ValueError('Fresh recovery namespace')
    for ref in old['helpers']:
        verify(ref)
    panel = io.read(verify(old['panel']))
    queries = {q['query_id']: q for q in panel}
    fg = io.read(verify(old['baseline_features']))
    ROOT.mkdir(parents=True)
    (ROOT/'spill').mkdir()
    (ROOT/'calibration').mkdir()
    (ROOT/'extras').mkdir()
    io.write(ROOT/'recipe.json', RECIPE)
    launch = dict(old, implementation=record(_Path(__file__)),
        recipe=record(ROOT/'recipe.json'), created_unix=time.time(),
        historical_launch=record(_OLD/'launch.json'),
        historical_failed_prepare=record(_OLD/'prepare_owned_tool_exit.json'))
    launch['helpers'] = [*old['helpers'], record(_SOURCE)]
    io.write(ROOT/'launch.json', launch)
    io.write(ROOT/'features_generation.json', io.read(_OLD/'features_generation.json'))
    io.write(ROOT/'calibration/generation.json', io.read(_OLD/'calibration/generation.json'))
    copied = 0
    for source in sorted((_OLD/'extras').glob('*.npz')):
        sealed = engine.sealed(source)
        if not sealed:
            continue
        metadata = source.with_suffix('.metadata.json')
        m = io.read(metadata)
        qid = m['query']['query_id']
        if m['query'] != queries[qid] or m['labels_read'] is not False:
            raise ValueError('Exact query-only prepared shard before transfer')
        with np.load(sealed, allow_pickle=False) as f, np.load(verify(fg['features'][qid]), allow_pickle=False) as base:
            if not np.array_equal(f['keys'], base['keys']) or len(f['keys']) == 0:
                raise ValueError('Only unchanged nonempty source/model-bound scores transferred')
        destination = ROOT/'extras'/source.name
        _shutil.copy2(source, destination)
        if record(destination)['sha256'] != record(source)['sha256']:
            raise ValueError('Byte-identical prepared score shard')
        engine.seal(destination)
        _shutil.copy2(metadata, destination.with_suffix('.metadata.json'))
        copied += 1
    public = io.read(_OLD/'public_source.json')
    source_mask = verify(public['public_mask'])
    destination_mask = ROOT/'public_mask.npy'
    _shutil.copy2(source_mask, destination_mask)
    if record(destination_mask)['sha256'] != public['public_mask']['sha256']:
        raise ValueError('Same both-public coordinate mask')
    engine.seal(destination_mask)
    public['public_mask'] = record(destination_mask)
    io.write(ROOT/'public_source.json', public)
    io.write(ROOT/'migration.json', dict(status='EMPTY_NUMERIC_REFERENCE_ONLY_RECOVERY',
        original_implementation=record(_SOURCE), original_launch=record(_OLD/'launch.json'),
        implementation=record(_Path(__file__)), failed_owned_exit=record(_OLD/'prepare_owned_tool_exit.json'),
        copied_prepared_shards=copied, original_public_source=record(_OLD/'public_source.json'),
        public_source=record(ROOT/'public_source.json'), copied_scores_byte_identical=True,
        only_scoring_change='EMPTY_KEYS_ZERO_NUMERIC_INCREMENT_LEXICAL_NATIONAL_UNRESTRICTED',
        all_nonempty_score_sources_models_recipe_identical=True,
        all526_retained=True, no_labels_read=True, created_unix=time.time()))
    authority(ROOT)
    print('MIGRATED_EMPTY_REFERENCE_RECOVERY', copied, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['migrate', 'prepare', 'generate', 'run', 'evaluate'])
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, engine.interrupted)
    signal.signal(signal.SIGINT, engine.interrupted)
    try:
        if args.action == 'generate':
            generate(args.output, args.variant)
        elif args.action == 'migrate':
            migrate()
        else:
            globals()[args.action](args.output)
    except engine.ResourceYield as error:
        print(str(error), flush=True)
        sys.exit(75)
    except Exception as error:
        if args.output.exists():
            io.write(args.output/(args.action+'_error.json'), dict(error=repr(error), updated_unix=time.time()))
        raise
