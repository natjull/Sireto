"""Execution-only searcher delegation after pre-collector owned failure.

Historical code/ns immutable. Same score/tolerance/two-case/budget contract.
"""
from pathlib import Path as _Path
import hashlib as _hashlib
import json as _json
_source=_Path(__file__).with_name('embeddinggemma2_v160_composite_explain_diagnostic.py')
_text=_source.read_text()
if _hashlib.sha256(_text.encode()).hexdigest()!='e75c04357340a23d9b3e348382b964dad8dd96331056e26798718e163110fd3f':
    raise ValueError('Frozen original diagnostic code drift')
_failed=_Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v160/composite_explain_diagnostic_20261014_night/replay_owned_child_exit.json')
_exit=_json.loads(_failed.read_text())
if _exit['exit_code']!=1 or not _exit['own_child_exit_confirmed']:
    raise ValueError('Actual pre-collector failed owned exit required')
_changes={
    "OUT=HOME/'v160/composite_explain_diagnostic_20261014_night'":"OUT=HOME/'v160/composite_explain_diagnostic_20261014_night_b'",
    '        class CaptureSearcher:\n':'        class CaptureSearcher:\n            def __getattr__(self,name):return getattr(real,name)\n',
    "'scripts.embeddinggemma2_v160_composite_explain_diagnostic'":"'scripts.embeddinggemma2_v160_explain_searcher_delegate_continuation'",
}
for _before,_after in _changes.items():
    if _text.count(_before)!=1:raise ValueError('Exact execution-only transformation required')
    _text=_text.replace(_before,_after)
exec(compile(_text,str(_source)+':delegate_only', 'exec'),globals())
