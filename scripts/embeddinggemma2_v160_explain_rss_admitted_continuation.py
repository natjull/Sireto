"""One last explicitly authorized resource-only attempt; no score changes.

A/B remain immutable.8.5GiB instead6, absolute04:10 stop, zero new swapouts.
"""
from pathlib import Path as _Path
import hashlib as _hashlib
import subprocess as _vm_subprocess
import re as _vm_re

def _swapouts():
    try:
        text=_vm_subprocess.check_output(['/usr/bin/vm_stat'],text=True,timeout=3)
        found=_vm_re.search(r'^Swapouts:\s+(\d+)\.',text,flags=_vm_re.M)
        return int(found[1]) if found else None
    except (OSError,_vm_subprocess.SubprocessError):return None

_INITIAL_SWAPOUTS=_swapouts()
def check_swapouts():
    current=_swapouts()
    if current is not None and _INITIAL_SWAPOUTS is not None and current>_INITIAL_SWAPOUTS:
        raise RuntimeError(f'STOP new swapouts baseline={_INITIAL_SWAPOUTS} current={current}')

_source=_Path(__file__).with_name('embeddinggemma2_v160_explain_searcher_delegate_continuation.py')
_text=_source.read_text()
if _hashlib.sha256(_text.encode()).hexdigest()!='57eedee29a76a5fe6f47f41d9da1a9fdc224fa328fccf3ebce1470ceb4a9db04':
    raise ValueError('Frozen delegated attempt B code drift')
_changes={
    "composite_explain_diagnostic_20261014_night_b'":"composite_explain_diagnostic_20261014_night_c'",
    "'scripts.embeddinggemma2_v160_explain_searcher_delegate_continuation'":"'scripts.embeddinggemma2_v160_explain_rss_admitted_continuation'",
    '_changes={\n':"_changes={\n    'rss_bytes=6*1024**3':'rss_bytes=int(8.5*1024**3)',\n    'max_execution_seconds=1200':'max_execution_seconds=2040',\n    'time.time()-start>1200':'time.time()-start>2040',\n    '    return rss\\n':'    check_swapouts()\\n    return rss\\n',\n",
}
for _before,_after in _changes.items():
    if _text.count(_before)!=1:raise ValueError('Resource-only admitted transformation bound')
    _text=_text.replace(_before,_after)
exec(compile(_text,str(_source)+':resource_only_authorized_C','exec'),globals())
