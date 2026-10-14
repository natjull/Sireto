"""Root-authorized Python-bool serialization only; frozen diagnostic intact."""
import hashlib
import json
from pathlib import Path

original = Path(__file__).with_name('embeddinggemma2_v171_study_geo_label_diagnostic.py')
text = original.read_text()
if hashlib.sha256(text.encode()).hexdigest() != '5735420f84ad9dbc811cd8a6e252c9e167f4967f4a6521f0a2eab28090df74e4':
    raise ValueError('Frozen original label-side implementation drift')
failed = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v171/separate_study_geo_components_20261014_night/label_side/owned_tool_exit.json')
failure = json.loads(failed.read_text())
if failure['exit_code'] != 1 or not failure['own_tool_exit_confirmed']:
    raise ValueError('Observed serialization failure required before continuation')
changes = {
    "root=out/'label_side'": "root=out/'label_side_b'",
    "row['target_in_original_numeric_universe']=inside": "row['target_in_original_numeric_universe']=bool(inside)",
}
for before, after in changes.items():
    if text.count(before) != 1:
        raise ValueError('Exact representation-only replacement count')
    text = text.replace(before, after)
exec(compile(text, str(original)+':authorized_serialization_only', 'exec'), globals())
