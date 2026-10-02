import json
import pytest
from scripts import retrieval_v49_primary_probe as v49


@pytest.mark.parametrize('status,n,opened',[
    ('STOP',128,False),('SEALED_INITIAL_TOP100',16,False),('SEALED_INITIAL_TOP100',128,True)])
def test_partial_generation_cannot_open_labels_or_cached_comparator(tmp_path,monkeypatch,status,n,opened):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status=status,queries=n,truth_opened=opened)))
    def forbidden(*args,**kwargs): raise AssertionError('No references before complete generation')
    monkeypatch.setattr(v49,'verify',forbidden)
    with pytest.raises(ValueError,match='Complete sealed generation required'): v49.evaluate(tmp_path)


def test_existing_measurement_not_overwritten(tmp_path):
    dest=tmp_path/'evaluation.json'; dest.write_text('sealed prior result')
    with pytest.raises(ValueError,match='Do not overwrite'): v49.evaluate(tmp_path)
    assert dest.read_text()=='sealed prior result'
