import json
import pytest
from scripts import retrieval_v43_spatial as v43


@pytest.mark.parametrize('status,n,opened',[
    ('STOP',128,False),('SEALED_INITIAL_TOP100',16,False),('SEALED_INITIAL_TOP100',128,True)])
def test_incomplete_or_label_open_generation_refused_before_label_reads(tmp_path,monkeypatch,status,n,opened):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status=status,queries=n,truth_opened=opened)))
    def forbidden(*args,**kwargs):
        raise AssertionError('Labels or references must not be opened for an incomplete generation')
    monkeypatch.setattr(v43,'verify',forbidden)
    with pytest.raises(ValueError,match='Complete sealed generation required'):
        v43.evaluate(tmp_path)
    assert not (tmp_path/'evaluation.json').exists()


def test_existing_evaluation_is_immutable(tmp_path):
    dest=tmp_path/'evaluation.json'; dest.write_text('prior result')
    with pytest.raises(ValueError,match='Do not overwrite'):
        v43.evaluate(tmp_path)
    assert dest.read_text()=='prior result'
