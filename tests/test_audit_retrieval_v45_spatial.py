import json
import pytest
from scripts import audit_retrieval_v45_spatial as audit


def test_no_partial_audit_or_truth_access(tmp_path,monkeypatch):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status='STOP',queries=128,truth_opened=False)))
    def forbidden(*args,**kwargs):
        raise AssertionError('Must refuse incomplete output before any reference read')
    monkeypatch.setattr(audit,'verify',forbidden)
    with pytest.raises(ValueError,match='Complete sealed generation required'):
        audit.sealed(tmp_path)


@pytest.mark.parametrize('mode,filename',[('parity','physical_parity128.json'),('diagnostics','spatial_diagnostics.json')])
def test_saved_audits_cannot_be_overwritten(tmp_path,mode,filename):
    dest=tmp_path/filename; dest.write_text('sealed prior result')
    with pytest.raises(ValueError,match='Do not overwrite'):
        getattr(audit,mode)(tmp_path)
    assert dest.read_text()=='sealed prior result'
