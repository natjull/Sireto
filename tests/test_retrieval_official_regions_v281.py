import csv,io,zipfile
import pytest
from scripts import retrieval_v281_official_region_snapshots as s

def payload(year,change=None):
 cfg=s.SOURCES[year];f=cfg['fields'];rows=[{k:'NOM OFFICIEL' for k in f} for _ in range(cfg['rows'])]
 for i,r in enumerate(rows):
  r[f[0]]=f'{i:02d}';r['CHEFLIEU']='11111';r['TNCC']='0'
 if change=='short':rows.pop()
 elif change=='duplicate':rows[-1][f[0]]=rows[0][f[0]]
 elif change=='missing':rows[-1]['NCC']=''
 elif change=='identifier':rows[-1][f[0]]='123456789'
 text=io.StringIO(newline='');w=csv.DictWriter(text,fieldnames=f,delimiter=cfg['delimiter']);w.writeheader();w.writerows(rows)
 raw=text.getvalue().encode(cfg['encoding'])
 if cfg['member'] is None:return raw,rows
 target=io.BytesIO()
 with zipfile.ZipFile(target,'w') as z:z.writestr(cfg['member'],raw)
 return target.getvalue(),rows

@pytest.mark.parametrize('year',['2015','2016','2026'])
def test_all_records_fields_preserved(year):
 raw,rows=payload(year);assert s.parse(raw,year)==rows

@pytest.mark.parametrize('year',['2015','2016','2026'])
@pytest.mark.parametrize('bad',['short','duplicate','missing','identifier'])
def test_no_silent_dropping_or_identifier_semantics(year,bad):
 raw,rows=payload(year,bad)
 with pytest.raises(ValueError):s.parse(raw,year)

def test_zip_never_extracts_and_refuses_arbitrary_member():
 raw=io.BytesIO()
 with zipfile.ZipFile(raw,'w') as z:z.writestr('../escape.txt','secret')
 with pytest.raises(ValueError):s.parse(raw.getvalue(),'2015')

def test_guard_no_truncation():
 with pytest.raises(ValueError):s.parse(b'A'*(s.MAX_BYTES+1),'2026')

def test_no_source_queries_or_whole_history_claim():
 assert set(s.SOURCES)=={'2015','2016','2026'} and s.LIMITS['max_rss_bytes']==2147483648
 for cfg in s.SOURCES.values():assert cfg['url'].startswith('https://www.insee.fr/fr/statistiques/fichier/') and '?' not in cfg['url']

@pytest.mark.parametrize('kind',['live','stop'])
def test_root_refused_before_admission(tmp_path,monkeypatch,kind):
 monkeypatch.setattr(s,'OUTPUT',tmp_path);root=tmp_path/'run';root.mkdir()
 if kind=='stop':(root/'status.json').touch()
 with pytest.raises((ValueError,FileNotFoundError)):s.sealed(root)
