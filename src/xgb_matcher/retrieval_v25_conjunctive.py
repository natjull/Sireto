"""Truth-free conjunctive retrieval. Identifiers are output keys, not evidence."""
import json
from pathlib import Path

from .retrieval_v20_workbench import ROOT, record


def config_at(path):
    c=json.loads(Path(path).read_text())
    if (c['schema']!='sireto-v25-conjunctive-1' or c['recipe']!='ADDITIVE_NUMBER_MUST_ADDRESS_CHAR_SIRET_DISCOVERY'
        or c['site_cap_per_query']!=512 or c['workers']!=4 or c['folds']!=[2,3,4]
        or any(c[k] for k in ('site_expansion','master_selection','national_run'))):
        raise ValueError('Frozen discovery recipe drift')
    return c


def code_records():
    paths=[Path(__file__),ROOT/'scripts/retrieval_v25_conjunctive.py',
           ROOT/'src/xgb_matcher/hierarchical_retrieval.py',ROOT/'scripts/run_retrieval_parent_discovery_v18.py',
           ROOT/'src/xgb_matcher/retrieval_v12_scene_ir.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
           ROOT/'src/xgb_matcher/retrieval_v21_membership.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    return {str(p.relative_to(ROOT)):record(p) for p in paths}


def index_inventory(root):
    return {str(p.relative_to(root)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(root.rglob('*')) if p.is_file() and not p.name.endswith('.lock')}


def search(backend, query, cap=512):
    if not 1<=cap<=512:
        raise ValueError('Discovery cap')
    if not query.number or not query.address:
        return []
    tv=backend._tantivy
    number=tv.Query.term_query(backend.index.schema,'number',query.number,'basic')
    address=backend._character_query('address_ngrams',query.address)
    text=tv.Query.boolean_query([(tv.Occur.Must,number),(tv.Occur.Must,address)])
    parsed=backend._combined_query(query,text,document_type='siret')
    found=[]
    for score,location in backend.searcher.search(parsed,limit=cap,count=False).hits:
        doc=backend.searcher.doc(location).to_dict()
        siren=backend._first(doc,'siren');siret=backend._first(doc,'siret')
        if not isinstance(siren,str) or not siren.isdigit() or len(siren)!=9 or not isinstance(siret,str) or not siret.isdigit() or len(siret)!=14:
            raise ValueError('Invalid official identities')
        found.append(dict(siret=siret,siren=siren,score=float(score)))
    found.sort(key=lambda r:(-r['score'],r['siret']))
    if len({r['siret'] for r in found})!=len(found):
        raise ValueError('Duplicate indexed SIRET')
    rank=0;previous=None
    for position,row in enumerate(found,1):
        if previous is None or row['score']!=previous:
            rank+=1
        row.update(candidate_rank=position,evidence_rank=rank)
        previous=row['score']
    return found
