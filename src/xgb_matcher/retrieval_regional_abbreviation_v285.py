"""Official whole regional qualifier around a whole nominal abbreviation.

No embedded arbitrary short tokens, no CRM dictionary, no borrowed aliases.
Original query stays intact. Explicit source snapshots are NOT full lineage.
"""
import re,json
from scripts import retrieval_v281_official_region_snapshots as source
from .retrieval_variants_v118 import normalized
from .retrieval_nominal_abbreviation_v256 import tokens
from .retrieval_compound_initialisms_v222 import partitions,SKIP,BETWEEN,TAIL
from .retrieval_corroborated_initial_v217 import EXACT_FIELDS


def names(snapshots):
 if set(snapshots)!=set(source.SOURCES):raise ValueError('All three explicit whole regional snapshots required')
 return sorted({normalized(r[f]) for s in snapshots.values() for r in s['rows']
  for f in ('NCC','NCCENR','LIBELLE') if f in r and normalized(r[f])})


def residual(name,regions):
 words=normalized(name).split();covered=set();matches=set()
 for region in regions:
  span=region.split()
  if not span or not re.fullmatch('[A-Z0-9 ]+',region):raise ValueError('Whole normalized official regional names required')
  for lo in range(len(words)-len(span)+1):
   if words[lo:lo+len(span)]==span:covered.update(range(lo,lo+len(span)));matches.add(region)
 return ' '.join(w for i,w in enumerate(words) if i not in covered),sorted(matches)


def patterns(name,regions):
 remaining,matched=residual(name,regions)
 if not matched:return []
 eligible=tokens(remaining)
 if not eligible:return []
 qualifier='('+'|'.join(sorted(set(regions)))+')'
 output=[]
 for token in eligible:
  for split in partitions(token):
   body=BETWEEN.join(chunk+'[A-Z0-9]*' for chunk in split)
   for variant in (body,qualifier+BETWEEN+body,body+BETWEEN+qualifier):
    output.append(SKIP+variant+TAIL)
 return output


def membership(name,aliases,regions):
 return any(re.fullmatch(p,normalized(a)) is not None for p in patterns(name,regions) for a in aliases)


def component(native,q,base,regions):
 ps=patterns(q.name,regions)
 if not ps or base[4] is None:return None
 tv=native._tantivy
 predicate=tv.Query.disjunction_max_query([tv.Query.regex_query(native.index.schema,f,p)
  for f in EXACT_FIELDS for p in ps],tie_breaker=0.)
 return tv.Query.const_score_query(tv.Query.boolean_query([
  (tv.Occur.Must,predicate),(tv.Occur.Must,tv.Query.const_score_query(base[4],0.))]),1.)


class Regions:
 def __init__(self,files,reference):
  root=files.file(reference).parent;m,s=source.sealed(root)
  for path in ('launch.json','phase/manifest.json','phase/supervision.json','phase/preflight.json'):files.file(source.record(root/path))
  files.file(m['source']);files.file(m['producer'])
  for ref in json.loads(files.file(source.record(root/'launch.json')).read_text())['implementation'].values():files.file(ref)
  for v in s['snapshots'].values():files.file(v['response'])
  self.names=names(s['snapshots']);self.files=files;files.check_epochs()
 def close(self):self.files.check_epochs()
