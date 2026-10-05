"""Official municipality names as query context, NOT business aliases.

All names joined by known CRM INSEE OR postcode; multiple municipalities and
all matching spans preserved. Original query stays open. The extra nominal
view is weak and place-conditioned, never exact identity/owner/sparse proof.
"""
from dataclasses import replace
import json
from pathlib import Path

from scripts import retrieval_v267_official_communes_snapshot as source
from .retrieval_corroborated_initial_v217 import PARTICLES, LEGAL
from .retrieval_nominal_abbreviation_v256 import component, tokens
from .retrieval_proof_v56 import components, signal
from .retrieval_variants_v118 import normalized


class Context:
    def __init__(self, files, reference):
        root=files.file(reference).parent
        manifest, snapshot=source.sealed(root)
        # Bind the ACTUAL complete successful source producer, supervision,
        # quiet preflight and original response into serving source epochs.
        for path in ('launch.json','phase/manifest.json','phase/supervision.json','phase/preflight.json'):
            files.file(source.record(root/path))
        files.file(manifest['producer']);files.file(manifest['source'])
        launch=json.loads(files.file(source.record(root/'launch.json')).read_text())
        for ref in launch['implementation'].values():files.file(ref)
        rows,stats=source.parse(files.file(snapshot['response']).read_bytes())
        if stats!=manifest['stats']:raise ValueError('Same whole official municipality snapshot required')
        self.insee={};self.postcode={}
        for row in rows:
            name=normalized(signal(row['nom']))
            if not name:continue # Masked/empty normalized context gives no proof, original record retained.
            self.insee.setdefault(row['code'],set()).add(name)
            for cp in row['codesPostaux']:self.postcode.setdefault(cp,set()).add(name)
        self.rows=stats['rows'];files.check_epochs()

    def names(self,q):
        # Official identifiers ONLY exact joins, no numerical distance/features.
        return self.insee.get(q.insee,set()) | self.postcode.get(q.postcode,set())


def residual_name(q, names):
    words=normalized(signal(q.name)).split();remove=set()
    for name in names:
        span=normalized(signal(name)).split()
        if not span:continue
        for lo in range(len(words)-len(span)+1):
            if words[lo:lo+len(span)]==span:remove.update(range(lo,lo+len(span)))
    remaining=[w for i,w in enumerate(words) if i not in remove]
    content=[w for w in remaining if w.isalpha() and w not in PARTICLES|LEGAL]
    residual=' '.join(remaining)
    # Do not turn empty context, a street-only token or generic STOP noun into
    # an initialism. Reuse the successful V256 whole3..6 language unchanged.
    if not remove or not (len(set(content))>=2 or tokens(residual)):return None
    return residual


def augmented_parts(backend,q,names):
    original=components(backend,q);tv=backend._tantivy
    def maximum(parts):
        parts=[p for p in parts if p is not None]
        return tv.Query.disjunction_max_query(parts,tie_breaker=0.) if parts else None
    def all_of(*parts):
        return tv.Query.boolean_query([(tv.Occur.Must,p) for p in parts]) if all(p is not None for p in parts) else None
    def const(p,value):
        return tv.Query.const_score_query(p,float(value)) if p is not None else None
    abbreviation=component(backend,q,original)
    if abbreviation is not None:
        original[0]=maximum([original[0],abbreviation])
        original[6]=all_of(original[0],const(original[4],0.))
        original[8]=all_of(original[0],original[1],const(original[4],0.))
    residual=residual_name(q,names)
    if residual is None:return original
    other_q=replace(q,name=residual);other=components(backend,other_q)
    alternate=maximum([other[0],component(backend,other_q,other)])
    at_place=all_of(alternate,const(original[4],0.))
    # Bounded MAX, no additive count for how many city names matched.
    original[0]=maximum([original[0],at_place])
    original[6]=maximum([original[6],at_place])
    original[8]=maximum([original[8],all_of(at_place,original[1])])
    return original
