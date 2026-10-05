"""A street number is corroboration ONLY for this leaf's informative road.

Generic French address syntax, never CRM business aliases/labels. Original
raw address/text, aliases/owners/geometry remain open. No hard road filter.
"""
from scripts import retrieval_v38_initial_top100 as v38
from .retrieval_corroborated_initial_v217 import PARTICLES
from .retrieval_proof_v56 import signal
from .retrieval_variants_v118 import normalized

# Syntactic road descriptors, not learned/memorized business-name aliases.
ROAD_TYPES=frozenset('RUE AV AVENUE BD BOULEVARD CHEMIN CHE CHAUSSEE ROUTE RTE PLACE PL ALLEE AL IMPASSE IMP QUAI PASSAGE PAS SQUARE SQ COURS CR ESPLANADE ESP PROMENADE PROM SENTIER VOIE'.split())


def core_terms(q):
    words=normalized(signal(q.address)).split()
    number=normalized(signal(q.number)).split()
    # Only the explicit leading house-number span is removed. A date/number
    # INSIDE a road name remains evidence; no all-digit or numerical feature.
    if number and words[:len(number)]==number:words=words[len(number):]
    return sorted(set(t for w in words if w not in PARTICLES|ROAD_TYPES
        for t in v38.tokens(w)))


def component(native,q,parts):
    terms=core_terms(q)
    if not terms or not q.number or parts[4] is None:return None
    tv=native._tantivy
    views=[]
    for field in v38.ADDRESSES:
        views.append(tv.Query.boolean_query([(tv.Occur.Should,tv.Query.const_score_query(
            tv.Query.term_query(native.index.schema,field,t,'basic'),1./len(terms))) for t in terms]))
    coverage=tv.Query.disjunction_max_query(views,tie_breaker=0.)
    number=tv.Query.const_score_query(tv.Query.term_query(native.index.schema,'number',signal(q.number),'basic'),0.)
    place=tv.Query.const_score_query(parts[4],0.)
    # The ONLY score is bounded road coverage. Numerical match and place
    # corroborate it, do not create an independent constant bonus.
    return tv.Query.boolean_query([(tv.Occur.Must,coverage),(tv.Occur.Must,number),(tv.Occur.Must,place)])
