"""All official address observations, strict site grain, once-normalized text.

Original dates, flags, masks and provenance live in the fully hashed raw
sources, addressed by row ordinal and projection role. No parent address is
spread to sites, no history/address validity or geometry is manufactured.
"""
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.xgb_matcher.retrieval_canonical_inventory_v140 import classify
from src.xgb_matcher.retrieval_variant_batches_v128 import normalize,WHITESPACE,MASK
from src.xgb_matcher.retrieval_site_attributes_v139 import public_codes,encode_code
from src.xgb_matcher.retrieval_variants_v118 import clean,normalized
from src.xgb_matcher.retrieval_proof_bridge_v137 import MISSING

CANONICAL=('siren','siret','subject_kind','source','raw_value','normalized_value','postcode','insee','street_number','is_current')
LINE=('numeroVoie','indiceRepetition','typeVoie','libelleVoie')
FULL=('complementAdresse','numeroVoie','indiceRepetition','typeVoie','libelleVoie','distributionSpeciale',
    'codePostal','libelleCommune','libelleCommuneEtranger','codeCedex','libelleCedex','codePaysEtranger','libellePaysEtranger')
CURRENT_FIELDS=tuple(dict.fromkeys(('siren','siret',*[p+s+'Etablissement' for s in ('','2') for p in FULL],
    'codeCommuneEtablissement','codeCommune2Etablissement','dernierNumeroVoieEtablissement','indiceRepetitionDernierNumeroVoieEtablissement')))
SCHEMA=pa.schema([
    ('observation_id',pa.uint32()),('source_row_ordinal',pa.uint32()),('projection_role',pa.uint8()),
    ('subject_code',pa.uint8()),('site_position',pa.uint32()),('parent_position',pa.uint32()),
    ('address_text',pa.string()),('source',pa.string()),('source_is_current',pa.bool_()),('current_exact',pa.bool_()),
    ('postcode_code',pa.uint32()),('insee_code',pa.uint32()),('number_text',pa.string()),('lexical_eligible',pa.bool_())])


def required(role):
    if role=='official_addresses':return CANONICAL
    if role=='current_addresses':return CURRENT_FIELDS
    raise ValueError('Explicit whole official address input role required')


def string(table,key):
    a=table[key].combine_chunks()
    if a.type!=pa.string():raise ValueError('Explicit raw address string field required: '+key)
    return a


def clean_array(values):return pc.utf8_trim(pc.fill_null(values,''),characters=WHITESPACE)


def visible_component(values):
    """A masked component is UNKNOWN, without erasing other public fields."""
    values=clean_array(values);mask=pc.match_substring_regex(pc.utf8_upper(values),MASK)
    unicode=pc.match_substring_regex(values,r'[^\x00-\x7f]')
    if pc.any(unicode).as_py():
        d=pc.dictionary_encode(pc.filter(values,unicode))
        exact=pa.array([bool(re.search(r'\[\s*ND\s*\]',v,re.I)) for v in d.dictionary.to_pylist()])
        mask=pc.replace_with_mask(mask,unicode,pc.take(exact,d.indices))
    return pc.if_else(mask,'',values)


def scoped_positions(table,role,catalog):
    n=len(table);parent=clean_array(string(table,'siren'));site=clean_array(string(table,'siret'))
    subject=clean_array(string(table,'subject_kind')) if role=='official_addresses' else pa.repeat(pa.scalar('SIRET'),n)
    vp=pc.match_substring_regex(parent,r'^[0-9]{9}$')
    vs=pc.and_(vp,pc.and_(pc.match_substring_regex(site,r'^[0-9]{14}$'),pc.equal(pc.utf8_slice_codeunits(site,0,9),parent)))
    sites=pc.and_(pc.equal(subject,'SIRET'),vs).to_numpy(zero_copy_only=False)
    parents=pc.and_(pc.equal(subject,'SIREN'),vp).to_numpy(zero_copy_only=False)
    if role=='current_addresses' and not np.all(sites):raise ValueError('Every original current site must retain explicit valid grain')
    codes=np.zeros(n,dtype=np.uint8);codes[parents]=1;codes[sites]=2
    sp=np.full(n,MISSING,dtype=np.uint32);pp=sp.copy();bound=sites|parents
    sk=pc.cast(pc.filter(site,pa.array(sites)),pa.uint64()).to_numpy(zero_copy_only=False)
    pk=pc.cast(pc.filter(parent,pa.array(bound)),pa.uint32()).to_numpy(zero_copy_only=False)
    sp[sites]=catalog.site_positions(sk);pp[bound]=catalog.parent_positions(pk)
    if np.any(sp[sites]==MISSING) or np.any(pp[sites]==MISSING):
        raise ValueError('Every explicit official address site AND parent must be admitted before projection')
    if (not np.array_equal(catalog.reconstruct(sp[sites],'site'),sk)
        or not np.array_equal(catalog.reconstruct(pp[bound][pp[bound]!=MISSING],'parent'),pk[pp[bound]!=MISSING])):
        raise ValueError('Every source identity must reconstruct exactly through opaque positions')
    return codes,sp,pp


def project(table,role,catalog,start_ordinal,observation_base,code_cache=None):
    required(role);n=len(table);factor=1 if role=='official_addresses' else 4
    if (not isinstance(start_ordinal,int) or isinstance(start_ordinal,bool) or start_ordinal<0
        or not isinstance(observation_base,int) or isinstance(observation_base,bool) or observation_base<0
        or start_ordinal+n>=MISSING or observation_base+factor*(start_ordinal+n)>=MISSING):
        raise ValueError('Complete source and observation typed ordinal range required')
    if not n:return pa.Table.from_batches([],schema=SCHEMA)
    for field in required(role):
        if field=='is_current':
            if table[field].type!=pa.bool_():raise ValueError('Original nullable official current flag required')
        else:string(table,field)
    cache={} if code_cache is None else code_cache
    subjects,sp,pp=scoped_positions(table,role,catalog)
    ordinals=np.arange(start_ordinal,start_ordinal+n,dtype=np.uint32)
    pieces=[]
    def append(rank,text,source,is_current,current,postcode,insee,number):
        data=dict(observation_id=pa.array(observation_base+factor*ordinals.astype(np.uint64)+rank,type=pa.uint32()),
            source_row_ordinal=pa.array(ordinals),projection_role=pa.repeat(pa.scalar(0 if factor==1 else rank+1,pa.uint8()),n),
            subject_code=pa.array(subjects),site_position=pa.array(sp),parent_position=pa.array(pp),address_text=text,
            source=source,source_is_current=is_current,current_exact=current,
            postcode_code=pa.array(public_codes(postcode,cache)),insee_code=pa.array(public_codes(insee,cache)),
            number_text=normalize(number),lexical_eligible=pc.and_(pa.array(subjects==2),pc.not_equal(text,'')))
        pieces.append(pa.Table.from_arrays([data[f.name] for f in SCHEMA],schema=SCHEMA))
    if role=='official_addresses':
        raw=string(table,'raw_value');value=pc.if_else(pc.equal(clean_array(raw),''),string(table,'normalized_value'),raw)
        current=table['is_current'].combine_chunks()
        append(0,normalize(value),string(table,'source'),current,pc.fill_null(current,False),
            string(table,'postcode'),string(table,'insee'),string(table,'street_number'))
    else:
        source=pa.repeat(pa.scalar('SIRENE_CURRENT'),n);is_current=pa.repeat(pa.scalar(True),n)
        components={field:visible_component(string(table,field)) for field in CURRENT_FIELDS if field not in ('siren','siret')}
        for rank,(suffix,full) in enumerate((('',False),('',True),('2',False),('2',True))):
            fields=[p+suffix+'Etablissement' for p in (FULL if full else LINE)]
            if full and not suffix:fields[3:3]=['dernierNumeroVoieEtablissement','indiceRepetitionDernierNumeroVoieEtablissement']
            text=normalize(pc.binary_join_element_wise(*[components[k] for k in fields],' '))
            append(rank,text,source,is_current,pa.repeat(pa.scalar(not bool(suffix)),n),
                string(table,'codePostal'+suffix+'Etablissement'),string(table,'codeCommune'+suffix+'Etablissement'),
                string(table,'numeroVoie'+suffix+'Etablissement'))
    result=pa.concat_tables(pieces)
    if factor>1:result=pc.take(result,pc.sort_indices(result,sort_keys=[('observation_id','ascending')]))
    if not np.array_equal(result['observation_id'].to_numpy(),np.arange(observation_base+factor*start_ordinal,
        observation_base+factor*(start_ordinal+n),dtype=np.uint32)):
        raise ValueError('Every original source observation/role must remain exactly once')
    return result


def scalar_text_oracle(original,projected,role):
    """Independent Python authority on the declared first source rows only."""
    expected=[]
    for row in original.to_pylist():
        if role=='official_addresses':
            value=row['raw_value'] if clean(row['raw_value']) else row['normalized_value']
            expected.append((normalized(value),encode_code(row['postcode']),encode_code(row['insee']),normalized(row['street_number'])))
        else:
            for suffix,full in (('',False),('',True),('2',False),('2',True)):
                fields=[p+suffix+'Etablissement' for p in (FULL if full else LINE)]
                if full and not suffix:fields[3:3]=['dernierNumeroVoieEtablissement','indiceRepetitionDernierNumeroVoieEtablissement']
                components=[clean(row[f]) if not re.search(r'\[\s*ND\s*\]',clean(row[f]),re.I) else '' for f in fields]
                expected.append((normalized(' '.join(components)),encode_code(row['codePostal'+suffix+'Etablissement']),
                    encode_code(row['codeCommune'+suffix+'Etablissement']),normalized(row['numeroVoie'+suffix+'Etablissement'])))
    actual=list(zip(*[projected[k].to_pylist() for k in ('address_text','postcode_code','insee_code','number_text')]))
    if actual!=expected:raise ValueError('Independent first-source-row address scalar oracle drift')
    return len(original)


def audit_batch(original,exported,role,catalog,start,base):
    """Full second source pass plus independent identity/grain reconstruction.

    Text recomputation uses the same tested native projector, not an asserted
    all-row independent scalar oracle. That distinction is public in stats.
    """
    expected=project(original,role,catalog,start,base)
    if not exported.equals(expected):raise ValueError('Every exported address field must match the complete second source pass')
    factor=1 if role=='official_addresses' else 4
    if role=='official_addresses':
        batch=original.select(['subject_kind','siren','siret']).to_batches()[0]
        keys,parent,context=classify(batch)
        subject=exported['subject_code'].to_numpy()
        if int(np.sum(subject==1))!=parent or int(np.sum(subject==0))!=context:
            raise ValueError('Every independent original address grain accounted')
    else:
        sid=[int(v.strip()) for v in original['siret'].to_pylist()]
        keys=np.repeat(np.array(sid,dtype=np.uint64),factor);subject=exported['subject_code'].to_numpy()
    pos=exported['site_position'].to_numpy()[subject==2]
    if not np.array_equal(catalog.reconstruct(pos,'site'),keys):
        raise ValueError('Every exported site proof must bind its ORIGINAL site, never a parent sibling')
    return True
