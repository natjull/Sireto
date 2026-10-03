"""Same V118 observation semantics, columnar batches; Unicode stays exact scalar.

No latest-period/name cap, CRM alias, source-family omission or site fanout.
ASCII normalization is native Arrow; non-ASCII dictionary values use the
unchanged Python authority, avoiding approximate Unicode substitutions.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA,UL_FIELDS,SITE_FIELDS,GIVEN_FIELDS,ROLES,normalized

WHITESPACE=''.join(chr(i) for i in range(0x110000) if chr(i).isspace())
MASK=r'\[[ \t\n\r\f\v\x1c-\x1f]*ND[ \t\n\r\f\v\x1c-\x1f]*\]'
ID=r'(^|[^0-9])([0-9]{14}|[0-9]{9})([^0-9]|$)'


def normalize(values):
    if isinstance(values,pa.ChunkedArray):values=values.combine_chunks()
    values=pc.fill_null(values,'');unicode=pc.match_substring_regex(values,r'[^\x00-\x7f]')
    upper=pc.utf8_upper(values);masked=pc.match_substring_regex(upper,MASK)
    result=pc.replace_substring_regex(upper,ID,r'\1 \3')
    result=pc.replace_substring_regex(result,ID,r'\1 \3')
    result=pc.utf8_trim(pc.replace_substring_regex(result,r'[^A-Z0-9]+',' '),characters=' ')
    result=pc.if_else(masked,'',result)
    if pc.any(unicode).as_py():
        values=pc.dictionary_encode(pc.filter(values,unicode))
        exact=pa.array([normalized(s) for s in values.dictionary.to_pylist()],type=pa.string())
        result=pc.replace_with_mask(result,unicode,pc.take(exact,values.indices))
    return result


def project(table,role,start_ordinal=0):
    if role not in ROLES:raise ValueError('Unknown official source role')
    if not isinstance(start_ordinal,int) or isinstance(start_ordinal,bool) or start_ordinal<0:raise ValueError('Nonnegative source ordinal required')
    n=len(table)
    if n==0:return pa.Table.from_batches([],schema=OBS_SCHEMA)
    raw_cache={};clean_cache={};norm_cache={}
    def raw(key):
        if key not in raw_cache:
            values=table[key].combine_chunks() if key in table.column_names else pa.nulls(n,type=pa.string())
            if not (pa.types.is_string(values.type) or pa.types.is_large_string(values.type) or pa.types.is_null(values.type)
                    or pa.types.is_date(values.type) or pa.types.is_timestamp(values.type)):
                raise ValueError('Unexpected source string/date field type: '+key)
            raw_cache[key]=(pa.array([None if v is None else str(v) for v in values.to_pylist()],type=pa.string())
                if pa.types.is_timestamp(values.type) else pc.cast(values,pa.string()))
        return raw_cache[key]
    def clean(key):
        if key not in clean_cache:clean_cache[key]=pc.utf8_trim(pc.fill_null(raw(key),''),characters=WHITESPACE)
        return clean_cache[key]
    def nullable(key):
        values=clean(key);return pc.if_else(pc.not_equal(values,''),values,None)
    def norm(key):
        if key not in norm_cache:norm_cache[key]=normalize(raw(key))
        return norm_cache[key]
    def const(value,type=pa.string()):return pa.repeat(pa.scalar(value,type=type),n)
    def bools(values):return pc.fill_null(values,False).to_numpy(zero_copy_only=False)
    historical=role.startswith('history_');legal=role.endswith('_ul');official=role=='official_names'
    subject=clean('subject_kind') if official else const('SIREN' if legal else 'SIRET')
    parent=clean('siren');site=pc.if_else(pc.equal(subject,'SIRET'),clean('siret'),'')
    valid_parent=pc.match_substring_regex(parent,r'^[0-9]{9}$')
    valid_site=pc.and_(valid_parent,pc.and_(pc.match_substring_regex(site,r'^[0-9]{14}$'),pc.equal(pc.utf8_slice_codeunits(site,0,9),parent)))
    if 'source_row_ordinal' in table.column_names:
        ordinal=table['source_row_ordinal'].combine_chunks()
        if ordinal.type!=pa.uint64() or ordinal.null_count:raise ValueError('Typed original source ordinals required')
    else:ordinal=pa.array(np.arange(start_ordinal,start_ordinal+n,dtype=np.uint64))
    current=const(not historical,pa.bool_())
    priority=pc.cast(table['source_priority'].combine_chunks(),pa.int32()) if 'source_priority' in table.column_names else const(None,pa.int32())
    if official:
        current=table['is_current'].combine_chunks() if 'is_current' in table.column_names else const(None,pa.bool_())
        if current.type!=pa.bool_() and not pa.types.is_null(current.type):raise ValueError('Explicit official current flag required')
        current=pc.cast(current,pa.bool_())
    base=dict(input_role=const(role),source_row_ordinal=ordinal,
        source=clean('source') if official else const('SIRENE_HISTORY' if historical else 'SIRENE_CURRENT'),
        siren=parent,siret=site,subject_kind=subject,
        valid_from=nullable('valid_from') if official else nullable('dateDebut') if historical else const(None),
        valid_to=nullable('valid_to') if official else nullable('dateFin') if historical else const(None),
        diffusion_status=nullable('statutDiffusionUniteLegale' if legal else 'statutDiffusionEtablissement'),
        administrative_state=nullable('etatAdministratifUniteLegale' if legal else 'etatAdministratifEtablissement'),
        source_period_start=nullable('dateDebut'),is_current=current,evidence_id=nullable('evidence_id'),
        source_record_id=nullable('source_record_id'),observed_at=nullable('observed_at'),source_priority=priority,
        historical_address_claimed=const(False,pa.bool_()),related_target_alias=const(False,pa.bool_()),
        original_normalized_value=raw('normalized_value') if official else const(None),
        original_raw_value=raw('raw_value') if official else const(None))
    pieces=[];emitted=np.zeros(n,dtype=np.bool_);row_order=np.arange(n,dtype=np.uint64)*64
    def append(value,text,fields,kind,derivation,mask,rank):
        mask=np.asarray(mask,dtype=np.bool_)
        if not np.any(mask):return
        emitted[mask]=True;kind=const(kind) if isinstance(kind,str) else kind
        parent_legal=pc.and_(pc.and_(pc.equal(subject,'SIREN'),valid_parent),pc.equal(kind,'LEGAL'))
        parent_commercial=pc.and_(pc.and_(pc.equal(subject,'SIREN'),valid_parent),pc.is_in(kind,value_set=pa.array(['USUAL','TRADE'])))
        site_name=pc.and_(pc.and_(pc.equal(subject,'SIRET'),valid_site),pc.is_in(kind,value_set=pa.array(['LEGAL','USUAL','TRADE','SIGN'])))
        scope=pc.if_else(parent_legal,'LEGAL_PERSON_NAME',pc.if_else(parent_commercial,'LEGAL_PERSON_COMMERCIAL_NAME',
            pc.if_else(site_name,'EXPLICIT_ENDPOINT_NAME','CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED')))
        data=dict(base,raw_value=value,name_text=text,source_fields=const(list(fields),pa.list_(pa.string())),
            name_kind=kind,derivation=const(derivation),lexical_scope=scope,
            lexical_eligible=pc.and_(pc.not_equal(text,''),pc.not_equal(scope,'CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED')))
        result=pa.Table.from_arrays([data[f.name] for f in OBS_SCHEMA],schema=OBS_SCHEMA)
        result=result.append_column('_order',pa.array(row_order+rank))
        pieces.append(pc.filter(result,pa.array(mask)))
    if official:
        value=pc.if_else(pc.equal(clean('raw_value'),''),raw('normalized_value'),raw('raw_value'))
        append(value,normalize(value),['raw_value'],clean('name_kind'),'OFFICIAL_OBSERVATION',np.ones(n,dtype=np.bool_),0)
    else:
        fields=UL_FIELDS if legal else SITE_FIELDS
        for rank,(field,kind) in enumerate(fields.items()):
            if field in table.column_names:append(raw(field),norm(field),[field],kind,'ORIGINAL_FIELD',bools(pc.not_equal(clean(field),'')),rank)
        if legal and not historical:
            diffusion=bools(pc.equal(raw('statutDiffusionUniteLegale'),'O'))
            given=[bools(pc.not_equal(norm(field),'')) for field in GIVEN_FIELDS]
            combo=np.zeros(n,dtype=np.uint8)
            for bit,valid in enumerate(given[1:]):combo|=valid.astype(np.uint8)<<bit
            for family_index,family in enumerate(('nomUniteLegale','nomUsageUniteLegale')):
                valid_family=diffusion & bools(pc.not_equal(norm(family),''));kind='LEGAL' if family_index==0 else 'USUAL'
                specifications=[((field,),valid,2*j) for j,(field,valid) in enumerate(zip(GIVEN_FIELDS,given))]
                specifications += [(tuple(GIVEN_FIELDS[1:][j] for j in range(4) if value&(1<<j)),combo==value,10)
                                   for value in range(16) if value.bit_count()>1]
                for given_fields,valid,rank in specifications:
                    mask=valid_family&valid
                    if not np.any(mask):continue
                    for direction,order in enumerate(((family,*given_fields),(*given_fields,family))):
                        value=pc.binary_join_element_wise(*[clean(field) for field in order],' ')
                        append(value,normalize(value),order,kind,'SAME_ROW_EI_CONCAT',mask,8+family_index*12+rank+direction)
        append(const(''),const(''),[],'UNKNOWN','CONTEXT_ROW',~emitted,63)
    combined=pa.concat_tables(pieces)
    return pc.take(combined,pc.sort_indices(combined,sort_keys=[('_order','ascending')])).drop(['_order'])
