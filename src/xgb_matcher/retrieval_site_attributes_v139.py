"""Exact structured site evidence, not candidate filters or ordinal features.

Five-character public codes are reversibly encoded, number strings normalized
by unchanged V118/V128. UNKNOWN remains evidence absence, never site removal.
Other official attributes/geometry reuse their already sealed national arrays.
"""
from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_coupled_source_v121 import public_code
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_variant_batches_v128 import normalize
from src.xgb_matcher.retrieval_official_context_v110 import join_keys,positions,DATE_NULL
from src.xgb_matcher.retrieval_activity_semantic_v113 import class_lookup
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED,MISSING
from src.xgb_matcher.retrieval_v20_workbench import verify,write_json

LETTERS='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
RAW=('siret','siren','codePostalEtablissement','codeCommuneEtablissement','numeroVoieEtablissement')
ATTRS=('postcode_codes','insee_codes','number_codes')


def encode_code(value):
    value=public_code(value)
    if not value:return int(MISSING)
    result=0
    for char in value:result=result*36+LETTERS.index(char)
    return result


def decode_code(value):
    if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)):
        raise ValueError('Explicit public code position required')
    if int(value)==MISSING:return ''
    if not 0<=int(value)<36**5:raise ValueError('Five-character public code range required')
    result=[];value=int(value)
    for _ in range(5):result.append(LETTERS[value%36]);value//=36
    return ''.join(reversed(result))


def public_codes(values,cache):
    """Scalar authority on EVERY original distinct string; native row scatter."""
    dictionary=pc.dictionary_encode(pc.fill_null(values,''));book=dictionary.dictionary.to_pylist()
    codes=np.empty(len(book),dtype=np.uint32)
    for i,value in enumerate(book):
        if value not in cache:cache[value]=encode_code(value)
        codes[i]=cache[value]
    return codes[dictionary.indices.to_numpy(zero_copy_only=False)]


def number_codes(values,book,lookup):
    dictionary=pc.dictionary_encode(normalize(values));values=dictionary.dictionary.to_pylist()
    codes=np.empty(len(values),dtype=np.uint32)
    for i,value in enumerate(values):
        if value not in lookup:
            if len(book)>=MISSING:raise ValueError('No number vocabulary truncation permitted')
            lookup[value]=len(book);book.append(value)
        codes[i]=lookup[value]
    return codes[dictionary.indices.to_numpy(zero_copy_only=False)]


def schema(source,expected,columns=RAW):
    f=pq.ParquetFile(source)
    if (not isinstance(expected,int) or isinstance(expected,bool) or not 0<expected<2**32
        or f.metadata.num_rows!=expected or any(f.schema_arrow.field(k).type!=pa.string() for k in columns)):
        raise ValueError('Whole original current stock schema/count required')
    return f


def build(source,keys,parents,output,batch_rows=131072,progress=lambda row:None):
    n=len(keys);f=schema(source,n)
    if (keys.dtype!=np.uint64 or parents.dtype!=np.uint32 or parents.shape!=keys.shape
        or not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0):
        raise ValueError('Exact unsigned current site/parent namespace required')
    output=Path(output);arrays={k:np.lib.format.open_memmap(output/(k+'.npy'),mode='w+',dtype=np.uint32,shape=(n,)) for k in ATTRS}
    seen=np.zeros(n,dtype=bool);cache={};book=[''];lookup={'':0};count=0;known={k:0 for k in ATTRS}
    for batch in f.iter_batches(batch_size=batch_rows,columns=list(RAW),use_threads=False):
        values=dict(zip(RAW,batch.columns));ids=join_keys(values['siret'],14,np.uint64);pid=join_keys(values['siren'],9,np.uint32)
        pos=positions(keys,parents,ids,pid,seen)
        projected=(public_codes(values[RAW[2]],cache),public_codes(values[RAW[3]],cache),number_codes(values[RAW[4]],book,lookup))
        for name,value in zip(ATTRS,projected):
            arrays[name][pos]=value
            if not np.array_equal(arrays[name][pos],value):raise ValueError('Every persisted structured code must be exact')
            known[name]+=int(np.count_nonzero(value!=(0 if name=='number_codes' else MISSING)))
        count+=len(batch)
        if count%(batch_rows*32)==0:progress(dict(phase='all_current_structured_attributes',sites=count,total=n))
    if count!=n or not np.all(seen):raise ValueError('All current sites once, no denominator filtering')
    for a in arrays.values():a.flush()
    write_json(output/'number_dictionary.json',dict(values=book,normalization='UNCHANGED_V118_V128',unknown_empty_code=0,
        numeric_codes_not_score_features=True,source_dictionary_not_crm_aliases=True))
    return dict(sites=n,known=known,number_strings=len(book),every_current_site_once=True,
        arbitrary_code_or_number_width_cap=None,identifier_features=False,unknown_codes_do_not_remove_sites=True,
        source_code_scalar_on_every_distinct_raw_string=True,independent_all_source_audit_pending=True)


def audit(source,keys,parents,arrays,book,context,batch_rows=131072):
    """Fresh complete source pass; every exported attribute reconstructed.

    Also independently checks reused HQ/state/dates against RAW stock, not
    just a derived dataset whose NULL handling could differ.
    """
    n=len(keys);extra=('etablissementSiege','etatAdministratifEtablissement','dateCreationEtablissement','dateDebut')
    f=schema(source,n,RAW+('etatAdministratifEtablissement',))
    if (f.schema_arrow.field(extra[0]).type!=pa.bool_() or any(f.schema_arrow.field(k).type!=pa.date32() for k in extra[2:])
        or book[:1]!=[''] or len(book)!=len(set(book))
        or any(not isinstance(s,str) or not s.isascii() or any(c not in LETTERS+' ' for c in s)
            or s!=s.strip() or '  ' in s for s in book)
        or any(a.dtype!=np.uint32 or a.shape!=(n,) for a in arrays.values()) or set(arrays)!=set(ATTRS)):
        raise ValueError('Whole exact normalized current attribute sources required')
    seen=np.zeros(n,dtype=bool);cache={};count=0;known={k:0 for k in ATTRS};words=pa.array(book,type=pa.string())
    state_book=pa.array(context['state_book'],type=pa.string())
    for batch in f.iter_batches(batch_size=batch_rows,columns=list(RAW+extra),use_threads=False):
        values=dict(zip(RAW+extra,batch.columns));ids=join_keys(values['siret'],14,np.uint64);pid=join_keys(values['siren'],9,np.uint32)
        pos=positions(keys,parents,ids,pid,seen)
        for name,field in zip(ATTRS[:2],RAW[2:4]):
            if not np.array_equal(arrays[name][pos],public_codes(values[field],cache)):
                raise ValueError('Every exported public code must reconstruct exactly')
        codes=arrays['number_codes'][pos]
        if np.any(codes>=len(book)) or not pc.take(words,pa.array(codes)).equals(normalize(values[RAW[4]])):
            raise ValueError('Every exported normalized street number must reconstruct exactly')
        hq=pc.if_else(pc.is_null(values[extra[0]]),0,pc.if_else(pc.fill_null(values[extra[0]],False),2,1)).to_numpy(zero_copy_only=False)
        if not np.array_equal(context['headquarters_codes'][pos],hq):raise ValueError('Reused official headquarters differs from raw stock')
        if not pc.take(state_book,pa.array(context['state_codes'][pos])).equals(values[extra[1]]):
            raise ValueError('Reused official state differs from raw stock')
        for name,field in (('creation_days',extra[2]),('current_period_start_days',extra[3])):
            days=pc.cast(values[field],pa.int32())
            if pc.any(pc.fill_null(pc.equal(days,int(DATE_NULL)),False)).as_py():raise ValueError('Date sentinel collision')
            expected=pc.fill_null(days,int(DATE_NULL)).to_numpy(zero_copy_only=False)
            if not np.array_equal(context[name][pos],expected):raise ValueError('Reused official dates differ from raw stock')
        for name in ATTRS:known[name]+=int(np.count_nonzero(arrays[name][pos]!=(0 if name=='number_codes' else MISSING)))
        count+=len(batch)
    if count!=n or not np.all(seen):raise ValueError('Whole independent original source audit required')
    return dict(sites=n,known=known,every_exported_attribute_reconstructed_exactly=True,
        every_reused_headquarters_state_and_date_matches_raw_stock=True,closed_sites_filtered=False,
        unknown_or_geography_sites_filtered=False,dates_not_invented_as_historical_address_validity=True)


class StructuredSites:
    """Readonly site evidence at opaque positions; extras stay UNKNOWN."""
    def __init__(self,reference):
        m=json.loads(verify(reference).read_text());self.metadata=m
        if (m['status']!='SEALED_V139_ALL_CURRENT_STRUCTURED_SITE_ATTRIBUTES' or any(m.get(k) is not False for k in PROTECTED)
            or m['complete_retrieval_index'] or m['ready_for_fold1']
            or not m['audit']['every_exported_attribute_reconstructed_exactly']):
            raise ValueError('Protected complete exact structured site evidence required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values():verify(ref)
        self.a={k.removesuffix('.npy'):np.load(verify(ref),mmap_mode='r',allow_pickle=False) for k,ref in m['artifacts'].items()}
        self.reused={k:np.load(verify(ref),mmap_mode='r',allow_pickle=False) for k,ref in m['reused_arrays'].items()}
        self.book=json.loads(verify(m['number_dictionary']).read_text())['values'];self.number_lookup={s:i for i,s in enumerate(self.book)}
        self.base=m['stats']['sites'];self.sites=m['namespace_sites']
        if (set(self.a)!=set(ATTRS) or any(a.dtype!=np.uint32 or a.shape!=(self.base,) for a in self.a.values())
            or self.book[:1]!=[''] or len(self.book)!=len(self.number_lookup) or self.base>self.sites):
            raise ValueError('Whole typed structured namespace/dictionary required')
        specs=dict(headquarters_codes=np.uint8,state_codes=np.uint16,creation_days=np.int32,
            current_period_start_days=np.int32,nomenclature_codes=np.uint8,original_activity_codes=np.uint16,site_xy=np.float64)
        if set(self.reused)!=set(specs) or any(self.reused[k].dtype!=d or self.reused[k].shape!=((self.base,2) if k=='site_xy' else (self.base,)) for k,d in specs.items()):
            raise ValueError('Whole exact reused site context/geometry required')
        cat=json.loads(verify(m['activity_catalog']).read_text());labels=json.loads(verify(cat['official_labels']).read_text())['rows']
        self.activity_classes=class_lookup(cat['stats']['nomenclatures'],cat['stats']['activity_book'],labels)

    def positions(self,ids):
        if ids.dtype!=np.uint32 or ids.ndim!=1 or np.any(ids>=self.sites):raise ValueError('Exact actual site positions required')
        return ids<self.base

    def flags(self,ids,postcode='',insee='',number=''):
        native=self.positions(ids);p=ids[native]
        place=np.zeros(len(ids));num=place.copy();hq=place.copy();classes=np.full(len(ids),-1,dtype=np.int32)
        for key,value in (('postcode_codes',postcode),('insee_codes',insee)):
            code=encode_code(value)
            if code!=MISSING:place[native]=np.maximum(place[native],self.a[key][p]==code)
        number=normalized(number);code=self.number_lookup.get(number)
        if number and code is not None:num[native]=self.a['number_codes'][p]==code
        hq[native]=self.reused['headquarters_codes'][p]==2
        classes[native]=self.activity_classes[self.reused['nomenclature_codes'][p],self.reused['original_activity_codes'][p]]
        return place,num,hq,classes

    def coordinates(self,ids):
        native=self.positions(ids);xy=np.full((len(ids),2),np.nan);xy[native]=self.reused['site_xy'][ids[native]];return xy
