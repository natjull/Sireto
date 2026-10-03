"""Explicit original SIRENE nomenclature and official historical NAF labels.

No taxonomy inferred from code shape/date, no CRM/label access or site filter.
The raw stock NULL is recovered despite V3's documented coalesce to empty.
"""
from pathlib import Path
import re
import struct

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_official_context_v110 import encode,join_keys,positions

NOMENCLATURES=[None,'','NAF1993','NAFRev1','NAFRev2','NAP']
EXPECTED={'NAF1993':696,'NAFRev1':712,'NAFRev2':732}


def dbf_labels(payload,nomenclature):
    """Strict two-character-field dBASE III reader, no general XLS authoring.

    All used OEM text must decode IDENTICALLY with CP437 and CP850; an
    ambiguous accent/unknown schema stops rather than silently guesses.
    """
    if nomenclature not in EXPECTED or len(payload)<97 or payload[0]!=3:
        raise ValueError('Official dBASE format/version drift')
    count,header,length=struct.unpack_from('<IHH',payload,4)
    if header!=97 or payload[96]!=13 or count!=EXPECTED[nomenclature]:
        raise ValueError('Official dBASE cardinality/header drift')
    if len(payload)!=header+count*length+1 or payload[-1]!=26:
        raise ValueError('Official dBASE truncated/trailing payload')
    fields=[]
    for offset in (32,64):
        desc=payload[offset:offset+32]
        if desc[11]!=ord('C') or not desc[16] or desc[17]:
            raise ValueError('Official dBASE character field drift')
        name=desc[:11].split(b'\0',1)[0]
        if name.decode('cp437')!=name.decode('cp850'):
            raise ValueError('Ambiguous OEM field encoding')
        fields.append((name.decode('cp850'),desc[16]))
    if fields[0][0]!='CODE' or fields[1][0] not in ('LIBELLE','LIBELLÉ') or 1+sum(f[1] for f in fields)!=length:
        raise ValueError('Official dBASE code/label schema drift')
    pattern=r'[0-9]{2}\.[0-9]{2}[A-Z]' if nomenclature=='NAFRev2' else r'[0-9]{2}\.[0-9][A-Z]'
    rows=[];seen=set()
    for i in range(count):
        record=payload[header+i*length:header+(i+1)*length]
        if record[0]!=32:raise ValueError('Deleted/unknown official table record; no removal')
        offset=1;values=[]
        for _,width in fields:
            raw=record[offset:offset+width].rstrip(b' ');offset+=width
            if raw.decode('cp850')!=raw.decode('cp437'):
                raise ValueError('Ambiguous OEM label encoding')
            values.append(raw.decode('cp850'))
        code,label=values
        if not re.fullmatch(pattern,code) or not label or not label.isprintable() or code in seen:
            raise ValueError('Official label/code/uniqueness drift')
        seen.add(code);rows.append(dict(nomenclature=nomenclature,code=code,label=label,source_row=i))
    return rows


def build(stock,keys_path,parents_path,activity_path,activity_book,output,progress=lambda row:None,batch_size=131072):
    file=pq.ParquetFile(stock);n=file.metadata.num_rows;output=Path(output)
    columns=['siret','siren','activitePrincipaleEtablissement','nomenclatureActivitePrincipaleEtablissement']
    if any(file.schema_arrow.field(k).type!=pa.string() for k in columns):
        raise ValueError('Raw original stock schema drift')
    keys=np.load(keys_path,mmap_mode='r',allow_pickle=False)
    parents=np.load(parents_path,mmap_mode='r',allow_pickle=False)
    activity=np.load(activity_path,mmap_mode='r',allow_pickle=False)
    if (keys.dtype!=np.uint64 or parents.dtype!=np.uint32 or activity.dtype!=np.uint16 or
        keys.shape!=(n,) or parents.shape!=(n,) or activity.shape!=(n,) or activity_book[:2]!=[None,'']):
        raise ValueError('Complete registry/raw activity context drift')
    arrays={k:np.lib.format.open_memmap(output/(k+'.npy'),mode='w+',dtype=d,shape=(n,))
            for k,d in (('nomenclature_codes',np.uint8),('original_activity_codes',np.uint16))}
    seen=np.zeros(n,dtype=bool);count=0;missing=0
    joint=np.zeros((len(NOMENCLATURES),len(activity_book)),dtype=np.int64)
    for batch in file.iter_batches(batch_size=batch_size,columns=columns,use_threads=False):
        rawkeys=join_keys(batch.column(0),14,np.uint64);rawparents=join_keys(batch.column(1),9,np.uint32)
        pos=positions(keys,parents,rawkeys,rawparents,seen)
        raw=encode(batch.column(2),activity_book)
        # Compare the known V3 coalesce, but STORE original NULL code separately.
        expected=np.where(raw==0,1,raw)
        if not np.array_equal(activity[pos],expected):
            raise ValueError('Raw original activity differs from V3 source projection')
        nom=encode(batch.column(3),NOMENCLATURES).astype(np.uint8)
        arrays['nomenclature_codes'][pos]=nom;arrays['original_activity_codes'][pos]=raw
        if not pc.take(pa.array(NOMENCLATURES,type=pa.string()),pa.array(arrays['nomenclature_codes'][pos])).equals(batch.column(3)):
            raise ValueError('Persisted nomenclature roundtrip drift')
        if not pc.take(pa.array(activity_book,type=pa.string()),pa.array(arrays['original_activity_codes'][pos])).equals(batch.column(2)):
            raise ValueError('Persisted original activity roundtrip drift')
        joint+=np.bincount(nom.astype(np.int64)*len(activity_book)+raw,
                         minlength=joint.size).reshape(joint.shape)
        missing+=batch.column(2).null_count;count+=len(batch)
        if count%(batch_size*32)==0:progress(dict(phase='complete_explicit_taxonomy_alignment',sites=count,total=n))
    if count!=n or not np.all(seen):raise ValueError('Incomplete original taxonomy; no site removal')
    for a in arrays.values():a.flush()
    return dict(sites=n,nomenclatures=NOMENCLATURES,activity_book=activity_book,joint_counts=joint.tolist(),
        original_activity_nulls=missing,persisted_roundtrip_rows=n,all_sites_preserved=True,
        nomenclature_inferred=False,v3_null_to_empty_projection_verified=True,
        identifiers_as_features=False,recall_measured=False)


def coverage(stats,labels):
    authority={(r['nomenclature'],r['code']) for r in labels}
    if len(authority)!=len(labels):raise ValueError('Official nomenclature/code duplicate')
    matched=0;unresolved=[]
    for i,nom in enumerate(stats['nomenclatures']):
        for j,code in enumerate(stats['activity_book']):
            count=stats['joint_counts'][i][j]
            if not count:continue
            if (nom,code) in authority:matched+=count
            else:unresolved.append(dict(nomenclature=nom,code=code,sites=count))
    return dict(described_sites=matched,unresolved_sites=stats['sites']-matched,unresolved=unresolved,
                missing_descriptions_do_not_remove_sites=True,nap_labels_not_yet_available=True)
