"""Same complete observation metadata encoding, vectorized rather than Python row loops."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_name_postings_v86 import SOURCE,NAME_KIND

SPECS={**{k:np.int32 for k in ('valid_from_days','valid_to_days','observed_days')},
       **{k:np.uint8 for k in ('source_codes','kind_codes','current_codes','priority_known')},'priorities':np.int16}


def encode(values,codes):
    ordered=[key for key,value in sorted(codes.items(),key=lambda item:item[1])]
    if sorted(codes.values())!=list(range(1,len(codes)+1)): raise ValueError('Code enumeration drift')
    index=pc.index_in(values,value_set=pa.array(ordered))
    if index.null_count: raise ValueError('Unknown metadata code; no silent reinterpretation')
    return pc.add(pc.cast(index,pa.uint8()),pa.scalar(1,pa.uint8())).to_numpy(zero_copy_only=False)


def build(source,output):
    file=pq.ParquetFile(source); n=file.metadata.num_rows; output=Path(output)
    arrays={k:np.lib.format.open_memmap(output/(k+'.npy'),mode='w+',dtype=d,shape=(n,)) for k,d in SPECS.items()}
    columns=['valid_from','valid_to','observed_at','source','name_kind','is_current','source_priority']; offset=0
    for batch in file.iter_batches(batch_size=131072,columns=columns,use_threads=False):
        stop=offset+len(batch); values={k:batch.column(i) for i,k in enumerate(columns)}
        for dest,key in zip(('valid_from_days','valid_to_days','observed_days'),columns[:3]):
            if values[key].type!=pa.date32(): raise ValueError('Observation date unit drift')
            arrays[dest][offset:stop]=pc.fill_null(pc.cast(values[key],pa.int32()),np.iinfo(np.int32).min).to_numpy(zero_copy_only=False)
        arrays['source_codes'][offset:stop]=encode(values['source'],SOURCE)
        arrays['kind_codes'][offset:stop]=encode(values['name_kind'],NAME_KIND)
        current=pc.if_else(pc.is_null(values['is_current']),0,pc.if_else(pc.fill_null(values['is_current'],False),2,1))
        arrays['current_codes'][offset:stop]=current.to_numpy(zero_copy_only=False)
        arrays['priority_known'][offset:stop]=pc.invert(pc.is_null(values['source_priority'])).to_numpy(zero_copy_only=False)
        arrays['priorities'][offset:stop]=pc.fill_null(values['source_priority'],0).to_numpy(zero_copy_only=False)
        offset=stop
    if offset!=n: raise ValueError('All observation metadata denominator drift')
    for a in arrays.values(): a.flush()
    return dict(observations=n,source_codes=SOURCE,kind_codes=NAME_KIND,current_codes={'UNKNOWN':0,'FALSE':1,'TRUE':2},
        null_date_sentinel=int(np.iinfo(np.int32).min),original_provenance_strings_in_full_v94_source=True,
        vectorized_codes_same_encoding=True)
