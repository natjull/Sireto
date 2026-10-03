"""All explicit site identities in original canonical names AND addresses.

Identity coverage never depends on lexical eligibility, masks, source family,
date, state or CRM/GT. Parent/context observations are counted, not fabricated
into site identities. The original rows remain in immutable source files.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_variant_batches_v128 import WHITESPACE
from src.xgb_matcher.retrieval_proof_bridge_v137 import MISSING


def classify(batch):
    if batch.schema.names!=['subject_kind','siren','siret'] or any(t!=pa.string() for t in batch.schema.types):
        raise ValueError('Original canonical identity grain schema required')
    kind,parent,site=[pc.utf8_trim(pc.fill_null(a,''),characters=WHITESPACE) for a in batch.columns]
    valid_parent=pc.match_substring_regex(parent,r'^[0-9]{9}$')
    endpoint=pc.and_(pc.equal(kind,'SIRET'),pc.and_(valid_parent,pc.and_(
        pc.match_substring_regex(site,r'^[0-9]{14}$'),pc.equal(pc.utf8_slice_codeunits(site,0,9),parent))))
    legal=pc.and_(pc.equal(kind,'SIREN'),valid_parent)
    keys=pc.cast(pc.filter(site,endpoint),pa.uint64()).to_numpy(zero_copy_only=False)
    parents=int(pc.sum(pc.cast(legal,pa.uint64())).as_py() or 0)
    return keys,parents,len(batch)-len(keys)-parents


def inventory(source,catalog,output,expected,batch_rows=131072):
    f=pq.ParquetFile(source);fields=['subject_kind','siren','siret']
    if (not isinstance(expected,int) or isinstance(expected,bool) or f.metadata.num_rows!=expected
        or any(f.schema_arrow.field(k).type!=pa.string() for k in fields)
        or not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0):
        raise ValueError('Whole original canonical source/count required')
    rows=sites=parents=contexts=present=absent=0;missing=[]
    for batch in f.iter_batches(batch_size=batch_rows,columns=fields,use_threads=False):
        keys,np_,nc=classify(batch);pos=catalog.site_positions(keys);known=pos!=MISSING
        if not np.array_equal(catalog.reconstruct(pos[known],'site'),keys[known]):
            raise ValueError('Every present canonical site identity must reconstruct exactly')
        unknown=keys[~known]
        if len(unknown):missing.append(np.unique(unknown))
        rows+=len(batch);sites+=len(keys);parents+=np_;contexts+=nc;present+=int(np.count_nonzero(known));absent+=len(unknown)
    if rows!=expected or sites+parents+contexts!=rows or present+absent!=sites:
        raise ValueError('Every original canonical row/identity scope required')
    unique=np.unique(np.concatenate(missing)) if missing else np.empty(0,dtype=np.uint64)
    # Full absent join oracle against both current and extra dictionaries.
    for begin in range(0,len(unique),batch_rows):
        if np.any(catalog.site_positions(unique[begin:begin+batch_rows])!=MISSING):
            raise ValueError('Every absent official key must be genuinely absent')
    path=Path(output)/'absent_site_keys.npy';np.save(path,unique,allow_pickle=False)
    if not np.array_equal(np.load(path,mmap_mode='r',allow_pickle=False),unique):
        raise ValueError('Every exported absent key must be exact')
    return dict(rows=rows,explicit_site_observations=sites,explicit_parent_context=parents,unverified_context=contexts,
        present_site_observations=present,absent_site_observations=absent,unique_absent_sites=len(unique),
        every_original_source_row_accounted=True,every_present_site_reconstructed_exactly=True,
        every_absent_site_genuinely_absent=True,lexical_mask_or_family_filter=False,
        parent_address_spread_to_sites=False,candidate_catalog_changed=False,human_denominator_filtered=False,
        identity_or_observation_cap=None,identifiers_as_features=False)
