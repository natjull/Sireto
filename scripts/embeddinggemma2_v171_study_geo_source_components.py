"""July study geography as a separate source component. Stage A, no GT."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts import embeddinggemma2_night as io

HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
AUTHORITY = HOME/'v142/bano_native_20261013_night/launch.json'
RECEIPT = Path('/Volumes/CATNAT_DATA/oubusiness/raw/receipts/sirene_geolocalisation.json')
STUDY = Path('/Volumes/CATNAT_DATA/oubusiness/raw/sirene/geolocalisation_etablissements_2026-07-01.parquet')
STOP_A = 1791953700  # 2026-10-14 04:55:00 UTC, earlier than the absolute deadline.
DTYPE = np.dtype([('study_raw', '<f8'), ('flags', 'u1'), ('raw_qc', 'u1'), ('distance_precision', '<f4')])
RECIPE = dict(stage='A_WHOLE_SOURCE_THEN_ALL526_NO_LABELS', source='JULY_STUDY_GEOCODING_SEPARATE_OBSERVATION_NOT_AUGUST_COORDINATE_REPAIR',
    study_rows=37575264, canonical_rows=43896818, queries=526, expected_numeric_rows=128313036,
    source_columns=['siret','x','y','qualite_xy','epsg','distance_precision'],
    quality_codes={'missing':0,'11':11,'12':12,'21':21,'22':22,'33':33,'other':63},
    lookup_qflags='LOW6_RAW_QC_128_CURRENT_PUBLIC_SOURCE_PRESENT_64_STRICT_EPSG2154',
    component='0.5/(1+EUCLIDEAN_DISTANCE_M/300)_RAW_FLOAT64',
    component_eligible='CANONICAL_KEY_CURRENT_eO_uO_SOURCE_PRESENT_EPSG2154_FINITE_L93_AND_RAW_QC11_OR21_AND_EXISTING_QUERY_GEOMETRY',
    query='UNCHANGED_V115_V17_QUERY_GEOMETRY_AND_BAN_WEAK_UNCERTAINTY_METADATA_NO_TARGET_POINT',
    query_uncertainty_not_probability=True, source_qc_not_certified_house_precision=True,
    source33='RANDOM_COMMUNE_IMPUTATION_NOT_CENTROID', source12_22='RANDOM_STREET_IMPUTATION',
    no_quality_transfer=True,no_coordinate_repair=True,no_current_xy_overwrite=True,
    flags={'canonical':1,'current_public':2,'study_present':4,'study_epsg2154':8,'study_supported_xy':16,'raw_qc11_or21':32,'query_geometry':64,'current_xy_supported':128},
    all_missing_retained_nan=True,no_candidate_pool=True,no_retrieval=True,no_model_score=True,no_fit=True,
    output_record_bytes=14,max_output_bytes=4*1024**3,threads=1,batch_rows=65536,
    rss_bytes=int(2.5*1024**3),max_prepare_seconds=900,max_total_seconds=1200,
    free_percent=25,pressure_levels=[0,1,2],disk_reserve_bytes=40*1024**3,no_spill=True,
    stop_before_utc='2026-10-14 04:55:00',labels_read=False,closed_sets_opened=False,
    metric='SOURCE_COMPONENT_AVAILABILITY_NOT_RETRIEVAL_RECALL')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    temporary = p.with_suffix('.pending.json')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, p)


def close(*arrays):
    for a in arrays:
        if isinstance(a, np.memmap):
            a._mmap.close()


def supported(xy):
    return (np.isfinite(xy).all(axis=1) & (xy[:,0]>=0) & (xy[:,0]<=1300000)
            & (xy[:,1]>=6000000) & (xy[:,1]<=7300000))


def source(out):
    started = time.monotonic()
    peak = 0
    process = psutil.Process()
    def guard(preparing=False):
        nonlocal peak
        rss = process.memory_info().rss
        peak = max(peak,rss)
        free, pressure = io.pressure()
        if (time.time() >= STOP_A or time.monotonic()-started > (900 if preparing else 1200)
                or rss > RECIPE['rss_bytes'] or free < 25 or pressure not in (0,1,2)
                or shutil.disk_usage(HOME).free < RECIPE['disk_reserve_bytes']):
            raise RuntimeError('STOP source component resource/time guard; preserve incomplete artifacts')
        return dict(rss_bytes=rss,peak_guarded_rss_bytes=peak,free_percent=free,pressure_level=pressure,elapsed_seconds=time.monotonic()-started)
    def record(p, expected=None):
        p = Path(p)
        before = p.stat()
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):
                h.update(block)
                guard()
        after = p.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('Input changed while hashing')
        if expected and h.hexdigest()!=expected:
            raise ValueError('Frozen SHA256 mismatch')
        return dict(path=str(p),sha256=h.hexdigest(),size_bytes=after.st_size,mtime_ns=after.st_mtime_ns,inode=after.st_ino)
    if out.exists():
        raise ValueError('Fresh immutable output required')
    out.mkdir(parents=True)
    try:
        guard(True)
        authority=read(AUTHORITY);receipt=read(RECEIPT)
        refs=dict(implementation=record(Path(__file__)),authority=record(AUTHORITY),receipt=record(RECEIPT),study=record(STUDY,receipt['sha256']))
        for name in ('keys','xy','public_mask','panel','baseline_features','geometry'):
            refs[name]=record(authority[name]['path'],authority[name]['sha256'])
        f=pq.ParquetFile(STUDY)
        if f.metadata.num_rows!=RECIPE['study_rows'] or f.schema_arrow.field('x').type!=pa.float32() or f.schema_arrow.field('y').type!=pa.float32():
            raise ValueError('Frozen study row count/float32 coordinate representation')
        panel=read(refs['panel']['path']);fg=read(refs['baseline_features']['path'])
        geometry={r['query_id']:r['geometry'] for r in read(refs['geometry']['path'])}
        scope={q['query_id'] for q in panel}
        if len(panel)!=526 or len(scope)!=526 or scope!=set(fg['features']) or scope!=set(geometry):
            raise ValueError('Exact ALL526 source scope')
        if any(q['fold'] not in (2,3,4) or any(k in q for k in ('label_siret','human_siret','ground_truth')) for q in panel):
            raise ValueError('Authorized query-only folds2/3/4 required')
        write(out/'recipe.json',RECIPE)
        write(out/'launch.json',dict(recipe=record(out/'recipe.json'),inputs=refs,
            study_semantic_date=receipt['semantic_date'],study_publication_date=receipt['publication_date'],
            source_separate_from_current_xy=True,no_quality_transfer=True,labels_read=False,created_unix=time.time()))
        keys=np.load(refs['keys']['path'],mmap_mode='r');public=np.load(refs['public_mask']['path'],mmap_mode='r')
        if len(keys)!=43896818 or keys.dtype!=np.uint64 or public.shape!=keys.shape or public.dtype!=bool:
            raise ValueError('Frozen canonical key/public headers')
        lookup=out/'study_lookup';lookup.mkdir()
        study_xy=np.lib.format.open_memmap(lookup/'xy.npy',mode='w+',dtype=np.float32,shape=(len(keys),2));study_xy[:]=np.nan
        qflags=np.lib.format.open_memmap(lookup/'qflags.npy',mode='w+',dtype=np.uint8,shape=keys.shape);qflags[:]=0
        precision=np.lib.format.open_memmap(lookup/'distance_precision.npy',mode='w+',dtype=np.float32,shape=keys.shape);precision[:]=np.nan
        seen=np.zeros(len(keys),dtype=bool);other_seen=set();counts=Counter();raw_codes=Counter();rows=0
        for batch_no,batch in enumerate(f.iter_batches(batch_size=65536,columns=RECIPE['source_columns'],use_threads=False)):
            guard(True)
            valid_id=pc.match_substring_regex(batch.column(0),'^[0-9]{14}$').fill_null(False).to_numpy(zero_copy_only=False)
            counts['INVALID_SOURCE_KEY']+=int(np.count_nonzero(~valid_id))
            selected=np.flatnonzero(valid_id)
            source_batch=batch.take(pa.array(selected))
            ids=pc.cast(source_batch.column(0),pa.uint64()).to_numpy(zero_copy_only=False)
            pos=np.searchsorted(keys,ids);inside=pos<len(keys);inside[inside]&=keys[pos[inside]]==ids[inside]
            locations=pos[inside]
            if len(np.unique(locations))!=len(locations) or np.any(seen[locations]):
                raise ValueError('Duplicate study SIRET; no last-wins source row selection')
            seen[locations]=True
            for identity in ids[~inside]:
                identity=int(identity)
                if identity in other_seen:raise ValueError('Duplicate noncanonical study SIRET')
                other_seen.add(identity)
            counts['CANONICAL_KEY_ABSENT']+=int(np.count_nonzero(~inside))
            eligible=inside.copy();eligible[inside]&=public[pos[inside]]
            counts['CURRENT_NON_PUBLIC_EXCLUDED']+=int(np.count_nonzero(inside&~eligible))
            at=np.flatnonzero(eligible)
            if len(at):
                sub=source_batch.take(pa.array(at));loc=pos[at]
                epsg=pc.equal(sub.column(4),'2154').fill_null(False).to_numpy(zero_copy_only=False)
                qc=np.fromiter((int(v) if v in ('11','12','21','22','33') else (0 if v in (None,'') else 63) for v in sub.column(3).to_pylist()),dtype=np.uint8,count=len(at))
                qflags[loc]=qc+np.uint8(128)+epsg.astype(np.uint8)*np.uint8(64)
                raw_codes.update(Counter(map(int,qc)))
                x=sub.column(1).to_numpy(zero_copy_only=False);y=sub.column(2).to_numpy(zero_copy_only=False)
                p=sub.column(5).to_numpy(zero_copy_only=False);precision[loc]=p
                good=epsg&np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=1300000)&(y>=6000000)&(y<=7300000)
                study_xy[loc[good],0]=x[good];study_xy[loc[good],1]=y[good]
                counts['CURRENT_PUBLIC_SOURCE_PRESENT']+=len(at)
                counts['OTHER_OR_MISSING_EPSG']+=int(np.count_nonzero(~epsg))
                counts['EPSG2154_SUPPORTED_STUDY_XY']+=int(np.count_nonzero(good))
                counts['EPSG2154_UNSUPPORTED_STUDY_XY']+=int(np.count_nonzero(epsg&~good))
                counts['EPSG2154_SUPPORTED_QC11_OR21']+=int(np.count_nonzero(good&np.isin(qc,[11,21])))
            rows+=batch.num_rows
            if batch_no%32==0:write(out/'prepare_state.json',dict(status='WHOLE_SOURCE_LOOKUP_RUNNING',rows=rows,**guard(True)))
        if rows!=37575264 or counts['INVALID_SOURCE_KEY']+counts['CANONICAL_KEY_ABSENT']+counts['CURRENT_NON_PUBLIC_EXCLUDED']+counts['CURRENT_PUBLIC_SOURCE_PRESENT']!=rows:
            raise ValueError('Complete source denominator classification')
        del seen,other_seen;gc.collect()
        for a in (study_xy,qflags,precision):a.flush()
        close(study_xy,qflags,precision,keys,public)
        lookup_refs={name:record(lookup/name) for name in ('xy.npy','qflags.npy','distance_precision.npy')}
        write(out/'lookup_manifest.json',dict(status='SEALED_WHOLE_STUDY_SEPARATE_POINT_LOOKUP_NO_QUERY_NO_GT',inputs=refs,
            artifacts=lookup_refs,source_rows=rows,counts=dict(counts),public_raw_quality_counts=dict(raw_codes),
            coordinate_dtype='ORIGINAL_STUDY_FLOAT32',current_coordinates_unchanged=True,quality_not_transferred=True,**guard(True)))
        allkeys=np.load(refs['keys']['path'],mmap_mode='r');public=np.load(refs['public_mask']['path'],mmap_mode='r')
        current_xy=np.load(refs['xy']['path'],mmap_mode='r');study_xy=np.load(lookup_refs['xy.npy']['path'],mmap_mode='r')
        qflags=np.load(lookup_refs['qflags.npy']['path'],mmap_mode='r');precision=np.load(lookup_refs['distance_precision.npy']['path'],mmap_mode='r')
        components=out/'components';components.mkdir();queries=[];total=0;disk=sum(r['size_bytes'] for r in lookup_refs.values())
        for q in panel:
            guard();qid=q['query_id'];original=fg['features'][qid];path=record(original['path'],original['sha256'])
            with np.load(path['path'],allow_pickle=False) as z:kk=z['keys']
            with zipfile.ZipFile(path['path']) as z:
                with z.open('features.npy') as stream:
                    version=np.lib.format.read_magic(stream)
                    shape,fortran,dtype=(np.lib.format.read_array_header_1_0(stream) if version==(1,0) else np.lib.format.read_array_header_2_0(stream))
            if kk.dtype!=np.uint64 or np.any(kk[1:]<=kk[:-1]) or shape!=(len(kk),51) or dtype!=np.float32:
                raise ValueError('Frozen unpruned numeric key/schema')
            at=np.searchsorted(allkeys,kk);found=at<len(allkeys);found[found]&=allkeys[at[found]]==kk[found]
            both=found.copy();both[found]&=public[at[found]]
            pos=np.flatnonzero(both);loc=at[pos];stored=qflags[loc];qc=stored&63
            present=(stored&128)!=0;epsg=(stored&64)!=0
            coord=np.asarray(study_xy[loc],dtype=np.float64);valid=supported(coord)
            current=supported(np.asarray(current_xy[loc],dtype=np.float64));precise=np.isin(qc,[11,21])
            g=geometry[qid]
            query_point=np.asarray([g.get('ban_x_l93',np.nan),g.get('ban_y_l93',np.nan)],dtype=np.float64)
            available=bool(g.get('spatial_crs_compatible') and supported(query_point.reshape(1,2))[0])
            packed=np.lib.format.open_memmap(components/(hashlib.sha256(qid.encode()).hexdigest()+'.npy'),mode='w+',dtype=DTYPE,shape=kk.shape)
            packed['study_raw']=np.nan;packed['distance_precision']=np.nan;packed['raw_qc']=0
            packed['flags']=found.astype(np.uint8)+2*both.astype(np.uint8)+(64 if available else 0)
            packed['raw_qc'][pos]=qc;packed['distance_precision'][pos]=precision[loc]
            packed['flags'][pos]+=4*present.astype(np.uint8)+8*epsg.astype(np.uint8)+16*valid.astype(np.uint8)+32*precise.astype(np.uint8)+128*current.astype(np.uint8)
            selected=present&epsg&valid&precise
            if available and np.any(selected):
                distance=np.hypot(coord[selected,0]-query_point[0],coord[selected,1]-query_point[1])
                packed['study_raw'][pos[selected]]=.5/(1+distance/300.)
            packed.flush();guard();destination=Path(packed.filename);close(packed)
            component=record(destination);total+=len(kk);disk+=component['size_bytes']
            if disk>RECIPE['max_output_bytes']:raise RuntimeError('STOP source-component disk cap')
            queries.append(dict(query_id=qid,fold=q['fold'],rows=len(kk),original_numeric_ref=original,
                ordered_keys_sha256=hashlib.sha256(kk.tobytes()).hexdigest(),component=component,
                canonical_key_present=int(found.sum()),current_public=int(both.sum()),study_present=int(present.sum()),
                study_supported_xy=int(np.count_nonzero(present&epsg&valid)),study_number_identified=int(selected.sum()),
                not_precise_raw_qc_counts={str(code):int(np.count_nonzero(present&(qc==code))) for code in (12,22,33,0,63)},
                query_geometry_available=available,query_ban_declared_weak=bool(g.get('ban_declared_weak')),
                query_ban_uncertainty_m=g.get('ban_uncertainty_m'),metric='SOURCE_COMPONENT_NOT_RETRIEVAL'))
            del kk,at,found,both,pos,loc,stored,qc,present,epsg,coord,current,precise,packed;gc.collect()
            if len(queries)%20==0:print('SOURCE_COMPONENTS',len(queries),total,flush=True)
            write(out/'component_state.json',dict(status='ALL526_SOURCE_COMPONENTS_RUNNING',queries=len(queries),rows=total,**guard()))
        close(allkeys,public,current_xy,study_xy,qflags,precision)
        if len(queries)!=526 or total!=128313036 or sum(q['rows']==0 for q in queries)!=4:
            raise ValueError('Whole original526/128313036/four-empty denominator required')
        for r in refs.values():
            stat=Path(r['path']).stat()
            if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=(r['size_bytes'],r['mtime_ns'],r['inode']):
                raise ValueError('Frozen source changed before whole-source seal')
        report=dict(status='COMPLETE_ALL526_SEPARATE_STUDY_SOURCE_COMPONENTS_NO_LABELS',queries=queries,rows=total,empty_queries=4,
            lookup=record(out/'lookup_manifest.json'),recipe=record(out/'recipe.json'),launch=record(out/'launch.json'),
            disk_bytes=disk,source_rows=rows,no_retrieval=True,no_qc_transfer=True,no_coordinate_repair=True,
            original_numeric_keys_unpruned=True,labels_read=False,outcomes_read=False,**guard())
        write(out/'source_report.json',report)
        write(out/'source_manifest.json',dict(status=report['status'],source_report=record(out/'source_report.json'),
            lookup_manifest=record(out/'lookup_manifest.json'),all526_sealed_before_labels=True))
        print(json.dumps({k:report[k] for k in ('status','rows','empty_queries','disk_bytes','peak_guarded_rss_bytes','elapsed_seconds')}),flush=True)
    except BaseException as exc:
        write(out/'stop.json',dict(status='STOP_INCOMPLETE_SOURCE_NOT_LABEL_ELIGIBLE',reason=str(exc),elapsed_seconds=time.monotonic()-started,peak_guarded_rss_bytes=peak))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    source(p.parse_args().output)
