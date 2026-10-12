"""Read-only official coordinate coverage attribution after sealed V32 outputs."""
from pathlib import Path
import numpy as np
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUT=io.HOME/'v48/postseal_geometry_coverage_20261012_a'

def main():
    if OUT.exists():raise ValueError('Fresh diagnostic namespace')
    stock=io.HOME/'v47/postseal_stock_status_20261012_a'
    summary=io.read(stock/'summary.json');cases=io.read(verify(summary['cases']))
    launch=io.read(io.HOME/'v23/hierarchical_native_20261012_a/launch.json')
    geometry={r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    keys=np.load(verify(launch['keys']),mmap_mode='r',allow_pickle=False)
    xy=np.load(verify(launch['xy']),mmap_mode='r',allow_pickle=False)
    rows=[]
    for c in cases:
        key=int(c['siret']);i=np.searchsorted(keys,key)
        if i>=len(keys) or keys[i]!=key:raise ValueError('Target in national coordinate namespace')
        point=xy[i];valid=bool(np.isfinite(point).all() and 0<=point[0]<=1300000 and 6000000<=point[1]<=7300000)
        g=geometry[c['query_id']];distance=None
        if valid and g['spatial_crs_compatible']:
            distance=float(np.hypot(point[0]-g['ban_x_l93'],point[1]-g['ban_y_l93']))
        rows.append(dict(query_id=c['query_id'],site_state=c['site_state'],site_diffusion=c['site_diffusion'],
            target_coordinate_available=valid,query_coordinate_compatible=g['spatial_crs_compatible'],
            query_ban_resolved=g['ban_resolved'],query_ban_weak=g['ban_declared_weak'],
            target_current_point_distance_m=distance,public_street_present=c['public_street_present'],
            public_number_present=c['public_number_present']))
    keys._mmap.close();xy._mmap.close();OUT.mkdir(parents=True)
    io.write(OUT/'cases.json',rows)
    result=dict(status='COMPLETE_LABEL_SIDE_GEOMETRY_COVERAGE',misses=len(rows),
        missing_target_coordinate=sum(not r['target_coordinate_available'] for r in rows),
        incompatible_query_coordinate=sum(not r['query_coordinate_compatible'] for r in rows),
        missing_target_but_public_street=sum(not r['target_coordinate_available'] and r['public_street_present'] for r in rows),
        missing_target_and_incompatible_query=sum(not r['target_coordinate_available'] and not r['query_coordinate_compatible'] for r in rows),
        missing_target_by_state={s:sum(not r['target_coordinate_available'] and r['site_state']==s for r in rows) for s in ('A','F')},
        no_feature_or_policy_changed=True,no_label_correction=True,not_a_recall_gain=True,
        original_stock_audit=record(stock/'summary.json'),coordinate_keys=launch['keys'],xy=launch['xy'],
        query_geometry=launch['geometry'],cases=record(OUT/'cases.json'),implementation=record(Path(__file__)))
    io.write(OUT/'summary.json',result);print(result,flush=True)

if __name__=='__main__':main()
