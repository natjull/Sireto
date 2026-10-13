"""Bound V129 source using native reference filters and lean way storage.

ALL526, radii, POIs, addresses and street lines are unchanged. Anonymous
buildings are an explicitly omitted OPTIONAL field, never used by V132.
No alias/candidate/label enters this projection. All old files are preserved.
"""
from pathlib import Path
import sys
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OLD=io.HOME/'v129/osm_local_20261013_night'
PREVIOUS=io.HOME/'v129/osm_local_20261013_night_b'
NEW=io.HOME/'v129/osm_local_20261013_night_c'
sys.path.insert(0,str(OLD/'deps'))
import osmium


class NativeLocations:
    def __init__(self,tracker):
        self.table=osmium.index.create_map('sparse_mem_array');self.count=0;self.tracker=tracker
    def __len__(self):return self.count
    def __setitem__(self,key,value):
        self.table.set(key,osmium.osm.Location(*value));self.tracker.add_node(key);self.count+=1
    def __contains__(self,key):
        try:return self.table.get(key).valid()
        except KeyError:return False
    def __getitem__(self,key):
        value=self.table.get(key);return value.lon,value.lat


def main():
    original=io.read(OLD/'launch.json');source=verify(original['implementation'])
    if io.read(PREVIOUS/'owned_tool_exit.json')['exit_code']!=143:
        raise ValueError('Observed authorized previous stop required')
    download=io.read(OLD/'download.json');verify(download['pbf'])
    if NEW.exists():raise ValueError('Fresh native source namespace required')
    NEW.mkdir(parents=True)
    io.write(NEW/'download.json',dict(download,reused_verified_download=record(OLD/'download.json')))
    io.write(NEW/'migration.json',dict(original_implementation=original['implementation'],
        previous_migration=record(PREVIOUS/'migration.json'),previous_exit=record(PREVIOUS/'owned_tool_exit.json'),
        implementation=record(Path(__file__)),unchanged_scientific_fields=['POIS','ADDRESSES','MATCHED_STREET_LINES'],
        omitted_optional_fields='ANONYMOUS_BUILDING_POLYGONS_NOT_READ_BY_ANY_V132_VARIANT',
        location_storage='CPP_SPARSE_MEM_ARRAY',way_storage='REFERENCE_IDS_NOT_DUPLICATE_COORDINATES',
        filter='CPP_IDTRACKER_ANY_QUERY_BUFFER_NODE_EQUIVALENT_TO_OLD_ANY_NODE_MEMBERSHIP',
        counters_after_filter_are_callback_counts_not_all_national_ways=True,
        all_queries=526,same_radii=True,same_name_tokens=True,no_redownload=True,no_label_read=True))
    text=source.read_text()
    changes={
        "ROOT = io.HOME/'v129/osm_local_20261013_night'":
        "ROOT = io.HOME/'v129/osm_local_20261013_night_c'",
        "sys.path.insert(0,str(ROOT/'deps'));import osmium":
        "sys.path.insert(0,str(io.HOME/'v129/osm_local_20261013_night/deps'));import osmium",
        "lookup=Neighbourhoods(queries);nodes={};ways={};pois={};roads={};addresses={};buildings={}":
        "lookup=Neighbourhoods(queries);tracker=osmium.IdTracker();nodes=NativeLocations(tracker);ways={};pois={};roads={};addresses={};buildings={}",
        "if complete:ways[obj.id]=dict(coords=coords,lines=parts)":
        "if complete:ways[obj.id]=dict(refs=refs);tracker.add_way(obj.id)",
        "coords=[p for n in member_refs for p in ways[n]['coords']]":
        "coords=[nodes[r] for n in member_refs for r in ways[n]['refs']]",
        "lines=[p for n in member_refs for p in ways[n]['lines']]":
        "lines=[[list(lonlat_to_l93(*nodes[r])) for r in ways[n]['refs']] for n in member_refs]",
        "reader.close();Handler().apply_file(str(pbf),locations=False)":
        "reader.close();Handler().apply_file(str(pbf),locations=False,filters=[tracker.contains_filter()])",
        "if t.get('building') and lines:buildings[key]=value":
        "if t.get('building') and lines and value['names']:buildings[key]=value",
        "        if not coords:return\n        lon=sum":
        "        if not coords:return\n        has_name=any(t.get(k) for k in ('name','brand','alt_name','official_name','short_name','old_name'))\n        if not ((t.get('addr:street') and t.get('addr:housenumber')) or (any(k in t for k in POI_KEYS) and has_name) or (t.get('building') and has_name and lines)):return\n        lon=sum"
    }
    old_lines="""            parts=[];part=[]
            for n in refs:
                if n in nodes:part.append(list(lonlat_to_l93(*nodes[n])))
                else:
                    if len(part)>1:parts.append(part)
                    part=[]
            if len(part)>1:parts.append(part)"""
    new_lines="""            parts=[];part=[]
            has_name=any(t.get(k) for k in ('name','brand','alt_name','official_name','short_name','old_name'))
            need_lines=t.get('highway') or (t.get('addr:street') and t.get('addr:housenumber')) or (any(k in t for k in POI_KEYS) and has_name) or (t.get('building') and has_name)
            if need_lines:
                for n in refs:
                    if n in nodes:part.append(list(lonlat_to_l93(*nodes[n])))
                    else:
                        if len(part)>1:parts.append(part)
                        part=[]
                if len(part)>1:parts.append(part)"""
    changes[old_lines]=new_lines
    for before,after in changes.items():
        if text.count(before)!=1:raise ValueError('Frozen transformation drift: '+before[:70])
        text=text.replace(before,after)
    namespace={'__name__':'v129_bound_native_worker','__file__':str(Path(__file__).resolve()),
               'NativeLocations':NativeLocations}
    exec(compile(text,str(source),'exec'),namespace)
    namespace['scan']()


if __name__=='__main__':main()
