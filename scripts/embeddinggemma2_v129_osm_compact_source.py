"""Same V129 source recipe with compact C++ node locations, no redownload.

The original frozen implementation and failed Python-dict run are preserved.
Only storage representation changes; all coordinates/radii/tags/queries stay
identical. Bound source text must match the historical launch implementation.
"""
from pathlib import Path
import sys
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OLD = io.HOME/'v129/osm_local_20261013_night'
NEW = io.HOME/'v129/osm_local_20261013_night_b'
sys.path.insert(0,str(OLD/'deps'))
import osmium


class CompactLocations:
    def __init__(self):
        self.table=osmium.index.create_map('sparse_mem_array');self.count=0
    def __len__(self):return self.count
    def __setitem__(self,key,value):
        self.table.set(key,osmium.osm.Location(value[0],value[1]));self.count+=1
    def __contains__(self,key):
        try:return self.table.get(key).valid()
        except KeyError:return False
    def __getitem__(self,key):
        value=self.table.get(key);return value.lon,value.lat


def main():
    old_launch=io.read(OLD/'launch.json')
    source=verify(old_launch['implementation'])
    old_exit=io.read(OLD/'owned_tool_exit.json')
    if old_exit['exit_code']!=1:raise ValueError('Observed old cache guard exit required')
    download=io.read(OLD/'download.json');verify(download['pbf'])
    if NEW.exists():raise ValueError('Fresh compact namespace required')
    NEW.mkdir(parents=True)
    io.write(NEW/'download.json',dict(download,reused_verified_download=record(OLD/'download.json')))
    io.write(NEW/'migration.json',dict(source_implementation=old_launch['implementation'],
        previous_launch=record(OLD/'launch.json'),previous_exit=record(OLD/'owned_tool_exit.json'),
        implementation=record(Path(__file__)),same_scientific_recipe=True,
        only_change='PYTHON_DICT_TO_CPP_SPARSE_MEM_ARRAY_LOCATION_STORAGE',
        no_redownload=True,no_label_read=True))
    text=source.read_text()
    changes={"ROOT = io.HOME/'v129/osm_local_20261013_night'":
             "ROOT = io.HOME/'v129/osm_local_20261013_night_b'",
             "sys.path.insert(0,str(ROOT/'deps'));import osmium":
             "sys.path.insert(0,str(io.HOME/'v129/osm_local_20261013_night/deps'));import osmium",
             "lookup=Neighbourhoods(queries);nodes={};ways={};pois={};roads={};addresses={};buildings={}":
             "lookup=Neighbourhoods(queries);nodes=CompactLocations();ways={};pois={};roads={};addresses={};buildings={}"}
    for before,after in changes.items():
        if text.count(before)!=1:raise ValueError('Bound execution transformation drift')
        text=text.replace(before,after)
    namespace={'__name__':'v129_bound_compact_worker','__file__':str(Path(__file__).resolve()),
               'CompactLocations':CompactLocations}
    exec(compile(text,str(source),'exec'),namespace)
    namespace['scan']()


if __name__=='__main__':main()
