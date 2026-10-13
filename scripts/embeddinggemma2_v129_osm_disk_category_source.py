"""Same bounded disk source, preserve category-specific object geometry.

One OSM way can be both road and place: its road midpoint and place centre
must remain distinct, exactly as in the original separate category dicts.
No scientific change, new query filtering, name expansion or label access.
"""
import argparse
from pathlib import Path
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OLD=io.HOME/'v129/osm_local_20261013_night_d'
NEW=io.HOME/'v129/osm_local_20261013_night_e'


def main(resume=False):
    binding=io.read(OLD/'launch.json')['implementation'];source=verify(binding)
    if io.read(OLD/'owned_tool_exit.json')['exit_code']!=143:
        raise ValueError('Actually observed previous stop required')
    text=source.read_text()
    before='''        key=f"{v['osm_type']}/{v['osm_id']}"
        db.execute('INSERT OR REPLACE INTO objects VALUES(?,?,?)',
            (key,json.dumps(v,ensure_ascii=False,separators=(',',':')),v.get('name_key','')))
        for cat in categories:
            db.execute('INSERT OR IGNORE INTO kinds VALUES(?,?)',(cat,key))'''
    after='''        body=json.dumps(v,ensure_ascii=False,separators=(',',':'))
        for cat in categories:
            key=f"{cat}|{v['osm_type']}/{v['osm_id']}"
            db.execute('INSERT OR REPLACE INTO objects VALUES(?,?,?)',(key,body,v.get('name_key','')))
            db.execute('INSERT OR IGNORE INTO kinds VALUES(?,?)',(cat,key))'''
    changes={"ROOT=io.HOME/'v129/osm_local_20261013_night_d'":
             "ROOT=io.HOME/'v129/osm_local_20261013_night_e'",before:after}
    for a,b in changes.items():
        if text.count(a)!=1:raise ValueError('Frozen execution transformation drift')
        text=text.replace(a,b)
    if text.count("('node/%',)")!=3:raise ValueError('Resume node projection binding')
    text=text.replace("('node/%',)","('%|node/%',)")
    namespace={'__name__':'v129_bound_disk_category_worker','__file__':str(Path(__file__).resolve())}
    exec(compile(text,str(source),'exec'),namespace)
    namespace['main'](resume)
    io.write(NEW/'category_storage_binding.json',dict(source_implementation=binding,
        previous_exit=record(OLD/'owned_tool_exit.json'),implementation=record(Path(__file__)),
        storage_key='CATEGORY_PIPE_OSM_TYPE_SLASH_OBJECT_ID',
        source_fields_identical_to_original_separate_category_dictionaries=True,
        no_new_scientific_filter=True,no_label_read=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-ways',action='store_true');a=p.parse_args();main(a.resume_ways)
