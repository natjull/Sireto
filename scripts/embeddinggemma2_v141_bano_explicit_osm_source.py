"""Immutable continuation accepting observed non-OSM producer codes.

Only exact source=OSM remains admissible. The current national CSV publishes
BAN explicitly; any other string is retained only in provenance counts, never
in the address index. No new data, radius or matching rule is introduced.
"""
from pathlib import Path
import hashlib

SOURCE=Path(__file__).with_name('embeddinggemma2_v141_bano_national_source.py')
EXPECTED='bbc819028db034bbbec59a4510be2b0853beb1cf16f960867e87abacbf5a5d14'


def main():
    text=SOURCE.read_text()
    if hashlib.sha256(text.encode()).hexdigest()!=EXPECTED:
        raise ValueError('Original frozen source implementation drift')
    replacements={
        "v141/bano_osm_20261013_night_b":"v141/bano_osm_20261013_night_c",
        "excluded_source_codes=['CAD','C+O','OD','O+O']":"excluded_source_codes='ALL_CODES_EXCEPT_EXACT_OSM_INCLUDING_CURRENT_BAN'",
        "if source not in ('OSM','OD','O+O','C+O','CAD'):raise ValueError(f'Unknown source {source!r}')":"# Non-OSM rows are all excluded below, including observed current source BAN.",
        "base.io.write(ROOT/'launch.json',launch)":"launch['continuation_binding']=dict(original_implementation=base.record(SOURCE_BINDING),original_exit=base.record(base.io.HOME/'v141/bano_osm_20261013_night_b/owned_tool_exit.json'),observed_first_department_sources=dict(OSM=166214,BAN=105632),source_OSM_rule_unchanged=True,unknown_non_OSM_sources_excluded_not_admitted=True); base.io.write(ROOT/'launch.json',launch)",
    }
    for old,new in replacements.items():
        if text.count(old)!=1:raise ValueError('Exact execution-only continuation patch required')
        text=text.replace(old,new)
    namespace=dict(__name__='__main__',__file__=str(Path(__file__)),SOURCE_BINDING=SOURCE)
    exec(compile(text,str(Path(__file__)),'exec'),namespace)


if __name__=='__main__':main()
