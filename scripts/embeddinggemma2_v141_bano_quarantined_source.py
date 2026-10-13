"""Uniform source-record quarantine, never label-side filtering or repair.

All nonempty CSV records not having exactly eight fields are archived/rejected.
At most 100 and at most 1e-5 of source records may be quarantined; otherwise
the source stops. No field reconstruction, target-specific branch or address
correction is permitted. Every one of the 526 query contexts is retained.
"""
from pathlib import Path
import hashlib

SOURCE=Path(__file__).with_name('embeddinggemma2_v141_bano_national_source.py')
EXPECTED='bbc819028db034bbbec59a4510be2b0853beb1cf16f960867e87abacbf5a5d14'


def captured_lines(stream,buffer):
    for line in stream:
        buffer.append(line)
        yield line


def main():
    text=SOURCE.read_text()
    if hashlib.sha256(text.encode()).hexdigest()!=EXPECTED:
        raise ValueError('Frozen original source drift')
    replacements={
        "v141/bano_osm_20261013_night_b":"v141/bano_osm_20261013_night_d",
        "excluded_source_codes=['CAD','C+O','OD','O+O']":"excluded_source_codes='ALL_CODES_EXCEPT_EXACT_OSM_INCLUDING_CURRENT_BAN'",
        "if source not in ('OSM','OD','O+O','C+O','CAD'):raise ValueError(f'Unknown source {source!r}')":"# All non-OSM codes remain excluded from the index, not guessed or reclassified.",
        "counters=Counter();source_counts=Counter();prefix_counts=defaultdict(Counter);batch=[]":"counters=Counter();source_counts=Counter();prefix_counts=defaultdict(Counter);batch=[];quarantined=[];physical_lines=[];last_physical_line=0",
        "for row_index,row in enumerate(csv.reader(stream),start=1):":"reader=csv.reader(CAPTURE_LINES(stream,physical_lines))\n        for row_index,row in enumerate(reader,start=1):",
        "if len(row)!=8:raise ValueError(f'National CSV8 drift at row{row_index}')":"""first_physical_line=last_physical_line+1;last_physical_line=reader.line_num
            raw_record=''.join(physical_lines) if len(row)!=8 else None;physical_lines.clear()
            if not row or all(not str(field).strip() for field in row):
                counters['blank_source_records']+=1
                continue
            if len(row)!=8:
                quarantined.append(dict(csv_record_index=row_index,first_physical_line=first_physical_line,last_physical_line=reader.line_num,raw_record=raw_record,fields=row,reason='NONEMPTY_CSV_NOT_EXACTLY_8_FIELDS'))
                counters['quarantined_nonempty_records']+=1
                if len(quarantined)>100:raise ValueError('Quarantine count exceeds frozen maximum100')
                continue""",
        "con.commit();base.guard()":"""con.commit();base.guard()
    if len(quarantined)/row_index>1e-5:raise ValueError('Quarantine rate exceeds frozen maximum1e-5')
    base.io.write(ROOT/'quarantined_source_records.json',dict(policy='QUARANTINED_SOURCE_RECORDS_NOT_LABEL_FILTERING',source_archive=launch['archive'],records=quarantined,all_raw_csv_records=row_index,physical_lines=reader.line_num,nonempty_rejected_count=len(quarantined),rejected_rate=len(quarantined)/row_index,maximum_count=100,maximum_rate=1e-5,no_reconstruction_or_repair=True))""",
        "PASS_ALL_NATIONAL_CSV8_WITH_SOURCE_AND_GZIP_CRC":"COMPLETE_SCAN_WITH_QUARANTINED_SOURCE_RECORDS_AND_GZIP_CRC",
        "COMPLETE_NATIONAL_BANO_OSM_ALL526_QUERY_ONLY_SOURCE":"COMPLETE_SCAN_WITH_QUARANTINE_NATIONAL_BANO_OSM_ALL526_CONTEXTS",
        "all_sources_archived_not_only_OSM=True,":"all_sources_archived_not_only_OSM=True,quarantine=base.record(ROOT/'quarantined_source_records.json'),csv_source_records=row_index,source_is_not_exhaustive_all_addresses=True,delegated_communes_of_current_parent_not_expanded=True,",
        "base.io.write(ROOT/'launch.json',launch)":"launch['continuation_binding']=dict(original_implementation=base.record(SOURCE_BINDING),original_exit=base.record(base.io.HOME/'v141/bano_osm_20261013_night_c/owned_tool_exit.json'),source_OSM_rule_unchanged=True); launch['source_record_quarantine']=dict(policy='QUARANTINED_SOURCE_RECORDS_NOT_LABEL_FILTERING',maximum_count=100,maximum_rate=1e-5,nonempty_CSV_not_8_fields_rejected=True,raw_records_archived=True,no_reconstruction_or_address_repair=True,all526queries_preserved=True); base.io.write(ROOT/'launch.json',launch)",
    }
    for old,new in replacements.items():
        if text.count(old)!=1:raise ValueError('Exact immutable continuation replacement required: '+old)
        text=text.replace(old,new)
    namespace=dict(__name__='__main__',__file__=str(Path(__file__)),SOURCE_BINDING=SOURCE,CAPTURE_LINES=captured_lines)
    exec(compile(text,str(Path(__file__)),'exec'),namespace)


if __name__=='__main__':main()
