"""Source-only published historical consolidation membership, not identity.

Compile ALL French rows of the visually verified Note17, no CRM/GT selection.
Consolidated control and economic interest differ; neither is direct ownership.
"""
import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import time
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCE=io.HOME/'v69/group_reports_20261013_a/savencia_2021_fr'
OUTPUT=io.HOME/'v70/published_group_members_20261013_a'
ROW=re.compile(r'^\s*(?P<name>.*?)\s+(?P<a>[0-9]{3})\s+(?P<b>[0-9]{3})\s+(?P<c>[0-9]{3})\s+FRANCE\s+(?P<values>(?:[0-9]+,[0-9]+\s+){3}[0-9]+,[0-9]+)\s*$')


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh source compilation namespace')
    metadata=io.read(SOURCE/'source.json');pdf=verify(metadata['source'])
    text=subprocess.check_output(['pdftotext','-f','120','-l','121','-layout',str(pdf),'-'],text=True,timeout=15)
    if 'NOTE 17.' not in text or '31/12/2021' not in text or 'Consolidation par intégration globale' not in text:
        raise ValueError('Exact visually reviewed scope and date')
    rows=[];carry='';root=None;foreign=False
    for line in text.splitlines():
        if line.strip()=='Étranger':foreign=True
        if foreign:continue
        if re.search(r'SAVENCIA SA\s+847\s+120\s+185\s+FRANCE',line):
            root=dict(siren='847120185',name='SAVENCIA SA');continue
        match=ROW.fullmatch(line)
        if match:
            name=match.group('name').strip();name=' '.join((carry+' '+name).split()) if carry else name
            carry='';values=[float(v.replace(',','.')) for v in match.group('values').split()]
            rows.append(dict(member_siren=match.group('a')+match.group('b')+match.group('c'),
                member_name=name,control_2021=values[0]/100,control_2020=values[1]/100,
                interest_2021=values[2]/100,interest_2020=values[3]/100,
                reference_date='2021-12-31',relationship='PUBLISHED_CONSOLIDATED_GROUP_MEMBERSHIP',
                direct_ownership_not_asserted=True,current_membership_not_asserted=True))
        elif line.strip()=='Société des Beurres et Crèmes des Régions':
            # Layout continuation is verified on the source PDF page121.
            carry=line.strip()
    if root is None or len(rows)<40 or len({r['member_siren'] for r in rows})!=len(rows):
        raise ValueError('Complete French source table, not two manually selected targets')
    if any(not r['member_name'] or not 0<=r['control_2021']<=1 or not 0<=r['interest_2021']<=1 for r in rows):
        raise ValueError('Bounded explicit percent columns')
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'members.json',dict(root=root,reference_date='2021-12-31',members=rows,
        source_scope='LIST_OF_MAIN_CONSOLIDATED_COMPANIES_NOT_EXHAUSTIVE_GROUP_UNIVERSE',
        all_french_siren_rows_of_note17_compiled=True,foreign_rows_without_french_siren_not_resolved=True))
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_SOURCE_MEMBERSHIP_NOT_RETRIEVAL',
        implementation=record(Path(__file__)),source_metadata=record(SOURCE/'source.json'),source_pdf=metadata['source'],
        visual_pages=[record(SOURCE/'visual/table-120.png'),record(SOURCE/'visual/table-121.png')],
        members=record(OUTPUT/'members.json'),french_member_rows=len(rows),root=root,
        effective_date='2021-12-31',publication_meeting_date='2022-04-21',
        current_ownership_not_asserted=True,source_policy_not_yet_admitted=True,
        source_subjects_not_gt_selected=True,crm_read=False,labels_read=False,
        no_alias_or_model_or_prediction_changed=True,not_recall_measurement=True,observed_unix=time.time()))
    print(dict(rows=len(rows),root=root),flush=True)


if __name__=='__main__':main()
