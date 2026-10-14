"""New scientific grammar: sole trailing corporate legal-form parentheses.

No labels observed. V158B and its recipe remain immutable. Only a qualifier
whose entire contents match the frozen corporate FORM may be parenthesized.
All other ambiguities, exact public resolution, degrees/MAX and dates unchanged.
"""
import argparse
from collections import Counter
import inspect
from pathlib import Path
import re
import time
import unicodedata
from scripts import embeddinggemma2_v158_corporate_administration_source as original
from scripts import embeddinggemma2_night as io

OUT=io.HOME/'v159/parenthetical_corporate_administration_20261014_night'
PAREN=re.compile(r'^([^()]+?)\s*\(('+original.FORM+r')\)$')


def clauses(text):
    value=''.join(c for c in unicodedata.normalize('NFKD',str(text or '').upper()) if not unicodedata.combining(c))
    value=value.replace('’',"'").replace("'",' ')
    headers=list(original.HEADER.finditer(value));counts=Counter();accepted=[]
    if not headers:counts['no_explicit_role_colon']+=1
    for i,h in enumerate(headers):
        end=headers[i+1].start() if i+1<len(headers) else len(value)
        piece=value[h.end():end].strip(' ;\r\n. ')
        if any(c in piece for c in ',;\n\r'):
            counts['ambiguous_list']+=1;continue
        if original.FORBIDDEN.search(piece):counts['person_or_nonname_suffix_rejected']+=1;continue
        name=None;origin=None
        if '(' in piece or ')' in piece:
            matched=PAREN.fullmatch(piece)
            if matched:
                name=matched.group(1);origin='SOLE_TRAILING_PARENTHETICAL_LEGAL_FORM'
            else:counts['non_legal_or_multiple_parentheses_rejected']+=1;continue
        else:
            matched=original.LEAD.fullmatch(piece)
            if matched:name=matched.group(2);origin='LEGAL_FORM_PREFIX'
            else:
                matched=original.TAIL.fullmatch(piece)
                if matched:name=matched.group(1);origin='LEGAL_FORM_SUFFIX'
        if name is None:counts['no_explicit_corporate_form']+=1;continue
        name=original.normalized(name.strip(' "«»'))
        if not name or len(name)<2:counts['empty_or_invalid_name']+=1;continue
        accepted.append((original.normalized(h.group(1)),name,origin));counts[origin]+=1
    return accepted,counts


def bind():
    original.OUT=OUT;original.clauses=clauses
    original.RECIPE=dict(original.RECIPE,scientific_parent='V158_STRICT_ROLE_AND_CORPORATE_FORM',
        added_grammar='ONLY_SOLE_TRAILING_PARENTHESIS_WHOSE_ENTIRE_CONTENTS_FULLMATCH_FROZEN_FORM',
        qualification_origin_preserved=True,nonexhaustive_administration_parser=True,
        no_labels_before_both_whole_sources_complete=True)
    body=inspect.getsource(original.worker)
    changes=[(") admin,CAST",") admin_text,CAST"),("r['admin']","r['admin_text']"),
        ('actor_name VARCHAR,publication_id VARCHAR','actor_name VARCHAR,qualification_origin VARCHAR,publication_id VARCHAR'),
        ('for role,name in accepted:','for role,name,origin in accepted:'),
        ("role=role,actor_name=name,publication_id=","role=role,actor_name=name,qualification_origin=origin,publication_id="),
        ('e.actor_name,e.role,(r.subject','e.actor_name,e.role,e.qualification_origin,(r.subject')]
    for old,new in changes:
        if body.count(old)!=1:raise ValueError('Exact frozen parent worker text binding: '+old)
        body=body.replace(old,new)
    exec(compile(body,str(Path(__file__)), 'exec'),original.__dict__)
    runner=inspect.getsource(original.run).replace('scripts.embeddinggemma2_v158_corporate_administration_source','scripts.embeddinggemma2_v159_parenthetical_corporate_source')
    exec(compile(runner,str(Path(__file__)), 'exec'),original.__dict__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','worker','run']);a=p.parse_args();bind()
    try:
        if a.action=='initialize':
            original.initialize();launch=io.read(OUT/'launch.json')
            launch['scientific_grammar_implementation']=original.prior.record(Path(__file__))
            launch['immutable_parent_B']=str(io.HOME/'v158/corporate_administration_20261014_night_b')
            io.write(OUT/'launch.json',launch)
        else:
            original.prior.verify(io.read(OUT/'launch.json')['scientific_grammar_implementation']);getattr(original,a.action)()
    except BaseException as error:
        if OUT.exists():io.write(OUT/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
