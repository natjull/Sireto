"""Official name/site evidence and fixed, identifier-free parent model inputs."""
import json
from pathlib import Path

from .retrieval_v20_workbench import ROOT, record


FEATURES = (
    'parent_name_jw','parent_name_tokens','best_site_name_jw','best_site_name_tokens',
    'best_joint_jw','best_joint_tokens','best_street','best_number','best_insee','best_postcode',
    'parent_rrf','max_direct_rrf','log_site_count','name_missing','crm_name_missing','crm_number_missing',
)

NAME_SELECTION_SQL = '''CREATE TEMP TABLE names AS SELECT DISTINCT n.siren,n.siret,n.subject_kind,n.name_role,
  trim(regexp_replace(upper(strip_accents(n.normalized_value)),'[^A-Z0-9]+',' ','g')) AS "name"
  FROM name_source n WHERE n.selected_for_subject AND n.normalized_value<>'' AND
  ((n.subject_kind='SIREN' AND n.name_role IN ('LEGAL_CURRENT','TRADE_CURRENT') AND n.siren IN (SELECT siren FROM pool))
  OR (n.subject_kind='SIRET' AND n.name_role='SITE_CURRENT' AND n.siret IN (SELECT siret FROM pool)))'''


def config_at(path):
    c=json.loads(Path(path).read_text())
    if (c['schema']!='sireto-v23-joint-parent-1' or c['recipe']!='OOF_PARENT_OFFICIAL_NAMES_JOINT_SITE_GLOBAL2000'
        or c['master_cap']!=2000 or c['reserved_prefix']!=0 or c['outer_folds']!=[2,3,4]
        or c['sampler']!='HEAD256_PLUS_STRATIFIED_TAIL256' or c['national_projection_allowed']):
        raise ValueError('V23 recipe drift')
    return c


def code_records():
    paths=[Path(__file__),ROOT/'scripts/retrieval_v23_joint_parent.py',
      ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_v21_membership.py',
      ROOT/'scripts/retrieval_v22_global_budget.py',ROOT/'src/xgb_matcher/retrieval_parent_fusion_v19.py',
      ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py',ROOT/'src/xgb_matcher/retrieval_v12_scene_ir.py',
      ROOT/'src/xgb_matcher/hierarchical_retrieval.py']
    return {str(p.relative_to(ROOT)):record(p) for p in paths}


def build_features(c):
    """Input tables: q, pool, names, sites. No labels or cohort status."""
    if c.execute('SELECT count(*) FROM q WHERE fold NOT IN (2,3,4)').fetchone()[0]:raise ValueError('Forbidden fold')
    c.execute('''CREATE MACRO jw(a,b) AS CASE WHEN coalesce(a,'')='' OR coalesce(b,'')='' THEN 0.0
      ELSE jaro_winkler_similarity(a,b) END''')
    c.execute('''CREATE MACRO tokens(a,b) AS CASE WHEN coalesce(a,'')='' OR coalesce(b,'')='' THEN 0.0
      ELSE len(list_intersect(string_split(a,' '),string_split(b,' ')))::DOUBLE /
        len(list_distinct(list_concat(string_split(a,' '),string_split(b,' ')))) END''')
    # Official legal/trade names are parent evidence. A site name stays on its
    # own SIRET and is never copied to siblings to fabricate a joint match.
    c.execute('''CREATE TEMP TABLE parent_names AS
      SELECT p.query_id,p.fold,p.siren,max(jw(q.crm_name,n.name)) parent_name_jw,
        max(tokens(q.crm_name,n.name)) parent_name_tokens,max(CAST(n.name IS NOT NULL AS INTEGER)) has_parent_name
      FROM (SELECT DISTINCT query_id,fold,siren FROM pool) p JOIN q USING(query_id,fold)
      LEFT JOIN names n ON p.siren=n.siren AND n.subject_kind='SIREN'
        AND n.name_role IN ('LEGAL_CURRENT','TRADE_CURRENT')
      GROUP BY p.query_id,p.fold,p.siren''')
    c.execute('''CREATE TEMP TABLE site_names AS
      SELECT p.query_id,p.fold,p.siren,p.siret,max(jw(q.crm_name,n.name)) site_name_jw,
        max(tokens(q.crm_name,n.name)) site_name_tokens,max(CAST(n.name IS NOT NULL AS INTEGER)) has_site_name
      FROM pool p JOIN q USING(query_id,fold)
      LEFT JOIN names n ON p.siret=n.siret AND p.siren=n.siren AND n.subject_kind='SIRET' AND n.name_role='SITE_CURRENT'
      GROUP BY p.query_id,p.fold,p.siren,p.siret''')
    c.execute('''CREATE TEMP TABLE site_evidence AS
      WITH facts AS (
        SELECT p.*,n.parent_name_jw,n.parent_name_tokens,n.has_parent_name,
          sn.site_name_jw,sn.site_name_tokens,sn.has_site_name,
          jw(q.crm_street,s.street_normalized) street,
          coalesce(q.crm_number<>'' AND q.crm_number=s.street_number,false) number_exact,
          coalesce(q.crm_suffix='' OR q.crm_suffix=upper(coalesce(s.street_number_suffix,'')),false) suffix_ok,
          coalesce(q.crm_insee<>'' AND q.crm_insee=s.insee,false) same_insee,
          coalesce(q.crm_postcode<>'' AND q.crm_postcode=s.postcode,false) same_postcode,
          q.crm_name='' crm_name_missing,q.crm_number='' crm_number_missing,s.administrative_state,
          s.siret IS NOT NULL official_site
        FROM pool p JOIN q USING(query_id,fold)
        JOIN parent_names n ON p.query_id=n.query_id AND p.fold=n.fold AND p.siren=n.siren
        JOIN site_names sn ON p.query_id=sn.query_id AND p.fold=sn.fold AND p.siren=sn.siren AND p.siret=sn.siret
        LEFT JOIN sites s ON p.siren=s.siren AND p.siret=s.siret
      )
      SELECT *, (same_insee OR same_postcode) local_site,
        greatest(parent_name_jw,site_name_jw)*street*
          CASE WHEN crm_number_missing THEN 1.0 ELSE CAST(number_exact AND suffix_ok AS INTEGER) END joint_jw,
        greatest(parent_name_tokens,site_name_tokens)*street*
          CASE WHEN crm_number_missing THEN 1.0 ELSE CAST(number_exact AND suffix_ok AS INTEGER) END joint_tokens
      FROM facts''')
    c.execute('''CREATE TEMP TABLE ranked_sites AS
      SELECT query_id,fold,siren,siret,dense_rank() OVER(PARTITION BY query_id,siren ORDER BY
        joint_tokens DESC,joint_jw DESC,(number_exact AND suffix_ok AND street=1) DESC,
        same_insee DESC,number_exact DESC,suffix_ok DESC,street DESC,(administrative_state='A') DESC) joint_site_rank
      FROM site_evidence WHERE local_site''')
    c.execute('''CREATE TEMP TABLE parent_features AS
      SELECT query_id,fold,siren,max(parent_name_jw) parent_name_jw,max(parent_name_tokens) parent_name_tokens,
        max(site_name_jw) best_site_name_jw,max(site_name_tokens) best_site_name_tokens,
        max(CASE WHEN local_site THEN joint_jw ELSE 0 END) best_joint_jw,
        max(CASE WHEN local_site THEN joint_tokens ELSE 0 END) best_joint_tokens,
        max(street) best_street,max(CAST(number_exact AS INTEGER)) best_number,
        max(CAST(same_insee AS INTEGER)) best_insee,max(CAST(same_postcode AS INTEGER)) best_postcode,
        max(parent_rrf) parent_rrf,max(direct_rrf) max_direct_rrf,ln(1+count(*)) log_site_count,
        1-max(greatest(has_parent_name,has_site_name)) name_missing,
        max(CAST(crm_name_missing AS INTEGER)) crm_name_missing,max(CAST(crm_number_missing AS INTEGER)) crm_number_missing
      FROM site_evidence GROUP BY query_id,fold,siren''')
    c.execute('''CREATE TEMP TABLE site_features AS SELECT e.query_id,e.fold,e.siren,e.siret,e.base_rank,
      e.direct_rrf,e.parent_rrf,e.local_site,r.joint_site_rank,e.joint_jw,e.joint_tokens,
      e.site_name_jw,e.parent_name_jw,e.official_site FROM site_evidence e
      LEFT JOIN ranked_sites r USING(query_id,fold,siren,siret)''')
    if c.execute('SELECT count(*) FROM site_features').fetchone()!=c.execute('SELECT count(*) FROM pool').fetchone():raise ValueError('Candidate population drift')


def make_master(c):
    """parent_scores contains only predictions from the matching outer fold model."""
    c.execute('''CREATE TEMP TABLE ranked_parents AS SELECT *,dense_rank() OVER(
      PARTITION BY query_id ORDER BY parent_score DESC) learned_parent_rank FROM parent_scores''')
    c.execute('''CREATE TEMP TABLE master AS
      SELECT * EXCLUDE(r),r candidate_rank,'OOF_PARENT_JOINT_SITE' selection_reason FROM (
        SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY rescue_score DESC,direct_rrf DESC,parent_score DESC,siret) r
        FROM (SELECT s.*,p.parent_score,p.learned_parent_rank,
          s.direct_rrf+CASE WHEN s.local_site THEN 1.0/(60+p.learned_parent_rank+s.joint_site_rank) ELSE 0 END rescue_score
          FROM site_features s JOIN ranked_parents p USING(query_id,fold,siren))
      ) WHERE r<=2000''')
    if c.execute('SELECT count(*) FROM (SELECT query_id,count(*) n,count(DISTINCT siret) nd,max(candidate_rank) mx FROM master GROUP BY query_id HAVING n>2000 OR n<>nd OR n<>mx)').fetchone()[0]:raise ValueError('Master integrity')
