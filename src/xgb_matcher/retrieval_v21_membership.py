"""Single preregistered structural repair; no cohort-specific scoring rules."""
import json
from pathlib import Path

import duckdb

from .retrieval_v20_workbench import ROOT, record, verify, sql_path


def load_config(path):
    c = json.loads(Path(path).read_text())
    if (c['schema'] != 'sireto-v21-membership-1'
        or c['recipe'] != 'ALL_PARENT_MEMBERSHIP_DEDUP_DENSE_RANK_V20_SCORE'
        or (c['master_cap'], c['protected_prefix'], c['rrf_k']) != (2000, 1750, 60)
        or any(c[k] for k in ('learned_components', 'new_weights_or_thresholds',
                             'cross_crm_claim_allowed', 'national_run_allowed', 'ready_for_fold1_allowed'))):
        raise ValueError('Frozen v21 recipe mismatch')
    return c


def implementation():
    paths = [Path(__file__), ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
             ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py',
             ROOT/'scripts/run_retrieval_v21_membership.py',
             ROOT/'scripts/evaluate_retrieval_v21_membership.py']
    return {str(p.relative_to(ROOT)): record(p) for p in paths}


def manifest_files(authority):
    path = verify(authority)
    m = json.loads(path.read_text())
    files = {k: verify(v) for k, v in m['files'].items()}
    if any(p.parent.resolve() != path.parent.resolve() for p in files.values()):
        raise ValueError('External manifest artifact')
    return m, files


def open_workbench_db(output):
    """Spill only beside this run's artifacts, never into the workspace disk."""
    spill = Path(output)/'_duckdb_spill'
    spill.mkdir(exist_ok=False)
    c = duckdb.connect()
    c.execute("SET memory_limit='4GB'")
    c.execute('SET threads=4')
    c.execute(f'SET temp_directory={sql_path(spill)}')
    c.execute("SET max_temp_directory_size='12GiB'")
    return c


def build_tables(c):
    """Inputs: truth-free q/raw/signals/sites. Identity only joins/ties, never scores."""
    if c.execute('SELECT count(*) FROM q WHERE fold NOT IN (2,3,4)').fetchone()[0]:
        raise ValueError('Forbidden fold')
    if c.execute('SELECT count(*) FROM (SELECT query_id,siret,count(*) n FROM raw GROUP BY ALL HAVING n>1)').fetchone()[0]:
        raise ValueError('Duplicate raw SIRET')
    if c.execute('SELECT count(*) FROM raw WHERE base_rank<1 OR base_rank>2000 OR fold NOT IN (2,3,4)').fetchone()[0]:
        raise ValueError('Forbidden raw ranks/folds')
    # Preserve membership independently of relevance. Duplicate records in one
    # channel are one piece of evidence, not independent confirmations.
    c.execute('''CREATE TEMP TABLE parents AS
      WITH ranked AS (
        SELECT query_id,fold,siren,channel,min(channel_rank) r FROM signals
        WHERE channel<>'v17_union' GROUP BY ALL
      ), support AS (
        SELECT query_id,fold,siren,sum(1.0/(60+r)) parent_rrf FROM ranked GROUP BY ALL
      ), membership AS (
        SELECT query_id,fold,siren FROM signals UNION SELECT query_id,fold,siren FROM raw
      )
      SELECT *,dense_rank() OVER(PARTITION BY query_id ORDER BY parent_rrf DESC) parent_rank
      FROM (SELECT m.*,coalesce(s.parent_rrf,0) parent_rrf FROM membership m
            LEFT JOIN support s USING(query_id,fold,siren))''')
    c.execute('''CREATE TEMP TABLE witnesses AS
      SELECT query_id,fold,siren,siret,'legacy_site_pool' channel,base_rank channel_rank FROM raw
      UNION
      SELECT query_id,fold,siren,source_siret siret,channel,channel_rank FROM signals
      WHERE length(source_siret)=14 AND channel NOT IN ('legacy_site_pool','v17_union')''')
    if c.execute('SELECT count(*) FROM witnesses w JOIN sites s USING(siret) WHERE w.siren<>s.siren').fetchone()[0]:
        raise ValueError('Source SIRET/official parent mismatch')
    c.execute('''CREATE TEMP TABLE direct AS
      SELECT query_id,fold,siren,siret,sum(1.0/(60+r)) direct_rrf,count(*) direct_channels
      FROM (SELECT query_id,fold,siren,siret,channel,min(channel_rank) r FROM witnesses GROUP BY ALL)
      GROUP BY query_id,fold,siren,siret''')
    c.execute('''CREATE TEMP TABLE address_sites AS
      SELECT *,dense_rank() OVER(PARTITION BY query_id,siren ORDER BY
        (number_exact AND suffix_ok AND street_similarity=1) DESC,
        same_insee DESC,number_exact DESC,suffix_ok DESC,street_similarity DESC,
        (administrative_state='A') DESC) site_address_rank
      FROM (
        SELECT p.query_id,p.fold,p.siren,s.siret,s.administrative_state,
          (q.crm_insee<>'' AND q.crm_insee=s.insee) same_insee,
          (q.crm_number<>'' AND q.crm_number=s.street_number) number_exact,
          (q.crm_suffix='' OR q.crm_suffix=upper(coalesce(s.street_number_suffix,''))) suffix_ok,
          CASE WHEN q.crm_street<>'' AND coalesce(s.street_normalized,'')<>''
            THEN jaro_winkler_similarity(q.crm_street,s.street_normalized) ELSE 0 END street_similarity
        FROM parents p JOIN q USING(query_id,fold) JOIN sites s ON p.siren=s.siren
        AND ((q.crm_insee<>'' AND q.crm_insee=s.insee) OR (q.crm_postcode<>'' AND q.crm_postcode=s.postcode))
      )''')
    c.execute('''CREATE TEMP TABLE candidates AS
      SELECT u.*,r.base_rank,coalesce(d.direct_rrf,0) direct_rrf,
        coalesce(p.parent_rrf,0) parent_rrf,p.parent_rank,a.site_address_rank,
        coalesce(d.direct_channels,0) direct_channels,
        coalesce(a.number_exact AND a.suffix_ok AND a.street_similarity=1,false) joint_address,
        coalesce(d.direct_rrf,0) + CASE WHEN a.site_address_rank IS NOT NULL THEN
          1.0/(60+p.parent_rank+a.site_address_rank) ELSE 0 END rescue_score
      FROM (SELECT query_id,fold,siren,siret FROM raw UNION
            SELECT query_id,fold,siren,siret FROM direct UNION
            SELECT query_id,fold,siren,siret FROM address_sites) u
      LEFT JOIN raw r USING(query_id,fold,siren,siret)
      LEFT JOIN direct d USING(query_id,fold,siren,siret)
      LEFT JOIN parents p USING(query_id,fold,siren)
      LEFT JOIN address_sites a USING(query_id,fold,siren,siret)''')
    c.execute('''CREATE TEMP TABLE master AS
      WITH counts AS (SELECT query_id,count(*) n FROM candidates WHERE base_rank<=1750 GROUP BY query_id),
      rescue AS (SELECT c.*,row_number() OVER(PARTITION BY query_id ORDER BY
        rescue_score DESC,direct_rrf DESC,parent_rrf DESC,siret) rescue_rank
        FROM candidates c WHERE base_rank IS NULL OR base_rank>1750)
      SELECT c.*,base_rank candidate_rank,'PRESERVED_BASE1750' selection_reason FROM candidates c WHERE base_rank<=1750
      UNION ALL
      SELECT r.* EXCLUDE(rescue_rank),coalesce(n,0)+rescue_rank candidate_rank,'DIRECT_OR_SIBLING_RESCUE' selection_reason
      FROM rescue r LEFT JOIN counts USING(query_id) WHERE rescue_rank<=2000-coalesce(n,0)''')
    if c.execute('SELECT count(*) FROM (SELECT query_id,count(*) n,count(DISTINCT siret) nd,min(candidate_rank) mn,max(candidate_rank) mx FROM master GROUP BY query_id HAVING n<>nd OR n>2000 OR n<>mx OR mn<>1)').fetchone()[0]:
        raise ValueError('Master cap, uniqueness or ranks')
    if c.execute('SELECT count(*) FROM raw r ANTI JOIN master m USING(query_id,siret) WHERE r.base_rank<=1750').fetchone()[0]:
        raise ValueError('Baseline preservation failed')
