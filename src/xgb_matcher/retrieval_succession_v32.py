"""Deterministic, row-coherent succession witnesses; identifiers only join/tie.

Inputs: raw(query_id,fold,siret,union_rank), edges(official succession fields).
The selected row and parent-level aggregate date are intentionally distinct.
No labels, CRM aliases, learned scoring, or new sources.
"""

CHANNEL = 'sirene_economic_continuity'

SQL = '''
CREATE TEMP TABLE v32_emitted AS
  SELECT DISTINCT r.query_id,r.fold,r.siret seed_siret,r.union_rank::INTEGER seed_rank,
    e.successor_siren siren,e.successor_siret linked_siret,
    'PREDECESSOR_TO_SUCCESSOR' direction,e.effective_date witness_effective_date,
    coalesce(e.relation_id,'') relation_id
  FROM raw r JOIN edges e ON r.siret=e.predecessor_siret
  WHERE r.union_rank BETWEEN 1 AND 250 AND e.economic_continuity
    AND length(e.successor_siren)=9
  UNION
  SELECT DISTINCT r.query_id,r.fold,r.siret seed_siret,r.union_rank::INTEGER seed_rank,
    e.predecessor_siren siren,e.predecessor_siret linked_siret,
    'SUCCESSOR_TO_PREDECESSOR' direction,e.effective_date witness_effective_date,
    coalesce(e.relation_id,'') relation_id
  FROM raw r JOIN edges e ON r.siret=e.successor_siret
  WHERE r.union_rank BETWEEN 1 AND 250 AND e.economic_continuity
    AND length(e.predecessor_siren)=9;

CREATE TEMP TABLE v32_parent_support AS
  SELECT query_id,fold,siren,min(seed_rank)::INTEGER seed_rank,
    max(witness_effective_date) parent_latest_effective_date,
    count(*) admissible_proof_count
  FROM v32_emitted GROUP BY query_id,fold,siren;

CREATE TEMP TABLE v32_witness AS
  SELECT * FROM v32_emitted
  QUALIFY row_number() OVER(PARTITION BY query_id,fold,siren ORDER BY
    seed_rank,witness_effective_date DESC NULLS LAST,
    linked_siret,seed_siret,direction,relation_id)=1;

CREATE TEMP TABLE v32_parent_order AS
  SELECT *,row_number() OVER(PARTITION BY query_id,fold ORDER BY
    seed_rank,parent_latest_effective_date DESC NULLS LAST,siren)::INTEGER channel_rank
  FROM v32_parent_support;

CREATE TEMP TABLE v32_proof AS
  SELECT w.*,p.parent_latest_effective_date,p.admissible_proof_count,p.channel_rank
  FROM v32_witness w JOIN v32_parent_order p USING(query_id,fold,siren,seed_rank)
  WHERE p.channel_rank<=256;

CREATE TEMP TABLE v32_signals AS
  SELECT query_id,fold,'sirene_economic_continuity' channel,channel_rank,
    siren,linked_siret source_siret,1.0::DOUBLE score,seed_rank,
    direction relation_direction,'SIRENE_SUCCESSION' relation_type
  FROM v32_proof;
'''


def build(c):
    if c.execute('SELECT count(*) FROM raw WHERE fold NOT IN (2,3,4)').fetchone()[0]:
        raise ValueError('Forbidden fold')
    if c.execute('''SELECT count(*) FROM (SELECT query_id,fold,siret FROM raw
      GROUP BY ALL HAVING count(*)>1)''').fetchone()[0]:
        raise ValueError('Duplicate raw site')
    c.execute(SQL)
    if c.execute("SELECT count(*) FROM v32_proof WHERE NOT regexp_full_match(linked_siret,'[0-9]{14}') OR NOT regexp_full_match(siren,'[0-9]{9}')").fetchone()[0]:
        raise ValueError('Invalid official witness identity')


def reference(raw, edges):
    """Independent small-data reference used only for tests and verification."""
    from collections import defaultdict
    by_query = defaultdict(dict)
    adjacency = defaultdict(list)
    for e in edges:
        if not e['economic_continuity']:continue
        for src,dst,direction in [('predecessor','successor','PREDECESSOR_TO_SUCCESSOR'),
                                  ('successor','predecessor','SUCCESSOR_TO_PREDECESSOR')]:
            adjacency[e[src+'_siret']].append((e,dst,direction))
    for r in raw:
        if r['fold'] not in (2,3,4):raise ValueError('Forbidden fold')
        if not 1<=r['union_rank']<=250:continue
        for e,dst,direction in adjacency[r['siret']]:
            if len(e[dst+'_siren'] or '')!=9:continue
            key=(r['query_id'],r['fold'],e[dst+'_siren'])
            proof=(r['union_rank'],e['effective_date'],e[dst+'_siret'],r['siret'],direction,e.get('relation_id') or '')
            by_query[key][proof]=proof
    parents=defaultdict(list)
    for (qid,fold,siren),values in by_query.items():
        def key(p):return (p[0],p[1] is None,-p[1].toordinal() if p[1] else 0,*p[2:])
        best=min(values,key=key);dates=[p[1] for p in values if p[1] is not None]
        latest=max(dates) if dates else None
        parents[(qid,fold)].append((siren,best,latest,len(values)))
    output=[]
    for (qid,fold),group in parents.items():
        group.sort(key=lambda x:(x[1][0],x[2] is None,-x[2].toordinal() if x[2] else 0,x[0]))
        for rank,(siren,p,latest,n) in enumerate(group[:256],1):
            output.append(dict(query_id=qid,fold=fold,seed_siret=p[3],seed_rank=p[0],siren=siren,
              linked_siret=p[2],direction=p[4],witness_effective_date=p[1],relation_id=p[5],
              parent_latest_effective_date=latest,admissible_proof_count=n,channel_rank=rank))
    return sorted(output,key=lambda r:(r['query_id'],r['fold'],r['channel_rank']))
