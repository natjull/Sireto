import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.retrieval_v36_evidence_ranker import EXTRA, FEATURES, PARAMS, extra_evidence, full_training
from scripts.retrieval_v35_site_fusion import db


def test_full_concurrence_and_operational_neutralization():
    f=pd.DataFrame([dict(query_id=q,siret=str(i),candidate_rank=i+1) for q in ('present','absent') for i in range(2000)])
    train,stats=full_training(f,dict(present='1999',absent='missing'),dict(present={'0','1','1999'}))
    assert len(train)==1998 and train.target.sum()==1
    assert stats==dict(queries=2,trainable=1,absent_positive=1,neutralized=2,rows=1998)
    assert set(train.query_id)=={'present'} and '1999' in set(train.siret)
    assert not {'0','1','missing'} & set(train.siret)


def test_feature_identity_exclusion_and_fixed_budget():
    assert len(FEATURES)==len(set(FEATURES))
    assert not {'siren','siret','query_id','fold','crm_name','x','y'} & set(FEATURES)
    assert PARAMS['lambdarank_num_pair_per_sample']==100 and PARAMS['lambdarank_pair_method']=='topk'


def test_official_site_proof_stays_on_its_site_and_no_distance_cutoff(tmp_path):
    files={}
    def save(name,rows,schema=None):
        path=tmp_path/name;pq.write_table(pa.Table.from_pylist(rows,schema=schema),path)
        return {'path':str(path)}
    files['queries.parquet']=save('queries.parquet',[dict(query_id='q',fold=2,crm_name='OLD SHOP')])
    files['v30_stable/pool.parquet']=save('pool.parquet',[
        dict(query_id='q',fold=2,siren='p',siret=s) for s in ('s1','s2','s3')])
    files['ban.parquet']=save('ban.parquet',[dict(query_id='q',fold=2,ban_resolved=True,ban_x_l93=700000.,ban_y_l93=6600000.,ban_uncertainty_m=20.)])
    sigschema=pa.schema([('query_id',pa.string()),('fold',pa.int64()),('siren',pa.string()),('source_siret',pa.string()),('channel',pa.string()),('channel_rank',pa.int64()),('score',pa.float64())])
    files['direct.parquet']=save('direct.parquet',[dict(query_id='q',fold=2,siren='p',source_siret='s1',channel='siret_historical_name_word',channel_rank=2,score=7.)],sigschema)
    files['structured.parquet']=save('structured.parquet',[],sigschema)
    jointschema=pa.schema([('query_id',pa.string()),('fold',pa.int64()),('siren',pa.string()),('siret',pa.string()),('evidence_rank',pa.int64()),('score',pa.float64())])
    files['joint.parquet']=save('joint.parquet',[],jointschema)
    cache={
      'all_names':save('names.parquet',[dict(siren='p',siret='s1',name_group='site_history',name='OLD SHOP'),dict(siren='p',siret='',name_group='parent_support',name='OLD SHOP')]),
      'xy':save('xy.parquet',[dict(siren='p',siret='s1',x=700010.,y=6600000.),dict(siren='p',siret='s2',x=710000.,y=6600000.),dict(siren='p',siret='s3',x=None,y=None)])}
    c=db(tmp_path)
    try:
        extra_evidence(c,{'files':files},cache)
        frame=c.execute('SELECT * FROM extra ORDER BY siret').df()
        assert len(frame)==3 and frame.siret.tolist()==['s1','s2','s3']
        assert frame.site_history_jw.tolist()==[1.,0.,0.]
        assert frame.parent_support_jw.tolist()==[1.,1.,1.]
        assert frame.xy_available.tolist()==[True,True,False]
        assert frame.iloc[1].xy_distance_log>frame.iloc[0].xy_distance_log>0
        assert frame.iloc[2].xy_distance_log==0
        assert frame.parent_siret_historical_name_word_relative_score.tolist()==[1.,1.,1.]
        assert frame.site_siret_historical_name_word_relative_score.tolist()==[1.,0.,0.]
        assert set(EXTRA)<=set(frame.columns)
    finally:c.close()
