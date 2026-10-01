import numpy as np
import pandas as pd
import pytest
from scripts.retrieval_v37_parent_evidence import CHANNELS,EXTRA,FEATURES,parent_table,attach


def test_all_configured_channels_present_and_duplicate_witness_not_recounted():
    rows=pd.DataFrame([dict(query_id='q',fold=2,siren=p,channel='temporal_name_char',channel_rank=r,score=s)
        for p,r,s in [('p1',1,10.),('p1',2,9.),('p2',3,5.)]])
    table=parent_table(rows)
    assert len(CHANNELS)==26 and len(EXTRA)==52 and len(FEATURES)==183
    assert len(table)==2 and not table.duplicated(['query_id','fold','siren']).any()
    assert np.allclose(table.discovery_temporal_name_char_relative_score,[1.,.5])
    assert np.allclose(table.discovery_temporal_name_char_rank_rrf,[1/61,1/63])
    assert table.discovery_sector_name_char_rank_rrf.tolist()==[0.,0.]
    frame=pd.DataFrame([dict(query_id='q',fold=2,siren=p,siret=s) for p,s in [('p1','a'),('p1','b'),('absent','c')]])
    joined=attach(frame,table)
    assert len(joined)==3 and joined.siret.tolist()==['a','b','c']
    assert joined.discovery_temporal_name_char_relative_score.tolist()==[1.,1.,0.]
    assert not {'siren','siret','query_id','fold'}&set(FEATURES)


def test_unknown_channel_and_forbidden_fold_rejected():
    row=dict(query_id='q',fold=1,siren='p',channel='temporal_name_char',channel_rank=1,score=1.)
    with pytest.raises(ValueError,match='Forbidden'):parent_table(pd.DataFrame([row]))
    row.update(fold=2,channel='oracle_channel')
    with pytest.raises(ValueError,match='Unknown'):parent_table(pd.DataFrame([row]))
