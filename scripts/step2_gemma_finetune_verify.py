"""Read-only audit of real Gemma adaptation receipts, not a unit-test suite."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import step2_colocated_signals as s
from scripts import step2_gemma_finetune as run
np = s.np


def verify(out):
    plan = run.read(out/'plan.json'); result = run.read(out/'result.json'); data = run.read(out/'data.json'); pilot = run.read(out/'pilot.json')
    assert result['status'] == 'COMPLETE_DEVELOPMENT_NOT_PRODUCTION'
    # A subsequent plumbing commit may change a working-tree implementation;
    # bind the actual run to its immutable preregistered Git version instead.
    for ref in plan['implementations']+[plan['config']]:
        path = Path(ref['path'])
        content = subprocess.check_output(['git','show',plan['git_head']+':'+str(path.relative_to(ROOT))],cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == ref['sha256'], str(path)
    run.verify(plan['protocol'])
    for ref in plan['base_files']+data['sources']:run.verify(ref)
    checked = 0
    for path in out.glob('*/receipt.json'):
        receipt = run.read(path)
        refs = receipt.get('artifacts',[])+receipt.get('shards',[])
        for key in ('result','checkpoint','binding'):
            if key in receipt:refs.append(receipt[key])
        for ref in refs:
            if path.parent.name=='summarize' and ref['path']==str(out/'result.json'):
                # Initial smoke publication appended timing to this JSON. Its
                # preregistered summary is exactly recoverable, without edits.
                original={k:v for k,v in result.items() if k not in ('wall_seconds','metal_seconds','peak_bytes','plan')}
                content=(json.dumps(original,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
                assert len(content)==ref['size_bytes'] and hashlib.sha256(content).hexdigest()==ref['sha256']
            else:run.verify(ref)
            checked += 1
    groups = np.array(data['groups']+pilot['groups']); n = data['queries']; roles = {k:np.array(v,dtype=int) for k,v in data['roles'].items()}
    assert not set(data['groups']) & set(pilot['groups'])
    assert set(roles['training']) | set(roles['evaluation']) == set(range(n))
    assert set(roles['proper']) | set(roles['calibration']) == set(roles['training'])
    for a,z in [('training','evaluation'),('proper','calibration')]:assert not set(groups[roles[a]]) & set(groups[roles[z]])
    meta = Path(data['metadata']); qi=np.load(meta/'qtext.npy');ci=np.load(meta/'ctext.npy');keys=np.load(out/'keys.npy');views=np.load(out/'views.npy')
    forbidden=set(groups[roles['evaluation']]) | set(groups[roles['calibration']]) | set(pilot['groups'])
    rows=run.read(out/'training.json');negative_texts=0
    for row in rows:
        i=row['row'];assert i in roles['proper'] and qi[i]==row['query']
        pos=np.flatnonzero(views[i,:,0]);assert len(pos)==1 and ci[i,pos[0]]==row['positive']
        allowed={int(ci[i,k]) for k in range(100) if not views[i,k,1] and ci[i,k]>=0 and ci[i,k]!=row['positive'] and keys[i,k].decode()[:9] not in forbidden}
        assert set(row['negatives']) <= allowed and row['negatives'];negative_texts += len(row['negatives'])
    training=run.read(out/'train/receipt.json');assert training['steps']>0 and training['adapter_changed'] and not training['historical_adapter_loaded']
    uses=nbuses=0;hits={};metrics={}
    for name,variant in result['variants'].items():
        pack=np.load(run.verify(variant['predictions']));top=s.b.select_top(pack['predictions'])[0]
        actual=views[roles['evaluation'],top];assert np.array_equal(actual,pack['hits'])
        assert s.bench.metrics(actual)==variant['top1'];hits[name]=actual
        for model in variant['models']:
            run.verify(model['model']);train=np.array(model['train_indices'])
            assert set(train) <= set(roles['training'])
            assert s.bench.sha(views[train,:,0].astype(np.float32))==model['target_sha256']
            for use in model['uses']:
                test=np.array(use['scored_indices']);assert not set(groups[train]) & set(groups[test]);assert use['replay_equal'];uses+=len(test)
            for nb in model['nb_training']+[u['nb_scoring'] for u in model['uses'] if u['nb_scoring']]:
                run.verify(nb['model']);assert not set(groups[nb['train_indices']]) & set(groups[nb['scored_indices']]);assert nb['serialized_replay_equal'];nbuses+=1
        for model in variant['pilot']['stack_models']:run.verify(model['model'])
        for confidence in variant['pilot']['confidence'].values():
            for ref in confidence['model_refs']:run.verify(ref)
        assert variant['sur']['count']==int(pack['sur'].sum())
        if not variant['sur']['count']:assert variant['sur']['precision']['union'] is None
        metrics[name]=dict(top1=variant['top1'],sur=variant['sur'],pilot_content=variant['pilot']['content'],pilot_stack=variant['pilot']['stack'])
    assert s.boot(hits,np.array(data['groups'])[roles['evaluation']])==result['deltas']
    proof=dict(passed=True,run_git_head=plan['git_head'],queries=n,companies=data['companies'],role_counts=data['role_counts'],
        gradient_steps=training['steps'],exposed_pairs=training['exposed_pairs'],training_pairs=len(rows),safe_negative_texts=negative_texts,
        receipt_files_verified=checked,ranking_vectors_serialized_replayed=uses,nb_serialized_company_disjoint_uses=nbuses,
        pilot_queries=len(pilot['ids']),labels_changed=False,closed_sets_opened=False,bootstrap_reproduced=True,metrics=metrics)
    run.write(out/'verification.json',proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if not a.output_dir.resolve().is_relative_to(run.OWN):raise ValueError('Owned namespace only')
    verify(a.output_dir)
