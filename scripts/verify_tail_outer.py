"""Verify frozen validation policy, refit membership and every outer curve."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_selection import predict
from battery_smetimes.tail_boost import labels,is_nca
from scripts.finalize_tail_boost import all_cells
from scripts.verify_tail_boost import preprocessing as new_preprocessing
from scripts.verify_curve_fidelity_outer import preprocessing as core_preprocessing
from scripts.verify_nca_curve_validation import preprocessing as nca_preprocessing
from scripts.train_nca_curve_fidelity import condition
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['training_source']);assert sha(source/'config.json')==cfg['training_config_sha256']
    assert sha(source/'reference_lock.json')==cfg['reference_lock_sha256']
    original=json.loads((source/'config.json').read_text())
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for p in (Path('battery_smetimes/tail_selection.py'),Path('battery_smetimes/tail_boost.py'),
            Path('battery_smetimes/curve_fidelity.py'),Path('scripts/train_nca_curve_fidelity.py')):
        assert sha(p)==cfg['source_sha256'][p.name]
    decisions=json.loads((root/'frozen_decisions.json').read_text());assert len(decisions)==len(original['plans'])*5
    checked=[];models=0
    for plan in original['plans']:
        cells=all_cells(plan);dev=plan['train']+plan['validation'];assert not set(dev)&set(plan['test'])
        for seed in original['seeds']:
            dest=root/plan['name']/str(seed);decision=json.loads((dest/'selection.json').read_text())
            assert decision==next(d['decision'] for d in decisions if d['fold']==plan['name'] and d['seed']==seed)
            assert len(decision['candidates'])==2
            # Recompute BOTH scores from training-side stored validation rows,
            # independently of the selector implementation and outer errors.
            scores=[]
            for candidate in decision['candidates']:
                file=Path(candidate['selection_file']);assert sha(file)==candidate['selection_sha256']
                c=json.loads(file.read_text());assert c['selected']['candidate']==candidate['candidate']
                folder=file.parent/f"candidate_{candidate['candidate']}"
                assert sha(folder/'model.joblib')==candidate['model_sha256']
                rows=json.loads((folder/'validation_metrics.json').read_text());assert {r['cell_id'] for r in rows}==set(plan['validation'])
                values=[(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]
                score=float(np.mean(values)+.25*np.quantile(values,.9));np.testing.assert_allclose(score,candidate['score'],atol=1e-12,rtol=0)
                scores.append(score)
            expected=decision['candidates'][0] if scores[0]<scores[1] else decision['candidates'][1]
            assert decision['selected']==expected
            b=joblib.load(dest/'model.joblib');assert b['train_ids']==dev and b['mode']==expected['mode']
            assert b['parameters']==expected['parameters']
            if b['mode']=='tail':new_preprocessing(b['inner'],cells)
            elif b['mode']=='reference_core':core_preprocessing(b['inner'],cells)
            else:
                nca_preprocessing(b['inner']['global_model'],cells,dev)
                for protocol,local in b['inner']['local'].items():
                    ids=[k for k in dev if condition(cells[k])==protocol];assert len(ids)>=6;nca_preprocessing(local,cells,ids)
            inner=b['inner'];assert inner['train_ids']==dev
            rows={r['cell_id']:r for r in json.loads((dest/'test_metrics.json').read_text())};assert set(rows)==set(plan['test'])
            with np.load(dest/'test_predictions.npz') as a:
                for key in plan['test']:
                    c=cells[key];x,y=labels(c)
                    np.testing.assert_array_equal(x,a[key+'__100__cycle']);np.testing.assert_array_equal(y,a[key+'__100__truth'])
                    remove=('cycle','truth') if is_nca(c) else ('soh','label_cycle','unfiltered_soh','reference_capacity_ah')
                    clean={k:v for k,v in c.items() if k not in remove};p=predict(b,clean,x)
                    np.testing.assert_allclose(p,a[key+'__100__prediction'],atol=1e-12,rtol=0)
                    if is_nca(c):
                        poison=dict(clean,early_cycle=np.r_[clean['early_cycle'],101,10000],
                            early_raw=np.vstack([clean['early_raw'],np.full((2,clean['early_raw'].shape[1]),np.inf)]),prefix=np.full_like(clean['prefix'],np.inf),truth=np.inf,cycle_life=np.inf)
                    else:
                        poison=dict(clean,cycle=np.r_[clean['cycle'],101,10000],
                            raw=np.vstack([clean['raw'],np.full((2,clean['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                    late=x>=(x[0]+x[-1])/2;total=np.sum((y-y.mean())**2)
                    values=dict(mape_percent=100*np.mean(abs(p-y)/y),mae_soh=np.mean(abs(p-y)),
                        rmse_soh=np.sqrt(np.mean((p-y)**2)),late_mape=100*np.mean(abs(p[late]-y[late])/y[late]),
                        endpoint_pp=100*abs(p[-1]-y[-1]),drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1])))
                    if total>1e-12:values['r2']=1-np.sum((p-y)**2)/total
                    for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                    checked.append(dict(cell_id=key,fold=plan['name'],seed=seed,model_sha256=sha(dest/'model.joblib'),
                        prediction_sha256=sha(dest/'test_predictions.npz'),targets=len(y)))
            models+=1;print('PASS '+plan['name']+' '+str(seed),flush=True)
    expected_cells={k for p in original['plans'] for k in p['test']}
    assert len(checked)==len(expected_cells)*5 and {r['cell_id'] for r in checked}==expected_cells
    save_json(root/'independent_outer_verification.json',dict(passed=True,models=models,rows=checked,
        source_sha256=sha(__file__),validation_only_choice_independently_recomputed=True,
        training_only_preprocessing=True,full_future_coverage=True,label_and_future_poison=True,goal_complete=False))
    print('MODELS='+str(models)+' ROWS='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
