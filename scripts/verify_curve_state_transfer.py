"""Replay every stored inner prediction; reconstruct train-only imputers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from battery_smetimes.curve_state_transfer import predict, fade_features
from battery_smetimes.curve_fidelity import targets, selection_score
from battery_smetimes.research_baselines import signal_prefix
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import save_json, sha


def matrices(cells,ids):
    state=[]; fade=[]
    for key in ids:
        c=cells[key]; good=np.isfinite(c['soh'])&(c['soh']>0)
        for i in np.flatnonzero(good&(c['label_cycle']>=10)&(c['label_cycle']<=100)):
            state.append(signal_prefix(c,float(c['label_cycle'][i])))
        for origin in range(20,101,10):
            if np.sum(good&(c['label_cycle']==origin))!=1:
                continue
            take=np.flatnonzero(good&(c['label_cycle']>origin))
            if len(take)<2:
                continue
            if len(take)>96:
                take=take[np.unique(np.linspace(0,len(take)-1,96).round().astype(int))]
            fade.append(fade_features(signal_prefix(c,origin),origin,c['label_cycle'][take]))
    return dict(state=np.stack(state).astype(float),fade=np.concatenate(fade))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(); run=args.run
    cfg=json.loads((run/'config.json').read_text())
    for name,digest in cfg['source_sha256'].items():
        assert sha(run/'source_snapshot'/name)==digest
    # Current replay implementation must match archived predictor implementation.
    assert sha(Path('battery_smetimes/curve_state_transfer.py'))==cfg['source_sha256']['curve_state_transfer.py']
    checked=[]
    for plan in cfg['plans']:
        cells=load_subset(plan['train']+plan['validation'])
        assert not set(cells)&set(plan['test'])
        training=matrices(cells,plan['train'])
        scores=[]
        for index,parameters in enumerate(cfg['candidates']):
            folder=run/plan['name']/f'candidate_{index}'
            bundle=joblib.load(folder/'model.joblib')
            assert bundle['train_ids']==plan['train'] and bundle['parameters']==parameters
            for head,x in training.items():
                imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
                np.testing.assert_allclose(imputer.statistics_,bundle[head]['imputer'].statistics_,atol=1e-12,rtol=0)
            rows=json.loads((folder/'validation_metrics.json').read_text())
            with np.load(folder/'validation_predictions.npz') as saved:
                for row in rows:
                    key=row['cell_id']; c=cells[key]; cycle,y=targets(c)
                    for field,expected in [('cycle',cycle),('truth',y)]:
                        np.testing.assert_array_equal(saved[key+'__100__'+field],expected)
                    clean={k:v for k,v in c.items() if k not in ('soh','label_cycle','reference_capacity_ah')}
                    p=predict(bundle,clean,cycle)
                    np.testing.assert_allclose(p,saved[key+'__100__prediction'],atol=1e-12,rtol=0)
                    late=cycle>=(cycle[0]+cycle[-1])/2
                    independent=dict(mape_percent=100*np.mean(abs(p-y)/y),
                        mae_soh=np.mean(abs(p-y)),rmse_soh=np.sqrt(np.mean((p-y)**2)),
                        late_mape=100*np.mean(abs(p[late]-y[late])/y[late]),
                        endpoint_pp=100*abs(p[-1]-y[-1]),
                        drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1])))
                    for field,value in independent.items():
                        np.testing.assert_allclose(value,row[field],atol=1e-10,rtol=0)
                    checked.append(dict(fold=plan['name'],candidate=index,cell_id=key))
            scores.append(dict(candidate=index,score=selection_score(rows)))
        chosen=json.loads((run/plan['name']/'selection.json').read_text())['selected']
        assert chosen['candidate']==min(scores,key=lambda r:r['score'])['candidate']
    save_json(run/'independent_verification.json',dict(passed=True,checked=checked,
        models=len(checked),training_only_medians_recomputed=True,outer_cells_loaded=False,
        timestamp=datetime.now(timezone.utc).isoformat()))
    print('CHECKED='+str(len(checked)))


if __name__=='__main__':
    main()
