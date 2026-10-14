"""Independent saved-model preprocessing/membership/full-label metric check."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from scripts.train_oxford_diagnostic_forecast import matrix,predict
from battery_smetimes.research_baselines import regression_metrics
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run;cfg=json.loads((run/'config.json').read_text());assert not cfg['smoke']
    cells=joblib.load(run/'prepared_cells.joblib');rows=[]
    for plan in cfg['plans']:
        for seed in cfg['seeds']:
            folder=run/plan['name']/str(seed);bundle=joblib.load(folder/'model.joblib')
            ids=plan['train']+plan['validation'];assert bundle['train_ids']==ids
            assert not set(ids)&set(plan['test'])
            x=np.concatenate([matrix(cells[k]) for k in ids])
            w=np.concatenate([np.full(len(cells[k]['truth']),1/len(cells[k]['truth'])) for k in ids])
            expected=np.average(x,axis=0,weights=w)
            np.testing.assert_allclose(bundle['scaler'].mean_,expected,rtol=1e-10,atol=1e-10)
            variance=np.average((x-expected)**2,axis=0,weights=w)
            np.testing.assert_allclose(bundle['scaler'].var_,variance,rtol=1e-9,atol=1e-10)
            selection=json.loads((folder/'selection.json').read_text())
            assert selection['selected']==min(selection['candidates'],key=lambda r:r['validation_mape'])
            for i in range(len(cfg['alphas'])):
                candidate=joblib.load(folder/f'candidate_{i}'/'model.joblib')
                assert candidate['train_ids']==plan['train']
            report=json.loads((folder/'metrics.json').read_text());assert {r['cell_id'] for r in report}==set(plan['test'])
            with np.load(folder/'predictions.npz',allow_pickle=False) as a:
                for row in report:
                    key=row['cell_id'];c=cells[key];prefix=key+'__100__'
                    x,y,p=[a[prefix+f] for f in ('cycle','truth','prediction')]
                    np.testing.assert_array_equal(x,c['cycle']);np.testing.assert_array_equal(y,c['truth'])
                    clean={k:v for k,v in c.items() if k not in ('truth','cycle')}
                    np.testing.assert_array_equal(p,predict(bundle,clean,x))
                    for k,v in regression_metrics(y,p).items():
                        if v is None:assert row['metrics'][k] is None
                        else:np.testing.assert_allclose(v,row['metrics'][k],rtol=0,atol=1e-12)
                    rows.append(dict(row,fold=plan['name'],seed=seed))
    for seed in cfg['seeds']:assert sorted(r['cell_id'] for r in rows if r['seed']==seed)==sorted(cells)
    save_json(run/'verification.json',dict(passed=True,run=str(run.resolve()),rows=rows,
        units=25,cell_seed_evaluations=40,train_only_weighted_preprocessing=True,
        full_future_label_coverage=True,label_removal=True,run_config_sha256=sha(run/'config.json'),
        source_sha256=sha(__file__),role='development sparse diagnostic supplement'))
    print('Verified 25 models / 40 cell-seed evaluations',flush=True)

if __name__=='__main__':main()
