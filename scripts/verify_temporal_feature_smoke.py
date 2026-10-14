"""Reconstruct preprocessing and replay every paired validation curve."""
import argparse, json
from pathlib import Path
import joblib, numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.temporal_features import matrix
from battery_smetimes.tail_boost import labels, is_nca
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.audit_tail_numeric_gates import metric


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for file in ('battery_smetimes/temporal_features.py','battery_smetimes/tail_boost.py','battery_smetimes/forecast.py'):
        assert sha(file)==cfg['source_sha256'][Path(file).name]
    for entry in cfg['provenance']:assert sha(entry['file'])==entry['sha256']
    reports=json.loads((root/'summary.json').read_text()); checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert not set(cells)&set(plan['test'])
        for augmented in cfg['candidates']:
            folder=root/plan['name']/str(int(augmented));b=joblib.load(folder/'model.joblib')
            assert b['train_ids']==plan['train'] and b['augmented']==augmented
            parts=[]
            for key in plan['train']:
                x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
                parts.append(matrix(cells[key],x[idx],augmented))
            z=np.concatenate(parts);expected=np.zeros(z.shape[1])
            for j in range(z.shape[1]):
                finite=z[:,j][np.isfinite(z[:,j])];expected[j]=np.median(finite) if len(finite) else 0
            np.testing.assert_allclose(expected,b['imputer'].statistics_,atol=1e-12,rtol=0)
            rows={r['cell_id']:r for r in json.loads((folder/'validation_metrics.json').read_text())}
            assert set(rows)==set(plan['validation']);scores=[];replayed=[]
            with np.load(folder/'validation_predictions.npz') as arrays:
                for key in plan['validation']:
                    c=cells[key];assert c['physical_family']==b['family'];x,y=labels(c)
                    np.testing.assert_array_equal(x,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth'])
                    remove=('cycle','truth') if is_nca(c) else ('soh','label_cycle','unfiltered_soh','reference_capacity_ah')
                    clean={k:v for k,v in c.items() if k not in remove}
                    p=b['model'].predict(b['imputer'].transform(matrix(clean,x,augmented)))
                    np.testing.assert_allclose(p,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                    values=metric(x,y,p)
                    for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                    scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                    replayed.append(values);checked.append(dict(cell_id=key,augmented=augmented,points=len(x),model_sha256=sha(folder/'model.joblib'),arrays_sha256=sha(folder/'validation_predictions.npz')))
            report=next(r for r in reports if r['dataset']==plan['dataset'] and r['augmented']==augmented)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['validation_score'],atol=1e-10,rtol=0)
            for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):
                np.testing.assert_allclose(np.mean([r[f] for r in replayed]),report[f],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' augmented='+str(augmented),flush=True)
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False))
    print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
