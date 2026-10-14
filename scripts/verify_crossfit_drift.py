"""OOF membership, residual reconstruction, closed-form Ridge and full replay."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import labels,predict as base_predict
from battery_smetimes.crossfit_drift import basis,predict
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.verify_tail_boost import preprocessing
from scripts.audit_tail_numeric_gates import metric
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for file in ('battery_smetimes/crossfit_drift.py','battery_smetimes/tail_boost.py'):
        assert sha(file)==cfg['source_sha256'][Path(file).name]
    for entry in cfg['provenance']:assert sha(entry['file'])==entry['sha256']
    reports=json.loads((root/'summary.json').read_text());checked=[];oof_count=0
    for plan in cfg['plans']:
        folder=root/plan['name'];cells=load_cells(plan)
        assert not set(cells)&set(plan['test'])
        splits=json.loads((folder/'oof_splits.json').read_text())
        assert len(splits)==2 and sorted(k for s in splits for k in s['held'])==sorted(plan['train'])
        xs=[];rs=[];ws=[]
        with np.load(folder/'oof_residuals.npz') as arrays:
            for index,s in enumerate(splits):
                assert not set(s['train'])&set(s['held'])
                assert set(s['train']+s['held'])==set(plan['train'])
                b=joblib.load(folder/f'oof_{index}.joblib');assert b['train_ids']==s['train']
                assert b['parameters']==cfg['parameters'];preprocessing(b,cells)
                for key in s['held']:
                    x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int));x,y=x[idx],y[idx]
                    p=base_predict(b,clean(cells[key]),x)
                    for f,v in [('cycle',x),('truth',y),('prediction',p)]:np.testing.assert_allclose(v,arrays[key+'__'+f],atol=1e-12,rtol=0)
                    xs.append(x);rs.append(y-p);w=np.linspace(1,3,len(x));ws.append(w/w.sum());oof_count+=1
            x,r,w=np.concatenate(xs),np.concatenate(rs),np.concatenate(ws);w*=len(w)/w.sum()
            for f,v in [('cycle',x),('residual',r),('weight',w)]:np.testing.assert_allclose(v,arrays['aggregate_'+f],atol=1e-12,rtol=0)
        for index,candidate in enumerate(cfg['candidates']):
            dest=folder/f'candidate_{index}';b=joblib.load(dest/'model.joblib')
            assert b['train_ids']==plan['train'] and b['base']['train_ids']==plan['train']
            preprocessing(b['base'],cells)
            if candidate is None:assert b['head'] is None
            else:
                head=b['head'];assert head['degree']==candidate['degree'] and head['alpha']==candidate['alpha']
                z=basis(x,head['degree']);mean=z.mean(axis=0);var=z.var(axis=0)
                np.testing.assert_allclose(mean,head['scaler'].mean_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(var,head['scaler'].var_,atol=1e-12,rtol=0)
                z=(z-mean)/np.sqrt(var);zm=np.average(z,axis=0,weights=w);rm=np.average(r,weights=w)
                zz=z-zm;rr=r-rm
                coef=np.linalg.solve((zz*w[:,None]).T@zz+head['alpha']*np.eye(z.shape[1]),(zz*w[:,None]).T@rr)
                np.testing.assert_allclose(coef,head['model'].coef_,atol=1e-10,rtol=0)
                np.testing.assert_allclose(rm-zm@coef,head['model'].intercept_,atol=1e-10,rtol=0)
            stored={v['cell_id']:v for v in json.loads((dest/'validation_metrics.json').read_text())}
            assert set(stored)==set(plan['validation']);scores=[]
            with np.load(dest/'validation_predictions.npz') as arrays:
                for key in plan['validation']:
                    xx,yy=labels(cells[key]);pp=predict(b,clean(cells[key]),xx)
                    for f,v in [('cycle',xx),('truth',yy),('prediction',pp)]:np.testing.assert_allclose(v,arrays[key+'__100__'+f],atol=1e-12,rtol=0)
                    values=metric(xx,yy,pp)
                    for f,v in values.items():np.testing.assert_allclose(v,stored[key][f],atol=1e-10,rtol=0)
                    scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                    checked.append(dict(cell_id=key,candidate=index,targets=len(xx),model_sha256=sha(dest/'model.joblib'),prediction_sha256=sha(dest/'validation_predictions.npz')))
            report=next(v for v in reports if v['dataset']==plan['dataset'] and v.get('fold',plan['name'])==plan['name'] and v['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    save_json(root/'independent_verification.json',dict(passed=True,rows=checked,oof_cells_replayed=oof_count,
        whole_cell_oof=True,ridge_closed_form_checked=True,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False))
    print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
