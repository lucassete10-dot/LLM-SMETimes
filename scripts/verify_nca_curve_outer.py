"""Check every outer NCA curve, split, frozen selection and train-only median."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from scripts.train_nca_curve_fidelity import predict,condition
from scripts.verify_nca_curve_validation import preprocessing
from scripts.prepare_zhu2022_nca import prefix
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['validation_source']);assert sha(source/'config.json')==cfg['validation_config_sha256']
    original=json.loads((source/'config.json').read_text())
    data=Path(cfg['cells_directory']);assert sha(data/'index.json')==cfg['index_sha256']
    index=json.loads((data/'index.json').read_text())
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for p in (Path('scripts/train_nca_curve_fidelity.py'),Path('scripts/train_zhu2022_nca.py'),
            Path('scripts/prepare_zhu2022_nca.py'),Path('battery_smetimes/curve_fidelity.py')):
        assert sha(p)==cfg['source_sha256'][p.name], 'Replay implementation changed'
    checked=[];models=0;recomputed=[]
    for plan in original['plans']:
        cells={}
        for key in plan['train']+plan['validation']+plan['test']:
            entry=index['cells'][key];p=data/entry['file'];assert sha(p)==entry['sha256'];c=joblib.load(p)
            assert np.all(c['early_cycle']<=100)
            np.testing.assert_array_equal(c['prefix'],prefix(c['early_cycle'],c['early_raw'],c['conditions']));cells[key]=c
        dev=plan['train']+plan['validation'];assert not set(dev)&set(plan['test'])
        for seed in original['seeds']:
            assert sha(source/plan['name']/str(seed)/'selection.json')==cfg['selection_sha256'][plan['name']][str(seed)]
            dest=root/plan['name']/str(seed);chosen=json.loads((dest/'selection.json').read_text())
            assert chosen==json.loads((source/plan['name']/str(seed)/'selection.json').read_text())
            assert chosen['selected']==min(chosen['candidates'],key=lambda r:r['selection_score'])
            b=joblib.load(dest/'model.joblib');assert b['train_ids']==dev and b['parameters']==chosen['selected']['parameters']
            preprocessing(b['global_model'],cells,dev)
            protocols={condition(cells[k]) for k in dev}
            expected_local={p for p in protocols if sum(condition(cells[k])==p for k in dev)>=6} if b['parameters']['local'] else set()
            assert set(b['local'])==expected_local
            for protocol,local in b['local'].items():
                ids=[k for k in dev if condition(cells[k])==protocol];assert len(ids)>=6;preprocessing(local,cells,ids)
            models+=1
            rows={r['cell_id']:r for r in json.loads((dest/'test_metrics.json').read_text())};assert set(rows)==set(plan['test'])
            with np.load(dest/'test_predictions.npz') as a:
                for key in plan['test']:
                    c=cells[key];x,y=c['cycle'],c['truth']
                    np.testing.assert_array_equal(x,a[key+'__100__cycle']);np.testing.assert_array_equal(y,a[key+'__100__truth'])
                    clean={k:v for k,v in c.items() if k not in ('cycle','truth')};p=predict(b,clean,x)
                    np.testing.assert_allclose(p,a[key+'__100__prediction'],atol=1e-12,rtol=0)
                    poison=dict(clean,prefix=np.full_like(clean['prefix'],np.inf),truth=np.inf,cycle_life=np.inf,
                        early_cycle=np.r_[clean['early_cycle'],101,10000],
                        early_raw=np.vstack([clean['early_raw'],np.full((2,clean['early_raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                    late=x>=(x[0]+x[-1])/2;total=np.sum((y-y.mean())**2)
                    values=dict(mape_percent=100*np.mean(abs(p-y)/y),mae_soh=np.mean(abs(p-y)),
                        rmse_soh=np.sqrt(np.mean((p-y)**2)),late_mape=100*np.mean(abs(p[late]-y[late])/y[late]),
                        endpoint_pp=100*abs(p[-1]-y[-1]),drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1])))
                    if total>1e-12:values['r2']=1-np.sum((p-y)**2)/total
                    for field,value in values.items():np.testing.assert_allclose(value,rows[key][field],atol=1e-10,rtol=0)
                    recomputed.append(dict(cell_id=key,seed=seed,**values,
                        truth_drop_pp=float(100*(y[0]-y[-1])),prediction_drop_pp=float(100*(p[0]-p[-1]))))
                    checked.append(dict(cell_id=key,fold=plan['name'],seed=seed,model_sha256=sha(dest/'model.joblib'),
                        predictions_sha256=sha(dest/'test_predictions.npz'),targets=len(y)))
            print('PASS '+plan['name']+' '+str(seed),flush=True)
    assert len(checked)==len(index['cells'])*5 and len({r['cell_id'] for r in checked})==len(index['cells'])
    cell_means=[]
    for key in index['cells']:
        selected=[r for r in recomputed if r['cell_id']==key];assert len(selected)==5
        cell_means.append({f:float(np.mean([r[f] for r in selected])) for f in
            ('mape_percent','late_mape','rmse_soh','endpoint_pp','drop_error_pp','truth_drop_pp','prediction_drop_pp')})
    baseline=[];base=Path(cfg['baseline']);basecfg=json.loads((base.parent/'config.json').read_text())
    for plan in original['plans']:
        for seed in basecfg['seeds']:
            with np.load(base/plan['name']/str(seed)/'predictions.npz') as a:
                for key in plan['test']:
                    y,p=[a[key+'__100__'+f] for f in ('truth','prediction')]
                    baseline.append(dict(cell_id=key,mape_percent=float(100*np.mean(abs(p-y)/y)),
                        endpoint_pp=float(100*abs(p[-1]-y[-1])),
                        drop_error_pp=float(100*abs((p[0]-p[-1])-(y[0]-y[-1])))))
    new={f:float(np.mean([r[f] for r in cell_means])) for f in
        ('mape_percent','late_mape','rmse_soh','endpoint_pp','drop_error_pp')}
    old_mape=float(np.mean([r['mape_percent'] for r in baseline]))
    gates=dict(full_mape=new['mape_percent']<=6,late_mape=new['late_mape']<=6,rmse=new['rmse_soh']<=.04,
        endpoint_mean=new['endpoint_pp']<=5,drop_mean=new['drop_error_pp']<=4,
        joint_fraction=np.mean([r['mape_percent']<=6 and r['late_mape']<=8 and r['endpoint_pp']<=8 and r['drop_error_pp']<=5 for r in cell_means])>=.9,
        severe_fraction=np.mean([r['truth_drop_pp']>=5 and r['prediction_drop_pp']<.3*r['truth_drop_pp'] for r in cell_means])<=.1,
        full_no_regression=new['mape_percent']<=old_mape+.5)
    for field,key in (('endpoint_pp','endpoint_p90'),('drop_error_pp','drop_error_p90')):
        old_p90=float(np.quantile([np.mean([r[field] for r in baseline if r['cell_id']==k]) for k in index['cells']],.9))
        gates[key]=np.quantile([r[field] for r in cell_means],.9)<=(3 if old_p90<=3 else .8*old_p90)
    gates={k:bool(v) for k,v in gates.items()}
    reported=json.loads((root/'summary.json').read_text());assert reported['gates']==gates
    for field,value in new.items():np.testing.assert_allclose(value,reported['new'][field]['mean'],atol=1e-12,rtol=0)
    save_json(root/'independent_outer_verification.json',dict(passed=True,models=models,rows=checked,
        source_sha256=sha(__file__),train_only_medians=True,original_selection_exact=True,
        prefix_raw_recomputed=True,future_and_label_poison=True,independent_gates=gates,goal_complete=False))
    print('MODELS='+str(models)+' ROWS='+str(len(checked)))


if __name__=='__main__':main()
