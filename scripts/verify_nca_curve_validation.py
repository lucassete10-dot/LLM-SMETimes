"""Independent full inner-array replay and training imputer check for NCA."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from scripts.train_zhu2022_nca import matrix
from scripts.train_nca_curve_fidelity import predict,condition
from scripts.prepare_zhu2022_nca import prefix
from scripts.prepare_same_type_forecast import sha,save_json


def preprocessing(model,cells,ids):
    assert model['train_ids']==ids
    xs=[]
    for key in ids:
        c=cells[key];idx=np.unique(np.linspace(0,len(c['truth'])-1,min(96,len(c['truth']))).round().astype(int))
        xs.append(matrix(c,c['cycle'][idx]))
    expected=SimpleImputer(strategy='median',keep_empty_features=True).fit(np.concatenate(xs)).statistics_
    np.testing.assert_allclose(expected,model['imputer'].statistics_,atol=1e-12,rtol=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run;cfg=json.loads((run/'config.json').read_text())
    assert json.loads((run/'progress.json').read_text())['training_complete']
    source=Path(cfg['cells_directory']);assert sha(source/'index.json')==cfg['index_sha256']
    index=json.loads((source/'index.json').read_text())
    for name,digest in cfg['source_sha256'].items():assert sha(run/'source_snapshot'/name)==digest
    count=0; selected_rows=[]
    for plan in cfg['plans']:
        cells={}
        for key in plan['train']+plan['validation']:
            entry=index['cells'][key];p=source/entry['file'];assert sha(p)==entry['sha256']
            c=joblib.load(p);np.testing.assert_array_equal(c['prefix'],prefix(c['early_cycle'],c['early_raw'],c['conditions']))
            cells[key]=c
        assert not set(cells)&set(plan['test'])
        for seed in cfg['seeds']:
            selection=json.loads((run/plan['name']/str(seed)/'selection.json').read_text())
            scores=[]
            for i,parameters in enumerate(cfg['candidates']):
                folder=run/plan['name']/str(seed)/f'candidate_{i}';b=joblib.load(folder/'model.joblib')
                assert b['parameters']==parameters and b['train_ids']==plan['train']
                preprocessing(b['global_model'],cells,plan['train'])
                for protocol,local in b['local'].items():
                    ids=[k for k in plan['train'] if condition(cells[k])==protocol];assert len(ids)>=6
                    preprocessing(local,cells,ids)
                rows=json.loads((folder/'validation_metrics.json').read_text());assert {r['cell_id'] for r in rows}==set(plan['validation'])
                with np.load(folder/'validation_predictions.npz') as a:
                    for row in rows:
                        key=row['cell_id'];c=cells[key];x,y=c['cycle'],c['truth']
                        np.testing.assert_array_equal(x,a[key+'__100__cycle']);np.testing.assert_array_equal(y,a[key+'__100__truth'])
                        clean={k:v for k,v in c.items() if k not in ('cycle','truth')}
                        p=predict(b,clean,x);np.testing.assert_allclose(p,a[key+'__100__prediction'],atol=1e-12,rtol=0)
                        poison=dict(clean,early_cycle=np.r_[clean['early_cycle'],101,10000],
                            early_raw=np.vstack([clean['early_raw'],np.full((2,clean['early_raw'].shape[1]),np.inf)]),
                            truth=np.inf,cycle_life=np.inf,prefix=np.full_like(clean['prefix'],np.inf))
                        np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                        late=x>=(x[0]+x[-1])/2;total=np.sum((y-y.mean())**2)
                        values=dict(mape_percent=100*np.mean(abs(p-y)/y),mae_soh=np.mean(abs(p-y)),
                            rmse_soh=np.sqrt(np.mean((p-y)**2)),late_mape=100*np.mean(abs(p[late]-y[late])/y[late]),
                            endpoint_pp=100*abs(p[-1]-y[-1]),drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1])))
                        if total>1e-12:values['r2']=1-np.sum((p-y)**2)/total
                        for field,value in values.items():np.testing.assert_allclose(row[field],value,atol=1e-10,rtol=0)
                        count+=1
                cell_scores=[(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]
                score=float(np.mean(cell_scores)+.25*np.quantile(cell_scores,.9));scores.append((score,i,rows))
            chosen=min(scores,key=lambda r:r[0]);assert selection['selected']['candidate']==chosen[1]
            np.testing.assert_allclose(chosen[0],selection['selected']['selection_score'],atol=1e-12,rtol=0)
            selected_rows.extend(dict(r,fold=plan['name'],seed=seed) for r in chosen[2])
    save_json(run/'independent_validation_verification.json',dict(passed=True,checked_curves=count,
        source_sha256=sha(__file__),train_only_medians=True,prefix_raw_recomputed=True,outer_test_loaded=False,
        selected_validation_rows=selected_rows,goal_complete=False))
    print('CHECKED='+str(count))
    print(json.dumps({f:float(np.mean([r[f] for r in selected_rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','joint_pass')}))


if __name__=='__main__':main()
