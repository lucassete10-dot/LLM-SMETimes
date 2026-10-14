"""Whole inner-set replay and independently reconstructed train imputers."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import train_arrays,predict,labels,is_nca,conditions
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import sha,save_json


def preprocessing(b,cells):
    x,_,_=train_arrays(cells,b['train_ids'],b['parameters']['tail_weight'])
    expected=SimpleImputer(strategy='median',keep_empty_features=True).fit(x.astype(float)).statistics_
    np.testing.assert_allclose(expected,b['imputer'].statistics_,atol=1e-12,rtol=0)
    if b['parameters']['model']=='hist_gradient_boosting':assert b['model'].early_stopping is False
    expected_groups={p for p in {conditions(cells[k]) for k in b['train_ids']}
        if sum(conditions(cells[k])==p for k in b['train_ids'])>=6} if b['parameters']['local'] else set()
    assert set(b['local'])==expected_groups
    for protocol,local in b['local'].items():
        ids=[k for k in b['train_ids'] if conditions(cells[k])==protocol]
        assert local['train_ids']==ids;preprocessing(local,cells)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run;cfg=json.loads((run/'config.json').read_text())
    assert json.loads((run/'progress.json').read_text())['training_complete']
    for name,digest in cfg['source_sha256'].items():assert sha(run/'source_snapshot'/name)==digest
    count=0;results=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert not set(cells)&set(plan['test'])
        for seed in cfg['seeds']:
            scores=[]
            for i,parameters in enumerate(cfg['candidates']):
                folder=run/plan['name']/str(seed)/f'candidate_{i}';b=joblib.load(folder/'model.joblib')
                assert b['train_ids']==plan['train'] and b['parameters']==parameters;preprocessing(b,cells)
                rows=json.loads((folder/'validation_metrics.json').read_text());assert {r['cell_id'] for r in rows}==set(plan['validation'])
                with np.load(folder/'validation_predictions.npz') as a:
                    for row in rows:
                        key=row['cell_id'];c=cells[key];x,y=labels(c)
                        np.testing.assert_array_equal(x,a[key+'__100__cycle']);np.testing.assert_array_equal(y,a[key+'__100__truth'])
                        remove=('cycle','truth') if is_nca(c) else ('soh','label_cycle','unfiltered_soh','reference_capacity_ah')
                        clean={k:v for k,v in c.items() if k not in remove};p=predict(b,clean,x)
                        np.testing.assert_allclose(p,a[key+'__100__prediction'],atol=1e-12,rtol=0)
                        late=x>=(x[0]+x[-1])/2;total=np.sum((y-y.mean())**2)
                        metrics=dict(mape_percent=100*np.mean(abs(p-y)/y),mae_soh=np.mean(abs(p-y)),
                            rmse_soh=np.sqrt(np.mean((p-y)**2)),late_mape=100*np.mean(abs(p[late]-y[late])/y[late]),
                            endpoint_pp=100*abs(p[-1]-y[-1]),drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1])))
                        if total>1e-12:metrics['r2']=1-np.sum((p-y)**2)/total
                        for f,v in metrics.items():np.testing.assert_allclose(v,row[f],atol=1e-10,rtol=0)
                        count+=1
                values=[(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]
                score=float(np.mean(values)+.25*np.quantile(values,.9));scores.append((score,i,rows))
            selected=json.loads((run/plan['name']/str(seed)/'selection.json').read_text())['selected']
            best=min(scores,key=lambda x:x[0]);assert selected['candidate']==best[1]
            np.testing.assert_allclose(selected['selection_score'],best[0],atol=1e-12,rtol=0)
            results.append(dict(fold=plan['name'],seed=seed,selected=selected))
            print('PASS '+plan['name']+' '+str(seed),flush=True)
    save_json(run/'independent_verification.json',dict(passed=True,checked_curves=count,
        selected=results,source_sha256=sha(__file__),training_only_imputers=True,
        cell_split_checked=True,row_random_validation_disabled=True,outer_test_loaded=False,goal_complete=False))
    print('CHECKED='+str(count))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
