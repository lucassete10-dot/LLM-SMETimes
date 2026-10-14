"""Independent arithmetic from NPZ: no training metric/aggregation imports."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


FIELDS=('mape_percent','late_mape','rmse_soh','endpoint_pp','drop_error_pp','truth_drop_pp','prediction_drop_pp')


def metrics(cycle,truth,prediction):
    assert cycle.ndim==truth.ndim==prediction.ndim==1 and len(cycle)==len(truth)==len(prediction)>=2
    assert np.isfinite(cycle).all() and np.all(np.diff(cycle)>0) and cycle[0]>100
    assert np.isfinite(truth).all() and (truth>0).all() and np.isfinite(prediction).all()
    residual=prediction-truth;late=cycle>=(cycle[0]+cycle[-1])/2
    values=(100*np.average(abs(residual)/truth),100*np.average(abs(residual[late])/truth[late]),np.sqrt(np.average(residual**2)),100*abs(residual[-1]),100*abs(residual[0]-residual[-1]),100*(truth[0]-truth[-1]),100*(prediction[0]-prediction[-1]))
    return dict(zip(FIELDS,map(float,values)))


def summarize(rows):
    ids=sorted({r['cell_id'] for r in rows});seeds=sorted({r['seed'] for r in rows});assert len(seeds)==5
    assert len(rows)==len(ids)*5 and len({(r['cell_id'],r['seed']) for r in rows})==len(rows)
    tensor=np.empty((len(ids),5,len(FIELDS)))
    for i,key in enumerate(ids):
        for j,seed in enumerate(seeds):
            r=next(r for r in rows if r['cell_id']==key and r['seed']==seed);tensor[i,j]=[r[f] for f in FIELDS]
    means=tensor.mean(axis=1);cells=[]
    for key,v in zip(ids,means):
        c=dict(cell_id=key,**dict(zip(FIELDS,map(float,v))))
        c['joint_pass']=all(c[f]<=threshold for f,threshold in [('mape_percent',6),('late_mape',8),('endpoint_pp',8),('drop_error_pp',5)])
        c['severe_flat_failure']=c['truth_drop_pp']>=5 and c['prediction_drop_pp']<.3*c['truth_drop_pp'];cells.append(c)
    summary={f:dict(mean=float(means[:,i].mean()),seed_sd=float(tensor[:,:,i].mean(axis=0).std(ddof=1))) for i,f in enumerate(FIELDS)}
    summary.update(cells=len(ids),joint_pass_fraction=sum(c['joint_pass'] for c in cells)/len(ids),severe_flat_fraction=sum(c['severe_flat_failure'] for c in cells)/len(ids),endpoint_p90=float(np.quantile(means[:,3],.9)),drop_error_p90=float(np.quantile(means[:,4],.9)))
    return summary,cells


def checks(new,old):
    result={name:new[field]['mean']<=threshold for name,field,threshold in [('full_mape','mape_percent',6),('late_mape','late_mape',6),('rmse','rmse_soh',.04),('endpoint_mean','endpoint_pp',5),('drop_mean','drop_error_pp',4)]}
    result.update(joint_fraction=new['joint_pass_fraction']>=.9,severe_fraction=new['severe_flat_fraction']<=.1,full_no_regression=new['mape_percent']['mean']<=old['mape_percent']['mean']+.5)
    for field in ('endpoint_p90','drop_error_p90'):result[field]=new[field]<=(3 if old[field]<=3 else .8*old[field])
    return result


def assert_summary(actual,stored):
    assert actual.keys()==stored.keys()
    for field,value in actual.items():
        if isinstance(value,dict):
            assert value.keys()==stored[field].keys()
            for key,v in value.items():np.testing.assert_allclose(v,stored[field][key],atol=1e-10,rtol=0)
        else:np.testing.assert_allclose(value,stored[field],atol=1e-10,rtol=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());stored=json.loads((root/'summary.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete'];assert cfg['dataset']=='MIT'
    baseline=Path(cfg['baseline']);primary=Path(cfg['current_primary'])
    assert sha(baseline/'config.json')==cfg['baseline_config_sha256'] and sha(primary/'config.json')==cfg['current_primary_config_sha256']
    records={k:[] for k in ('new','baseline','current_primary')};evidence=[];all_ids=[]
    for plan in cfg['plans']:
        assert not set(plan['train']+plan['validation'])&set(plan['test']);all_ids.extend(plan['test']);reference={}
        for name,path,seeds,suffix in [('new',root,cfg['seeds'],'outer/test_predictions.npz'),('baseline',baseline,range(20261004,20261009),'test/predictions.npz'),('current_primary',primary,cfg['seeds'],'test_predictions.npz')]:
            for seed in seeds:
                file=path/plan['name']/str(seed)/suffix
                with np.load(file) as arrays:
                    expected={key+'__100__'+f for key in plan['test'] for f in ('cycle','truth','prediction')};assert expected<=set(arrays.files)
                    if name=='new':assert expected==set(arrays.files)
                    for key in plan['test']:
                        cycle,truth,p=[arrays[key+'__100__'+f] for f in ('cycle','truth','prediction')]
                        if key not in reference:reference[key]=(cycle.copy(),truth.copy())
                        else:
                            np.testing.assert_array_equal(cycle,reference[key][0]);np.testing.assert_array_equal(truth,reference[key][1])
                        records[name].append(dict(cell_id=key,seed=seed,**metrics(cycle,truth,p)))
                evidence.append(dict(file=str(file.resolve()),sha256=sha(file),cohort=name))
    assert len(all_ids)==len(set(all_ids))==134 and all(len(v)==670 for v in records.values())
    summaries={};cells=None
    for name,rows in records.items():
        summary,cc=summarize(rows);assert_summary(summary,stored[name]);summaries[name]=summary
        if name=='new':cells=cc
    gates=checks(summaries['new'],summaries['baseline']);assert gates==stored['gates'] and all(gates.values())==stored['numeric_gates_passed']
    saved_cells={c['cell_id']:c for c in json.loads((root/'by_cell.json').read_text())};assert set(saved_cells)==set(all_ids)
    for c in cells:
        for f,v in c.items():
            if f!='cell_id':np.testing.assert_allclose(v,saved_cells[c['cell_id']][f],atol=1e-10,rtol=0)
    failures=[]
    for c in cells:
        violations=[f for f,t in [('mape_percent',6),('late_mape',8),('endpoint_pp',8),('drop_error_pp',5)] if c[f]>t]
        if violations or c['severe_flat_failure']:failures.append(dict(**c,failed_cell_fields=violations))
    save_json(root/'independent_numeric_audit.json',dict(passed=True,scope='arithmetic only; not a complete model/input/figure audit',summary=summaries,gates=gates,numeric_gates_passed=all(gates.values()),failed_cells=failures,evidence=evidence,source_sha256=sha(__file__),goal_complete=False))
    print('ARITHMETIC_PASS gates='+str(sum(gates.values()))+'/10 joint='+str(sum(c['joint_pass'] for c in cells))+'/134 failures='+str(len(failures)),flush=True)


if __name__=='__main__':main()
