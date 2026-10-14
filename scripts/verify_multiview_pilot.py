"""Whole-scope saved-model replay, train-statistics and selection audit."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np
import torch
from battery_smetimes.multiview_pilot import views,predict
from battery_smetimes.curve_fidelity import targets
from scripts.audit_rate_blend_metrics import metrics as arithmetic
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_curve_fidelity import load_subset


def selection_score(rows):
    values=np.array([(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows])
    return float(values.mean()+.25*np.quantile(values,.9))


def curve_metrics(c,y,p):
    result=arithmetic(c,y,p);last=c>c[-1]-100
    result.update(mae_soh=float(np.mean(abs(p-y))),last100_mape=float(100*np.mean(abs(p[last]-y[last])/y[last])),
                  severe_flat_failure=bool(result['truth_drop_pp']>=5 and result['prediction_drop_pp']<.3*result['truth_drop_pp']),
                  future_targets=len(y),future_span=float(c[-1]-100),short_coverage=bool(len(y)<10 or c[-1]-100<100))
    result['joint_pass']=all(result[f]<=limit for f,limit in [('mape_percent',6),('late_mape',8),('endpoint_pp',8),('drop_error_pp',5)])
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'progress.json').read_text())['passed']
    for file,digest in cfg['source_sha256'].items():
        assert sha(file)==digest==sha(root/'source_snapshot'/Path(file).name)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    task_rows=[];curves=0;models=0
    for plan in cfg['plans']:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
        assert not set(plan['train'])&set(plan['validation'])
        for seed in cfg['seeds']:
            for mode in cfg['modes']:
                folder=root/plan['name']/str(seed)/mode;b=joblib.load(folder/'model.joblib')
                assert b['train_ids']==plan['train'] and b['validation_ids']==plan['validation']
                assert b['seed']==seed and b['mode']==mode and b['epochs']==cfg['epochs']
                for view,s in enumerate(b['preprocessing']):
                    x=np.stack([views(cells[k])[view] for k in plan['train']]);median=[]
                    for column in x.T:
                        finite=column[np.isfinite(column)];median.append(np.median(finite) if len(finite) else 0.)
                    median=np.array(median);filled=np.where(np.isnan(x),median,x)
                    np.testing.assert_array_equal(median,s['median'])
                    np.testing.assert_array_equal(filled.mean(axis=0),s['mean'])
                    np.testing.assert_array_equal(np.maximum(filled.std(axis=0),1e-6),s['scale'])
                for role in ('train','validation'):
                    saved=json.loads((folder/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in saved]==plan[role]
                    with np.load(folder/(role+'_predictions.npz')) as a:
                        assert set(a.files)=={k+'__100__'+f for k in plan[role] for f in ('cycle','truth','prediction')}
                        for row in saved:
                            key=row['cell_id'];c,y=targets(cells[key]);start=time.perf_counter();p=predict(b,cells[key],c)
                            milliseconds=(time.perf_counter()-start)*1000
                            for name,value in [('cycle',c),('truth',y),('prediction',p)]:np.testing.assert_array_equal(value,a[key+'__100__'+name])
                            metric=curve_metrics(c,y,p);e=p-y;ss=float(np.sum((y-y.mean())**2))
                            metric.update(r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),start_error_pp=float(100*abs(e[0])))
                            for name,value in metric.items():
                                if value is None:assert row[name] is None
                                else:np.testing.assert_allclose(value,row[name],rtol=0,atol=1e-12)
                            q=dict(cells[key],raw=cells[key]['raw'].copy());q['raw'][:,[38,39]]=np.inf
                            q.update(soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf)
                            q['cycle']=np.r_[q['cycle'],100000];q['raw']=np.vstack([q['raw'],np.full((1,51),np.inf)])
                            np.testing.assert_array_equal(p,predict(b,q,c))
                            try:predict(b,dict(cells[key],physical_family='WRONG'),c)
                            except ValueError:pass
                            else:raise AssertionError('family gate failed')
                            task_rows.append(dict(fold=plan['name'],seed=seed,mode=mode,role=role,
                                                  cell_id=key,inference_curve_ms=milliseconds,**metric));curves+=1
                history=json.loads((folder/'history.json').read_text());best=min(history,key=lambda r:r['validation_score'])
                assert best['epoch']==b['best_epoch'];np.testing.assert_allclose(best['validation_score'],b['best_validation_score'],rtol=0,atol=1e-12)
                val=[r for r in task_rows if r['fold']==plan['name'] and r['seed']==seed and r['mode']==mode and r['role']=='validation']
                np.testing.assert_allclose(selection_score(val),b['best_validation_score'],rtol=0,atol=1e-12)
                assert sum(p.numel() for p in b['model'].parameters())==b['parameter_count']
                models+=1
        print('VERIFIED '+plan['name'],flush=True)
    expected=len(cfg['plans'])*len(cfg['seeds'])*len(cfg['modes']);assert models==expected
    save_json(root/'verified_cell_metrics.json',task_rows)
    save_json(root/'independent_input_verification.json',dict(passed=True,models=models,curves=curves,
              train_only_statistics_rebuilt=True,all_saved_arrays_exact=True,capacity_future_poison_pass=True,
              source_sha256=sha(__file__),scope='saved weights/inputs/metrics/checkpoint ranking; not independent optimizer refits',
              goal_complete=False))


if __name__=='__main__':main()
