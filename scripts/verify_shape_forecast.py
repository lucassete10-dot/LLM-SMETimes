"""Independent polynomial shape algebra, training preprocessing and estimator replay."""
import argparse,json,warnings
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.shape_forecast import fit,predict
from battery_smetimes.waveform_shape import vector,matrix
from battery_smetimes.tail_boost import labels,is_nca
from scripts.verify_conditional_drift import manual_scale
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def independent_vector(cell):
    nca=is_nca(cell);cycle=np.asarray(cell['early_cycle'] if nca else cell['cycle'],dtype=float)
    raw=np.asarray(cell['early_raw'] if nca else cell['raw'],dtype=float);take=(cycle>0)&(cycle<=100);c=cycle[take];raw=raw[take]
    if nca:columns=[np.arange(11),np.arange(16,27)];reference=5
    elif cell['metadata']['dataset']=='XJTU':columns=[np.arange(a,a+5) for a in (0,5,10,23,28,33)];reference=5
    else:columns=[np.arange(11),np.arange(11,22),np.arange(22,33)];reference=1
    pieces=[]
    for col in columns:
        t=np.linspace(-1,1,len(col));design=np.column_stack([np.ones(len(t)),t,(3*t*t-1)/2,(5*t**3-3*t)/2]);result=[]
        for row in raw[:,col]:
            ok=np.isfinite(row)
            if ok.sum()<4:result.append(np.full(3,np.nan));continue
            value=np.linalg.lstsq(design[ok],row[ok],rcond=None)[0][1:];threshold=64*np.finfo(float).eps*max(1.,np.max(np.abs(row[ok])))
            value[np.abs(value)<=threshold]=0;result.append(value)
        pieces.append(result)
    z=np.column_stack(pieces)
    def trend(t,x):
        slopes=[]
        for j in range(x.shape[1]):
            ok=np.isfinite(x[:,j])
            slopes.append(np.linalg.lstsq(np.c_[t[ok]/100,np.ones(ok.sum())],x[ok,j],rcond=None)[0][0] if ok.sum()>=2 else np.nan)
        return slopes
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        first=np.nanmedian(z[:reference],axis=0);last=np.nanmedian(z[-3:],axis=0);spread=np.nanstd(z,axis=0)
    return np.r_[first,last,last-first,trend(c,z),spread,trend(c[-10:],z[-10:])]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256']
    assert sha(cfg.get('preregistration_file','docs/soh_waveform_shape_stage_20261005.md'))==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for f in ('shape_forecast','waveform_shape','tail_boost','research_baselines','forecast'):
        p=Path('battery_smetimes')/(f+'.py');assert sha(p)==cfg['source_sha256'][p.name]
    fivefold=cfg.get('fivefold_shape',False)
    expected=[None,dict(kind='extra_trees',alpha=None)] if fivefold else [None,dict(kind='extra_trees',alpha=None),dict(kind='ridge',alpha=10),dict(kind='ridge',alpha=100)]
    assert cfg['candidates']==expected
    if fivefold:
        frozen=json.loads((source/'config.json').read_text())
        assert cfg['plans']==frozen['plans'] and len(cfg['plans'])==5
        assert all(p['dataset']=='XJTU' for p in cfg['plans']) and cfg['seed']==20261005
    reports=json.loads((root/'summary.json').read_text());checked=[];shape_checks=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        control_path=Path(cfg['control_paths'][plan['name']]) if fivefold else source/plan['name']/'base.joblib'
        assert sha(control_path)==cfg['base_sha256'][plan['name']]
        for key,cell in cells.items():
            expected=independent_vector(clean(cell));np.testing.assert_allclose(vector(clean(cell)),expected,atol=1e-10,rtol=0,equal_nan=True)
            shape_checks.append(dict(cell_id=key,features=len(expected)))
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib');assert b['train_ids']==plan['train']
            if params is None:
                assert b['base']['train_ids']==plan['train'] and b['base']['seed']==cfg['seed']
                for key in plan['validation']:
                    cc=clean(cells[key]);x,_=labels(cells[key])
                    original=dict(mode='control',base=joblib.load(control_path),train_ids=plan['train'])
                    np.testing.assert_allclose(predict(b,cc,x),predict(original,cc,x),atol=1e-12,rtol=0)
            if params is not None:
                assert b['seed']==cfg['seed'] and b['kind']==params['kind'] and b['alpha']==params['alpha']
                xs=[];ys=[];ws=[]
                for key in plan['train']:
                    x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
                    xs.append(matrix(clean(cells[key]),x[idx],True));ys.append(y[idx]);w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
                z,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum();median=[]
                for j in range(z.shape[1]):
                    v=z[:,j][np.isfinite(z[:,j])];median.append(np.median(v) if len(v) else 0)
                median=np.asarray(median);np.testing.assert_allclose(median,b['imputer'].statistics_,atol=1e-12,rtol=0);z=np.where(np.isfinite(z),z,median)
                replay=fit(cells,plan['train'],seed=cfg['seed'],**params)
                if params['kind']=='ridge':
                    offset=z[0];shift=z-offset;mu=np.average(shift,axis=0,weights=w);var=np.average((shift-mu)**2,axis=0,weights=w);scale=manual_scale(var,mu,w.sum())
                    for expected,actual in [(mu+offset,b['scaler'].mean_),(var,b['scaler'].var_),(scale,b['scaler'].scale_)]:np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
                    xx=b['scaler'].transform(z);xm=np.average(xx,axis=0,weights=w);ym=np.average(y,weights=w);xc=xx-xm;yc=y-ym;coef=b['model'].coef_
                    gradient=xc.T@(w*(xc@coef-yc))+params['alpha']*coef;rhs=xc.T@(w*yc)
                    assert np.linalg.norm(gradient)<=1e-8*max(1.,np.linalg.norm(rhs))
                    assert b['model'].solver=='svd' and b['model'].alpha==params['alpha']
                else:assert b['model'].n_estimators==128 and b['model'].min_samples_leaf==1 and b['model'].max_depth is None
            rows={r['cell_id']:r for r in json.loads((folder/'validation_metrics.json').read_text())};assert set(rows)==set(plan['validation']);values_all=[];scores=[]
            with np.load(folder/'validation_predictions.npz') as arrays:
                assert len(arrays.files)==3*len(rows)
                for key in plan['validation']:
                    cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x)
                    np.testing.assert_array_equal(x,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth']);np.testing.assert_allclose(p,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                    if params is not None:np.testing.assert_allclose(p,predict(replay,cc,x),atol=1e-12,rtol=0)
                    if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                    else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                    try:predict(b,dict(cc,physical_family='different'),x)
                    except ValueError:pass
                    else:raise AssertionError('family isolation failed')
                    values=metric(x,y,p)
                    for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                    values_all.append(values);scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                    checked.append(dict(cell_id=key,candidate=index,points=len(x),model_sha256=sha(folder/'model.joblib'),arrays_sha256=sha(folder/'validation_predictions.npz')))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):np.testing.assert_allclose(np.mean([v[f] for v in values_all]),report[f],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    expected_count=sum(len(p['validation'])*len(cfg['candidates']) for p in cfg['plans'])
    assert len(checked)==expected_count
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,shape_checks=shape_checks,retrained_models=sum(p is not None for p in cfg['candidates'])*len(cfg['plans']),source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False));print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
