"""Independent early-label joins, train stats/refits and all future curves."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.pooled_anchor import fit,predict
from battery_smetimes.fade_rate import predict as control_predict
from battery_smetimes.tail_boost import inputs,labels
from scripts.verify_conditional_drift import manual_scale
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def manual_impute(x,imputer):
    x=np.asarray(x,dtype=float)
    med=[]
    for col in x.T:
        finite=col[np.isfinite(col)];med.append(np.median(finite) if len(finite) else 0.)
    med=np.asarray(med);np.testing.assert_allclose(med,imputer.statistics_,atol=1e-12,rtol=0)
    return np.where(np.isfinite(x),x,med)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete'];source=Path(cfg['source_run'])
    assert sha(source/'config.json')==cfg['source_config_sha256'] and cfg['plans']==json.loads((source/'config.json').read_text())['plans']
    assert cfg['candidates']==[None]+[dict(kind='ridge',alpha=a) for a in (1,10,100)]+[dict(kind='extra_trees',alpha=None)]
    assert sha('docs/soh_nasa_pooled_anchor_stage_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for name in ('pooled_anchor','fade_rate','conditional_drift','crossfit_drift','tail_boost','forecast','research_baselines'):
        assert sha(Path('battery_smetimes')/(name+'.py'))==cfg['source_sha256'][name+'.py']
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
        control=source/plan['name']/'20261005'/'inner'/'model.joblib';assert sha(control)==cfg['control_sha256'][plan['name']]
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/('candidate_'+str(index));b=joblib.load(folder/'model.joblib');assert b['train_ids']==plan['train'];fn=control_predict if params is None else predict
            if params is None:replay=joblib.load(control)
            else:
                assert b['kind']==params['kind'] and b['alpha']==params['alpha'] and b['seed']==cfg['seed']
                xs=[];ys=[];weights=[];anchors=[];membership=[]
                for key in plan['train']:
                    c=cells[key];lookup={float(n):float(y) for n,y in zip(c['label_cycle'],c['soh']) if 0<n<=100 and np.isfinite(y) and y>0};take=[i for i,n in enumerate(c['cycle']) if 0<n<=100 and float(n) in lookup]
                    cycle=np.asarray(c['cycle'])[take];x=np.asarray(c['raw'])[take];y=np.array([lookup[float(n)] for n in cycle]);assert len(y)>=3
                    xs.append(x);ys.append(y);weights.append(np.full(len(y),1/len(y)));anchors.append(float(np.median(y[-3:])));membership.append(dict(cell_id=key,cycles=cycle.tolist(),rows=len(y)))
                assert membership==b['supervision'];np.testing.assert_array_equal(anchors,b['anchors'])
                x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights);w*=len(w)/w.sum();z=manual_impute(x,b['imputer'])
                if params['kind']=='ridge':
                    shifted=z-z[0];mu=np.average(shifted,axis=0,weights=w);var=np.average((shifted-mu)**2,axis=0,weights=w);scale=manual_scale(var,mu,w.sum())
                    for expected,actual in [(mu+z[0],b['scaler'].mean_),(var,b['scaler'].var_),(scale,b['scaler'].scale_)]:np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
                    xx=b['scaler'].transform(z);xm=np.average(xx,axis=0,weights=w);ym=np.average(y,weights=w);xc=xx-xm;yc=y-ym;coef=b['model'].coef_
                    grad=xc.T@(w*(xc@coef-yc))+params['alpha']*coef;rhs=xc.T@(w*yc);assert np.linalg.norm(grad)<=1e-8*max(1.,np.linalg.norm(rhs))
                else:assert b['model'].n_estimators==128 and b['model'].min_samples_leaf==5
                dx=[];rate=[];dw=[]
                for key,a in zip(plan['train'],anchors):
                    c,yy=labels(cells[key]);idx=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int));dx.append(inputs(clean(cells[key]),c[idx]));rate.append((a-yy[idx])/(1+(c[idx]-100)/100));v=np.linspace(1,3,len(idx));dw.append(v/v.sum())
                d=b['dynamic'];assert d['train_ids']==plan['train'];np.testing.assert_array_equal(anchors,d['training_anchors']);dx=manual_impute(np.concatenate(dx),d['imputer']);dw=np.concatenate(dw);dw*=len(dw)/dw.sum()
                dynamic=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=cfg['seed'],n_jobs=2).fit(dx,np.concatenate(rate),sample_weight=dw)
                replay=fit(cells,plan['train'],seed=cfg['seed'],**params)
            values_all=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows={r['cell_id']:r for r in json.loads((folder/(role+'_metrics.json')).read_text())};assert set(rows)==set(ids)
                with np.load(folder/(role+'_predictions.npz')) as arrays:
                    assert len(arrays.files)==3*len(ids)
                    for key in ids:
                        cc=clean(cells[key]);c,y=labels(cells[key]);p=fn(b,cc,c);np.testing.assert_allclose(p,fn(replay,cc,c),atol=1e-12,rtol=0)
                        np.testing.assert_array_equal(c,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth']);np.testing.assert_allclose(p,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                        if params is not None:
                            take=(np.asarray(cc['cycle'])>0)&(np.asarray(cc['cycle'])<=100);xx=b['imputer'].transform(np.asarray(cc['raw'],dtype=float)[take][-3:])
                            if b['scaler'] is not None:xx=b['scaler'].transform(xx)
                            aa=float(np.median(b['model'].predict(xx)));np.testing.assert_allclose(aa,rows[key]['predicted_anchor'],atol=1e-12,rtol=0)
                            independent=aa-c/100*dynamic.predict(d['imputer'].transform(inputs(cc,c)));np.testing.assert_allclose(p,independent,atol=1e-12,rtol=0)
                        poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                        np.testing.assert_allclose(p,fn(b,poison,c),atol=1e-12,rtol=0)
                        try:fn(b,dict(cc,physical_family='different'),c)
                        except ValueError:pass
                        else:raise AssertionError('family gate failed')
                        values=metric(c,y,p)
                        for field,value in values.items():np.testing.assert_allclose(value,rows[key][field],atol=1e-10,rtol=0)
                        if role=='validation':values_all.append(values)
                        checked.append(dict(fold=plan['name'],candidate=index,role=role,cell_id=key,model_sha256=sha(folder/'model.joblib'),prediction_sha256=sha(folder/(role+'_predictions.npz'))))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index);fields=('mape_percent','late_mape','endpoint_pp','drop_error_pp');scores=[sum(v[f] for f in fields)/4 for v in values_all]
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            for field in fields:np.testing.assert_allclose(np.mean([v[field] for v in values_all]),report[field],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    assert len(checked)==60;save_json(root/'independent_verification.json',dict(passed=True,checked=checked,retrained_bundles=16,source_sha256=sha(__file__),outer_loaded=False,goal_complete=False));print('CURVES=60',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
