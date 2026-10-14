"""Rebuild train-only rate targets, estimators and every saved curve."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.fade_rate import fit,predict,anchor
from battery_smetimes.tail_boost import inputs,labels,is_nca
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.warm_monotone import derive_prior
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def imputed(z,imputer):
    median=[]
    for col in z.T:
        finite=col[np.isfinite(col)];median.append(np.median(finite) if len(finite) else 0.)
    median=np.asarray(median);np.testing.assert_allclose(median,imputer.statistics_,atol=1e-12,rtol=0)
    return np.where(np.isfinite(z),z,median)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256']
    assert cfg['plans']==json.loads((source/'config.json').read_text())['plans']
    fivefold=cfg.get('fivefold_rate',False)
    assert cfg['candidates']==([None,dict(learned_anchor=True)] if fivefold else [None,dict(learned_anchor=False),dict(learned_anchor=True)])
    if fivefold:assert len(cfg['plans'])==5 and all(p['dataset']=='MIT' for p in cfg['plans']) and cfg['seed']==20261005
    assert sha(cfg.get('preregistration_file','docs/soh_fade_rate_stage_20261005.md'))==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for name in ('fade_rate','conditional_drift','crossfit_drift','tail_boost','forecast','research_baselines'):
        assert sha(Path('battery_smetimes')/(name+'.py'))==cfg['source_sha256'][name+'.py']
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        cache=Path(cfg['parameter_files'][plan['name']]);assert sha(cache)==cfg['parameter_sha256'][plan['name']]
        parameter_rows=derive_prior(cells,plan['train'])[1];assert parameter_rows==json.loads(cache.read_text())
        control=Path(cfg['control_paths'][plan['name']]) if 'control_paths' in cfg else source/plan['name']/'base.joblib'
        assert sha(control)==cfg['base_sha256'][plan['name']]
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/('candidate_'+str(index));b=joblib.load(folder/'model.joblib');assert b['train_ids']==plan['train']
            if params is None:
                original=dict(mode='control',base=joblib.load(control),train_ids=plan['train'])
                assert b['base']['train_ids']==plan['train'] and b['base']['seed']==cfg['seed']
            else:
                assert b['seed']==cfg['seed'] and b['learned_anchor']==params['learned_anchor']
                a=np.array([r['anchor'] for r in parameter_rows]) if params['learned_anchor'] else np.ones(len(plan['train']))
                np.testing.assert_array_equal(a,b['training_anchors']);xs=[];rates=[];weights=[]
                for key,aa in zip(plan['train'],a):
                    c,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
                    xs.append(inputs(clean(cells[key]),c[idx]))
                    # Trees can change tied split decisions when y differs at roundoff.
                    # Audit alternate algebra, then refit the exact registered float64 targets.
                    value=(aa-y[idx])/(1+(c[idx]-100)/100)
                    np.testing.assert_allclose(value,100*(aa-y[idx])/c[idx],atol=1e-14,rtol=0)
                    rates.append(value);w=np.linspace(1,3,len(idx));weights.append(w/w.sum())
                xx=np.concatenate(xs);rate=np.concatenate(rates);w=np.concatenate(weights);w*=len(w)/w.sum();xx=imputed(xx,b['imputer'])
                independent=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=cfg['seed'],n_jobs=2).fit(xx,rate,sample_weight=w)
                replay=fit(cells,plan['train'],seed=cfg['seed'],parameter_fits=parameter_rows,**params)
                if params['learned_anchor']:
                    z=imputed(np.stack([prefix(clean(cells[k])) for k in plan['train']]),b['anchor_imputer'])
                    anchor_tree=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=cfg['seed'],n_jobs=2).fit(z,a)
                assert b['model'].n_estimators==128 and b['model'].min_samples_leaf==1
            validation=[];training=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows={r['cell_id']:r for r in json.loads((folder/(role+'_metrics.json')).read_text())};assert set(rows)==set(ids)
                with np.load(folder/(role+'_predictions.npz')) as arrays:
                    assert len(arrays.files)==3*len(ids)
                    for key in ids:
                        cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x)
                        for name,value in [('cycle',x),('truth',y)]:np.testing.assert_array_equal(value,arrays[key+'__100__'+name])
                        np.testing.assert_allclose(p,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                        if params is None:np.testing.assert_allclose(p,predict(original,cc,x),atol=1e-12,rtol=0)
                        else:
                            aa=float(anchor_tree.predict(b['anchor_imputer'].transform(prefix(cc)[None]))[0]) if params['learned_anchor'] else 1.
                            independent_p=aa-x/100*independent.predict(b['imputer'].transform(inputs(cc,x)))
                            np.testing.assert_allclose(p,independent_p,atol=1e-12,rtol=0)
                            np.testing.assert_allclose(p,predict(replay,cc,x),atol=1e-12,rtol=0)
                            if role=='training':
                                np.testing.assert_allclose(rows[key]['predicted_anchor'],anchor(b,cc),atol=1e-12,rtol=0)
                                np.testing.assert_allclose(rows[key]['training_anchor'],a[plan['train'].index(key)],atol=1e-12,rtol=0)
                        if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                        else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                        np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                        try:predict(b,dict(cc,physical_family='different'),x)
                        except ValueError:pass
                        else:raise AssertionError('family gate failed')
                        values=metric(x,y,p)
                        for field,value in values.items():np.testing.assert_allclose(value,rows[key][field],atol=1e-10,rtol=0)
                        (validation if role=='validation' else training).append(values)
                        checked.append(dict(fold=plan['name'],candidate=index,role=role,cell_id=key,model_sha256=sha(folder/'model.joblib'),predictions_sha256=sha(folder/(role+'_predictions.npz'))))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            fields=('mape_percent','late_mape','endpoint_pp','drop_error_pp');scores=[sum(v[f] for f in fields)/4 for v in validation]
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            for field in fields:np.testing.assert_allclose(np.mean([v[field] for v in validation]),report[field],atol=1e-10,rtol=0)
            np.testing.assert_allclose(np.mean([v['mape_percent'] for v in training]),report['training_mape'],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    assert len(checked)==sum(len(p['train']+p['validation'])*len(cfg['candidates']) for p in cfg['plans'])
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,retrained_models=len(cfg['plans'])*(len(cfg['candidates'])-1),source_sha256=sha(__file__),outer_loaded=False,goal_complete=False));print('CURVES='+str(len(checked)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
