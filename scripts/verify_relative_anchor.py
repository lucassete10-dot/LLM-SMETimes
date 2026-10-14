"""Independent relative-feature joins/statistics and frozen-dynamics replay."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.relative_anchor import fit,predict
from battery_smetimes.fade_rate import predict as fade
from battery_smetimes.tail_boost import labels
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_pooled_anchor import manual_impute
from scripts.verify_conditional_drift import manual_scale
from scripts.audit_tail_numeric_gates import metric
from scripts.prepare_same_type_forecast import save_json,sha


def independent_signals(cell,compact):
    cycle=np.asarray(cell['cycle'],dtype=float);raw=np.asarray(cell['raw'],dtype=float)
    take=(cycle>0)&(cycle<=100);cycle,raw=cycle[take],raw[take]
    assert len(cycle)>=3 and raw.shape[1]==44 and not np.isinf(raw).any()
    reference=np.array([np.nanmedian(col[:5]) for col in raw.T])
    cols=[5,10,34,35,38,40] if compact else list(range(44))
    return cycle,(raw-reference)[:,cols]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text());previous=Path(cfg['previous_run']);rate_root=Path(cfg['rate_run'])
    assert json.loads((root/'progress.json').read_text())['training_complete'] and cfg['outer_loaded'] is False
    assert sha(previous/'config.json')==cfg['previous_config_sha256'] and sha(rate_root/'config.json')==cfg['rate_config_sha256']
    assert cfg['plans']==json.loads((previous/'config.json').read_text())['plans']
    assert sha('docs/soh_relative_anchor_stage_20261005.md')==cfg['preregistration_sha256']
    assert cfg['candidates']==[dict(compact=c,alpha=a) for c in (False,True) for a in (10,100)]
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for name in ('relative_anchor','pooled_anchor','fade_rate','tail_boost','forecast','research_baselines'):
        assert sha(Path('battery_smetimes')/(name+'.py'))==cfg['source_sha256'][name+'.py']
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
        cells=load_subset(plan['train']+plan['validation'])
        for key,c in cells.items():assert sha(Path('data/interim/same_type_forecast/20260929T120631_212616Z')/(key+'.npz'))==cfg['prepared_sha256'][key]
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/('candidate_'+str(index));b=joblib.load(folder/'model.joblib');prov=json.loads((folder/'provenance.json').read_text())
            rate_file=rate_root/plan['name']/str(cfg['seed'])/'inner'/'model.joblib'
            assert Path(prov['rate_file'])==rate_file.resolve() and sha(rate_file)==prov['rate_sha256']
            rate=joblib.load(rate_file);assert rate['train_ids']==b['train_ids']==plan['train'] and rate['seed']==b['seed']==cfg['seed'] and not rate['learned_anchor']
            assert b['compact']==params['compact'] and b['alpha']==params['alpha']
            xs=[];ys=[];ws=[];membership=[]
            for key in plan['train']:
                cell=cells[key];cycle,x=independent_signals(cell,params['compact'])
                lookup={float(c):float(y) for c,y in zip(cell['label_cycle'],cell['soh']) if 0<c<=100 and np.isfinite(y) and y>0}
                take=np.array([float(c) in lookup for c in cycle]);cy=cycle[take];y=np.array([lookup[float(c)] for c in cy]);assert len(y)>=3
                xs.append(x[take]);ys.append(y);ws.append(np.full(len(y),1/len(y)));membership.append(dict(cell_id=key,cycles=cy.tolist(),rows=len(y)))
            assert membership==b['supervision'];x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
            z=manual_impute(x,b['imputer']);shift=z-z[0];mu=np.average(shift,axis=0,weights=w);var=np.average((shift-mu)**2,axis=0,weights=w)
            for expected,actual in [(mu+z[0],b['scaler'].mean_),(var,b['scaler'].var_),(manual_scale(var,mu,w.sum()),b['scaler'].scale_)]:np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
            xx=(z-b['scaler'].mean_)/b['scaler'].scale_;xm=np.average(xx,axis=0,weights=w);ym=np.average(y,weights=w);xc=xx-xm;yc=y-ym;coef=b['model'].coef_
            grad=xc.T@(w*(xc@coef-yc))+params['alpha']*coef;assert np.linalg.norm(grad)<=1e-8*max(1.,np.linalg.norm(xc.T@(w*yc)))
            replay=fit(cells,plan['train'],seed=cfg['seed'],rate=rate,**params);values=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows={r['cell_id']:r for r in json.loads((folder/(role+'_metrics.json')).read_text())};assert set(rows)==set(ids)
                with np.load(folder/(role+'_predictions.npz')) as arrays, np.load(rate_file.parent/(role+'_predictions.npz')) as source_arrays:
                    assert len(arrays.files)==3*len(ids)
                    for key in ids:
                        cell=clean(cells[key]);cycle,truth=labels(cells[key]);p=predict(b,cell,cycle);np.testing.assert_allclose(p,predict(replay,cell,cycle),atol=1e-12,rtol=0)
                        _,features=independent_signals(cell,params['compact']);features=np.where(np.isfinite(features[-3:]),features[-3:],b['imputer'].statistics_)
                        anchor=float(np.median(((features-b['scaler'].mean_)/b['scaler'].scale_)@coef+b['model'].intercept_))
                        d=fade(rate,cell,cycle);np.testing.assert_allclose(d,source_arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                        np.testing.assert_allclose(p,anchor+d-fade(rate,cell,np.array([101.]))[0],atol=1e-12,rtol=0)
                        np.testing.assert_allclose(p[0]-p[-1],d[0]-d[-1],atol=1e-12,rtol=0)
                        for field,a in [('cycle',cycle),('truth',truth),('prediction',p)]:np.testing.assert_allclose(a,arrays[key+'__100__'+field],atol=1e-12,rtol=0)
                        poison=dict(cell,cycle=np.r_[cell['cycle'],101],raw=np.vstack([cell['raw'],np.full((1,44),np.inf)]),soh=np.inf,cycle_life=np.inf)
                        np.testing.assert_allclose(p,predict(b,poison,cycle),atol=1e-12,rtol=0)
                        try:predict(b,dict(cell,physical_family='different'),cycle)
                        except ValueError:pass
                        else:raise AssertionError('family gate failed')
                        m=metric(cycle,truth,p)
                        for field,v in m.items():np.testing.assert_allclose(v,rows[key][field],atol=1e-10,rtol=0)
                        if role=='validation':values.append(m)
                        checked.append(dict(fold=plan['name'],candidate=index,role=role,cell_id=key,model_sha256=sha(folder/'model.joblib'),prediction_sha256=sha(folder/(role+'_predictions.npz'))))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index);fields=('mape_percent','late_mape','endpoint_pp','drop_error_pp');scores=[sum(m[f] for f in fields)/4 for m in values]
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            for f in fields:np.testing.assert_allclose(np.mean([m[f] for m in values]),report[f],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    assert len(checked)==48;save_json(root/'independent_verification.json',dict(passed=True,checked=checked,retrained_bundles=16,source_sha256=sha(__file__),outer_loaded=False,goal_complete=False))
    old=json.loads((previous/'screening_decision.json').read_text());selected=[min([r for r in reports if r['fold']==p['name']],key=lambda r:(r['score'],r['candidate'])) for p in cfg['plans']]
    fields=('score','mape_percent','late_mape','endpoint_pp','drop_error_pp');means={name:{f:float(np.mean([r[f] for r in rows])) for f in fields} for name,rows in [('relative',selected),('absolute',old['selected'])]}
    wins=sum(a['score']<next(r for r in old['selected'] if r['fold']==a['fold'])['score'] for a in selected)
    passed=wins>=3 and all(means['relative'][f]<=means['absolute'][f]+1e-10 for f in ('score','endpoint_pp','drop_error_pp'))
    save_json(root/'screening_decision.json',dict(means=means,selected=selected,better_folds=wins,expand_formal=passed,previous_screen_sha256=sha(previous/'screening_decision.json'),aggregation='one-seed inner validation only; not formal or blind',goal_complete=False))
    print('CURVES=48 SCREEN='+str(passed),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
