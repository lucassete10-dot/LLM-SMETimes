"""Fresh independent weighted forest refits and all saved inner curve checks."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_model_state_portable import equal_state
from scripts.verify_nasa_robust_inner import close
from scripts.audit_rate_blend_metrics import metrics
from scripts.verify_mit_segment_features import replay,condition_key
from battery_smetimes.mit_segment_features import inputs,predict
from battery_smetimes.curve_fidelity import selection_score


def target(cell):
    m=(cell['label_cycle']>100)&np.isfinite(cell['soh'])&(cell['soh']>0)
    return cell['label_cycle'][m].astype(float),cell['soh'][m].astype(float)


def rebuild(cells,stored):
    xs=[];ys=[];ws=[];par=stored['parameters'];ids=stored['train_ids']
    for key in ids:
        c,y=target(cells[key]);ii=np.unique(np.round(np.linspace(0,len(y)-1,min(96,len(y)))).astype(int))
        xs.append(inputs(clean(cells[key]),c[ii],par));ys.append(y[ii])
        if par['scheme'].startswith('ramp'):
            endpoint=float(par['scheme'][4:])
            manual=1+(endpoint-1)*np.linspace(0,1,len(ii))
            w=np.linspace(1,endpoint,len(ii))
            np.testing.assert_allclose(w,manual,rtol=0,atol=1e-14)
        else:
            assert par['scheme']=='terminal10'
            w=np.ones(len(ii));w[c[ii]>=c[ii][0]+.75*(c[ii][-1]-c[ii][0])]=10
        ws.append(w/w.sum())
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,
        random_state=stored['seed'],n_jobs=2).fit(z,y,sample_weight=w)
    equal_state(stored['imputer'],imputer)
    equal_state(stored['model'],model)
    groups={}
    for k in ids:groups.setdefault(condition_key(cells[k]),[]).append(k)
    if stored['local']:
        assert set(stored['local'])=={k for k,v in groups.items() if len(v)>=6}
        for k,b in stored['local'].items():assert b['train_ids']==groups[k]
    result=dict(stored,imputer=imputer,model=model)
    result['local']={k:rebuild(cells,b) for k,b in stored['local'].items()}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    root=ap.parse_args().run;cfg=json.loads((root/'config.json').read_text());plan=cfg['plan']
    assert json.loads((root/'progress.json').read_text())['complete'] and not cfg['outer_loaded']
    assert cfg['candidates']==[dict(block='V',context='segments',scheme=s) for s in ('ramp10','ramp30','terminal10')]
    parent=Path(cfg['parent']);child=parent/plan['name']/str(cfg['seed'])
    assert sha(parent/'config.json')==cfg['parent_config_sha256'] and sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    assert sha(child/'selection.json')==cfg['control_selection_sha256']
    for f,h in cfg['source_sha256'].items():assert sha(f)==h==sha(root/'source_snapshot'/Path(f).name)
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    assert cfg['prepared_sha256']=={k:v['prepared_sha256'] for k,v in cells.items()}
    scores=[];curves=0;fits=0
    for i,par in enumerate(cfg['candidates']):
        out=root/f'candidate_{i}';stored=joblib.load(out/'model.joblib')
        assert stored['parameters']==par and stored['train_ids']==plan['train'] and stored['seed']==cfg['seed']
        bundle=rebuild(cells,stored);fits+=1+len(bundle['local'])
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in records]==ids
            with np.load(out/(role+'_predictions.npz')) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                rows=[]
                for key,record in zip(ids,records):
                    c,y=target(cells[key]);q=clean(cells[key]);p=replay(bundle,q,c);stem=key+'__100__'
                    np.testing.assert_array_equal(c,a[stem+'cycle']);np.testing.assert_array_equal(y,a[stem+'truth'])
                    np.testing.assert_allclose(p,a[stem+'prediction'],rtol=0,atol=1e-12)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,raw=q['raw'].copy())
                    poison['raw'][:,[38,39]]=np.inf
                    poison['raw']=np.vstack([poison['raw'],np.full((1,51),np.inf)]);poison['cycle']=np.r_[q['cycle'],10000]
                    np.testing.assert_allclose(p,predict(bundle,poison,c),rtol=0,atol=1e-12)
                    try:predict(bundle,dict(q,physical_family='WRONG'),c)
                    except ValueError:pass
                    else:raise AssertionError('family isolation failed')
                    e=p-y;ss=float(np.sum((y-y.mean())**2))
                    r=dict(cell_id=key,**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),
                        r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),
                        future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),
                        start_error_pp=float(100*abs(e[0])))
                    close(r,record);rows.append(r);curves+=1
            if role=='validation':vr=rows
        scores.append(dict(candidate=i,parameters=par,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    sel=json.loads((root/'selection.json').read_text())
    assert len(scores)==len(sel['candidates'])
    for actual,saved in zip(scores,sel['candidates']):close(actual,saved)
    old=json.loads((child/'selection.json').read_text())['selected'];close(old,sel['control'])
    best=min(scores,key=lambda r:(r['score'],r['candidate']));close(best,sel['selected'])
    expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    assert sel['expand']==expand and curves==321
    save_json(root/'independent_verification.json',dict(passed=True,component_refits=fits,curves=curves,
        expand=expand,outer_loaded=False,verifier_sha256=sha(__file__),goal_complete=False))
    print(json.dumps(dict(passed=True,curves=curves,component_refits=fits,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
