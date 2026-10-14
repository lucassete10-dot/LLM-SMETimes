"""Independent OOF target arithmetic, 14 fresh refits, all curves and input tests."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from scripts.verify_mit_segment_features import independent_inputs
from scripts.verify_model_state_portable import equal_state
from scripts.verify_nasa_robust_inner import close
from scripts.audit_rate_blend_metrics import metrics
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from battery_smetimes.curve_fidelity import selection_score
from battery_smetimes.mit_pace_forecast import predict as actual_predict

FEATURES=dict(block='V',context='segments')


def target(cell):
    keep=(cell['label_cycle']>100)&np.isfinite(cell['soh'])&(cell['soh']>0)
    return cell['label_cycle'][keep].astype(float),cell['soh'][keep].astype(float)


def label(cell):
    c,y=target(cell);initial=float(np.median(y[:5]));selected=None
    for j in range(len(y)):
        window=y[max(0,j-2):min(len(y),j+3)]
        if np.median(window)<=initial-.05:selected=float(c[j]-100);break
    return selected


def features(cell):return independent_inputs(clean(cell),np.array([101.]),FEATURES)[0,:-3]


def head(cells,stored,alpha):
    pairs=[(k,label(cells[k])) for k in stored['train_ids']];usable=[(k,v) for k,v in pairs if v is not None]
    assert stored['labeled_ids']==[k for k,v in usable]
    assert stored['unobserved_ids']==[k for k,v in pairs if v is None]
    x=np.stack([features(cells[k]) for k,v in usable]);y=np.log([v for k,v in usable])
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    scaler=StandardScaler().fit(z);model=Ridge(alpha=alpha).fit(scaler.transform(z),y)
    for name,new in [('imputer',imputer),('scaler',scaler),('model',model)]:equal_state(stored[name],new)
    assert stored['lower']==y.min() and stored['upper']==y.max()
    return dict(stored,imputer=imputer,scaler=scaler,model=model)


def pace(h,cell):
    x=features(cell)[None,:];v=h['model'].predict(h['scaler'].transform(h['imputer'].transform(x)))[0]
    return float(np.exp(np.clip(v,h['lower'],h['upper'])))


def curve_input(cell,c,p):return np.column_stack([independent_inputs(clean(cell),c,FEATURES),(c-100)/p])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];seed=cfg['seed']
    assert not cfg['outer_loaded'] and cfg['alphas']==[10.,100.] and json.loads((root/'progress.json').read_text())['complete']
    for f,h in cfg['source_sha256'].items():assert sha(f)==h==sha(root/'source_snapshot'/Path(f).name)
    parent=Path(cfg['parent']);assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    child=parent/plan['name']/str(seed);assert sha(child/'selection.json')==cfg['control_selection_sha256']
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    assert cfg['prepared_sha256']=={k:v['prepared_sha256'] for k,v in cells.items()}
    supervision=json.loads((root/'training_pace_supervision.json').read_text())
    assert supervision==[dict(cell_id=k,pace_label=label(cells[k])) for k in plan['train']]
    scores=[];curves=0;refits=0
    for index,alpha in enumerate(cfg['alphas']):
        out=root/f'candidate_{index}';bundle=joblib.load(out/'model.joblib');ids=plan['train']
        assert bundle['train_ids']==ids and bundle['alpha']==alpha and bundle['seed']==seed
        assert bundle['family']==cells[ids[0]]['physical_family']
        oof={};splits=[]
        for (tr,va),saved in zip(KFold(5,shuffle=True,random_state=seed).split(ids),bundle['oof_folds']):
            train=[ids[j] for j in tr];held=[ids[j] for j in va]
            assert saved['train']==train and saved['held']==held and saved['head']['train_ids']==train
            assert not set(train)&set(held);splits.append(dict(train=train,held=held))
            h=head(cells,saved['head'],alpha);refits+=1;oof.update({k:pace(h,cells[k]) for k in held})
        assert len(bundle['oof_folds'])==5 and set(oof)==set(ids)
        equal_state(oof,bundle['oof_paces']);equal_state(oof,json.loads((out/'oof_paces.json').read_text()))
        assert splits==json.loads((out/'oof_split.json').read_text())
        final=head(cells,bundle['pace_head'],alpha);refits+=1
        assert final['train_ids']==ids
        xs=[];ys=[];ws=[]
        for k in ids:
            c,y=target(cells[k]);ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
            xs.append(curve_input(cells[k],c[ii],oof[k]));ys.append(y[ii]);w=np.linspace(1,3,len(ii));ws.append(w/w.sum())
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
        imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
        model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w);refits+=1
        equal_state(bundle['curve_imputer'],imputer);equal_state(bundle['curve_model'],model)
        for role,members in [('training',ids),('validation',plan['validation'])]:
            stored=json.loads((out/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==members
            rows=[]
            with np.load(out/(role+'_predictions.npz')) as a:
                assert set(a.files)=={k+'__100__'+f for k in members for f in ('cycle','truth','prediction')}
                for k,record in zip(members,stored):
                    c,y=target(cells[k]);q=clean(cells[k]);pvalue=pace(final,q);p=model.predict(imputer.transform(curve_input(q,c,pvalue)))
                    stem=k+'__100__';np.testing.assert_array_equal(c,a[stem+'cycle']);np.testing.assert_array_equal(y,a[stem+'truth']);np.testing.assert_allclose(p,a[stem+'prediction'],rtol=0,atol=1e-12)
                    polluted=dict(q,soh=np.inf,label_cycle=np.inf,cycle_life=np.inf,reference_capacity_ah=np.inf,raw=q['raw'].copy())
                    polluted['raw'][:,[38,39]]=np.inf;polluted['raw']=np.vstack([polluted['raw'],np.full((1,51),np.inf)]);polluted['cycle']=np.r_[q['cycle'],10000]
                    np.testing.assert_allclose(p,actual_predict(bundle,polluted,c),rtol=0,atol=1e-12)
                    try:actual_predict(bundle,dict(q,physical_family='WRONG'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate failed')
                    e=p-y;ss=float(np.sum((y-y.mean())**2))
                    r=dict(cell_id=k,**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,
                        r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),start_error_pp=float(100*abs(e[0])),predicted_pace=pvalue)
                    close(r,record);rows.append(r);curves+=1
            if role=='validation':vr=rows
        scores.append(dict(candidate=index,alpha=alpha,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
        print('VERIFIED alpha '+str(alpha),flush=True)
    selection=json.loads((root/'selection.json').read_text())
    for actual,saved in zip(scores,selection['candidates']):close(actual,saved)
    control=json.loads((child/'selection.json').read_text())['selected'];close(control,cfg['control']);close(control,selection['control'])
    best=min(scores,key=lambda r:(r['score'],r['candidate']));close(best,selection['selected'])
    expand=best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp']
    assert selection['expand']==expand and curves==214 and refits==14
    save_json(root/'independent_verification.json',dict(passed=True,component_refits=14,curves=214,expand=expand,
        outer_loaded=False,verifier_sha256=sha(__file__),scope='OOF/targets/strict fitted state/all curves/input isolation/arithmetic',goal_complete=False))
    print(json.dumps(dict(passed=True,curves=214,component_refits=14,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
