"""Fresh reproducibility refits plus independent feature and metric arithmetic."""
import argparse,json
from pathlib import Path
import warnings
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.nca_segment_features import fit,prefix,predict
from battery_smetimes.tail_selection import predict as control_predict
from battery_smetimes.curve_fidelity import selection_score
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_model_state_portable import equal_state
from scripts.verify_nasa_robust_inner import close
from scripts.audit_rate_blend_metrics import metrics


def independent_prefix(cell,p):
    c=np.asarray(cell['early_cycle']);mask=c<=100;c=c[mask].astype(float)
    x=cell['early_raw'][mask,:16 if p['block']=='V' else 32].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        first=np.nanmedian(x[:5],axis=0);last=np.nanmedian(x[-3:],axis=0);slope=[]
        for j in range(x.shape[1]):
            times=c[-10:];values=x[-10:,j];valid=np.isfinite(values);times=times[valid];values=values[valid]
            if len(times)<2:slope.append(np.nan);continue
            t=times-times.mean();slope.append(float(np.sum(t*(values-values.mean()))/np.sum(t*t))*100)
        parts=[first,last,last-first,np.asarray(slope)]
        if p['context']=='segments':
            med=[];counts=[]
            for lower in (0,25,50,75):
                take=(c>lower)&(c<=lower+25);counts.append(int(take.sum()))
                med.append(np.nanmedian(x[take],axis=0) if take.any() else np.full(x.shape[1],np.nan))
            parts+=med+[med[j+1]-med[j] for j in range(3)]+[np.asarray(counts)/100]
    return np.concatenate(parts+[np.asarray(cell['conditions']),np.asarray([c[0],c[-1],len(c),c[-1]-c[0]])/100])


def independent_prediction(bundle,cell,future):
    condition=tuple(cell['conditions'])
    if condition in bundle['local']:return independent_prediction(bundle['local'][condition],cell,future)
    v=independent_prefix(cell,bundle['parameters']);p=np.tile(v,(len(future),1));h=(future-100)/100
    x=np.column_stack([p,h,h*h,np.log1p(h),p*h[:,None]])
    return bundle['model'].predict(bundle['imputer'].transform(x))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];seed=cfg['seed']
    assert json.loads((root/'progress.json').read_text())['complete'] and not cfg['outer_loaded']
    assert cfg['candidates']==[dict(block=b,context=c) for b in ('V','VI') for c in ('basic','segments')]
    for f,h in cfg['source_sha256'].items():assert sha(f)==h==sha(root/'source_snapshot'/Path(f).name)
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256'] and sha(NCA/'index.json')==cfg['nca_index_sha256']
    index=json.loads((NCA/'index.json').read_text());cells=load_cells(plan)
    assert not set(cells)&set(plan['test']) and cfg['prepared_sha256']=={k:index['cells'][k]['sha256'] for k in cells}
    coverage=json.loads((root/'early_coverage.json').read_text())
    assert coverage==[dict(cell_id=k,early_records=int(np.sum(v['early_cycle']<=100)),segment_counts=[int(np.sum((v['early_cycle']>s)&(v['early_cycle']<=s+25))) for s in (0,25,50,75)]) for k,v in cells.items()]
    scores=[];curves=0;refits=0
    for i,parameters in enumerate([None]+cfg['candidates']):
        out=root/f'candidate_{i}';stored=joblib.load(out/'model.joblib')
        if i:
            assert stored['train_ids']==plan['train'] and stored['parameters']==parameters and stored['seed']==seed
            groups={}
            for k in plan['train']:groups.setdefault(tuple(cells[k]['conditions']),[]).append(k)
            assert set(stored['local'])=={g for g,ks in groups.items() if len(ks)>=6}
            for group,local in stored['local'].items():assert local['train_ids']==groups[group] and not local['local']
            bundle=fit(cells,plan['train'],parameters,seed);equal_state(stored,bundle);refits+=1+len(bundle['local'])
        else:
            source=Path(cfg['control_model_file']);assert sha(source)==cfg['control_model_sha256']
            equal_state(stored['inner'],joblib.load(source));bundle=stored
        for role,members in [('training',plan['train']),('validation',plan['validation'])]:
            saved=json.loads((out/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in saved]==members
            rows=[]
            with np.load(out/(role+'_predictions.npz')) as a:
                assert set(a.files)=={k+'__100__'+f for k in members for f in ('cycle','truth','prediction')}
                for k,record in zip(members,saved):
                    c,y=cells[k]['cycle'],cells[k]['truth'];q=clean(cells[k])
                    if i:np.testing.assert_allclose(prefix(q,parameters),independent_prefix(q,parameters),rtol=0,atol=1e-12,equal_nan=True)
                    p=independent_prediction(bundle,q,c) if i else control_predict(bundle,q,c);stem=k+'__100__'
                    np.testing.assert_array_equal(c,a[stem+'cycle']);np.testing.assert_array_equal(y,a[stem+'truth']);np.testing.assert_allclose(p,a[stem+'prediction'],rtol=0,atol=1e-12)
                    poisoned=dict(q,truth=np.inf,cycle=np.inf,cycle_life=np.inf,prefix=np.inf,early_raw=q['early_raw'].copy())
                    poisoned['early_raw']=np.vstack([poisoned['early_raw'],np.full((1,32),np.inf)]);poisoned['early_cycle']=np.r_[q['early_cycle'],9999]
                    actual=predict if i else control_predict
                    np.testing.assert_allclose(p,actual(bundle,poisoned,c),rtol=0,atol=1e-12)
                    try:actual(bundle,dict(q,physical_family='WRONG'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate failed')
                    e=p-y;ss=float(np.sum((y-y.mean())**2))
                    r=dict(cell_id=k,**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),start_error_pp=float(100*abs(e[0])))
                    close(r,record);rows.append(r);curves+=1
            if role=='validation':vr=rows
        scores.append(dict(candidate=i,parameters=parameters,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
        print('VERIFIED candidate '+str(i),flush=True)
    sel=json.loads((root/'selection.json').read_text());assert len(sel['candidates'])==len(scores)
    for actual,saved in zip(scores,sel['candidates']):close(actual,saved)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0];close(best,sel['selected']);close(old,sel['control'])
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    assert expand==sel['expand'] and curves==220 and refits==12
    save_json(root/'independent_verification.json',dict(passed=True,fresh_reproducibility_component_refits=refits,curves=curves,expand=expand,
        outer_loaded=False,verifier_sha256=sha(__file__),scope='fresh strict model replay; independent feature/metric arithmetic, input isolation; not independent trainer implementation',goal_complete=False))
    print(json.dumps(dict(passed=True,curves=curves,component_refits=refits,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
