"""Separate OOF design and constrained solver replay, gain/anchor/score checks."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_horizon_gain import predict,phi
from battery_smetimes.xjtu_segment_features import predict as base_predict
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(31,13,11)
    assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    parent=Path(cfg['parent']);assert json.loads((parent/'independent_verification.json').read_text())['passed']
    splits=json.loads((parent/'oof_splits.json').read_text());xs=[];ys=[];seen=[]
    for s in splits:
        assert not set(s['train'])&set(s['held']) and set(s['train']+s['held'])==set(plan['train'])
        with np.load(parent/f"oof_{s['index']}"/'predictions.npz',allow_pickle=False) as a:
            for k in s['held']:
                seen.append(k);c,y,p=[a[k+'__'+f] for f in ('cycle','truth','prediction')];cc,yy=targets(cells[k])
                np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                ix=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int));w=np.linspace(1,3,len(ix));w=np.sqrt(w/(np.sum(w)*31))
                factor=(c[ix]-100)/(c[ix]);dp=p[0]-p[ix]
                xs.append(np.stack([dp,dp*factor],axis=1)*w[:,None]);ys.append((y[0]-y[ix])*w)
    assert len(seen)==31 and set(seen)==set(plan['train'])
    base=joblib.load(cfg['model_file']);serial_trees(base);scores=[];curves=0
    for i,penalty in enumerate([None]+cfg['penalties']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');serial_trees(b['base'])
        if penalty is None:coef=np.array([1.,0.])
        else:
            # Separate algebra for design, shared bounded least-squares optimizer.
            x=np.concatenate(xs);target=np.concatenate(ys)
            solver=lsq_linear(np.vstack([x,np.diag([np.sqrt(penalty)]*2)]),np.r_[target,np.sqrt(penalty),0.],
                bounds=([.5,0],[1.5,.5]),tol=1e-12,max_iter=500)
            assert solver.success;coef=solver.x
            np.testing.assert_allclose(b['solver']['cost'],solver.cost,rtol=0,atol=1e-12)
        np.testing.assert_allclose(b['coefficients'],coef,rtol=0,atol=1e-8)
        assert b['train_ids']==b['base']['train_ids']==plan['train'] and b['penalty']==penalty
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text());assert len(records)==len(ids);rows=[]
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')];cc,yy=targets(cells[k])
                    np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy);q=clean(cells[k])
                    bp=base_predict(base,q,c);anchor=base_predict(base,q,[101.])[0]
                    np.testing.assert_allclose(phi(c),(c-100)/c,rtol=0,atol=1e-15)
                    gain=b['coefficients'][0]+b['coefficients'][1]*(c-100)/c
                    assert np.all(gain>=.5) and np.all(gain<=2)
                    np.testing.assert_allclose(p,anchor+gain*(bp-anchor),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(predict(b,q,[101.])[0],anchor,rtol=0,atol=1e-12)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                        cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,q['raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                    try:predict(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:predict(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinates')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(records[pos],r);rows.append(r);curves+=1
            if role=='validation':vr=rows
        s=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,penalty=penalty,coefficients=b['coefficients'].tolist(),score=float(s.mean()+.25*np.percentile(s,90)),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best);assert selection['expand']==expand and curves==132
    save_json(root/'independent_verification.json',dict(passed=True,curves=132,shared_solver_replays=2,expand=expand,outer_loaded=False,
        scope='Independent OOF design/relative gain/anchor/selection, shared bounded solver and feature/metric helpers; prior verified OOF models reused.',verifier_sha256=sha(Path(__file__))))
    print('PASS:2solver replays,132curves,bounds/phi/anchor/poison/family/metrics; expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
