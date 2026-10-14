"""Replayed member curves and separate mean/median, paired diagnostics and score audit."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_subset_ensemble import predict
from battery_smetimes.xjtu_segment_features import predict as base_predict
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(31,13,11) and not set(cells)&set(plan['test'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    parent=Path(cfg['parent']);assert json.loads((parent/'independent_verification.json').read_text())['passed']
    base=joblib.load(cfg['model_file']);serial_trees(base);members=[]
    for s in cfg['splits']:
        assert not set(s['train'])&set(s['held']) and set(s['train']+s['held'])==set(plan['train'])
        b=joblib.load(parent/f"oof_{s['index']}"/'model.joblib');serial_trees(b)
        assert b['train_ids']==s['train'] and not set(b['train_ids'])&set(plan['validation']+plan['test']);members.append(b)
    cache={};paired=[];oof=[];component_curves=0
    with np.load(root/'component_predictions.npz',allow_pickle=False) as a:
        assert set(a.files)=={k+'__'+f for k in cells for f in ('cycle','truth','full','members')}
        for k in cells:
            c,y=targets(cells[k]);q=clean(cells[k]);p=base_predict(base,q,c);ps=np.stack([base_predict(m,q,c) for m in members]);component_curves+=5
            np.testing.assert_array_equal(a[k+'__cycle'],c);np.testing.assert_array_equal(a[k+'__truth'],y)
            np.testing.assert_allclose(a[k+'__full'],p,rtol=0,atol=1e-12);np.testing.assert_allclose(a[k+'__members'],ps,rtol=0,atol=1e-12);cache[k]=(c,y,p,ps)
            if k in plan['validation']:
                for name,v in [('full',p)]+[(f'member_{i}',v) for i,v in enumerate(ps)]:
                    paired.append(dict(cell_id=k,model=name,**detail(c,y,v),relative_endpoint_residual_pp=float(100*((y[-1]-y[0])-(v[-1]-v[0]))),late_bias_pp=float(100*np.mean(v[len(v)//2:]-y[len(y)//2:]))))
    for s in cfg['splits']:
        with np.load(parent/f"oof_{s['index']}"/'predictions.npz',allow_pickle=False) as a:
            for k in s['held']:
                y,p=[a[k+'__'+f] for f in ('truth','prediction')];oof.append(dict(cell_id=k,relative_endpoint_residual_pp=float(100*((y[-1]-y[0])-(p[-1]-p[0])))))
    diag=json.loads((root/'transfer_diagnostic.json').read_text())
    assert len(diag['validation_paired'])==len(paired)==78 and len(diag['training_oof'])==len(oof)==31
    for x,y in zip(diag['validation_paired'],paired):close(x,y)
    for x,y in zip(diag['training_oof'],oof):close(x,y)
    scores=[];curves=0
    for i,kind in enumerate(cfg['kinds']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');serial_trees(b)
        assert b['train_ids']==plan['train'] and b['kind']==kind and b['family']==base['family']
        assert len(b['members'])==(1 if kind=='full' else 5)
        for j,m in enumerate(b['members']):assert m['train_ids']==(plan['train'] if kind=='full' else cfg['splits'][j]['train'])
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            saved=json.loads((out/(role+'_metrics.json')).read_text());rows=[];assert len(saved)==len(ids)
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p,ps=cache[k];q=clean(cells[k])
                    if kind=='mean':p=sum(ps)/5
                    elif kind=='median':p=np.sort(ps,axis=0)[2]
                    for f,v in [('cycle',c),('truth',y)]:np.testing.assert_array_equal(a[k+'__100__'+f],v)
                    np.testing.assert_allclose(a[k+'__100__prediction'],p,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(predict(b,q,c),p,rtol=0,atol=1e-12)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,q['raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(predict(b,poison,c),p,rtol=0,atol=1e-12)
                    try:predict(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:predict(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinates')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(saved[pos],r);rows.append(r);curves+=1
            if role=='validation':vr=rows
        v=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,kind=kind,score=float(v.mean()+.25*np.percentile(v,90)),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    sel=json.loads((root/'selection.json').read_text());assert len(sel['candidates'])==3
    for x,y in zip(sel['candidates'],scores):close(x,y)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0];expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(sel['selected'],best);assert sel['expand']==expand and curves==132 and component_curves==220
    save_json(root/'independent_verification.json',dict(passed=True,component_curves=220,combined_curves=132,paired_rows=78,oof_rows=31,expand=expand,outer_loaded=False,
        scope='Previously verified models replayed, independent mean/median/diagnostic/score arithmetic; shared feature and metrics helpers, no new model refits.',verifier_sha256=sha(Path(__file__))))
    print('PASS:220member/132combinedcurves/78paired/31OOF/metrics/selection/poison/family/SHA;expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
