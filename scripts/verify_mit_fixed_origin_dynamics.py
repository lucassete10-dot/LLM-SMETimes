"""Fresh dynamic fit replay, frozen origin and separate change formula."""
import json,sys
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_fixed_origin_dynamics import fit,predict,origin
from scripts.verify_mit_segment_curve_net import equation
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def serial_trees(value):
    if isinstance(value,dict):
        for obj in value.values():serial_trees(obj)
    elif hasattr(value,'n_jobs'):
        value.n_jobs=1


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(83,24,27)
    assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    assert not set(plan['train'])&set(plan['validation'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256']
    assert sha(Path(cfg['parent'])/'independent_verification.json')==cfg['parent_proof_sha256']
    paths={Path(p).name:Path(p) for p in cfg['origin_files']}
    for f,d in cfg['origin_files'].items():assert sha(Path(f))==d
    close(cfg['control'],json.loads(paths['selection.json'].read_text())['selected'])
    frozen=dict(control=joblib.load(paths['control.joblib']),feature=joblib.load(paths['feature.joblib']),new_weight=cfg['control']['new_weight'])
    serial_trees(frozen)
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    scores=[];curves=0
    for i,width in enumerate(cfg['widths']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib')
        serial_trees(b['frozen'])
        assert b['train_ids']==plan['train'] and b['seed']==cfg['seed'] and b['epochs']==cfg['epochs']
        fresh=fit(cells,plan['train'],width,cfg['seed'],frozen,cfg['epochs'])
        for k,v in b['state'].items():np.testing.assert_array_equal(v,fresh['state'][k])
        for obj,fields in [('imputer',['statistics_']),('scaler',['mean_','scale_','var_'])]:
            for field in fields:np.testing.assert_array_equal(getattr(b[obj],field),getattr(fresh[obj],field))
        np.testing.assert_array_equal(b['history'],fresh['history']);np.testing.assert_array_equal(b['history'],json.loads((out/'epoch_loss.json').read_text()))
        assert b['parameter_count']==sum(v.size for v in b['state'].values())
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text());rows=[]
            assert len(records)==len(ids)
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as arrays:
                assert set(arrays.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p=[arrays[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                    cc,yy=targets(cells[k]);np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                    q=clean(cells[k]);np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12);np.testing.assert_allclose(p,predict(fresh,q,c),rtol=0,atol=1e-12)
                    a=origin(frozen,q)
                    np.testing.assert_allclose(p,a+equation(b,q,c)-equation(b,q,np.array([101.]))[0],rtol=0,atol=2e-6)
                    np.testing.assert_allclose(p[0],a,rtol=0,atol=1e-12);assert np.all(np.diff(p)<=1e-7)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,unfiltered_soh=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                        cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
                    np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                    try:predict(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:predict(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinate gate')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(records[pos],r)
                    rows.append(r);curves+=1
            if role=='validation':vr=rows
        s=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,width=width,shape_parameter_count=b['parameter_count'],score=float(s.mean()+.25*np.percentile(s,90)),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=cfg['control']
    expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best);assert selection['expand']==expand and curves==214
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,fresh_dynamic_refits=2,expand=expand,outer_loaded=False,
        scope='Shared training replay, exact fresh dynamic states/loss; serial frozen tree origin prediction tolerance1e-12; separate NumPy difference equation, origin and selection checks.',
        verifier_sha256=sha(Path(__file__)),equation_helper_sha256=sha(Path('scripts/verify_mit_segment_curve_net.py'))))
    print('PASS: 2 dynamic refits, 214 curves, frozen origin, exact states, formula/leakage/family/selection; expand='+str(expand))


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
