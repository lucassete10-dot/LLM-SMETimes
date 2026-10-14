"""Fresh network replay plus separately evaluated curve equation and metrics."""
import json,sys
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_segment_curve_net import fit,predict
from battery_smetimes.mit_segment_features import prefix
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def equation(b,cell,c):
    z=b['scaler'].transform(b['imputer'].transform(prefix(cell,b['parameters'])[None]))
    state=b['state']
    # NumPy forward implementation, distinct from CurveNet's torch equation.
    hidden=np.tanh(z@state['network.0.weight'].T+state['network.0.bias'])
    u=(hidden@state['network.2.weight'].T+state['network.2.bias'])[0]
    sigmoid=lambda t:1/(1+np.exp(-t))
    a=.5+.75*sigmoid(u[0]);rate=.02*np.logaddexp(0,u[1])
    accel=.003*np.logaddexp(0,u[2]);tau=np.logaddexp(0,u[3])
    h=(c-100)/100;hinge=.1*np.logaddexp(0,(h-tau)/.1)
    return a-rate*h-accel*hinge**2


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(83,24,27)
    assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    assert not set(plan['train'])&set(plan['validation'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256']
    assert sha(Path(cfg['parent'])/'independent_verification.json')==cfg['parent_proof_sha256']
    sf=Path(cfg['control_selection_file']);assert sha(sf)==cfg['control_selection_sha256']
    close(cfg['control'],json.loads(sf.read_text())['selected'])
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    scores=[];curves=0
    for i,width in enumerate(cfg['widths']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib')
        assert b['train_ids']==plan['train'] and b['seed']==cfg['seed'] and b['epochs']==cfg['epochs']
        fresh=fit(cells,plan['train'],width,cfg['seed'],cfg['epochs'])
        for k,v in b['state'].items():np.testing.assert_array_equal(v,fresh['state'][k])
        np.testing.assert_array_equal(b['imputer'].statistics_,fresh['imputer'].statistics_)
        np.testing.assert_array_equal(b['scaler'].mean_,fresh['scaler'].mean_)
        np.testing.assert_array_equal(b['scaler'].scale_,fresh['scaler'].scale_)
        np.testing.assert_array_equal(b['history'],fresh['history'])
        assert b['parameter_count']==sum(a.size for a in b['state'].values())
        np.testing.assert_array_equal(json.loads((out/'epoch_loss.json').read_text()),b['history'])
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text());rows=[]
            assert len(records)==len(ids)
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as arrays:
                assert set(arrays.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p=[arrays[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                    cc,yy=targets(cells[k]);np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                    q=clean(cells[k]);np.testing.assert_array_equal(p,predict(b,q,c));np.testing.assert_array_equal(p,predict(fresh,q,c))
                    np.testing.assert_allclose(p,equation(b,q,c),rtol=0,atol=2e-6)
                    assert np.all(np.diff(p)<=1e-7)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,unfiltered_soh=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                        cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
                    np.testing.assert_array_equal(p,predict(b,poison,c))
                    try:predict(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:predict(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinate gate')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])))
                    close(records[pos],r)
                    for f,v in dict(mape_percent=100*np.mean(abs(p-y)/y),endpoint_pp=100*abs(p[-1]-y[-1]),
                        drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1]))).items():np.testing.assert_allclose(r[f],v,rtol=0,atol=1e-10)
                    rows.append(r);curves+=1
            if role=='validation':vr=rows
        s=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,width=width,parameter_count=b['parameter_count'],score=float(s.mean()+.25*np.percentile(s,90)),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=cfg['control']
    expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best);assert selection['expand']==expand and curves==214
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,fresh_network_refits=2,expand=expand,
        scope='Shared training/feature replay with exact states; separate NumPy forward equation tolerance2e-6 and metric/selection checks.',
        outer_loaded=False,verifier_sha256=sha(Path(__file__))))
    print('PASS: 2 fresh networks, 214 curves, exact states, NumPy equation, poison/family/monotonicity/metrics; expand='+str(expand))


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
