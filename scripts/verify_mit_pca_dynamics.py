"""Fresh PCA/head replay; independently written relative curve/score arithmetic."""
import json,sys
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_pca_dynamics import fit,predict
from battery_smetimes.mit_fixed_origin_dynamics import origin
from battery_smetimes.mit_segment_features import prefix
from battery_smetimes.curve_fidelity import targets
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def equation(b,cell,c):
    x=b['imputer'].transform(prefix(cell,b['parameters'])[None]);x=b['scaler'].transform(x)
    z=(x-b['pca'].mean_)@b['pca'].components_.T
    u=(z@b['state']['weight'].T+b['state']['bias'])[0]
    rate=.02*np.logaddexp(0,u[0]);accel=.003*np.logaddexp(0,u[1]);tau=np.logaddexp(0,u[2])
    h=(c-100)/100;f=.1*np.logaddexp(0,(h-tau)/.1);f0=.1*np.logaddexp(0,(.01-tau)/.1)
    return origin(b['frozen'],cell)-rate*(h-.01)-accel*(f*f-f0*f0)


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(83,24,27)
    assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    assert not set(plan['train'])&set(plan['validation'])
    parent=Path(cfg['parent']);assert sha(parent/'config.json')==cfg['parent_config_sha256']
    assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    close(cfg['control'],json.loads((parent/'config.json').read_text())['control'])
    assert sha(Path(cfg['origin_model_file']))==cfg['origin_model_sha256']
    frozen=joblib.load(cfg['origin_model_file'])['frozen'];serial_trees(frozen)
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    scores=[];count=0
    for i,n in enumerate(cfg['components']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');serial_trees(b['frozen'])
        assert b['train_ids']==plan['train'] and b['seed']==cfg['seed'] and b['epochs']==cfg['epochs']
        fresh=fit(cells,plan['train'],n,cfg['seed'],frozen,cfg['epochs'])
        for k,v in b['state'].items():np.testing.assert_array_equal(v,fresh['state'][k])
        for obj,fields in [('imputer',['statistics_']),('scaler',['mean_','scale_','var_']),
            ('pca',['components_','mean_','explained_variance_','singular_values_'])]:
            for field in fields:np.testing.assert_array_equal(getattr(b[obj],field),getattr(fresh[obj],field))
        assert b['head_parameter_count']==sum(v.size for v in b['state'].values())==3*(n+1)
        np.testing.assert_array_equal(b['history'],fresh['history']);np.testing.assert_array_equal(b['history'],json.loads((out/'epoch_loss.json').read_text()))
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text());assert len(records)==len(ids);rows=[]
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as arrays:
                assert set(arrays.files)=={k+'__100__'+field for k in ids for field in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p=[arrays[k+'__100__'+field] for field in ('cycle','truth','prediction')]
                    cc,yy=targets(cells[k]);np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                    q=clean(cells[k]);np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(p,predict(fresh,q,c),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(p,equation(b,q,c),rtol=0,atol=2e-6)
                    np.testing.assert_allclose(p[0],origin(frozen,q),rtol=0,atol=1e-12);assert np.all(np.diff(p)<=1e-7)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                        cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
                    np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                    try:predict(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:predict(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinates')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(records[pos],r)
                    for field,v in dict(mape_percent=100*np.mean(abs(p-y)/y),endpoint_pp=100*abs(p[-1]-y[-1]),
                        drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1]))).items():np.testing.assert_allclose(r[field],v,rtol=0,atol=1e-10)
                    rows.append(r);count+=1
            if role=='validation':vr=rows
        s=np.array([np.mean([r[field] for field in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,components=n,head_parameter_count=b['head_parameter_count'],score=float(s.mean()+.25*np.percentile(s,90)),
            **{field:float(np.mean([r[field] for r in vr])) for field in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=cfg['control']
    expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best);assert selection['expand']==expand and count==214
    save_json(root/'independent_verification.json',dict(passed=True,curves=count,fresh_pca_head_refits=2,expand=expand,
        outer_loaded=False,verifier_sha256=sha(Path(__file__)),
        scope='Shared fit/feature replay, exact PCA/head/loss statistics; separately written NumPy curve/PCA/selection, query isolation.'))
    print('PASS: 2fresh PCA/heads,214curves, stats/state/loss exact, equation/poison/family/selection; expand='+str(expand))


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
