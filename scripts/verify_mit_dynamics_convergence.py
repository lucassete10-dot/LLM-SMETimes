"""Exact optimizer replay and separate TRAIN curve/summary checks."""
import json,sys
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_fixed_origin_dynamics import fit,predict,origin
from scripts.verify_mit_segment_curve_net import equation
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']);parent=Path(cfg['parent'])
    assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
    assert len(cells)==83 and not cfg['validation_loaded'] and not cfg['outer_loaded']
    assert sha(parent/'config.json')==cfg['parent_config_sha256']
    assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for f,d in cfg['parent_artifacts'].items():assert sha(Path(f))==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    summary=json.loads((root/'summary.json').read_text());curves=0
    for i,width in enumerate(cfg['widths']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');old=joblib.load(parent/f'candidate_{i}'/'model.joblib')
        serial_trees(b['frozen']);serial_trees(old['frozen'])
        assert b['train_ids']==plan['train'] and b['seed']==cfg['seed'] and b['epochs']==cfg['epochs']
        fresh=fit(cells,plan['train'],width,cfg['seed'],old['frozen'],cfg['epochs'])
        for k,v in b['state'].items():np.testing.assert_array_equal(v,fresh['state'][k])
        for obj,fields in [('imputer',['statistics_']),('scaler',['mean_','scale_','var_'])]:
            for field in fields:np.testing.assert_array_equal(getattr(b[obj],field),getattr(fresh[obj],field))
        np.testing.assert_array_equal(b['history'],fresh['history']);np.testing.assert_array_equal(b['history'][:400],old['history'])
        np.testing.assert_array_equal(b['history'],json.loads((out/'epoch_loss.json').read_text()))
        records=json.loads((out/'training_metrics.json').read_text());assert len(records)==83;rows=[]
        with np.load(out/'training_predictions.npz',allow_pickle=False) as arrays:
            assert set(arrays.files)=={k+'__100__'+f for k in plan['train'] for f in ('cycle','truth','prediction')}
            for pos,k in enumerate(plan['train']):
                c,y,p=[arrays[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                cc,yy=targets(cells[k]);np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                q=clean(cells[k]);np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12)
                np.testing.assert_allclose(p,predict(fresh,q,c),rtol=0,atol=1e-12)
                anchor=origin(old['frozen'],q)
                np.testing.assert_allclose(p,anchor+equation(b,q,c)-equation(b,q,np.array([101.]))[0],rtol=0,atol=2e-6)
                np.testing.assert_allclose(p[0],anchor,rtol=0,atol=1e-12);assert np.all(np.diff(p)<=1e-7)
                poison=dict(q,soh=np.inf,label_cycle=np.inf,unfiltered_soh=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                    cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
                np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                try:predict(b,dict(q,physical_family='INVALID'),c)
                except ValueError:pass
                else:raise AssertionError('family gate')
                r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(records[pos],r)
                for f,v in dict(mape_percent=100*np.mean(abs(p-y)/y),endpoint_pp=100*abs(p[-1]-y[-1]),
                    drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1]))).items():np.testing.assert_allclose(r[f],v,rtol=0,atol=1e-10)
                rows.append(r);curves+=1
        fields=('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')
        oldrows=json.loads((parent/f'candidate_{i}'/'training_metrics.json').read_text())
        prior=float(np.mean(b['history'][-200:-100]));last=float(np.mean(b['history'][-100:]));change=(prior-last)/prior
        report=dict(candidate=i,width=width,old_training={f:float(np.mean([r[f] for r in oldrows])) for f in fields},
            new_training={f:float(np.mean([r[f] for r in rows])) for f in fields},epoch400_loss=b['history'][399],epoch1600_loss=b['history'][-1],
            prior100_loss=prior,last100_loss=last,last_relative_drop=change,plateau_by_fixed_rule=bool(change<=.01),first400_loss_exact=True)
        close(summary['reports'][i],report)
    assert curves==166 and not summary['adopted'] and summary['new_predictive_fits']==2
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,fresh_refits=2,first400_loss_exact=True,
        validation_loaded=False,outer_loaded=False,adopted=False,verifier_sha256=sha(Path(__file__)),
        scope='Shared fit replay with exact states/loss; separate NumPy equation/metric/plateau arithmetic. TRAIN-only, not generalization.'))
    print('PASS: 2 fresh1600 fits, first400exact, 166curves, formula/metrics/isolation; no adoption')


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
