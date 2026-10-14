"""Frozen-model evaluation replay and separate selection arithmetic."""
import json,sys
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_fixed_origin_dynamics import predict,origin
from scripts.verify_mit_segment_curve_net import equation
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['validation'])
    assert len(cells)==24 and set(cells)==set(plan['validation']) and not set(cells)&set(plan['train']+plan['test'])
    parent=Path(cfg['parent']);assert sha(parent/'config.json')==cfg['parent_config_sha256']
    assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    assert sha(Path(cfg['old_config_file']))==cfg['old_config_sha256']
    close(cfg['control'],json.loads(Path(cfg['old_config_file']).read_text())['control'])
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    scores=[];count=0
    for i,(f,d) in enumerate(cfg['model_files'].items()):
        assert sha(Path(f))==d;b=joblib.load(f);serial_trees(b['frozen'])
        assert b['train_ids']==plan['train'] and b['seed']==cfg['seed'] and b['epochs']==1600
        out=root/f'candidate_{i}';records=json.loads((out/'validation_metrics.json').read_text());assert len(records)==24;rows=[]
        with np.load(out/'validation_predictions.npz',allow_pickle=False) as arrays:
            assert set(arrays.files)=={k+'__100__'+field for k in plan['validation'] for field in ('cycle','truth','prediction')}
            for pos,k in enumerate(plan['validation']):
                c,y,p=[arrays[k+'__100__'+field] for field in ('cycle','truth','prediction')]
                cc,yy=targets(cells[k]);np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                q=clean(cells[k]);np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12)
                a=origin(b['frozen'],q);np.testing.assert_allclose(p[0],a,rtol=0,atol=1e-12)
                np.testing.assert_allclose(p,a+equation(b,q,c)-equation(b,q,np.array([101.]))[0],rtol=0,atol=2e-6)
                poison=dict(q,soh=np.inf,label_cycle=np.inf,capacity=np.inf,cycle_life=np.inf,
                    cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
                np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                try:predict(b,dict(q,physical_family='INVALID'),c)
                except ValueError:pass
                else:raise AssertionError('family gate')
                r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(records[pos],r)
                rows.append(r);count+=1
        s=np.array([np.mean([r[field] for field in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in rows])
        scores.append(dict(candidate=i,width=b['width'],score=float(s.mean()+.25*np.percentile(s,90)),
            **{field:float(np.mean([r[field] for r in rows])) for field in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=cfg['control']
    expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best);assert selection['expand']==expand and count==48 and cfg['new_fits']==0
    save_json(root/'independent_verification.json',dict(passed=True,curves=48,new_fits=0,expand=expand,outer_loaded=False,
        scope='Frozen models previously refit in Stage84; separate equation/selection arithmetic, shared metric helper and query replay.',verifier_sha256=sha(Path(__file__))))
    print('PASS: 48 frozen validation curves, equation/origin/metrics/selection/poison/isolation; expand='+str(expand))


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
