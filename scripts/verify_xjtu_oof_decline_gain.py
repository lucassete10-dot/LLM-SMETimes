"""Fresh OOF-model replay and separate scalar gain/composition/score arithmetic."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_segment_features import fit,predict as base_predict
from battery_smetimes.xjtu_decline_gain import predict
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];ids=plan['train'];cells=load_subset(ids+plan['validation'])
    assert (len(ids),len(plan['validation']),len(plan['test']))==(31,13,11)
    assert set(cells)==set(ids+plan['validation']) and not set(cells)&set(plan['test'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    splits=json.loads((root/'oof_splits.json').read_text());oof={};component_count=0
    for i,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=cfg['seed']).split(ids)):
        train=[ids[j] for j in tr];held=[ids[j] for j in va]
        assert splits[i]==dict(index=i,train=train,held=held) and not set(train)&set(held)
        out=root/f'oof_{i}';b=joblib.load(out/'model.joblib');serial_trees(b)
        assert b['train_ids']==train and b['parameters']==cfg['parameters'] and b['seed']==cfg['seed']
        fresh=fit(cells,train,cfg['parameters'],cfg['seed']);serial_trees(fresh);component_count+=1+len(fresh['local'])
        assert set(b['local'])==set(fresh['local'])
        np.testing.assert_array_equal(b['imputer'].statistics_,fresh['imputer'].statistics_)
        for key,local in b['local'].items():
            assert local['train_ids']==fresh['local'][key]['train_ids'] and not set(local['train_ids'])&set(held)
            np.testing.assert_array_equal(local['imputer'].statistics_,fresh['local'][key]['imputer'].statistics_)
        with np.load(out/'predictions.npz',allow_pickle=False) as a:
            assert set(a.files)=={k+'__'+f for k in held for f in ('cycle','truth','prediction')}
            for k in held:
                c,y,p=[a[k+'__'+f] for f in ('cycle','truth','prediction')];cc,yy=targets(cells[k])
                np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                np.testing.assert_allclose(p,base_predict(b,clean(cells[k]),c),rtol=0,atol=1e-12)
                np.testing.assert_allclose(p,base_predict(fresh,clean(cells[k]),c),rtol=0,atol=1e-12)
                assert k not in oof;oof[k]=(c,y,p)
    assert set(oof)==set(ids)
    xx=xy=0.
    # Match original fold-produced record order, but separately compute products.
    for split in splits:
        for k in split['held']:
            c,y,p=oof[k];idx=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int))
            w=np.linspace(1,3,len(idx));w=w/(np.sum(w)*31);dp=p[0]-p[idx];dy=y[0]-y[idx]
            xx+=float(np.sum(w*dp**2));xy+=float(np.sum(w*dp*dy))
    base=joblib.load(cfg['model_file']);serial_trees(base);scores=[];curves=0
    for i,penalty in enumerate([None]+cfg['penalties']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');serial_trees(b['base'])
        g=1. if penalty is None else float(np.clip((xy+penalty)/(xx+penalty),.5,2))
        np.testing.assert_allclose(b['gain'],g,rtol=0,atol=1e-12)
        assert b['train_ids']==ids and b['base']['train_ids']==ids and b['penalty']==penalty
        if penalty is not None:close(b['statistics'],dict(sum_weighted_pred_squared=xx,sum_weighted_pred_true=xy,penalty=penalty))
        for role,keys in [('training',ids),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text());assert len(records)==len(keys);rows=[]
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in keys for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(keys):
                    c,y,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')];cc,yy=targets(cells[k])
                    np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy);q=clean(cells[k])
                    pred=base_predict(base,q,c);anchor=base_predict(base,q,[101.])[0]
                    np.testing.assert_allclose(p,anchor+g*(pred-anchor),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(predict(b,q,[101.])[0],anchor,rtol=0,atol=1e-12)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                        cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,q['raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                    try:predict(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(records[pos],r)
                    rows.append(r);curves+=1
            if role=='validation':vr=rows
        s=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,penalty=penalty,gain=g,score=float(s.mean()+.25*np.percentile(s,90)),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best);assert selection['expand']==expand and curves==132
    assert json.loads((root/'progress.json').read_text())['oof_component_fits']==component_count
    save_json(root/'independent_verification.json',dict(passed=True,fresh_oof_bundles=5,fresh_component_refits=component_count,
        oof_cells=31,curves=curves,expand=expand,outer_loaded=False,verifier_sha256=sha(Path(__file__)),
        scope='Shared tree/feature fit replay; separate gain weighted-product/anchored composition/score arithmetic, predictions not full-state identity.'))
    print('PASS:5freshOOF bundles,31heldcells,132curves,gain/anchor/poison/family/selection; expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
