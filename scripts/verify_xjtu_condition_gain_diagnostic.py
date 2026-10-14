"""Independent OOF weighted-product/condition/metrics check; no calibration performance claim."""
import json,sys
from pathlib import Path
import numpy as np
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train'])
    assert len(cells)==31 and not set(cells)&set(plan['test']+plan['validation'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    parent=Path(cfg['parent']);records={};conditions={}
    for s in json.loads((parent/'oof_splits.json').read_text()):
        assert not set(s['held'])&set(s['train']) and set(s['held']+s['train'])==set(plan['train'])
        with np.load(parent/f"oof_{s['index']}"/'predictions.npz',allow_pickle=False) as a:
            for k in s['held']:
                assert k not in records
                records[k]=[a[k+'__'+f] for f in ('cycle','truth','prediction')]
                m=cells[k]['metadata'];conditions[k]=[m.get('charge_protocol'),m.get('discharge_rate_c'),m.get('discharge_cutoff_v')]
    assert set(records)==set(plan['train'])
    result=json.loads((root/'diagnostic.json').read_text());assert result['validation_loaded']==result['outer_loaded']==False
    counted=[];curves=0
    for group in [result['family']]+result['conditions']:
        ids=group['cell_ids'];assert len(ids)==len(set(ids))==group['count']
        if 'condition' in group:
            expected=[k for k in records if conditions[k]==group['condition']];assert ids==expected;counted+=ids
        else:assert set(ids)==set(records)
        assert group['local_support_at_least6']==(len(ids)>=6)
        xs=[];ys=[];ws=[];ends=[]
        for k in ids:
            c,y,p=records[k];ix=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int))
            w=np.linspace(1,3,len(ix));ws.extend(w/(sum(w)*len(ids)));xs.extend(p[0]-p[ix]);ys.extend(y[0]-y[ix]);ends.append((p[0]-p[-1],y[0]-y[-1]))
        x,y,w=map(np.array,(xs,ys,ws));e=np.array(ends)
        prod=dict(xx=float(np.sum(w*x*x)),xy=float(np.sum(w*x*y)),endpoint_xx=float(np.mean(e[:,0]**2)),endpoint_xy=float(np.mean(e[:,0]*e[:,1])))
        close(group['weighted_products'],prod)
        gains=dict(trajectory=None if prod['xx']==0 else float(np.clip(prod['xy']/prod['xx'],.5,2)),endpoint=None if prod['endpoint_xx']==0 else float(np.clip(prod['endpoint_xy']/prod['endpoint_xx'],.5,2)))
        close(group['gains'],gains)
        for name,gain in [('original',1.)]+list(gains.items()):
            if gain is None:continue
            method=group['methods'][name];np.testing.assert_allclose(method['gain'],gain,rtol=0,atol=1e-12);rows=[]
            for i,k in enumerate(ids):
                c,y,p=records[k];q=p[0]+gain*(p-p[0]);n=max(1,len(y)//2)
                row=dict(cell_id=k,**detail(c,y,q),start_error_pp=float(100*abs(q[0]-y[0])),early_bias_pp=float(100*np.mean(q[:n]-y[:n])),late_bias_pp=float(100*np.mean(q[n:]-y[n:])) if n<len(y) else None)
                close(method['rows'][i],row);rows.append(row);curves+=1
            close(method['mean'],{f:float(np.mean([r[f] for r in rows if r[f] is not None])) for f in method['mean']})
    assert len(counted)==31 and set(counted)==set(records) and curves==186
    save_json(root/'independent_verification.json',dict(passed=True,train_cells=31,groups=len(result['conditions']),metric_rows=curves,validation_loaded=False,outer_loaded=False,
        scope='Independent grouping/products/gain algebra; shared metrics helper, descriptive calibration not held-out performance',verifier_sha256=sha(Path(__file__))))
    print('PASS:TRAIN31 conditions/products/gains/186metrics/SHA; validation and outer not loaded')


if __name__=='__main__':main(Path(sys.argv[1]))
