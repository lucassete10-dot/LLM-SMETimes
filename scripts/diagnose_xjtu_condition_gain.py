"""TRAIN-only OOF calibration tradeoffs, explicitly not held-out calibrated accuracy."""
import json, shutil, logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from battery_smetimes.curve_fidelity import condition_key
from scripts.train_curve_fidelity import load_subset
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha, save_json

PARENT=Path('experiments/xjtu_oof_decline_gain/20261007T060106_225247Z')


def summaries(records):
    xx=xy=exx=exy=0.
    for r in records:
        y,p=r['truth'],r['prediction']
        ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        w=np.linspace(1,3,len(ii));w/=w.sum()*len(records)
        dp=p[0]-p[ii];dy=y[0]-y[ii]
        xx+=float(w@(dp*dp));xy+=float(w@(dp*dy))
        exx+=float((p[0]-p[-1])**2/len(records))
        exy+=float((p[0]-p[-1])*(y[0]-y[-1])/len(records))
    gains={'trajectory':None if xx==0 else float(np.clip(xy/xx,.5,2)),
           'endpoint':None if exx==0 else float(np.clip(exy/exx,.5,2))}
    methods={}
    for name,g in [('original',1.)]+list(gains.items()):
        if g is None:continue
        rows=[]
        for r in records:
            c,y,p=r['cycle'],r['truth'],r['prediction']
            # Relative diagnostic anchored at first available OOF prediction, not cycle101 inference.
            q=p[0]+g*(p-p[0]);n=max(1,len(y)//2)
            rows.append(dict(cell_id=r['cell_id'],**detail(c,y,q),
                start_error_pp=float(100*abs(q[0]-y[0])),
                early_bias_pp=float(100*np.mean(q[:n]-y[:n])),
                late_bias_pp=float(100*np.mean(q[n:]-y[n:])) if n<len(y) else None))
        fields=('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp','early_bias_pp','late_bias_pp')
        methods[name]=dict(gain=g,rows=rows,mean={f:float(np.mean([r[f] for r in rows if r[f] is not None])) for f in fields})
    return dict(count=len(records),cell_ids=[r['cell_id'] for r in records],local_support_at_least6=len(records)>=6,
        weighted_products=dict(xx=xx,xy=xy,endpoint_xx=exx,endpoint_xy=exy),gains=gains,methods=methods,
        coverage=[dict(cell_id=r['cell_id'],points=len(r['cycle']),span=float(r['cycle'][-1]-r['cycle'][0])) for r in records])


def main():
    cfg=json.loads((PARENT/'config.json').read_text());plan=cfg['plan']
    assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    cells=load_subset(plan['train']);assert not set(cells)&set(plan['validation']+plan['test'])
    root=Path('experiments/xjtu_condition_gain_diagnostic')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir();print('RUN='+str(root.resolve()),flush=True)
    files=[Path(__file__),Path('docs/soh_xjtu_condition_gain_diagnostic_20261007.md'),Path('battery_smetimes/curve_fidelity.py'),Path('scripts/train_nasa_robust_inner.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    deps=[PARENT/'config.json',PARENT/'independent_verification.json',PARENT/'oof_splits.json'];records=[]
    for s in json.loads((PARENT/'oof_splits.json').read_text()):
        assert not set(s['held'])&set(s['train']) and set(s['held']+s['train'])==set(plan['train'])
        path=PARENT/f"oof_{s['index']}"/'predictions.npz';deps.append(path)
        with np.load(path,allow_pickle=False) as a:
            for k in s['held']:records.append(dict(cell_id=k,**{f:a[k+'__'+f] for f in ('cycle','truth','prediction')}))
    assert len(records)==31 and {r['cell_id'] for r in records}==set(plan['train'])
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),plan=plan,input_sha256={str(f.resolve()):sha(f) for f in deps},
        source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},validation_loaded=False,outer_loaded=False))
    groups={}
    for r in records:groups.setdefault(condition_key(cells[r['cell_id']]),[]).append(r)
    result=dict(family=summaries(records),conditions=[dict(condition=list(k),**summaries(v)) for k,v in groups.items()],
        scope='TRAIN OOF descriptive calibration resubstitution; not independent calibration evaluation',validation_loaded=False,outer_loaded=False)
    save_json(root/'diagnostic.json',result)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'diagnostic.log',encoding='utf-8'),logging.StreamHandler()])
    for g in [result['family']]+result['conditions']:logging.info('group=%s n=%s gains=%s',g.get('condition','family'),g['count'],g['gains'])
    save_json(root/'progress.json',dict(complete=True,new_predictive_fits=0,train_cells=31,groups=len(groups),goal_complete=False))


if __name__=='__main__':main()
