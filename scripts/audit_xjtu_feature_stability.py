"""Label-free TRAIN-only slope estimator sensitivity and missingness audit."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from prepare_xjtu import FEATURE_NAMES
from battery_smetimes.xjtu_segment_features import columns
from battery_smetimes.curve_fidelity import condition_key
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/xjtu_oof_decline_gain/20261007T060106_225247Z')


def slope_stats(c,v):
    mask=np.isfinite(v);t=c[mask].astype(float);x=v[mask].astype(float)
    if len(t)<2:return dict(finite_points=len(t),span=None,ols=None,theil_sen=None,sensitivity=None,flag=False)
    centered=t-t.mean();ols=float(centered@(x-x.mean())/(centered@centered)*100)
    pairs=[(x[j]-x[i])/(t[j]-t[i])*100 for i in range(len(t)) for j in range(i+1,len(t))]
    robust=float(np.median(pairs));span=float(t[-1]-t[0]);den=max(abs(ols),abs(robust),float(np.ptp(x)*100/span),1e-12)
    sensitivity=float(abs(ols-robust)/den)
    return dict(finite_points=len(t),span=span,ols=ols,theil_sen=robust,sensitivity=sensitivity,flag=sensitivity>.5+1e-12)


def main():
    cfg=json.loads((PARENT/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train'])
    assert len(cells)==31 and not set(cells)&set(plan['validation']+plan['test'])
    cols=columns('VIT');assert all('throughput' not in FEATURE_NAMES[i] for i in cols)
    root=Path('experiments/xjtu_feature_stability')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir();print('RUN='+str(root.resolve()),flush=True)
    files=[Path(__file__),Path('prepare_xjtu.py'),Path('battery_smetimes/xjtu_segment_features.py'),Path('docs/soh_xjtu_feature_stability_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plan=plan,feature_columns=cols,feature_names=[FEATURE_NAMES[i] for i in cols],
        input_sha256={str((PARENT/'config.json').resolve()):sha(PARENT/'config.json')},source_sha256={str(f):sha(f) for f in files},
        prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},validation_loaded=False,outer_loaded=False,labels_used=False,new_exclusions=[]))
    rows=[];arrays={};groups={}
    for k in plan['train']:
        cell=cells[k];mask=cell['cycle']<=100;c=cell['cycle'][mask].astype(float);x=cell['raw'][mask][:,cols]
        assert np.isfinite(c).all() and np.all(np.diff(c)>0) and not np.isinf(x).any()
        fields=[]
        for j,col in enumerate(cols):
            valid=x[np.isfinite(x[:,j]),j]
            fields.append(dict(feature=FEATURE_NAMES[col],finite_fraction=float(np.isfinite(x[:,j]).mean()),constant=bool(len(valid)>0 and np.ptp(valid)<1e-12),
                **slope_stats(c[-10:],x[-10:,j])))
        row=dict(cell_id=k,condition=list(condition_key(cell)),early_records=len(c),first=float(c[0]),last=float(c[-1]),
            quarter_counts=[int(((c>lo)&(c<=hi)).sum()) for lo,hi in [(0,25),(25,50),(50,75),(75,100)]],fields=fields)
        rows.append(row);groups.setdefault(condition_key(cell),[]).append(k);arrays[k+'__cycle']=c;arrays[k+'__signals']=x
    np.savez_compressed(root/'early_signals.npz',**arrays)
    finite=[f for r in rows for f in r['fields'] if f['sensitivity'] is not None]
    summary=dict(cells=31,columns=len(cols),record_range=[min(r['early_records'] for r in rows),max(r['early_records'] for r in rows)],
        empty_quarters=sum(v==0 for r in rows for v in r['quarter_counts']),all_missing_cell_columns=sum(f['finite_fraction']==0 for r in rows for f in r['fields']),
        constant_cell_columns=sum(f['constant'] for r in rows for f in r['fields']),calculable_trends=len(finite),flagged_trends=sum(f['flag'] for f in finite),
        median_sensitivity=float(np.median([f['sensitivity'] for f in finite])),p90_sensitivity=float(np.percentile([f['sensitivity'] for f in finite],90)))
    report=dict(summary=summary,cells=rows,conditions=[dict(condition=list(g),count=len(ids),members=ids,local_supported=len(ids)>=6) for g,ids in groups.items()],
        labels_used=False,validation_loaded=False,outer_loaded=False,new_exclusions=[],goal_complete=False)
    save_json(root/'audit.json',report);save_json(root/'progress.json',dict(complete=True,new_model_fits=0,cells=31));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
