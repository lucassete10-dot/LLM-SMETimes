"""Paired developmental fold probe: past completed diagnostics vs 120s inputs."""
import json
import time
import traceback
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from scipy.io import loadmat
from battery_smetimes import continuous_length as model
from battery_smetimes.xjtu_full_signal import cycle_features
from prepare_xjtu import diagnostic
from scripts.train_core_length_cv import load
from scripts.train_continuous_length_pilot import target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    baseline=Path('experiments/core_length_cv/20261008T083003_121461Z')
    old=json.loads((baseline/'config.json').read_text());plan=old['plans'][0]
    root=Path('experiments/xjtu_full_signal_probe')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    files=[Path(__file__),Path('battery_smetimes/xjtu_full_signal.py'),Path('battery_smetimes/continuous_length.py')]
    cfg=dict(plan=plan,origins=[50,100],seeds=[20261008,20261009],arms=['prefix120','completed_diagnostic'],
        input_contract='All measured V/I/t/T in completed diagnostic source cycles <= N. No future updates or capacity-label input.',
        baseline=str(baseline.resolve()),signal_index=old['signal_index'],signal_index_sha256=sha(Path(old['signal_index'])),
        source_sha256={str(p):sha(p) for p in files},outer_loaded=False,independent_test=False,goal_complete=False,
        decision_rule='Exploratory only. Require paired tail/stage/MAPE benefit before multi-fold confirmation. Never adopt from one fold.',
        common_future_cutoff=100)
    save_json(root/'config.json',cfg)
    cells=load(plan,old['signal_index']);registry=json.loads(Path(old['registry']).read_text())['cells']
    records=[];rows=[];start=time.perf_counter()
    try:
        assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        expanded={}
        for cid,cell in cells.items():
            source=Path(registry[cid]['raw_source']);data=loadmat(source,simplify_cells=True)['data']
            cycles=[];raw=[];excluded=[];batch=int(cid.split('_')[0][1:])
            for j,r in enumerate(data[:100]):
                if not diagnostic(batch,r['description']):continue
                signal={k:r[k] for k in ('relative_time_min','voltage_V','current_A','temperature_C')}
                try:x=cycle_features(signal)
                except ValueError as exc:
                    excluded.append(dict(cycle=j+1,reason=str(exc)));continue
                cycles.append(j+1);raw.append(x)
            pc,pr=np.asarray(cycles),np.asarray(raw)
            sc,sr,conditions=model.signal_history(cell)
            common=np.intersect1d(sc,pc)
            ci=np.searchsorted(sc,common);pi=np.searchsorted(pc,common)
            expanded[cid]=dict(cell,early_cycle=common,early_raw=np.c_[sr[ci],pr[pi]],conditions=conditions)
            path=root/(cid+'.features.npz')
            np.savez_compressed(path,cycle=common,raw=expanded[cid]['early_raw'],conditions=conditions)
            records.append(dict(cell_id=cid,source_sha256=sha(source),features_sha256=sha(path),observations=len(common),exclusions=excluded))
            print('FEATURE',cid,len(common),flush=True)
        save_json(root/'feature_manifest.json',records)
        for n in cfg['origins']:
            usable=[]
            for cid in cells:
                try:
                    model.features(expanded[cid],n);model.features(cells[cid],n)
                    if len(target(cells[cid],n)[0])<2:continue
                    usable.append(cid)
                except ValueError:continue
            train=[c for c in plan['train'] if c in usable];val=[c for c in plan['validation'] if c in usable]
            targets={c:target(cells[c],n) for c in usable}
            for seed in cfg['seeds']:
                for arm,source_cells in [('prefix120',cells),('completed_diagnostic',expanded)]:
                    folder=root/str(n)/str(seed)/arm;folder.mkdir(parents=True)
                    bundle=model.fit(source_cells,train,n,'point','VIT',1,seed,targets)
                    joblib.dump(bundle,folder/'model.joblib');saved=joblib.load(folder/'model.joblib')
                    arrays={}
                    for cid in val:
                        c,y=targets[cid];p=model.predict(bundle,source_cells[cid],c)
                        np.testing.assert_allclose(p,model.predict(saved,source_cells[cid],c),atol=1e-12,rtol=0)
                        poisoned=deepcopy(source_cells[cid]);poisoned['raw'][poisoned['cycle']>n]=np.inf
                        if 'early_raw' in poisoned:poisoned['early_raw'][poisoned['early_cycle']>n]=np.inf
                        poisoned['soh']=np.inf;poisoned['label_cycle']=np.array([np.inf]);poisoned['reference_capacity_ah']=np.inf
                        np.testing.assert_allclose(p,model.predict(bundle,poisoned,c),atol=1e-12,rtol=0)
                        arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                        for h,mask in [('operational',np.ones(len(c),bool)),('common',c>100)]:
                            if mask.sum()<2:continue
                            rows.append(dict(cell_id=cid,origin=n,seed=seed,arm=arm,horizon=h,**evaluate(c[mask],y[mask],p[mask])))
                    np.savez_compressed(folder/'predictions.npz',**arrays)
                    save_json(folder/'config.json',dict(train=train,validation=val,model_sha256=sha(folder/'model.joblib')))
                    save_json(root/'all_rows.json',rows)
                    rr=[r for r in rows if r['origin']==n and r['seed']==seed and r['arm']==arm and r['horizon']=='common']
                    print(n,seed,arm,'val',len(val),'MAPE',np.mean([r['mape_percent'] for r in rr]),'late',np.mean([r['late20_mae_pp'] for r in rr]),'stage',np.mean([r['stage_error_pp'] for r in rr]),flush=True)
        save_json(root/'progress.json',dict(terminal=True,success=True,models=8,seconds=time.perf_counter()-start,goal_complete=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()))
        save_json(root/'progress.json',dict(terminal=True,success=False));raise


if __name__=='__main__':main()
