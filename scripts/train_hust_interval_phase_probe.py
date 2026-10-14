"""Small phase-alignment hypothesis test against verified paired baseline."""
from copy import deepcopy
from datetime import datetime,timezone
import json,shutil,time,traceback
from pathlib import Path
import joblib,numpy as np
from battery_smetimes import interval_phase_clock as model
from scripts.train_continuous_length_pilot import target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    baseline=Path('experiments/hust_voltage_intervals/20261008T074922_391629Z')
    cfg=json.loads((baseline/'config.json').read_text());assert json.loads((baseline/'verification.json').read_text())['passed']
    root=Path('experiments/hust_interval_phase_probe')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    source=[Path(__file__),Path('battery_smetimes/interval_phase_clock.py'),Path('battery_smetimes/voltage_interval_features.py'),Path('battery_smetimes/continuous_length.py')]
    (root/'source_snapshot').mkdir()
    for path in source:shutil.copy2(path,root/'source_snapshot'/path.name)
    cfg.update(origin=100,baseline=str(baseline.resolve()),baseline_verification_sha256=sha(baseline/'verification.json'),
        source_sha256={str(p):sha(p) for p in source},hypothesis='Fix cycle-dependent donor survival by predicting a prefix-signal 90%-SOH clock.',
        candidate='interval_phase_clock',budget=dict(clock_trees=20,curve_trees=60,total_trees=80),
        limitation='One seed/fold only; 90% crossing is training supervision, not an inference input; unknown future conditions assumed unchanged.')
    save_json(root/'config.json',cfg);cells={};begin=time.perf_counter()
    print('RUN='+str(root.resolve()),flush=True)
    try:
        for key in ['source_index','physics_index']:
            for r in json.loads(Path(cfg[key]).read_text())['cells']:
                cid=r['cell_id']
                if cid not in cfg['train']+cfg['validation']:continue
                assert sha(Path(r['path']))==r['sha256']
                with np.load(r['path'],allow_pickle=False) as data:cells.setdefault(cid,{}).update({k:data[k].copy() for k in data.files})
                if 'physical_family' in r:cells[cid]['physical_family']=r['physical_family']
        assert not set(cells)&set(cfg['test'])
        targets={cid:target(cell,100) for cid,cell in cells.items()}
        bundle=model.fit(cells,cfg['train'],targets,100,20261008);path=root/'model.joblib';joblib.dump(bundle,path)
        replay=joblib.load(path);arrays={};rows=[];clock_rows=[]
        for cid in cfg['validation']:
            c,y=targets[cid];p=model.predict(bundle,cells[cid],c)
            np.testing.assert_allclose(p,model.predict(replay,cells[cid],c),rtol=0,atol=1e-12)
            poison=deepcopy(cells[cid]);poison['truth']=np.inf;poison['cycle']=np.array([np.inf])
            poison['early_raw'][poison['early_cycle']>100]=np.inf
            poison['physics_raw'][poison['physics_cycle']>100]=np.inf
            np.testing.assert_allclose(p,model.predict(bundle,poison,c),rtol=0,atol=1e-12)
            try:model.predict(bundle,dict(cells[cid],physical_family='wrong'),c)
            except ValueError:pass
            else:raise AssertionError('family gate')
            arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
            for horizon,mask in [('operational',np.ones(len(c),bool)),('common',c>200)]:
                rows.append(dict(cell_id=cid,origin=100,seed=20261008,candidate='interval_phase_clock',horizon=horizon,**evaluate(c[mask],y[mask],p[mask])))
            clock_rows.append(dict(cell_id=cid,inferred_cycles_remaining=model.inferred_duration(bundle,cells[cid]),
                observed_cycles_remaining_for_diagnostics_only=model.crossing90(c,y,100)))
        np.savez_compressed(root/'predictions.npz',**arrays);save_json(root/'metrics.json',rows);save_json(root/'clock_diagnostics.json',clock_rows)
        for horizon in ['operational','common']:
            rr=[r for r in rows if r['horizon']==horizon]
            print(horizon,json.dumps({k:float(np.mean([r[k] for r in rr])) for k in ['mape_percent','late20_mae_pp','stage_error_pp','acceleration_direction_reversed']}),flush=True)
        save_json(root/'progress.json',dict(terminal=True,success=True,seconds=time.perf_counter()-begin,model_sha256=sha(path),goal_complete=False))
        (root/'training.log').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(terminal=True,success=False));raise


if __name__=='__main__':main()
