"""Validation-only retrofit of verified fitted weights; no new test-cell access."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes import supported_curve as model
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    source=Path('experiments/continuous_length/20261008T014005_004506Z')
    assert json.loads((source/'independent_replay_verification.json').read_text())['passed']
    cfg=json.loads((source/'config.json').read_text());root=Path('experiments/supported_curve')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    files=[Path(__file__),Path('battery_smetimes/supported_curve.py')]
    save_json(root/'config.json',dict(source=str(source.resolve()),source_config_sha256=sha(source/'config.json'),
        plans=cfg['plans'],source_sha256={str(f):sha(f) for f in files},outer_loaded=False,independent_test=False,
        rule='curve-space leaf averaging, train-support linear tails, separate point anchor',goal_complete=False))
    rowsall=[];done=0;print('RUN='+str(root.resolve()),flush=True)
    for plan in cfg['plans']:
        cells,_=load_nominal(plan);assert not set(cells)&set(plan['test']);targets={k:target(c,100) for k,c in cells.items()}
        base=source/plan['name']/'100'/str(cfg['seeds'][0]);anchor=joblib.load(base/'point_VIT_leaf1'/'model.joblib')
        for candidate in ('continuous_V_leaf1','continuous_VIT_leaf3'):
            cc=json.loads((base/candidate/'config.json').read_text());assert sha(base/candidate/'model.joblib')==cc['model_sha256']
            shape=joblib.load(base/candidate/'model.joblib');b=model.retrofit(shape,anchor,cells,targets)
            folder=root/plan['name']/candidate;folder.mkdir(parents=True);joblib.dump(b,folder/'model.joblib');reloaded=joblib.load(folder/'model.joblib')
            rows=[];arrays={}
            for key in cc['validation']:
                cell=cells[key];c,y=targets[key];p=model.predict(b,cell,c)
                np.testing.assert_allclose(p,model.predict(reloaded,cell,c),rtol=0,atol=1e-12)
                poison=deepcopy(cell)
                if 'early_cycle' in cell:
                    poison['early_raw'][poison['early_cycle']>100]=np.inf;poison['truth']=np.full_like(poison['truth'],np.inf);poison['cycle']=np.array([np.inf])
                else:
                    poison['raw'][poison['cycle']>100]=np.inf;poison['soh']=np.full_like(poison['soh'],np.inf);poison['reference_capacity_ah']=np.inf
                poison['cycle_life']=np.inf
                np.testing.assert_allclose(p,model.predict(b,poison,c),rtol=0,atol=1e-12)
                assert max(np.diff(model.predict(b,cell,np.linspace(c[0],c[-1],400))))<=1e-10
                rows.append(dict(cell_id=key,**evaluate(c,y,p)))
                arrays.update({key+'__'+n:v for n,v in [('cycle',c),('truth',y),('prediction',p)]})
            save_json(folder/'config.json',dict(model_sha256=sha(folder/'model.joblib'),source_model_sha256=cc['model_sha256'],used_train=b['train_ids'],validation=cc['validation']))
            save_json(folder/'metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
            rowsall.extend(dict(dataset=plan['dataset'],candidate=candidate,origin=100,seed=cfg['seeds'][0],**r) for r in rows)
            done+=1;save_json(root/'all_rows.json',rowsall)
            print(f'{plan["dataset"]} {candidate}: MAPE={np.mean([r["mape_percent"] for r in rows]):.3f} screen={np.mean([r["screen_pass"] for r in rows]):.3f}',flush=True)
    save_json(root/'progress.json',dict(completed_models=done,terminal=True,success=True,goal_complete=False))
    print('COMPLETE '+str(root.resolve()),flush=True)


if __name__=='__main__':main()
