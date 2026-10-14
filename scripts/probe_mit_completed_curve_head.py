"""Test whether completed-prefix features benefit a continuous curve head."""
import argparse,json,time
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from battery_smetimes import continuous_length as model
from scripts.train_core_length_cv import load
from scripts.train_continuous_length_pilot import target,evaluate
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('cache',type=Path);args=ap.parse_args()
    original=json.loads((args.cache/'config.json').read_text());plans=original['plans']
    cells=load(plans[0]);expanded={}
    for cid,cell in cells.items():
        with np.load(args.cache/(cid+'.features.npz')) as d:
            expanded[cid]=dict(cell,early_cycle=d['cycle'].copy(),early_raw=d['raw'].copy(),conditions=d['conditions'].copy())
    root=Path('experiments/mit_completed_curve_head')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    print('RUN='+str(root.resolve()),flush=True)
    cfg=dict(plans=plans,origin=100,seed=20261008,cache=str(args.cache),independent_test=False,
        hypothesis='With richer completed-past signals, a continuous level/rate/knee/acceleration head may reduce tail platforms. Reject if trajectory error worsens.',
        source_sha256={str(p):sha(p) for p in [Path(__file__),Path('battery_smetimes/continuous_length.py')]})
    save_json(root/'config.json',cfg);rows=[];targets={c:target(cell,100) for c,cell in cells.items()};start=time.perf_counter()
    for plan in plans:
        for arm,src in [('prefix300',cells),('completed_past',expanded)]:
            folder=root/plan['name']/arm;folder.mkdir(parents=True)
            bundle=model.fit(src,plan['train'],100,'continuous','VIT',1,cfg['seed'],targets)
            joblib.dump(bundle,folder/'model.joblib');arrays={}
            for cid in plan['validation']:
                c,y=targets[cid];p=model.predict(bundle,src[cid],c)
                poison=deepcopy(src[cid]);poison['soh']=np.inf;poison['raw'][poison['cycle']>100]=np.inf
                if 'early_raw' in poison:poison['early_raw'][poison['early_cycle']>100]=np.inf
                np.testing.assert_allclose(p,model.predict(bundle,poison,c),rtol=0,atol=1e-12)
                np.testing.assert_allclose(p,model.predict(joblib.load(folder/'model.joblib'),src[cid],c),rtol=0,atol=1e-12)
                arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                rows.append(dict(cell_id=cid,fold=plan['name'],arm=arm,**evaluate(c,y,p)))
            np.savez_compressed(folder/'predictions.npz',**arrays);save_json(root/'all_rows.json',rows)
            rr=[r for r in rows if r['arm']==arm]
            print(plan['name'],arm,'MAPE',np.mean([r['mape_percent'] for r in rr]),'late',np.mean([r['late20_mae_pp'] for r in rr]),'stage',np.mean([r['stage_error_pp'] for r in rr]),flush=True)
    save_json(root/'progress.json',dict(success=True,terminal=True,seconds=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
