"""Fixed outer holdout, five development folds/seeds, operational and matched horizons."""
from copy import deepcopy
from datetime import datetime,timezone
import json,shutil,time,traceback
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import KFold
from battery_smetimes import continuous_length as point
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    source=Path('experiments/continuous_length/20261008T063901_571685Z')
    assert json.loads((source/'independent_replay_verification.json').read_text())['passed']
    original=json.loads((source/'config.json').read_text())['plans'][0]
    dev=sorted(original['train']+original['validation']);outer=original['test']
    assert len(dev)==len(set(dev)) and not set(dev)&set(outer)
    cells,conversion=load_nominal(dict(original,train=dev,validation=[]));assert set(cells)==set(dev)
    origins=[10,20,30,50,75,100];seeds=list(range(20261008,20261013))
    eligible=[];excluded=[]
    for cid in dev:
        try:
            for origin in origins:point.features(cells[cid],origin)
            if len(target(cells[cid],100)[0])<2:raise ValueError('insufficient common future labels')
            eligible.append(cid)
        except ValueError as e:excluded.append(dict(cell_id=cid,reason=str(e),not_bad_cell=True))
    plans=[]
    for i,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=20261008).split(eligible),1):
        plans.append(dict(name=f'NCA_inner{i}',dataset='Zhu2022_NCA',train=[eligible[j] for j in tr],validation=[eligible[j] for j in va],test=outer))
    root=Path('experiments/nca_length_cv')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    files=[Path(__file__),Path('battery_smetimes/continuous_length.py'),Path('scripts/train_continuous_length_pilot.py')]
    (root/'source_snapshot').mkdir()
    for p in files:shutil.copy2(p,root/'source_snapshot'/p.name)
    save_json(root/'config.json',dict(plans=plans,origins=origins,seeds=seeds,original_plan=original,development_cells=eligible,
        excluded=excluded,source_sha256={str(p):sha(p) for p in files},label_conversion=conversion,
        training_target_cutoff='input_origin',common_evaluation_cutoff=100,operational_evaluation_cutoff='input_origin',
        input_contract='prefix VIT and known conditions only',outer_loaded=False,independent_test=False,
        length_rule=dict(version='development_band_v1',numeric_goal_mape=6,minimum_cell_numeric_pass_fraction=.9,
            mean_late_mae_pp_limit=2,mean_stage_pp_limit=3,maximum_reversal_fraction=.1,
            near_best_mape_pp=.25,near_best_late_mae_pp=.5,near_best_stage_pp=.5,
            scope='Provisional development recommendation only; require both matched and operational evaluation and separately report shape failures; knee/platform gates not yet certified.'),
        limitation='Previously examined development cells, not independent tests. Training afterN differs from earlier common-after100 pilot; no direct performance attribution.',goal_complete=False))
    start=time.perf_counter();done=0;rows=[];print('RUN='+str(root.resolve()),flush=True)
    try:
        for plan in plans:
            for origin in origins:
                targets={k:target(cells[k],origin) for k in plan['train']+plan['validation']}
                for seed in seeds:
                    folder=root/plan['name']/str(origin)/str(seed);folder.mkdir(parents=True)
                    t=time.perf_counter();b=point.fit(cells,plan['train'],origin,'point','VIT',1,seed,targets)
                    path=folder/'model.joblib';joblib.dump(b,path);replay=joblib.load(path);batch=[];arrays={}
                    for cid in plan['validation']:
                        c,y=targets[cid];p=point.predict(b,cells[cid],c)
                        np.testing.assert_allclose(p,point.predict(replay,cells[cid],c),rtol=0,atol=1e-12)
                        poison=deepcopy(cells[cid]);poison['early_raw'][poison['early_cycle']>origin]=np.inf
                        poison['truth']=np.inf;poison['cycle']=np.array([np.inf]);poison['cycle_life']=np.inf
                        np.testing.assert_allclose(p,point.predict(b,poison,c),rtol=0,atol=1e-12)
                        arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                        for horizon,mask in [('operational',np.ones(len(c),dtype=bool)),('common',c>100)]:
                            row=dict(dataset='Zhu2022_NCA',fold=plan['name'],origin=origin,seed=seed,cell_id=cid,horizon=horizon,**evaluate(c[mask],y[mask],p[mask]));batch.append(row);rows.append(row)
                        for distance in (50,100,200):
                            mask=c<=origin+distance
                            if mask.sum()>=2:
                                row=dict(dataset='Zhu2022_NCA',fold=plan['name'],origin=origin,seed=seed,cell_id=cid,horizon=f'fixed_{distance}',**evaluate(c[mask],y[mask],p[mask]),full_horizon_covered=bool(c[-1]>=origin+distance));batch.append(row);rows.append(row)
                    save_json(folder/'config.json',dict(train=plan['train'],validation=plan['validation'],origin=origin,seed=seed,model_sha256=sha(path),seconds=time.perf_counter()-t))
                    save_json(folder/'metrics.json',batch);np.savez_compressed(folder/'predictions.npz',**arrays)
                    done+=1;save_json(root/'all_rows.json',rows);save_json(root/'progress.json',dict(completed_fits=done,planned_fits=150,terminal=False,seconds=time.perf_counter()-start))
                selected=[r for r in rows if r['fold']==plan['name'] and r['origin']==origin and r['horizon']=='common']
                print(f'{plan["name"]} N={origin} 5-seed mean MAPE={np.mean([r["mape_percent"] for r in selected]):.3f}',flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,planned_fits=150,terminal=True,success=True,seconds=time.perf_counter()-start,goal_complete=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False));raise


if __name__=='__main__':main()
