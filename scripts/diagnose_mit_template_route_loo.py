"""TRAIN-only LOO diagnostic of relative trajectories, no leaked origin model."""
import json,shutil,time,logging
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.mit_segment_features import prefix,condition_key
from battery_smetimes.mit_fixed_origin_dynamics import PARAMETERS
from battery_smetimes.knee_forecast import trajectory
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/mit_neighbor_templates/20261007T054345_631824Z')


def evaluate(change,truth):
    err=change-truth
    return dict(change_mae_pp=float(100*np.mean(abs(err))),change_rmse_pp=float(100*np.sqrt(np.mean(err**2))),
        change_endpoint_pp=float(100*abs(err[-1])),change_drop_error_pp=float(100*abs((change[0]-change[-1])-(truth[0]-truth[-1]))))


def main():
    pc=json.loads((PARENT/'config.json').read_text());assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    plan=pc['plan'];ids=plan['train'];cells=load_subset(ids)
    assert set(cells)==set(ids) and not set(cells)&set(plan['validation']+plan['test'])
    cache_file=PARENT/'candidate_0/model.joblib';cache=joblib.load(cache_file)
    assert cache['train_ids']==ids
    templates={t['cell_id']:t for t in cache['templates']}
    root=Path('experiments/mit_template_route_loo')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_segment_features.py'),Path('battery_smetimes/knee_forecast.py'),
        Path('docs/soh_mit_template_route_loo_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),cache_file=str(cache_file.resolve()),cache_sha256=sha(cache_file),
        plan=plan,k=3,parameters=PARAMETERS,source_sha256={str(f):sha(f) for f in files},
        prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},validation_loaded=False,outer_loaded=False,
        origin_model_used=False,new_template_fits=0))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'diagnostic.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();records=[];arrays={};bundles=[]
    for held in ids:
        donors=[k for k in ids if k!=held];assert held not in donors
        imputer=SimpleImputer(strategy='median',keep_empty_features=True);scaler=StandardScaler()
        raw=np.stack([prefix(cells[k],PARAMETERS) for k in donors]);z=scaler.fit_transform(imputer.fit_transform(raw))
        q=clean(cells[held]);x=scaler.transform(imputer.transform(prefix(q,PARAMETERS)[None]))[0]
        local=[k for k in donors if templates[k]['condition']==condition_key(q)]
        pool=local if len(local)>=6 else donors
        pp=[donors.index(k) for k in pool];d=np.sqrt(np.mean((z[pp]-x)**2,axis=1))
        pick=np.argsort(d,kind='stable')[:min(3,len(pool))];neighbors=[pool[j] for j in pick]
        w=1/(d[pick]+.001);w/=w.sum()
        c,y=targets(cells[held]);truth=y-y[0]
        changes=np.stack([trajectory(templates[k]['parameters'],c)-trajectory(templates[k]['parameters'],[101.])[0] for k in pool])
        near=np.sum(changes[pick]*w[:,None],axis=0);mean=np.mean(changes,axis=0)
        r=dict(cell_id=held,donor_ids=donors,pool_ids=pool,neighbor_ids=neighbors,distances=d[pick].tolist(),weights=w.tolist(),
            scope='same_condition' if len(local)>=6 else 'family_fallback',
            majority_neighbors_extrapolated=bool(np.mean([c[-1]>templates[k]['record_end'] for k in neighbors])>.5),
            nearest=evaluate(near,truth),pool_mean=evaluate(mean,truth))
        records.append(r);bundles.append(dict(held=held,donor_ids=donors,imputer=imputer,scaler=scaler))
        arrays.update({held+'__'+f:v for f,v in [('cycle',c),('truth_change',truth),('nearest_change',near),('pool_mean_change',mean)]})
    fields=('change_mae_pp','change_rmse_pp','change_endpoint_pp','change_drop_error_pp')
    summary=dict(cells=83,preprocessing_fits=83,new_template_fits=0,curves=166,
        nearest={f:float(np.mean([r['nearest'][f] for r in records])) for f in fields},
        pool_mean={f:float(np.mean([r['pool_mean'][f] for r in records])) for f in fields},
        nearest_endpoint_wins=sum(r['nearest']['change_endpoint_pp']<r['pool_mean']['change_endpoint_pp'] for r in records),
        majority_neighbor_extrapolated_cells=sum(r['majority_neighbors_extrapolated'] for r in records),
        seconds=time.perf_counter()-start,validation_loaded=False,outer_loaded=False,adopted=False,goal_complete=False,
        interpretation='Relative-change TRAIN-held-cell routing diagnostic, not absolute SOH accuracy. Cached donor-specific supervision excludes held row; no frozen origin inference.')
    save_json(root/'rows.json',records);np.savez_compressed(root/'curves.npz',**arrays);joblib.dump(bundles,root/'preprocessors.joblib')
    save_json(root/'summary.json',summary);logging.info('SAVED %s',json.dumps(summary));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
