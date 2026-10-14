"""Fixed TRAIN LOO pace-supervised retrieval, no original validation labels."""
import json,shutil,time,logging
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_segment_features import prefix,condition_key
from battery_smetimes.mit_pace_forecast import pace_label,FEATURES
from battery_smetimes.mit_template_tangent import curve
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.diagnose_mit_template_route_loo import evaluate
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/mit_template_tangent_diagnostic/20261007T055013_210632Z')


def fit_head(raw,pace,donors,alpha):
    labeled=[k for k in donors if pace[k] is not None]
    if len(labeled)<10:raise ValueError('insufficient pace supervision')
    y=np.log([pace[k] for k in labeled]);imp=SimpleImputer(strategy='median',keep_empty_features=True)
    scaler=StandardScaler();z=scaler.fit_transform(imp.fit_transform(np.stack([raw[k] for k in labeled])))
    model=Ridge(alpha=alpha).fit(z,y)
    return dict(imputer=imp,scaler=scaler,model=model,lower=float(y.min()),upper=float(y.max()),
        donor_ids=donors,labeled_ids=labeled,unobserved_ids=[k for k in donors if pace[k] is None])


def coordinates(b,raw,keys):
    x=b['scaler'].transform(b['imputer'].transform(np.stack([raw[k] for k in keys])))
    return np.clip(b['model'].predict(x),b['lower'],b['upper'])


def main():
    pc=json.loads((PARENT/'config.json').read_text());assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    plan=pc['plan'];ids=plan['train'];cells=load_subset(ids)
    assert set(cells)==set(ids) and not set(cells)&set(plan['validation']+plan['test'])
    loo=Path(pc['parent']);lc=json.loads((loo/'config.json').read_text());cf=Path(lc['cache_file'])
    templates={t['cell_id']:t for t in joblib.load(cf)['templates']}
    raw={k:prefix(cells[k],FEATURES) for k in ids};pace={k:pace_label(cells[k]) for k in ids}
    root=Path('experiments/mit_supervised_template_route')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_pace_forecast.py'),Path('battery_smetimes/mit_segment_features.py'),
        Path('battery_smetimes/mit_template_tangent.py'),Path('docs/soh_mit_supervised_template_route_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    deps=[PARENT/'config.json',PARENT/'independent_verification.json',PARENT/'curves.npz',cf]
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),input_sha256={str(f.resolve()):sha(f) for f in deps},cache_file=str(cf.resolve()),
        plan=plan,alphas=[10.,100.],k=3,source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},
        validation_loaded=False,outer_loaded=False,origin_model_used=False,adopted=False))
    save_json(root/'training_pace_labels.json',pace)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'diagnostic.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();reports=[]
    with np.load(PARENT/'curves.npz',allow_pickle=False) as baseline:
        for i,alpha in enumerate([10.,100.]):
            out=root/f'candidate_{i}';out.mkdir();rows=[];arrays={};heads=[]
            for held in ids:
                donors=[k for k in ids if k!=held];b=fit_head(raw,pace,donors,alpha)
                z=coordinates(b,raw,donors);q=coordinates(b,raw,[held])[0]
                local=[k for k in donors if templates[k]['condition']==condition_key(cells[held])];pool=local if len(local)>=6 else donors
                d=np.array([abs(z[donors.index(k)]-q) for k in pool]);pick=np.argsort(d,kind='stable')[:min(3,len(pool))]
                near=[pool[j] for j in pick];w=1/(d[pick]+.001);w/=w.sum()
                c,y=targets(cells[held]);truth=y-y[0];changes=[]
                for k in near:
                    t=templates[k];changes.append(curve(t['parameters'],t['record_end'],c)-curve(t['parameters'],t['record_end'],[101.])[0])
                p=np.sum(np.stack(changes)*w[:,None],axis=0);base=baseline[held+'__tangent_change']
                rows.append(dict(cell_id=held,donor_ids=donors,neighbor_ids=near,distances=d[pick].tolist(),weights=w.tolist(),
                    scope='same_condition' if len(local)>=6 else 'family_fallback',query_predicted_logpace=float(q),
                    baseline=evaluate(base,truth),supervised=evaluate(p,truth)))
                heads.append(dict(held=held,head=b));arrays.update({held+'__'+f:v for f,v in [('cycle',c),('truth_change',truth),('prediction_change',p)]})
            save_json(out/'rows.json',rows);joblib.dump(heads,out/'heads.joblib');np.savez_compressed(out/'curves.npz',**arrays)
            fields=tuple(rows[0]['baseline']);report=dict(candidate=i,alpha=alpha,
                **{arm:{f:float(np.mean([r[arm][f] for r in rows])) for f in fields} for arm in ('baseline','supervised')},
                endpoint_improved_cells=sum(r['supervised']['change_endpoint_pp']<r['baseline']['change_endpoint_pp'] for r in rows))
            reports.append(report);logging.info('SAVED %s',json.dumps(report))
    save_json(root/'summary.json',dict(reports=reports,held_cells=83,new_pace_heads=166,curves=166,seconds=time.perf_counter()-start,
        validation_loaded=False,outer_loaded=False,adopted=False,goal_complete=False,
        interpretation='TRAIN-held relative-change routing only; no absolute SOH claim, query future/pace never enters fitted head or route.'))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
