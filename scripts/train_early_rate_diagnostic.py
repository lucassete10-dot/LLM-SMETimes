"""Training-cell-only leave-one-out ablation of fixed-window future decline."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from battery_smetimes import continuous_length as features
from battery_smetimes import condition_length as protocol
from scripts.train_continuous_length_pilot import load_nominal,target
from scripts.prepare_same_type_forecast import sha,save_json

BASE=Path('experiments/condition_length/20261008T020929_340690Z')


def label(cell):
    c,y=target(cell,100);mask=(c>=101)&(c<=125);c,y=c[mask],y[mask]
    if len(c)<10 or c[-1]-c[0]<15:raise ValueError('insufficient fixed-window labels')
    x=c-c.mean();return float(-100*x@(y-y.mean())/(x@x)),dict(points=len(c),first=float(c[0]),last=float(c[-1]))


def main():
    root=Path('experiments/early_rate_diagnostic')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    cfg=json.loads((BASE/'config.json').read_text());plan=cfg['plan'];ids=json.loads((BASE/'train_coverage.json').read_text())['used']
    trainplan=dict(plan,train=ids,validation=[]);cells,_=load_nominal(trainplan)
    assert set(cells)==set(ids) and not set(cells)&set(plan['validation']+plan['test'])
    labels={};coverage=[]
    for k in ids:
        try:v,support=label(cells[k]);labels[k]=v;coverage.append(dict(cell_id=k,included=True,**support))
        except ValueError as e:coverage.append(dict(cell_id=k,included=False,reason=str(e),not_bad_cell=True))
    ids=list(labels);assert len(ids)>=6
    files=[Path(__file__),Path('battery_smetimes/continuous_length.py'),Path('battery_smetimes/condition_length.py'),Path('scripts/train_continuous_length_pilot.py')]
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for f in files:shutil.copy2(f,snapshot/f.name)
    save_json(root/'config.json',dict(original_plan=plan,used_train=ids,origins=[50,100],blocks=['condition','V','VIT'],
        candidates=['ridge10','extra_trees'],seed=20261008,target='OLS capacity/rated-capacity decline per100cycles during source101..125',
        source_sha256={str(f):sha(f) for f in files},coverage=coverage,validation_loaded=False,outer_loaded=False,
        independent_test=False,goal_complete=False,limitation='Fixed near-term decline diagnostic; not future full-curve accuracy or future knee evidence.'))
    print('RUN='+str(root.resolve()),flush=True);rows=[];done=0
    try:
        for origin in (50,100):
            for block in ('condition','V','VIT'):
                raw={k:features.features(cells[k],origin,block) for k in ids}
                for kind in ('ridge10','extra_trees'):
                    for held in ids:
                        train=[k for k in ids if k!=held]
                        encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
                        groups=np.array([[protocol.key(cells[k])] for k in train]);known=encoder.fit_transform(groups)
                        x=np.c_[np.stack([raw[k] for k in train]),known]
                        query=np.c_[raw[held][None],encoder.transform([[protocol.key(cells[held])]])]
                        imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x);sc=StandardScaler().fit(z)
                        y=np.array([labels[k] for k in train])
                        model=Ridge(alpha=10) if kind=='ridge10' else ExtraTreesRegressor(n_estimators=80,min_samples_leaf=2,max_features=.8,random_state=20261008,n_jobs=2)
                        model.fit(sc.transform(z),y);p=float(model.predict(sc.transform(imp.transform(query)))[0])
                        compatible=[labels[k] for k in train if protocol.key(cells[k])==protocol.key(cells[held])]
                        baseline=float(np.median(compatible if compatible else y))
                        folder=root/str(origin)/block/kind;folder.mkdir(parents=True,exist_ok=True)
                        path=folder/(held+'.joblib');bundle=dict(train_ids=train,held_out=held,encoder=encoder,imputer=imp,scaler=sc,model=model,origin=origin,block=block)
                        joblib.dump(bundle,path);b=joblib.load(path)
                        np.testing.assert_allclose(p,b['model'].predict(b['scaler'].transform(b['imputer'].transform(query)))[0],rtol=0,atol=1e-12)
                        rows.append(dict(origin=origin,block=block,model=kind,cell_id=held,truth_rate=labels[held],predicted_rate=p,
                            group_median_baseline=baseline,group_train_count=len(compatible),model_path=str(path),model_sha256=sha(path)))
                        done+=1
                    selected=[r for r in rows if r['origin']==origin and r['block']==block and r['model']==kind]
                    err=np.array([r['predicted_rate']-r['truth_rate'] for r in selected])
                    save_json(root/'rows.json',rows);save_json(root/'progress.json',dict(completed_fits=done,terminal=False))
                    print(f'N={origin} {block}/{kind} rate_MAE_pp_per100={100*np.mean(abs(err)):.3f}',flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,goal_complete=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False));raise


if __name__=='__main__':main()
