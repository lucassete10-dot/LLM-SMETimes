"""Independent saved-model replay and preprocessing/metric checks."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as point, level_loss_curve as split
from battery_smetimes import joint_shape_mixture as joint
from scripts.train_continuous_length_pilot import load_nominal, target
from scripts.nca_full_label_loader import load as load_full
from scripts.prepare_same_type_forecast import sha, save_json


def check_preprocessing(head, x):
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler().fit(imp.fit_transform(x))
    for actual, expected in ((head['imputer'].statistics_,imp.statistics_),
                             (head['scaler'].mean_,scaler.mean_),
                             (head['scaler'].scale_,scaler.scale_)):
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run',type=Path)
    root = parser.parse_args().run
    cfg = json.loads((root/'config.json').read_text())
    status = json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for p,digest in cfg['source_sha256'].items():
        assert sha(root/'source_snapshot'/Path(p).name) == digest
        if Path(p).name != 'train_level_loss_probe.py':
            assert sha(Path(p)) == digest
    models = curves = 0
    for plan in cfg['plans']:
        if plan['dataset']=='HUST':
            indexpath=Path(cfg['hust_index']); assert sha(indexpath)==cfg['hust_index_sha256']
            records={r['cell_id']:r for r in json.loads(indexpath.read_text())['cells']}; cells={}
            for cid in plan['train']+plan['validation']:
                record=records[cid]; assert sha(Path(record['path']))==record['sha256']
                with np.load(record['path'],allow_pickle=False) as data:
                    cells[cid]={k:data[k].copy() for k in data.files}
                cells[cid]['physical_family']=record['physical_family']
        elif plan['dataset']=='Zhu2022_NCA':
            assert sha(Path(cfg['nca_label_index']))==cfg['nca_label_index_sha256']
            cells,_=load_full(plan,cfg['nca_label_index'])
        else:
            cells,_=load_nominal(plan)
        assert not set(cells)&set(plan['test'])
        targets={k:target(v,100) for k,v in cells.items()}
        for candidate in cfg['candidates']:
            folder=root/plan['name']/candidate; path=folder/'model.joblib'
            record=json.loads((folder/'config.json').read_text())
            assert sha(path)==record['model_sha256']
            bundle=joblib.load(path); ids=record['train']
            assert ids==bundle['train_ids'] and set(ids)<=set(plan['train'])
            assert not set(ids)&set(plan['validation']+plan['test'])
            block='condition' if candidate=='point_condition' else 'VIT'
            xs=[]
            for cid in ids:
                c,_=targets[cid]
                ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
                xs.append(point.point_inputs(cells[cid],100,c[ii],block))
            x=np.concatenate(xs)
            if candidate.startswith('continuous_'):
                x=np.stack([point.features(cells[k],100) for k in ids])
                check_preprocessing(bundle,x)
                assert len(bundle['model'].estimators_)==80
                module=joint if candidate=='continuous_jointmixture' else point
                if candidate=='continuous_jointmixture':
                    leaves=bundle['model'].apply(bundle['scaler'].transform(bundle['imputer'].transform(x)))
                    np.testing.assert_array_equal(bundle['training_leaves'],leaves)
                    np.testing.assert_allclose(bundle['joint_parameters'],
                        [f['parameters'] for f in bundle['training_shape_fits']],rtol=0,atol=0)
            elif candidate=='level_loss':
                module=split; check_preprocessing(bundle['loss'],x)
                check_preprocessing(bundle['level'],np.stack([point.features(cells[k],100) for k in ids]))
                assert sum(len(bundle[h]['model'].estimators_) for h in ('level','loss'))==80
                for cid in ids:
                    cell=cells[cid]
                    c,y=((cell['cycle'],cell['truth']) if 'early_cycle' in cell else
                         (cell['label_cycle'],cell['soh']))
                    c,y=np.asarray(c,float),np.asarray(y,float)
                    valid=np.isfinite(c)&np.isfinite(y)&(y>0); c,y=c[valid],y[valid]
                    assert c[0]<=100<=c[-1] and np.all(np.diff(c)>0)
                    # Independent linear interpolation of training supervision.
                    right=int(np.searchsorted(c,100)); left=max(0,right-1)
                    level=y[right] if c[right]==100 else y[left]+(y[right]-y[left])*(100-c[left])/(c[right]-c[left])
                    np.testing.assert_allclose(bundle['training_levels'][cid],level,rtol=0,atol=1e-12)
            else:
                module=point; check_preprocessing(bundle,x)
                assert len(bundle['model'].estimators_)==80
            rows=json.loads((folder/'metrics.json').read_text())
            assert [r['cell_id'] for r in rows]==record['validation']
            with np.load(folder/'predictions.npz',allow_pickle=False) as data:
                for row in rows:
                    cid=row['cell_id']; assert cid in plan['validation']
                    c,y=targets[cid]; p=module.predict(bundle,cells[cid],c)
                    if candidate.startswith('continuous_'):
                        dense=module.predict(bundle,cells[cid],np.linspace(c[0],c[-1],400))
                        assert np.max(np.diff(dense))<=1e-10
                        assert np.max(np.diff(dense,n=2))<=1e-9
                        if candidate=='continuous_jointmixture':
                            z=bundle['scaler'].transform(bundle['imputer'].transform(point.features(cells[cid],100)[None]))
                            leaf=bundle['model'].apply(z)[0]; matches=bundle['training_leaves']==leaf[None]
                            weights=(matches/matches.sum(axis=0)).mean(axis=1)
                            # Joint mean parameters must reproduce original tree output.
                            np.testing.assert_allclose(weights@bundle['joint_parameters'],
                                bundle['model'].predict(z)[0]*bundle['target_scale'],atol=1e-12,rtol=0)
                            donor_curves=np.stack([point.basis(c,100,a[3])@a[:3] for a in bundle['joint_parameters']])
                            np.testing.assert_allclose(p,weights@donor_curves,atol=1e-12,rtol=0)
                    for name,values in [('cycle',c),('truth',y),('prediction',p)]:
                        np.testing.assert_allclose(data[cid+'__'+name],values,atol=1e-12,rtol=0)
                    e=p-y; late=c>=c[0]+.8*(c[-1]-c[0])
                    expected=dict(mape_percent=100*np.mean(abs(e)/y),mae_soh=np.mean(abs(e)),
                                  rmse_soh=np.sqrt(np.mean(e**2)),late20_mae_pp=100*np.mean(abs(e[late])))
                    for name,value in expected.items():
                        np.testing.assert_allclose(row[name],value,atol=1e-10,rtol=0)
                    levels=[]
                    for values in (y,p):
                        levels.append([np.median(values[sorted(range(len(c)),key=lambda j:(abs(c[j]-boundary),j))[:min(5,len(c))]])
                                       for boundary in np.linspace(c[0],c[-1],4)])
                    yd,pd=-100*np.diff(levels[0]),-100*np.diff(levels[1])
                    np.testing.assert_allclose(row['stage_error_pp'],max(abs(yd-pd)),atol=1e-10,rtol=0)
                    assert row['acceleration_direction_reversed']==bool(yd[-1]-yd[0]>1 and pd[-1]-pd[0]<-1)
                    curves+=1
            models+=1
    assert models==status['completed_fits']==3*len(cfg['plans'])
    proof=dict(passed=True,verified_models=models,verified_curves=curves,
               train_only_preprocessing_rebuilt=True,training_level_supervision_checked=True,
               equal_tree_counts_checked=True,exact_saved_weight_replay=True,
               independent_basic_stage_reversal_metrics=True,independent_test=False,
               limitation='Single-seed exploration, not stable superiority, knee certification or final acceptance.',
               goal_complete=False)
    save_json(root/'verification.json',proof); print(json.dumps(proof))


if __name__=='__main__':
    main()
