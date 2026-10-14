"""82-donor preprocessing refits and separate relative-change routing arithmetic."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_segment_features import prefix,condition_key
from battery_smetimes.curve_fidelity import targets
from battery_smetimes.knee_forecast import fit_knee
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def delta(params,c):
    a,r,b,knee=params;h=(c-100)/100
    return -r*(h-.01)-b*(np.maximum(h-knee,0)**2-max(.01-knee,0)**2)


def errors(p,y):
    e=p-y
    return dict(change_mae_pp=float(np.abs(e).mean()*100),change_rmse_pp=float(np.sqrt(np.dot(e,e)/len(e))*100),
        change_endpoint_pp=float(abs(e[-1])*100),change_drop_error_pp=float(abs((p[-1]-p[0])-(y[-1]-y[0]))*100))


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];ids=plan['train'];cells=load_subset(ids)
    assert len(ids)==83 and set(cells)==set(ids) and not set(cells)&set(plan['validation']+plan['test'])
    parent=Path(cfg['parent']);assert sha(parent/'config.json')==cfg['parent_config_sha256']
    assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    assert sha(Path(cfg['cache_file']))==cfg['cache_sha256']
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    cache=joblib.load(cfg['cache_file']);assert cache['train_ids']==ids
    templates={t['cell_id']:t for t in cache['templates']}
    for k in ids:
        np.testing.assert_array_equal(templates[k]['parameters'],fit_knee(*targets(cells[k])))
    records=json.loads((root/'rows.json').read_text());bundles=joblib.load(root/'preprocessors.joblib')
    assert len(records)==len(bundles)==83
    with np.load(root/'curves.npz',allow_pickle=False) as arrays:
        assert set(arrays.files)=={k+'__'+f for k in ids for f in ('cycle','truth_change','nearest_change','pool_mean_change')}
        for pos,held in enumerate(ids):
            rec=records[pos];bundle=bundles[pos];donors=[k for k in ids if k!=held]
            assert rec['cell_id']==bundle['held']==held and rec['donor_ids']==bundle['donor_ids']==donors and held not in donors
            imp=SimpleImputer(strategy='median',keep_empty_features=True);s=StandardScaler()
            z=s.fit_transform(imp.fit_transform(np.stack([prefix(cells[k],cfg['parameters']) for k in donors])))
            np.testing.assert_array_equal(imp.statistics_,bundle['imputer'].statistics_)
            for field in ('mean_','scale_','var_'):np.testing.assert_array_equal(getattr(s,field),getattr(bundle['scaler'],field))
            q=clean(cells[held]);x=s.transform(imp.transform(prefix(q,cfg['parameters'])[None]))[0]
            poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
            np.testing.assert_array_equal(x,s.transform(imp.transform(prefix(poison,cfg['parameters'])[None]))[0])
            local=[k for k in donors if templates[k]['condition']==condition_key(q)];pool=local if len(local)>=6 else donors
            assert rec['pool_ids']==pool and held not in pool and rec['scope']==('same_condition' if len(local)>=6 else 'family_fallback')
            distances=np.array([np.linalg.norm(z[donors.index(k)]-x)/np.sqrt(len(x)) for k in pool])
            pick=np.argsort(distances,kind='stable')[:min(3,len(pool))];nearids=[pool[j] for j in pick]
            w=1/(distances[pick]+.001);w/=w.sum()
            assert rec['neighbor_ids']==nearids and held not in nearids
            np.testing.assert_allclose(rec['distances'],distances[pick],rtol=0,atol=1e-12)
            np.testing.assert_allclose(rec['weights'],w,rtol=0,atol=1e-12)
            c,y=targets(cells[held]);yc=y-y[0]
            np.testing.assert_array_equal(c,arrays[held+'__cycle']);np.testing.assert_array_equal(yc,arrays[held+'__truth_change'])
            changes=np.stack([delta(templates[k]['parameters'],c) for k in pool])
            pn=np.einsum('i,ij->j',w,changes[pick]);pm=changes.mean(axis=0)
            np.testing.assert_allclose(pn,arrays[held+'__nearest_change'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(pm,arrays[held+'__pool_mean_change'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(pn[0],0,rtol=0,atol=1e-12);np.testing.assert_allclose(pm[0],0,rtol=0,atol=1e-12)
            close(rec['nearest'],errors(pn,yc));close(rec['pool_mean'],errors(pm,yc))
            assert rec['majority_neighbors_extrapolated']==bool(np.mean([c[-1]>templates[k]['record_end'] for k in nearids])>.5)
    summary=json.loads((root/'summary.json').read_text());fields=tuple(records[0]['nearest'])
    for arm in ('nearest','pool_mean'):close(summary[arm],{f:float(np.mean([r[arm][f] for r in records])) for f in fields})
    assert summary['nearest_endpoint_wins']==sum(r['nearest']['change_endpoint_pp']<r['pool_mean']['change_endpoint_pp'] for r in records)
    assert summary['majority_neighbor_extrapolated_cells']==sum(r['majority_neighbors_extrapolated'] for r in records)
    assert summary['cells']==summary['preprocessing_fits']==83 and summary['curves']==166 and not summary['adopted']
    assert not cfg['validation_loaded'] and not cfg['outer_loaded'] and not cfg['origin_model_used']
    save_json(root/'independent_verification.json',dict(passed=True,held_cells=83,curves=166,preprocessor_refits=83,template_solver_replays=83,
        validation_loaded=False,outer_loaded=False,verifier_sha256=sha(Path(__file__)),
        scope='TRAIN-held relative change only: 82-donor preprocessors independently fit; shared template solver replay, separate distances/formula/metrics; no frozen origin.'))
    print('PASS:83held/82donor fits,166changes,83template replay, routes/poison/SHA/metric arithmetic')


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
