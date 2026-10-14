"""Independent pace-head fits and route/difference arithmetic for TRAIN LOO."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_segment_features import prefix,condition_key
from battery_smetimes.mit_pace_forecast import pace_label,FEATURES
from battery_smetimes.curve_fidelity import targets
from scripts.verify_mit_template_tangent import relative
from scripts.verify_mit_template_route_loo import errors
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];ids=plan['train'];cells=load_subset(ids)
    assert len(ids)==83 and set(cells)==set(ids) and not set(cells)&set(plan['validation']+plan['test'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    parent=Path(cfg['parent']);assert json.loads((parent/'independent_verification.json').read_text())['passed']
    templates={t['cell_id']:t for t in joblib.load(cfg['cache_file'])['templates']}
    raw={k:prefix(cells[k],FEATURES) for k in ids};pace={k:pace_label(cells[k]) for k in ids}
    close(json.loads((root/'training_pace_labels.json').read_text()),pace)
    reports=[];count=0
    with np.load(parent/'curves.npz',allow_pickle=False) as baseline:
        for i,alpha in enumerate(cfg['alphas']):
            out=root/f'candidate_{i}';records=json.loads((out/'rows.json').read_text());heads=joblib.load(out/'heads.joblib');assert len(records)==len(heads)==83
            rows=[]
            with np.load(out/'curves.npz',allow_pickle=False) as arrays:
                assert set(arrays.files)=={k+'__'+f for k in ids for f in ('cycle','truth_change','prediction_change')}
                for pos,held in enumerate(ids):
                    donors=[k for k in ids if k!=held];labeled=[k for k in donors if pace[k] is not None]
                    b=heads[pos]['head'];r=records[pos]
                    assert heads[pos]['held']==r['cell_id']==held and b['donor_ids']==r['donor_ids']==donors
                    assert b['labeled_ids']==labeled and b['unobserved_ids']==[k for k in donors if pace[k] is None] and held not in labeled
                    imp=SimpleImputer(strategy='median',keep_empty_features=True);scaler=StandardScaler()
                    x=scaler.fit_transform(imp.fit_transform(np.stack([raw[k] for k in labeled])));y=np.log([pace[k] for k in labeled])
                    model=Ridge(alpha=alpha).fit(x,y)
                    np.testing.assert_array_equal(imp.statistics_,b['imputer'].statistics_)
                    for f in ('mean_','scale_','var_'):np.testing.assert_array_equal(getattr(scaler,f),getattr(b['scaler'],f))
                    np.testing.assert_allclose(model.coef_,b['model'].coef_,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(model.intercept_,b['model'].intercept_,rtol=0,atol=1e-12)
                    assert b['lower']==y.min() and b['upper']==y.max()
                    keys=donors+[held];xx=scaler.transform(imp.transform(np.stack([raw[k] for k in keys])))
                    coord=np.clip(xx@model.coef_+model.intercept_,y.min(),y.max());q=coord[-1]
                    np.testing.assert_allclose(r['query_predicted_logpace'],q,rtol=0,atol=1e-12)
                    query=clean(cells[held]);poison=dict(query,soh=np.inf,label_cycle=np.inf,cycle_life=np.inf,reference_capacity_ah=np.inf,
                        cycle=np.r_[query['cycle'],101.],raw=np.vstack([query['raw'],np.full((1,51),np.inf)]))
                    np.testing.assert_array_equal(prefix(query,FEATURES),prefix(poison,FEATURES))
                    local=[k for k in donors if templates[k]['condition']==condition_key(query)];pool=local if len(local)>=6 else donors
                    d=np.array([abs(coord[donors.index(k)]-q) for k in pool]);pick=np.argsort(d,kind='stable')[:min(3,len(pool))]
                    near=[pool[j] for j in pick];w=1/(d[pick]+.001);w/=w.sum()
                    assert r['neighbor_ids']==near and held not in near and r['scope']==('same_condition' if len(local)>=6 else 'family_fallback')
                    np.testing.assert_allclose(r['distances'],d[pick],rtol=0,atol=1e-12);np.testing.assert_allclose(r['weights'],w,rtol=0,atol=1e-12)
                    c,yc=targets(cells[held]);truth=yc-yc[0];p=arrays[held+'__prediction_change']
                    np.testing.assert_array_equal(c,arrays[held+'__cycle']);np.testing.assert_array_equal(truth,arrays[held+'__truth_change'])
                    expected=sum(ww*relative(templates[k],c) for k,ww in zip(near,w))
                    np.testing.assert_allclose(p,expected,rtol=0,atol=1e-10)
                    np.testing.assert_allclose(p[0],0,rtol=0,atol=1e-12);assert np.all(np.diff(p)<=1e-10)
                    base=baseline[held+'__tangent_change'];close(r['baseline'],errors(base,truth));close(r['supervised'],errors(p,truth))
                    rows.append(r);count+=1
            fields=tuple(rows[0]['baseline']);reports.append(dict(candidate=i,alpha=alpha,
                **{arm:{f:float(np.mean([r[arm][f] for r in rows])) for f in fields} for arm in ('baseline','supervised')},
                endpoint_improved_cells=sum(r['supervised']['change_endpoint_pp']<r['baseline']['change_endpoint_pp'] for r in rows)))
    summary=json.loads((root/'summary.json').read_text())
    for a,b in zip(summary['reports'],reports):
        assert a['candidate']==b['candidate'] and a['alpha']==b['alpha'] and a['endpoint_improved_cells']==b['endpoint_improved_cells']
        close(a['baseline'],b['baseline']);close(a['supervised'],b['supervised'])
    assert summary['new_pace_heads']==summary['curves']==count==166 and not summary['adopted']
    assert not cfg['validation_loaded'] and not cfg['outer_loaded'] and not cfg['origin_model_used']
    save_json(root/'independent_verification.json',dict(passed=True,head_refits=166,held_cells=83,curves=count,validation_loaded=False,outer_loaded=False,
        scope='Separate Ridge/imputer/scaler refits, route/difference/metric arithmetic; shared early features/pace supervision and verified tangent formula. Relative TRAIN LOO only.',
        verifier_sha256=sha(Path(__file__))))
    print('PASS:166fresh pace heads,83held,166changes,train-only stats/route/formula/poison/SHA')


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
