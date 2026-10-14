"""Independent rate supervision/refit, primary replay and all locked curves."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_selection import choose,fit as primary_fit,predict as primary_predict
from battery_smetimes.tail_boost import labels,inputs
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.warm_monotone import derive_prior
from scripts.verify_fade_rate import imputed
from scripts.verify_rate_blend import score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric,aggregate,gates


def refit_rate(b,cells,ids,seed,params):
    assert b['train_ids']==ids and b['seed']==seed and b['learned_anchor']
    assert [r['cell_id'] for r in params]==ids
    anchors=np.array([r['anchor'] for r in params]);np.testing.assert_array_equal(anchors,b['training_anchors'])
    xs=[];targets=[];ws=[]
    for key,anchor in zip(ids,anchors):
        cycle,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(inputs(clean(cells[key]),cycle[idx]));target=(anchor-y[idx])/(1+(cycle[idx]-100)/100)
        np.testing.assert_allclose(target,100*(anchor-y[idx])/cycle[idx],atol=1e-14,rtol=0);targets.append(target)
        w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
    x=imputed(np.concatenate(xs),b['imputer']);w=np.concatenate(ws);w*=len(w)/w.sum()
    tree=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2).fit(x,np.concatenate(targets),sample_weight=w)
    z=imputed(np.stack([prefix(clean(cells[k])) for k in ids]),b['anchor_imputer'])
    anchor_tree=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2).fit(z,anchors)
    return tree,anchor_tree


def curve_components(parts,replay_primary,replay_rate,cells,key,cycle):
    cell=clean(cells[key]);rate=parts['rate'];tree,anchor_tree=replay_rate
    anchor=float(anchor_tree.predict(rate['anchor_imputer'].transform(prefix(cell)[None]))[0]);q=anchor-(1+(cycle-100)/100)*tree.predict(rate['imputer'].transform(inputs(cell,cycle)))
    p=primary_predict(replay_primary,cell,cycle);np.testing.assert_allclose(p,primary_predict(parts['primary'],cell,cycle),atol=1e-12,rtol=0)
    from battery_smetimes.fade_rate import predict as rate_predict
    np.testing.assert_allclose(q,rate_predict(rate,cell,cycle),atol=1e-12,rtol=0)
    poison=dict(cell,cycle=np.r_[cell['cycle'],101],raw=np.vstack([cell['raw'],np.full((1,cell['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
    for fn,b,value in [(primary_predict,parts['primary'],p),(rate_predict,rate,q)]:
        np.testing.assert_allclose(value,fn(b,poison,cycle),atol=1e-12,rtol=0)
        try:fn(b,dict(cell,physical_family='different'),cycle)
        except ValueError:pass
        else:raise AssertionError('family gate failed')
    return p,q


def audit_curves(folder,parts,replay_primary,replay_rate,cells,ids,role,weights,checked,fold,seed):
    values={i:[] for i in range(len(weights))}
    with np.load(folder/(role+'_components.npz')) as components:
        assert len(components.files)==4*len(ids)
        cached={}
        for key in ids:
            cycle,truth=labels(cells[key]);p,q=curve_components(parts,replay_primary,replay_rate,cells,key,cycle)
            for f,a in [('cycle',cycle),('truth',truth),('primary',p),('rate',q)]:np.testing.assert_allclose(a,components[key+'__100__'+f],atol=1e-12,rtol=0)
            cached[key]=(cycle,truth,p,q)
        for index,w in enumerate(weights):
            dest=folder/f'candidate_{index}' if folder.name=='inner' else folder
            records={r['cell_id']:r for r in json.loads((dest/(role+'_metrics.json')).read_text())};assert set(records)==set(ids)
            prediction_file=dest/(role+'_predictions.npz');prediction_digest=sha(prediction_file)
            with np.load(prediction_file) as arrays:
                assert len(arrays.files)==3*len(ids)
                for key,(cycle,truth,p,q) in cached.items():
                    prediction=(1-w)*p+w*q
                    for f,a in [('cycle',cycle),('truth',truth),('prediction',prediction)]:np.testing.assert_allclose(a,arrays[key+'__100__'+f],atol=1e-12,rtol=0)
                    m=metric(cycle,truth,prediction)
                    for f,v in m.items():np.testing.assert_allclose(v,records[key][f],atol=1e-10,rtol=0)
                    values[index].append(dict(cell_id=key,seed=seed,**m));checked.append(dict(fold=fold,seed=seed,role=role,candidate=index,cell_id=key,file=str(prediction_file.resolve()),sha256=prediction_digest))
            assert sha(prediction_file)==prediction_digest
    return values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--fold',help='Audit only a completed fold; never a complete-run acceptance');args=parser.parse_args();root=args.run;cfg=json.loads((root/'config.json').read_text());progress=json.loads((root/'progress.json').read_text())
    if args.fold is None:assert progress['training_complete'] and progress['outer_complete']
    assert cfg['seeds']==list(range(20261005,20261010)) and cfg['weights']==[0,.25,.5,.75,1]
    route=Path(cfg['primary_run']);pcfg=json.loads((route/'config.json').read_text());assert cfg['plans']==pcfg['plans'] and sha(route/'config.json')==cfg['primary_config_sha256']
    for key,path in [('screened',cfg['screened_run']),('baseline',cfg['baseline']),('current_primary',cfg['current_primary'])]:assert sha(Path(path)/'config.json')==cfg[key+'_config_sha256']
    assert sha('docs/soh_rate_blend_formal_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for name in ('rate_blend','fade_rate','warm_monotone','monotone_network','conditional_drift','crossfit_drift','tail_selection','tail_boost','curve_fidelity','forecast','research_baselines','physical_types'):
        assert sha(Path('battery_smetimes')/(name+'.py'))==cfg['source_sha256'][name+'.py']
    assert sha(root/'inner_lock.json')==progress['inner_lock_sha256'];lockfile=root/'inner_lock.json';locks=json.loads(lockfile.read_text())['selections'];assert len(locks)==25
    checked=[];rows=[];old_rows=[];current_rows=[];plans=cfg['plans'] if args.fold is None else [p for p in cfg['plans'] if p['name']==args.fold];assert plans
    for plan in plans:
        assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
        cells=load_subset(plan['train']+plan['validation']+plan['test'])
        for key in cells:assert sha(Path('data/interim/same_type_forecast/20260929T120631_212616Z')/(key+'.npz'))==cfg['prepared_sha256'][key]
        for role,ids in [('inner',plan['train']),('outer',plan['train']+plan['validation'])]:
            params=derive_prior(cells,ids)[1]
            for seed in cfg['seeds']:
                folder=root/plan['name']/str(seed)/role;lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed)
                assert Path(lock['selection_file'])==(root/plan['name']/str(seed)/'inner'/'selection.json').resolve() and sha(lock['selection_file'])==lock['selection_sha256']
                if role=='inner':
                    parts=joblib.load(folder/'components.joblib');prov=json.loads((folder/'provenance.json').read_text());decision=choose(route,pcfg,plan,seed,cells);assert prov['decision']==decision and lock['primary_choice']==decision['selected']
                    pf=Path(prov['primary_file']);assert sha(pf)==prov['primary_sha256']==decision['selected']['model_sha256'];assert sha(folder/'components.joblib')==lock['components_sha256'] and sha(folder/'training_parameters.json')==lock['parameter_sha256']
                    assert (folder/'selection.json').stat().st_mtime_ns<=lockfile.stat().st_mtime_ns
                else:
                    parts=joblib.load(folder/'model.joblib');assert parts['weight']==lock['selected']['weight'];assert (folder/'model.joblib').stat().st_mtime_ns>lockfile.stat().st_mtime_ns
                assert params==json.loads((folder/'training_parameters.json').read_text());assert parts['primary']['train_ids']==parts['primary']['inner']['train_ids']==parts['rate']['train_ids']==ids
                assert parts['primary']['family']==parts['rate']['family']==cells[ids[0]]['physical_family']
                replay_rate=refit_rate(parts['rate'],cells,ids,seed,params);replay_primary=primary_fit(cells,ids,lock['primary_choice'],seed)
                weights=cfg['weights'] if role=='inner' else [parts['weight']]
                audit_curves(folder,parts,replay_primary,replay_rate,cells,ids,'training',weights,checked,plan['name'],seed)
                val=audit_curves(folder,parts,replay_primary,replay_rate,cells,plan['validation'] if role=='inner' else plan['test'],'validation' if role=='inner' else 'test',weights,checked,plan['name'],seed)
                if role=='inner':
                    selection=json.loads((folder/'selection.json').read_text());assert selection['used_only_validation']
                    assert len(selection['candidates'])==5
                    for index,r in enumerate(selection['candidates']):
                        assert r['candidate']==index and r['weight']==cfg['weights'][index];np.testing.assert_allclose(score(val[index]),r['score'],atol=1e-10,rtol=0)
                        assert sha(folder/f'candidate_{index}'/'validation_predictions.npz')==r['validation_sha256']
                    assert min(selection['candidates'],key=lambda r:(r['score'],r['weight']))==selection['selected']==lock['selected']
                else:rows.extend(val[0])
                print('PASS '+plan['name']+' '+role+' seed='+str(seed),flush=True)
        for destination,path,seeds,name in [(old_rows,Path(cfg['baseline']),range(20261004,20261009),'test/predictions.npz'),(current_rows,Path(cfg['current_primary']),cfg['seeds'],'test_predictions.npz')]:
            for seed in seeds:
                with np.load(path/plan['name']/str(seed)/name) as a:
                    for key in plan['test']:
                        cycle,truth=labels(cells[key]);np.testing.assert_array_equal(cycle,a[key+'__100__cycle']);np.testing.assert_array_equal(truth,a[key+'__100__truth']);destination.append(dict(cell_id=key,seed=seed,**metric(cycle,truth,a[key+'__100__prediction'])))
    expected=sum(5*(5*len(p['train']+p['validation'])+len(p['train']+p['validation']+p['test'])) for p in plans);assert len(checked)==expected and len(rows)==5*sum(len(p['test']) for p in plans)
    if args.fold is not None:
        save_json(root/(args.fold+'_independent_verification.json'),dict(passed=True,scope='one completed fold only; no complete-run or numeric-goal claim',fold=args.fold,checked=checked,retrained_rate_bundles=10,retrained_primary_bundles=10,curves=expected,all_inner_locks_sha256=sha(lockfile),source_sha256=sha(__file__),goal_complete=False));print('PARTIAL_FOLD_CURVES='+str(expected),flush=True);return
    new,cell_rows=aggregate(rows);old,_=aggregate(old_rows);current,_=aggregate(current_rows);saved=json.loads((root/'summary.json').read_text())
    for label,value in [('new',new),('baseline',old),('current_primary',current)]:
        for f,v in value.items():
            if isinstance(v,dict):
                for k,n in v.items():np.testing.assert_allclose(n,saved[label][f][k],atol=1e-10,rtol=0)
            else:np.testing.assert_allclose(v,saved[label][f],atol=1e-10,rtol=0)
    assert gates(new,old)==saved['gates'] and all(saved['gates'].values())==saved['numeric_gates_passed']
    stored=json.loads((root/'by_cell_seed.json').read_text());assert {(r['cell_id'],r['seed']) for r in stored}=={(r['cell_id'],r['seed']) for r in rows}
    for r in rows:
        actual=next(s for s in stored if (s['cell_id'],s['seed'])==(r['cell_id'],r['seed']))
        for f,v in r.items():
            if f not in ('cell_id','seed'):np.testing.assert_allclose(v,actual[f],atol=1e-10,rtol=0)
    stored_cells=json.loads((root/'by_cell.json').read_text());assert {c['cell_id'] for c in stored_cells}=={c['cell_id'] for c in cell_rows}
    for cell in cell_rows:
        actual=next(c for c in stored_cells if c['cell_id']==cell['cell_id'])
        for f,v in cell.items():
            if f!='cell_id':np.testing.assert_allclose(v,actual[f],atol=1e-10,rtol=0)
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,retrained_rate_bundles=50,retrained_primary_bundles=50,curves=expected,inner_locks=25,numeric_gates_passed=saved['numeric_gates_passed'],source_sha256=sha(__file__),goal_complete=False));print('CURVES='+str(expected),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
