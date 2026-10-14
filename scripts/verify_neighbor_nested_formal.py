"""Replay locked nested selections and independently refit all outer HGB models."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import inputs,labels
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells,NCA
from scripts.verify_neighbor_curve_nca_formal import reference_replay,audit,score
from scripts.audit_rate_blend_metrics import metrics,summarize,checks,assert_summary
from scripts.prepare_same_type_forecast import sha,save_json


def independent_primary(cells,ids,seed,choice):
    assert choice==dict(mode='tail',parameters=dict(model='hist_gradient_boosting',max_leaves=15,tail_weight=1,local=False))
    xs=[];ys=[];ws=[]
    for key in ids:
        c,y=labels(cells[key]);idx=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int))
        xs.append(inputs(clean(cells[key]),c[idx]));ys.append(y[idx]);ws.append(np.full(len(idx),1/len(idx)))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
    model=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1,max_bins=64,early_stopping=False,random_state=seed).fit(imputer.transform(x),y,sample_weight=w)
    family=cells[ids[0]]['physical_family'];assert {cells[k]['physical_family'] for k in ids}=={family}
    inner=dict(model=model,imputer=imputer,local={},train_ids=ids,family=family,parameters=choice['parameters'],seed=seed)
    return dict(mode='tail',inner=inner,train_ids=ids,family=family,parameters=choice['parameters'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text());progress=json.loads((root/'progress.json').read_text());assert progress['training_complete']
    for f,h in cfg['source_sha256'].items():assert sha(f)==h and sha(root/'source_snapshot'/Path(f).name)==h
    route=Path(cfg['training_source']);pcfg=json.loads((route/'config.json').read_text());assert sha(route/'config.json')==cfg['training_config_sha256'] and pcfg['plans']==cfg['plans']
    assert cfg['seeds']==list(range(20261005,20261010)) and cfg['split_seed']==20261005
    assert sha(cfg['baseline_config_path'])==cfg['baseline_config_sha256'] and sha(Path(cfg['current_primary'])/'config.json')==cfg['current_config_sha256']
    index=json.loads((NCA/'index.json').read_text())['cells']
    for key,h in cfg['prepared_sha256'].items():assert sha(NCA/index[key]['file'])==h
    assert sha(root/'inner_lock.json')==progress['inner_lock_sha256'];lockdoc=json.loads((root/'inner_lock.json').read_text());assert lockdoc['outer_loaded'] is False;locks=lockdoc['selections']
    assert len(locks)==25 and len({(r['fold'],r['seed']) for r in locks})==25
    lock_time=(root/'inner_lock.json').stat().st_mtime_ns;all_ids=[k for p in cfg['plans'] for k in p['test']];assert len(all_ids)==len(set(all_ids))==55
    rows=[];controlrows=[];curves=0;innercurves=0;evidence=[]
    # Validate every training-only selection before loading any outer cell here.
    for plan in cfg['plans']:
        assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
        for seed in cfg['seeds']:
            lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed);child=Path(lock['inner_run']);cc=json.loads((child/'config.json').read_text())
            assert cc['parent_plan']==plan and cc['seed']==seed and cc['outer_loaded'] is False and cc['original_validation_loaded'] is False
            assert cc['primary_choice']==cfg['primary_choice'] and cc['candidates']==cfg['candidates']
            for name,key in [('config.json','inner_config_sha256'),('selection.json','selection_sha256'),('independent_verification.json','verification_sha256')]:
                assert sha(child/name)==lock[key] and (child/name).stat().st_mtime_ns<=lock_time
            proof=json.loads((child/'independent_verification.json').read_text());assert proof['passed'] and proof['outer_loaded'] is False and proof['original_validation_loaded'] is False
            assert proof['source_sha256']==sha('scripts/verify_neighbor_nested_cv_seed.py') and proof['reference_verifier_sha256']==sha('scripts/verify_neighbor_curve_nca_formal.py')
            assert proof['independent_hgb_refits']==3;innercurves+=proof['curves']
            for name,h in lock['component_sha256'].items():assert sha(child/name/'components.joblib')==h and (child/name/'components.joblib').stat().st_mtime_ns<=lock_time
            scores=[]
            for i in range(len(cfg['candidates'])):
                pooled=json.loads((child/f'pooled_candidate_{i}_metrics.json').read_text());assert {r['cell_id'] for r in pooled}==set(plan['train']) and len(pooled)==39;scores.append(score(pooled))
            winner=min(range(len(scores)),key=lambda i:(scores[i],i));assert winner==lock['selected']['candidate'] and cfg['candidates'][winner]==lock['parameters'];np.testing.assert_allclose(scores[winner],lock['selected']['score'],rtol=0,atol=1e-10)
            process=json.loads((root/plan['name']/str(seed)/'inner_verification_process.json').read_text());assert process['exit_code']==0
    for plan in cfg['plans']:
        ids=plan['train']+plan['validation'];cells=load_cells(dict(plan,train=ids+plan['test'],validation=[]))
        for seed in cfg['seeds']:
            lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed);folder=root/plan['name']/str(seed)/'outer';assert (folder/'model.joblib').stat().st_mtime_ns>=lock_time
            parts=joblib.load(folder/'model.joblib');assert parts['parameters']==lock['parameters'] and parts['primary']['train_ids']==ids and parts['primary']['parameters']==cfg['primary_choice']['parameters']
            primary=independent_primary(cells,ids,seed,cfg['primary_choice']);np.testing.assert_array_equal(primary['inner']['imputer'].statistics_,parts['primary']['inner']['imputer'].statistics_)
            reference=reference_replay(parts['analog'],cells,ids)
            audit(folder,parts,primary,reference,cells,ids,'training',[lock['parameters']],True,seed)
            result=audit(folder,parts,primary,reference,cells,plan['test'],'test',[lock['parameters']],False,seed)[0];rows.extend(result);curves+=len(ids)+len(plan['test'])
            with np.load(folder/'test_components.npz') as a:
                for key in plan['test']:controlrows.append(dict(cell_id=key,seed=seed,**metrics(*[a[key+'__100__'+f] for f in ('cycle','truth','primary')])))
            evidence.append(dict(file=str((folder/'model.joblib').resolve()),sha256=sha(folder/'model.joblib')));print('OUTER VERIFIED '+plan['name']+' '+str(seed),flush=True)
    stored=json.loads((root/'summary.json').read_text());new,cellrows=summarize(rows);control,_=summarize(controlrows);assert_summary(new,stored['new']);assert_summary(control,stored['fixed_hgb_control']);other={}
    for label,path,seeds,suffix in [('baseline',Path(cfg['baseline']),range(20261004,20261009),'predictions.npz'),('current_primary',Path(cfg['current_primary']),cfg['seeds'],'test_predictions.npz')]:
        records=[]
        for plan in cfg['plans']:
            for seed in seeds:
                file=path/plan['name']/str(seed)/suffix
                with np.load(file) as a,np.load(root/plan['name']/str(cfg['seeds'][0])/'outer/test_predictions.npz') as actual:
                    for key in plan['test']:
                        c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];np.testing.assert_array_equal(c,actual[key+'__100__cycle']);np.testing.assert_array_equal(y,actual[key+'__100__truth']);records.append(dict(cell_id=key,seed=seed,**metrics(c,y,p)))
                evidence.append(dict(file=str(file.resolve()),sha256=sha(file)))
        other[label],_=summarize(records);assert_summary(other[label],stored[label])
    gate=checks(new,other['baseline']);assert gate==stored['gates'] and all(gate.values())==stored['numeric_gates_passed']
    saved=json.loads((root/'by_cell.json').read_text());assert len(saved)==55
    for a,b in zip(cellrows,saved):
        assert a['cell_id']==b['cell_id']
        for f,v in a.items():
            if f!='cell_id':np.testing.assert_allclose(v,b[f],rtol=0,atol=1e-10)
    save_json(root/'independent_verification.json',dict(passed=True,outer_curves=curves,verified_child_curves=innercurves,independent_outer_hgb_refits=25,independent_child_hgb_refits=75,new=new,gates=gate,numeric_gates_passed=all(gate.values()),failed_joint_cells=[r for r in cellrows if not r['joint_pass']],evidence=evidence,source_sha256=sha(__file__),goal_complete=False))
    print(json.dumps(dict(passed=True,outer_curves=curves,verified_child_curves=innercurves,gates=gate)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
