"""Independent model refits, input isolation, all curves and fixed goal gates."""
import argparse,json
from pathlib import Path
from datetime import datetime
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_xjtu_segment_fold_state import refit,replay,equal_state
from scripts.verify_nasa_robust_inner import close
from scripts.audit_rate_blend_metrics import metrics,summarize,checks,assert_summary
from battery_smetimes.curve_fidelity import condition_key
from battery_smetimes.tail_selection import fit as old_fit,predict as old_predict
from scripts.train_xjtu_segment_outer import predict as actual_predict

def targets(c):
    take=(c['label_cycle']>100)&np.isfinite(c['soh'])&(c['soh']>0)
    return c['label_cycle'][take].astype(float),c['soh'][take].astype(float)

def detail(c,y,p):
    residual=p-y;ss=float(np.sum((y-y.mean())**2))
    return dict(**metrics(c,y,p),mae_soh=float(np.mean(abs(residual))),r2=1-float(residual@residual)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),start_error_pp=float(100*abs(p[0]-y[0])))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run;cfg=json.loads((root/'config.json').read_text());lock=json.loads((root/'choice_lock.json').read_text());inner=Path(cfg['inner']);icfg=json.loads((inner/'config.json').read_text());proof=json.loads((inner/'independent_verification.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    assert cfg['choice_lock_sha256']==sha(root/'choice_lock.json') and lock['outer_loaded'] is False and len(lock['records'])==25
    assert cfg['inner_config_sha256']==sha(inner/'config.json') and cfg['inner_proof_sha256']==sha(inner/'independent_verification.json') and proof['passed']
    assert cfg['plans']==icfg['plans'] and cfg['seeds']==icfg['seeds'] and len(cfg['plans'])==5
    for f,d in cfg['source_sha256'].items():assert sha(f)==d==sha(root/'source_snapshot'/Path(f).name)
    allrows=[];outer=[];refits=0;testids=[]
    for plan in cfg['plans']:
        ids=plan['train']+plan['validation'];assert not set(ids)&set(plan['test']);testids+=plan['test'];cells=load_subset(ids+plan['test'])
        for seed in cfg['seeds']:
            rec=next(r for r in lock['records'] if r['fold']==plan['name'] and r['seed']==seed);child=inner/plan['name']/str(seed);sel=json.loads((child/'selection.json').read_text());ccfg=json.loads((child/'config.json').read_text())
            assert rec['selection_sha256']==sha(child/'selection.json') and rec['choice']==sel['selected'] and rec['legacy_choice']==ccfg['control_choice']['selected']
            assert rec['config_sha256']==sha(child/'config.json') and rec['proof_sha256']==sha(child/'independent_verification.json')
            out=root/plan['name']/str(seed);timing=json.loads((out/'model_timing.json').read_text());evaluation=json.loads((out/'evaluation_timing.json').read_text())
            assert datetime.fromisoformat(lock['locked_at_utc'])<=datetime.fromisoformat(timing['fit_started_utc'])<=datetime.fromisoformat(timing['fit_finished_utc'])<=datetime.fromisoformat(evaluation['test_loaded_at_utc'])
            assert json.loads((out/'data_sha256.json').read_text())=={k:c['prepared_sha256'] for k,c in cells.items()}
            bundle=joblib.load(out/'model.joblib');assert bundle['train_ids']==ids
            if rec['choice']['candidate']:
                assert bundle['mode']=='segment';p=rec['choice']['parameters'];assert bundle['parameters']==p
                rebuilt=refit(cells,ids,bundle['inner'],p,seed);refits+=1;groups={}
                for key in ids:groups.setdefault(condition_key(cells[key]),[]).append(key)
                required={k:v for k,v in groups.items() if len(v)>=6};assert set(required)==set(rebuilt['local'])
                rebuilt['local']={k:refit(cells,m,rebuilt['local'][k],p,seed) for k,m in required.items()};refits+=len(required);bundle=dict(bundle,inner=rebuilt)
            else:
                rebuilt=old_fit(cells,ids,rec['legacy_choice'],seed);equal_state(bundle,rebuilt)
                # Legacy fallbacks reuse their previously verified fitting path;
                # this checks fresh train+validation refit, not a new architecture.
                refits+=1;bundle=rebuilt
            for role,members in [('training',plan['train']),('validation',plan['validation']),('test',plan['test'])]:
                stored=json.loads((out/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==members
                with np.load(out/(role+'_predictions.npz')) as a:
                    assert set(a.files)=={k+'__100__'+f for k in members for f in ('cycle','truth','prediction')}
                    for j,key in enumerate(members):
                        c,y=targets(cells[key]);q=clean(cells[key]);p=replay(bundle['inner'],q,c) if bundle['mode']=='segment' else old_predict(bundle,q,c)
                        np.testing.assert_array_equal(c,a[key+'__100__cycle']);np.testing.assert_array_equal(y,a[key+'__100__truth']);np.testing.assert_allclose(p,a[key+'__100__prediction'],rtol=0,atol=1e-12)
                        polluted=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,raw=q['raw'].copy())
                        if bundle['mode']=='segment':polluted['raw'][:,[22,45]]=np.inf
                        polluted['cycle']=np.r_[q['cycle'],10000];polluted['raw']=np.vstack([polluted['raw'],np.full((1,46),np.inf)])
                        np.testing.assert_allclose(p,actual_predict(bundle,polluted,c),rtol=0,atol=1e-12)
                        try:actual_predict(bundle,dict(q,physical_family='WRONG'),c)
                        except ValueError:pass
                        else:raise AssertionError('family gate failed')
                        r=dict(cell_id=key,fold=plan['name'],seed=seed,role=role,**detail(c,y,p));close(r,stored[j]);allrows.append(r)
                        if role=='test':outer.append(r)
            print('VERIFIED '+plan['name']+' seed '+str(seed),flush=True)
    assert len(testids)==len(set(testids))==55 and len(outer)==275 and len(allrows)==1375
    report=json.loads((root/'comprehensive_curve_metrics.json').read_text());assert report['count']==len(allrows)
    for r,s in zip(allrows,report['rows']):close(r,s)
    saved=json.loads((root/'summary.json').read_text());new,cells=summarize(outer);assert_summary(new,saved['new']);summaries={'new':new};evidence=[]
    for name,key,seeds in [('baseline','baseline',range(20261004,20261009)),('current_primary','current_primary',cfg['seeds'])]:
        source=Path(cfg[key]);assert sha(source/'config.json')==cfg[key+'_config_sha256'];rows=[]
        for plan in cfg['plans']:
            for seed in seeds:
                f=source/plan['name']/str(seed)/('test/predictions.npz' if name=='baseline' else 'test_predictions.npz')
                with np.load(f) as a:
                    for cell in plan['test']:
                        c,y,p=[a[cell+'__100__'+v] for v in ('cycle','truth','prediction')]
                        with np.load(root/plan['name']/str(cfg['seeds'][0])/'test_predictions.npz') as current:np.testing.assert_array_equal(c,current[cell+'__100__cycle']);np.testing.assert_array_equal(y,current[cell+'__100__truth'])
                        rows.append(dict(cell_id=cell,seed=seed,**metrics(c,y,p)))
                evidence.append(dict(file=str(f.resolve()),sha256=sha(f)))
        summary,_=summarize(rows);assert_summary(summary,saved[name]);summaries[name]=summary
    gates=checks(new,summaries['baseline']);assert gates==saved['gates'] and all(gates.values())==saved['numeric_gates_passed']
    stored={r['cell_id']:r for r in json.loads((root/'by_cell.json').read_text())};assert set(stored)==set(testids)
    for r in cells:close(r,stored[r['cell_id']])
    save_json(root/'independent_verification.json',dict(passed=True,models=25,component_refits=refits,curves=1375,outer_cell_seed_rows=275,summary=summaries,gates=gates,numeric_gates_passed=all(gates.values()),joint_cells=sum(c['joint_pass'] for c in cells),baseline_evidence=evidence,source_sha256=sha(__file__),scope='models/inputs/arrays/metrics; final scientific figures and Excel still pending',goal_complete=False))
    print(json.dumps(dict(passed=True,curves=1375,gates=gates,joint_cells=sum(c['joint_pass'] for c in cells))),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
