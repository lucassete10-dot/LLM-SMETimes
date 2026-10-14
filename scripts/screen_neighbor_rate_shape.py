"""Training-only pace-aligned shape screen; no query labels used for pace."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.neighbor_shape import mix
from battery_smetimes.neighbor_rate_shape import pace,map_cycle
from battery_smetimes.neighbor_curve import predict as reference_predict
from battery_smetimes.tail_selection import predict as primary_predict
from battery_smetimes.tail_boost import conditions
from battery_smetimes.conditional_drift import prefix
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_neighbor_curve_nca_formal import score
from scripts.audit_rate_blend_metrics import metrics
from scripts.prepare_same_type_forecast import sha,save_json


def candidates():
    return [dict(mode='raw',k=1,weight=0.)]+[dict(mode=mode,k=k,weight=w) for mode in ('raw','shape') for k in (1,3) for w in (.25,.5,1.)]+[dict(mode='rate_shape',k=k,weight=w) for k in (1,3) for w in (.25,.5,1.)]


def expansion(records):
    fields=('mape_percent','endpoint_pp','drop_error_pp')
    new={f:float(np.mean([r['selected'][f] for r in records])) for f in fields}
    old={f:float(np.mean([r['raw_control'][f] for r in records])) for f in fields}
    wins=sum(r['selected']['score']<r['raw_control']['score'] for r in records)
    return dict(score_wins=wins,new=new,raw_control=old,expand=wins>=4 and new['endpoint_pp']<=old['endpoint_pp'] and new['drop_error_pp']<=old['drop_error_pp'] and new['mape_percent']<=old['mape_percent']+.1)


def main():
    parent=Path('experiments/neighbor_nested_formal/20261005T151102_801711Z')
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    locks=json.loads((parent/'inner_lock.json').read_text())['selections'];selected=[r for r in locks if r['seed']==20261005];assert len(selected)==5
    root=Path('experiments/neighbor_rate_shape_screen')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snapshot=root/'source_snapshot';snapshot.mkdir();files=[Path(__file__),Path('battery_smetimes/neighbor_shape.py'),Path('battery_smetimes/neighbor_rate_shape.py'),Path('docs/soh_neighbor_rate_shape_screen_20261005.md')]
    for f in files:shutil.copy2(f,snapshot/f.name)
    save_json(root/'config.json',dict(seed=20261005,candidates=candidates(),parent=str(parent.resolve()),parent_verification_sha256=sha(parent/'independent_verification.json'),inner_lock_sha256=sha(parent/'inner_lock.json'),source_sha256={str(f):sha(f) for f in files},children=selected,original_validation_loaded=False,outer_loaded=False,goal_complete=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'screen.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();reports=[]
    try:
        for lock in selected:
            child=Path(lock['inner_run']);cfg=json.loads((child/'config.json').read_text());assert sha(child/'config.json')==lock['inner_config_sha256']
            plan=cfg['parent_plan'];cells=load_cells(dict(plan,validation=[]));assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
            folder=root/plan['name'];folder.mkdir();pooled=[[] for _ in candidates()];anchors=[];support=[]
            for split in cfg['splits']:
                model_file=child/split['name']/'components.joblib';assert sha(model_file)==lock['component_sha256'][split['name']];parts=joblib.load(model_file);assert parts['primary']['train_ids']==split['train'];dest=folder/split['name'];dest.mkdir();outputs=[{} for _ in candidates()]
                with np.load(child/split['name']/'validation_components.npz') as a:
                    for key in split['validation']:
                        query=clean(cells[key]);base_anchor=float(primary_predict(parts['primary'],query,[101.])[0]);reference_anchor={k:float(reference_predict(parts['analog'],query,[101.],k)[0]) for k in (1,3)}
                        poison=dict(query,truth=np.inf,soh=np.inf,cycle_life=np.inf,early_cycle=np.r_[query['early_cycle'],101],early_raw=np.vstack([query['early_raw'],np.full((1,query['early_raw'].shape[1]),np.inf)]))
                        np.testing.assert_allclose(base_anchor,primary_predict(parts['primary'],poison,[101.])[0],rtol=0,atol=1e-12)
                        for k in (1,3):np.testing.assert_allclose(reference_anchor[k],reference_predict(parts['analog'],poison,[101.],k)[0],rtol=0,atol=1e-12)
                        base_201=float(primary_predict(parts['primary'],query,[201.])[0]);reference_201={k:float(reference_predict(parts['analog'],query,[201.],k)[0]) for k in (1,3)};ratios={k:pace(base_anchor,base_201,reference_anchor[k],reference_201[k]) for k in (1,3)}
                        anchors.append(dict(cell_id=key,inner=split['name'],primary=base_anchor,reference1=reference_anchor[1],reference3=reference_anchor[3],primary201=base_201,reference1_201=reference_201[1],reference3_201=reference_201[3],pace1=ratios[1],pace3=ratios[3]))
                        c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','primary')];bundle=parts['analog'];local=[i for i,r in enumerate(bundle['references']) if r['condition']==conditions(query)];pool=local if len(local)>=3 else list(range(len(bundle['references'])));q=bundle['scaler'].transform(bundle['imputer'].transform(prefix(query)[None]))[0];order=sorted(pool,key=lambda i:(float(np.linalg.norm(q-bundle['z'][i])),bundle['train_ids'][i]))[:3]
                        support.append(dict(cell_id=key,inner=split['name'],protocol_peers=len(local),family_fallback=len(local)<3,reference_ids=[bundle['train_ids'][i] for i in order],distances=[float(np.linalg.norm(q-bundle['z'][i])) for i in order]))
                        warped={k:reference_predict(parts['analog'],query,map_cycle(c,ratios[k]),k) for k in (1,3)}
                        for i,param in enumerate(candidates()):
                            ref=a[key+'__100__neighbor'+str(param['k'])];v=mix(p,warped[param['k']] if param['mode']=='rate_shape' else ref,base_anchor,reference_anchor[param['k']],param['weight']) if param['mode']!='raw' else (1-param['weight'])*p+param['weight']*ref
                            r=dict(cell_id=key,**metrics(c,y,v),initial_signed_pp=float(100*(v[0]-y[0])),trend_signed_pp=float(100*((v[-1]-v[0])-(y[-1]-y[0]))));pooled[i].append(r);outputs[i].update({key+'__100__'+f:value for f,value in [('cycle',c),('truth',y),('prediction',v)]})
                for i,out in enumerate(outputs):np.savez_compressed(dest/f'candidate_{i}_predictions.npz',**out)
            results=[]
            for i,rows in enumerate(pooled):
                assert len(rows)==39 and {r['cell_id'] for r in rows}==set(plan['train']);save_json(folder/f'candidate_{i}_metrics.json',rows)
                results.append(dict(candidate=i,parameters=candidates()[i],score=score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')},initial_abs_pp=float(np.mean([abs(r['initial_signed_pp']) for r in rows])),trend_abs_pp=float(np.mean([abs(r['trend_signed_pp']) for r in rows]))))
            winner=min(results,key=lambda r:(r['score'],r['candidate']));raw=min(results[:13],key=lambda r:(r['score'],r['candidate']));record=dict(fold=plan['name'],selected=winner,raw_control=raw,all_candidates=results);reports.append(record);save_json(folder/'selection.json',record);save_json(folder/'anchors.json',anchors);save_json(folder/'support.json',support);logging.info('SAVED %s winner%s raw%s',plan['name'],winner['candidate'],raw['candidate']);save_json(root/'progress.json',dict(complete=False,parent_folds=len(reports),outer_loaded=False))
        result=expansion(reports);save_json(root/'summary.json',dict(records=reports,screen=result,independent_validation_records=195,independent_new_cells=False,outer_loaded=False,goal_complete=False));save_json(root/'progress.json',dict(complete=True,parent_folds=5,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False));print(json.dumps(result),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
