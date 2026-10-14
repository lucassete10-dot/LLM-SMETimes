"""Fixed train-side feature ablation; 144 fresh fits, no outer load."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from battery_smetimes.nasa_feature_calibration import fit,predict,early_health
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_cycle_calibration import score
from scripts.prepare_same_type_forecast import sha,save_json

def specifications():
    return [None]+[dict(block=b,mode=m,alpha=a) for b in ('V','VI','VIT') for m in ('absolute','delta') for a in (10.,100.)]

def summary(records):
    fields=('score','mape_percent','endpoint_pp','drop_error_pp')
    out={n:{f:float(np.mean([r[n][f] for r in records])) for f in fields} for n in ('selected','control')};n,o=out['selected'],out['control']
    wins=sum(r['selected']['score']<r['control']['score'] for r in records);nc=sum(r['selected']['candidate']>0 for r in records)
    expand=wins>=3 and nc>=2 and n['endpoint_pp']<=o['endpoint_pp']+1e-10 and n['drop_error_pp']<=o['drop_error_pp']+1e-10 and n['mape_percent']<=o['mape_percent']+.1
    return dict(**out,wins=wins,new_choices=nc,expand=expand,outer_loaded=False,goal_complete=False)

def main():
    parent=Path('experiments/nasa_slope_blend/20261005T162823_305584Z');assert json.loads((parent/'independent_verification.json').read_text())['passed'];pcfg=json.loads((parent/'config.json').read_text());specs=specifications()
    root=Path('experiments/nasa_feature_ablation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir();files=[Path(__file__),Path('battery_smetimes/nasa_feature_calibration.py'),Path('docs/soh_nasa_feature_ablation_20261006.md'),Path('scripts/train_curve_fidelity.py'),Path('scripts/train_crossfit_drift_smoke.py'),Path('scripts/train_nasa_robust_inner.py'),Path('scripts/verify_nasa_cycle_calibration.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plans=pcfg['plans'],candidates=specs,parent=str(parent.resolve()),parent_sha256={f:sha(parent/f) for f in ('config.json','summary.json','independent_verification.json')},source_sha256={str(f):sha(f) for f in files},outer_loaded=False,seed=20261006,deterministic=True,new_input='V/VI/VIT no Ah/Wh',control_input='legacy Stage49 includes fixed-window integrals'))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();records=[];allrows=[];fits=0
    try:
        for plan in pcfg['plans']:
            ids=plan['train']+plan['validation'];cells=load_subset(ids);assert not set(ids)&set(plan['test']);folder=root/plan['name'];folder.mkdir();vr=[];provenance={}
            save_json(folder/'data_sha256.json',{k:cells[k]['prepared_sha256'] for k in ids})
            control=json.loads((parent/plan['name']/'selection.json').read_text())['selected']['candidate']
            for held in ids:
                train=[k for k in ids if k!=held];dest=folder/('hold_'+held);dest.mkdir();save_json(dest/'split.json',dict(train=train,validation=[held],excluded_outer=plan['test']))
                for i,params in enumerate(specs):
                    out=dest/f'candidate_{i}';out.mkdir();bundle=None
                    if i:bundle=fit(cells,train,params);fits+=1;joblib.dump(bundle,out/'model.joblib');save_json(out/'training_members.json',bundle['training_members'])
                    for role,keys in [('training',train),('validation',[held])]:
                        arrays={};early={};rr=[]
                        if not i:
                            path=parent/plan['name']/('hold_'+held)/f'candidate_{control}'/(role+'_predictions.npz');provenance[str(path.resolve())]=sha(path)
                            with np.load(path) as a:inherited={k:a[k].copy() for k in a.files}
                        for key in keys:
                            cell=cells[key];lc,y0=cell['label_cycle'],cell['soh'];take=(lc>100)&np.isfinite(y0)&(y0>0);c,y=lc[take],y0[take];q=clean(cell)
                            if i:
                                p=predict(bundle,q,c);ec,eh=early_health(bundle,q);early.update({key+'__cycle':ec,key+'__predicted_health':eh})
                            else:
                                np.testing.assert_array_equal(c,inherited[key+'__100__cycle']);np.testing.assert_array_equal(y,inherited[key+'__100__truth']);p=inherited[key+'__100__prediction']
                            r=dict(cell_id=key,candidate=i,fold=plan['name'],heldout=held,role=role,seed=20261006,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])))
                            rr.append(r);allrows.append(r);arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                        save_json(out/(role+'_metrics.json'),rr);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                        if i:np.savez_compressed(out/(role+'_early_health.npz'),**early)
                        if role=='validation':vr.extend(rr)
                logging.info('SAVED %s hold %s fits=%s',plan['name'],held,fits)
            choices=[]
            for i in range(13):
                rr=[r for r in vr if r['candidate']==i];assert len(rr)==3;choices.append(dict(candidate=i,score=score(rr),**{f:float(np.mean([r[f] for r in rr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
            record=dict(fold=plan['name'],selected=min(choices,key=lambda r:(r['score'],r['candidate'])),control=choices[0],candidates=choices,selector_jackknife=[dict(omitted_validation_cell=k,candidate=min((score([r for r in vr if r['candidate']==i and r['cell_id']!=k]),i) for i in range(13))[1]) for k in sorted(ids)])
            records.append(record);save_json(folder/'selection.json',record);save_json(folder/'inherited_sha256.json',provenance)
        save_json(root/'summary.json',dict(records=records,**summary(records)));save_json(root/'comprehensive_curve_metrics.json',dict(rows=allrows,count=len(allrows)))
        save_json(root/'progress.json',dict(complete=True,fits=fits,curves=len(allrows),seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False));print(json.dumps(summary(records)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
