"""Fixed61-arm transformation screen; verified inherited weights, no outer load."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_cycle_calibration import score


def specifications(parent):
    return parent+[dict(arm='blend',calibration=p['calibration'],parent_direct=i,parent_level=i+12,weight=w) for i,p in enumerate(parent[1:13],1) for w in (.25,.5,.75)]


def summary(records):
    fields=('score','mape_percent','endpoint_pp','drop_error_pp');result={name:{f:float(np.mean([r[name][f] for r in records])) for f in fields} for name in ('selected','control')};new,old=result['selected'],result['control'];wins=sum(r['selected']['score']<r['control']['score'] for r in records);newchoices=sum(r['selected']['candidate']>=25 for r in records);expand=wins>=3 and newchoices>=2 and new['endpoint_pp']<=old['endpoint_pp']+1e-10 and new['drop_error_pp']<=old['drop_error_pp']+1e-10 and new['mape_percent']<=old['mape_percent']+.1
    return dict(**result,wins=wins,new_choices=newchoices,expand=expand,new_component_fits=0,outer_loaded=False,goal_complete=False)


def main():
    parent=Path('experiments/nasa_robust_inner/20261005T161437_982746Z');assert json.loads((parent/'independent_verification.json').read_text())['passed'];cfg=json.loads((parent/'config.json').read_text());specs=specifications(cfg['candidates']);assert len(specs)==61;root=Path('experiments/nasa_slope_blend')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir();files=[Path(__file__),Path('battery_smetimes/nasa_slope_blend.py'),Path('docs/soh_nasa_slope_blend_20261006.md'),Path('scripts/train_nasa_robust_inner.py'),Path('scripts/audit_rate_blend_metrics.py'),Path('scripts/verify_nasa_cycle_calibration.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(dataset='NASA',plans=cfg['plans'],seed=cfg['seed'],candidates=specs,parent=str(parent.resolve()),parent_config_sha256=sha(parent/'config.json'),parent_proof_sha256=sha(parent/'independent_verification.json'),parent_summary_sha256=sha(parent/'summary.json'),source_sha256={str(f):sha(f) for f in files},outer_loaded=False,new_component_fits=0,goal_complete=False));logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'screen.log',encoding='utf-8'),logging.StreamHandler()]);records=[];allrows=[];start=time.perf_counter()
    try:
        for plan in cfg['plans']:
            ids=plan['train']+plan['validation'];folder=root/plan['name'];folder.mkdir();vr=[];provenance={}
            for held in ids:
                pd=parent/plan['name']/('hold_'+held);dest=folder/('hold_'+held);dest.mkdir();split=json.loads((pd/'split.json').read_text());assert split==dict(train=[k for k in ids if k!=held],validation=[held],excluded_outer=plan['test']);save_json(dest/'split.json',split)
                for f in [pd/'primary.joblib',*[pd/f'calibration_{i}.joblib' for i in range(1,13)]]:provenance[str(f.resolve())]=sha(f)
                for i,spec in enumerate(specs):
                    out=dest/f'candidate_{i}';out.mkdir()
                    for role,keys in [('training',split['train']),('validation',[held])]:
                        arrays={};rr=[];lf=pd/f"candidate_{spec.get('parent_level',i)}/{role}_predictions.npz";provenance[str(lf.resolve())]=sha(lf)
                        with np.load(lf) as a:
                            for key in keys:
                                c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')]
                                if i>=25:
                                    df=pd/f"candidate_{spec['parent_direct']}/{role}_predictions.npz";provenance[str(df.resolve())]=sha(df)
                                    with np.load(df) as b:
                                        np.testing.assert_array_equal(c,b[key+'__100__cycle']);np.testing.assert_array_equal(y,b[key+'__100__truth']);p=(1-spec['weight'])*p+spec['weight']*b[key+'__100__prediction']
                                r=dict(cell_id=key,candidate=i,fold=plan['name'],heldout=held,role=role,seed=cfg['seed'],**detail(c,y,p));rr.append(r);allrows.append(r);arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                        save_json(out/(role+'_metrics.json'),rr);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                        if role=='validation':vr.extend(rr)
                logging.info('SAVED %s hold%s 61arms183curves',plan['name'],held)
            choices=[]
            for i in range(61):
                rows=[r for r in vr if r['candidate']==i];assert len(rows)==3;choices.append(dict(candidate=i,score=score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
            old=json.loads((parent/plan['name']/'selection.json').read_text())['selected']['candidate'];selected=min(choices,key=lambda r:(r['score'],r['candidate']));jackknife=[dict(omitted_validation_cell=k,candidate=min((score([r for r in vr if r['candidate']==i and r['cell_id']!=k]),i) for i in range(61))[1]) for k in sorted(ids)];record=dict(fold=plan['name'],selected=selected,control=choices[old],candidates=choices,selector_jackknife=jackknife);records.append(record);save_json(folder/'selection.json',record);save_json(folder/'inherited_sha256.json',provenance)
        save_json(root/'summary.json',dict(records=records,**summary(records)));save_json(root/'comprehensive_curve_metrics.json',dict(rows=allrows,count=len(allrows),new_component_fits=0,outer_loaded=False));save_json(root/'progress.json',dict(complete=True,curves=len(allrows),seconds=time.perf_counter()-start,new_component_fits=0,outer_loaded=False,goal_complete=False));print(json.dumps(summary(records)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
