"""Preregistered 79-arm train-side screen; no new fits or outer cells."""
import json, logging, shutil, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scripts.prepare_same_type_forecast import sha, save_json
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_cycle_calibration import score

def specifications(parent):
    groups=[list(range(1,7)),list(range(7,13)),list(range(1,13))]
    return parent+[dict(arm='ensemble',members=g,reducer=r,weight=w)
                   for g in groups for r in ('mean','median') for w in (0.,.25,.5)]

def summarize(records):
    fields=('score','mape_percent','endpoint_pp','drop_error_pp')
    out={name:{f:float(np.mean([r[name][f] for r in records])) for f in fields}
         for name in ('selected','control')}
    n,o=out['selected'],out['control']
    wins=sum(r['selected']['score']<r['control']['score'] for r in records)
    nc=sum(r['selected']['candidate']>=61 for r in records)
    expand=(wins>=3 and nc>=2 and n['endpoint_pp']<=o['endpoint_pp']+1e-10
            and n['drop_error_pp']<=o['drop_error_pp']+1e-10
            and n['mape_percent']<=o['mape_percent']+.1)
    return dict(**out,wins=wins,new_choices=nc,expand=expand,
                new_component_fits=0,outer_loaded=False,goal_complete=False)

def main():
    parent=Path('experiments/nasa_slope_blend/20261005T162823_305584Z')
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    cfg=json.loads((parent/'config.json').read_text()); specs=specifications(cfg['candidates'])
    assert len(specs)==79
    root=Path('experiments/nasa_calibration_ensemble')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True); print('RUN='+str(root.resolve()),flush=True)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__),Path('docs/soh_nasa_calibration_ensemble_20261006.md'),Path('scripts/train_nasa_robust_inner.py'),Path('scripts/verify_nasa_cycle_calibration.py')]
    for f in files:shutil.copy2(f,snapshot/f.name)
    save_json(root/'config.json',dict(plans=cfg['plans'],seed=cfg['seed'],candidates=specs,
              parent=str(parent.resolve()),parent_sha256={f:sha(parent/f) for f in ('config.json','summary.json','independent_verification.json')},
              source_sha256={str(f):sha(f) for f in files},new_component_fits=0,outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'screen.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();records=[];allrows=[]
    try:
        for plan in cfg['plans']:
            ids=plan['train']+plan['validation']; folder=root/plan['name'];folder.mkdir();vr=[];lineage={}
            for held in ids:
                pd=parent/plan['name']/('hold_'+held);dest=folder/('hold_'+held);dest.mkdir()
                split=json.loads((pd/'split.json').read_text());assert split==dict(train=[k for k in ids if k!=held],validation=[held],excluded_outer=plan['test'])
                save_json(dest/'split.json',split)
                for role,keys in [('training',split['train']),('validation',[held])]:
                    cache=[]
                    for i in range(61):
                        file=pd/f'candidate_{i}'/(role+'_predictions.npz');lineage[str(file.resolve())]=sha(file)
                        with np.load(file) as a:cache.append({k:a[k].copy() for k in a.files})
                    for i,spec in enumerate(specs):
                        out=dest/f'candidate_{i}';out.mkdir(exist_ok=True);arrays={};rr=[]
                        for key in keys:
                            prefix=key+'__100__';c,y=cache[0][prefix+'cycle'],cache[0][prefix+'truth']
                            if i<61:p=cache[i][prefix+'prediction']
                            else:
                                reducer=np.mean if spec['reducer']=='mean' else np.median
                                direct=reducer(np.stack([cache[j][prefix+'prediction'] for j in spec['members']]),axis=0)
                                level=reducer(np.stack([cache[j+12][prefix+'prediction'] for j in spec['members']]),axis=0)
                                p=(1-spec['weight'])*level+spec['weight']*direct
                            r=dict(cell_id=key,candidate=i,fold=plan['name'],heldout=held,role=role,seed=cfg['seed'],**detail(c,y,p))
                            rr.append(r);allrows.append(r);arrays.update({prefix+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                        save_json(out/(role+'_metrics.json'),rr);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                        if role=='validation':vr.extend(rr)
                logging.info('SAVED %s hold %s 79 arms',plan['name'],held)
            choices=[]
            for i in range(79):
                rows=[r for r in vr if r['candidate']==i];assert len(rows)==3
                choices.append(dict(candidate=i,score=score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
            old=json.loads((parent/plan['name']/'selection.json').read_text())['selected']['candidate']
            record=dict(fold=plan['name'],selected=min(choices,key=lambda r:(r['score'],r['candidate'])),control=choices[old],candidates=choices,
                        selector_jackknife=[dict(omitted_validation_cell=k,candidate=min((score([r for r in vr if r['candidate']==i and r['cell_id']!=k]),i) for i in range(79))[1]) for k in sorted(ids)])
            records.append(record);save_json(folder/'selection.json',record);save_json(folder/'inherited_sha256.json',lineage)
        save_json(root/'summary.json',dict(records=records,**summarize(records)))
        save_json(root/'comprehensive_curve_metrics.json',dict(rows=allrows,count=len(allrows)))
        save_json(root/'progress.json',dict(complete=True,curves=len(allrows),seconds=time.perf_counter()-start,new_component_fits=0,outer_loaded=False,goal_complete=False))
        print(json.dumps(summarize(records)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
