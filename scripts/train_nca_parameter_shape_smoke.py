"""Four locked early-feature-to-shape heads, excluding all outer cells."""
import json,shutil,logging,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.nca_parameter_shape import fit,predict
from battery_smetimes.tail_selection import choose,predict as control_predict
from battery_smetimes.curve_fidelity import selection_score
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/tail_boost/20261005T034431_681399Z')


def main():
    pc=json.loads((PARENT/'config.json').read_text());plan=next(p for p in pc['plans'] if p['name']=='NCA_fold1');seed=20261005
    cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    choice=choose(PARENT,pc,plan,seed,cells);s=choice['selected'];source=Path(s['selection_file']).parent/f"candidate_{s['candidate']}"/'model.joblib'
    control=dict(mode=s['mode'],inner=joblib.load(source),family=cells[plan['train'][0]]['physical_family'])
    root=Path('experiments/nca_parameter_shape_smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir();files=[Path(__file__),*[Path('battery_smetimes')/(f+'.py') for f in ('nca_parameter_shape','tail_boost','curve_fidelity','knee_forecast')],Path('docs/soh_nca_parameter_shape_smoke_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    index=json.loads((NCA/'index.json').read_text());candidates=[dict(shape=s,leaf=l) for s in ('quadratic','knee') for l in (1,3)]
    save_json(root/'config.json',dict(plan=plan,seed=seed,candidates=candidates,parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        control_choice=choice,control_model_file=str(source.resolve()),control_model_sha256=sha(source),
        prepared_sha256={k:index['cells'][k]['sha256'] for k in cells},nca_index_sha256=sha(NCA/'index.json'),
        source_sha256={str(f):sha(f) for f in files},outer_loaded=False,smoke=True))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();scores=[];curves=0
    for i,parameters in enumerate([None]+candidates):
        out=root/f'candidate_{i}';out.mkdir();b=control if i==0 else fit(cells,plan['train'],parameters,seed);joblib.dump(b,out/'model.joblib')
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            rows=[];arrays={}
            for k in ids:
                c,y=cells[k]['cycle'],cells[k]['truth'];p=(control_predict if i==0 else predict)(b,clean(cells[k]),c)
                rows.append(dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))));curves+=1
                arrays.update({k+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
            save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
            if role=='validation':vr=rows
        r=dict(candidate=i,parameters=parameters,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
        scores.append(r);logging.info('SAVED %s',json.dumps(r))
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    save_json(root/'selection.json',dict(selected=best,control=old,candidates=scores,expand=expand,outer_loaded=False,goal_complete=False))
    save_json(root/'progress.json',dict(complete=True,new_component_fits=4,curves=curves,seconds=time.perf_counter()-start,outer_loaded=False))
    print(json.dumps(dict(selected=best,expand=expand,curves=curves)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
