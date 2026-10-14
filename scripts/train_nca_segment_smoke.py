"""NCA fixed four-arm signal ablation; no outer test loading."""
import json,shutil,logging,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.nca_segment_features import fit,predict
from battery_smetimes.tail_selection import choose,predict as control_predict
from battery_smetimes.curve_fidelity import selection_score
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/tail_boost/20261005T034431_681399Z')
CANDIDATES=[dict(block=b,context=c) for b in ('V','VI') for c in ('basic','segments')]


def main():
    pc=json.loads((PARENT/'config.json').read_text());plan=next(p for p in pc['plans'] if p['name']=='NCA_fold1');seed=20261005
    cells=load_cells(plan);assert not set(cells)&set(plan['test'])
    choice=choose(PARENT,pc,plan,seed,cells);selected=choice['selected'];source=Path(selected['selection_file']).parent/f"candidate_{selected['candidate']}"/'model.joblib'
    control=dict(mode=selected['mode'],inner=joblib.load(source),family=cells[plan['train'][0]]['physical_family'])
    root=Path('experiments/nca_segment_smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/nca_segment_features.py'),Path('battery_smetimes/tail_selection.py'),
        Path('docs/soh_nca_segment_smoke_20261006.md'),Path('scripts/train_tail_boost.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    index=json.loads((NCA/'index.json').read_text())
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),plan=plan,seed=seed,candidates=CANDIDATES,
        control_choice=choice,control_model_sha256=sha(source),control_model_file=str(source.resolve()),nca_index_sha256=sha(NCA/'index.json'),
        prepared_sha256={k:index['cells'][k]['sha256'] for k in cells},source_sha256={str(f):sha(f) for f in files},outer_loaded=False,smoke=True))
    coverage=[]
    for k,v in cells.items():
        c=v['early_cycle'];coverage.append(dict(cell_id=k,early_records=int(np.sum(c<=100)),segment_counts=[int(np.sum((c>s)&(c<=s+25))) for s in (0,25,50,75)]))
    save_json(root/'early_coverage.json',coverage)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();scores=[];curves=0;fits=0
    try:
        for i,parameters in enumerate([None]+CANDIDATES):
            out=root/f'candidate_{i}';out.mkdir()
            bundle=control if i==0 else fit(cells,plan['train'],parameters,seed)
            if i:fits+=1+len(bundle['local'])
            joblib.dump(bundle,out/'model.joblib')
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];arrays={}
                for key in ids:
                    c,y=cells[key]['cycle'],cells[key]['truth'];q=clean(cells[key]);p=(control_predict if i==0 else predict)(bundle,q,c)
                    rows.append(dict(cell_id=key,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))));curves+=1
                    arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                if role=='validation':vr=rows
            r=dict(candidate=i,parameters=parameters,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
            scores.append(r);logging.info('SAVED %s',json.dumps(r))
        best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
        expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
        save_json(root/'selection.json',dict(selected=best,control=old,candidates=scores,expand=expand,outer_loaded=False,goal_complete=False))
        save_json(root/'progress.json',dict(complete=True,new_component_fits=fits,curves=curves,seconds=time.perf_counter()-start,outer_loaded=False))
        print(json.dumps(dict(selected=best,expand=expand,curves=curves,component_fits=fits)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
