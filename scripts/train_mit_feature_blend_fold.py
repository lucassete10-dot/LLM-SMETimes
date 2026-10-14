"""One isolated inner fold/seed; exactly one new Vsegments specification."""
import argparse,json,logging,shutil,time,traceback
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_segment_features import fit,predict as new_predict
from battery_smetimes.tail_selection import choose,predict as old_predict
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARAMETERS=dict(block='V',context='segments')
WEIGHTS=[0.,.25,.5,.75,1.]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    parent=Path('experiments/tail_boost/20261005T034441_870182Z');pcfg=json.loads((parent/'config.json').read_text());plan=next(p for p in pcfg['plans'] if p['name']==args.fold)
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    choice=choose(parent,pcfg,plan,args.seed,cells);s=choice['selected'];sf=Path(s['selection_file']);mf=sf.parent/f"candidate_{s['candidate']}"/'model.joblib'
    old=dict(mode=s['mode'],inner=joblib.load(mf),family=cells[plan['train'][0]]['physical_family'])
    root=args.output;root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('docs/soh_mit_feature_blend_full_20261006.md'),*[Path('battery_smetimes')/(n+'.py') for n in ('mit_segment_features','tail_selection','tail_boost','curve_fidelity','forecast','research_baselines')],Path('scripts/train_curve_fidelity.py'),Path('scripts/train_crossfit_drift_smoke.py'),Path('prepare_mit.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plan=plan,seed=args.seed,parameters=PARAMETERS,weights=WEIGHTS,parent=str(parent.resolve()),parent_config_sha256=sha(parent/'config.json'),control_choice=choice,control_model_file=str(mf.resolve()),control_model_sha256=sha(mf),source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},outer_loaded=False,smoke=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter()
    try:
        new=fit(cells,plan['train'],PARAMETERS,args.seed);joblib.dump(old,root/'control.joblib');joblib.dump(new,root/'feature.joblib')
        cache={}
        for key,v in cells.items():
            c,y=targets(v);q=clean(v);cache[key]=(c,y,old_predict(old,q,c),new_predict(new,q,c))
        scores=[];count=0
        for i,w in enumerate(WEIGHTS):
            dest=root/f'candidate_{i}';dest.mkdir();vr=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];arrays={}
                for key in ids:
                    c,y,a,b=cache[key];p=(1-w)*a+w*b
                    rows.append(dict(cell_id=key,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))));count+=1
                    arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                save_json(dest/(role+'_metrics.json'),rows);np.savez_compressed(dest/(role+'_predictions.npz'),**arrays)
                if role=='validation':vr=rows
            summary=dict(candidate=i,new_weight=w,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')});scores.append(summary)
            logging.info('SAVED %s',json.dumps(summary))
        best=min(scores,key=lambda r:(r['score'],r['candidate']));control=scores[0]
        acceptable=best['candidate']>0 and best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp']
        selected=best if acceptable else control
        save_json(root/'selection.json',dict(selected=selected,best_score_candidate=best,control=control,candidates=scores,accepted_new=acceptable,outer_loaded=False,goal_complete=False))
        save_json(root/'progress.json',dict(complete=True,new_component_fits=1+len(new['local']),curves=count,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False))
        print(json.dumps(dict(selected=selected,curves=count,component_fits=1+len(new['local']))),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
