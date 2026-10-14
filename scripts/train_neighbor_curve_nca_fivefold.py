import json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from battery_smetimes.neighbor_curve import fit,predict
from battery_smetimes.tail_selection import choose,predict as primary_predict
from battery_smetimes.tail_boost import labels,is_nca
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json

def main():
    screen_root=Path('experiments/neighbor_curve_smoke/20261005T141253_536002Z')
    assert json.loads((screen_root/'independent_verification.json').read_text())['passed']
    assert next(r for r in json.loads((screen_root/'screening_decision.json').read_text()) if r['plan']=='NCA_fold1')['expand_fivefold']
    start=time.perf_counter();root=Path('experiments/neighbor_curve_nca_fivefold')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snapshot=root/'source_snapshot';snapshot.mkdir();files=[Path(__file__),Path('docs/soh_neighbor_curve_nca_fivefold_20261005.md'),Path('scripts/train_neighbor_curve_smoke.py'),*Path('battery_smetimes').glob('*.py'),*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','train_crossfit_drift_smoke','prepare_same_type_forecast')]]
    for f in files:shutil.copy2(f,snapshot/f.name)
    routes=[Path('experiments/tail_boost/20261005T034431_681399Z')]*5;seed=20261005;candidates=[dict(k=1,weight=0.)]+[dict(k=k,weight=w) for k in (1,3) for w in (.25,.5,1.)]
    configs=[json.loads((r/'config.json').read_text()) for r in routes];plans=[cfg['plans'][i] for i,cfg in enumerate(configs)]
    save_json(root/'config.json',dict(plans=plans,seed=seed,candidates=candidates,source_sha256={str(f):sha(f) for f in files},routes=[dict(path=str(r.resolve()),sha256=sha(r/'config.json')) for r in routes],prepared_sha256={k:cfg['prepared_sha256'][k] for cfg,p in zip(configs,plans) for k in p['train']+p['validation']},outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);results=[]
    for route,cfg,plan in zip(routes,configs,plans):
        cells=load_cells(plan);assert not set(cells)&set(plan['test']);decision=choose(route,cfg,plan,seed,cells);sel=decision['selected'];file=Path(sel['selection_file']).parent/f"candidate_{sel['candidate']}"/'model.joblib'
        analog=fit(cells,plan['train']);primary=dict(mode=sel['mode'],inner=joblib.load(file),train_ids=plan['train'],family=analog['family']);assert primary['inner']['train_ids']==plan['train']
        folder=root/plan['name'];folder.mkdir();joblib.dump(dict(analog=analog,primary=primary),folder/'components.joblib');save_json(folder/'provenance.json',dict(decision=decision,primary_file=str(file.resolve()),primary_sha256=sha(file)))
        arrays={}
        for role in ('training','validation'):
            ids=plan['train'] if role=='training' else plan['validation'];out={}
            for key in ids:
                query=clean(cells[key]);c,y=labels(cells[key]);base=primary_predict(primary,query,c);values={k:predict(analog,query,c,k,exclude=key if role=='training' else None) for k in (1,3)}
                poison=dict(query,cycle_life=np.inf,soh=np.inf,truth=np.inf)
                if is_nca(query):poison.update(early_cycle=np.r_[query['early_cycle'],101],early_raw=np.vstack([query['early_raw'],np.full((1,query['early_raw'].shape[1]),np.inf)]))
                else:poison.update(cycle=np.r_[query['cycle'],101],raw=np.vstack([query['raw'],np.full((1,query['raw'].shape[1]),np.inf)]))
                for k in (1,3):
                    np.testing.assert_allclose(values[k],predict(analog,poison,c,k,exclude=key if role=='training' else None),rtol=0,atol=1e-12)
                try:predict(analog,dict(query,physical_family='wrong'),c,1)
                except ValueError:pass
                else:raise AssertionError('family gate')
                out.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('primary',base),('neighbor1',values[1]),('neighbor3',values[3])]})
            np.savez_compressed(folder/(role+'_components.npz'),**out);arrays[role]=out
        scores=[]
        for i,params in enumerate(candidates):
            dest=folder/f'candidate_{i}';dest.mkdir()
            for role in ('training','validation'):
                out=arrays[role];ids=plan['train'] if role=='training' else plan['validation'];preds={};rows=[]
                for key in ids:
                    prefix=key+'__100__';c,y,p,q=[out[prefix+f] for f in ('cycle','truth','primary','neighbor'+str(params['k']))];forecast=(1-params['weight'])*p+params['weight']*q
                    rows.append(dict(cell_id=key,**curve_metrics(c,y,forecast)));preds.update({prefix+f:v for f,v in [('cycle',c),('truth',y),('prediction',forecast)]})
                np.savez_compressed(dest/(role+'_predictions.npz'),**preds);save_json(dest/(role+'_metrics.json'),rows)
                if role=='validation':scores.append(dict(candidate=i,score=selection_score(rows),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in rows])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in rows])),mape_percent=float(np.mean([r['mape_percent'] for r in rows]))))
        selected=min(scores,key=lambda r:(r['score'],r['candidate']));control=scores[0];screen=selected['score']<control['score'] and selected['endpoint_pp']<=control['endpoint_pp'] and selected['drop_error_pp']<=control['drop_error_pp'];result=dict(plan=plan['name'],selected=selected,control=control,candidates=scores,expand_fivefold=screen);save_json(folder/'selection.json',result);results.append(result);logging.info('SAVED %s',result)
    save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False));save_json(root/'fold_results.json',results)
    wins=sum(r['selected']['score']<r['control']['score'] for r in results)
    mean={role:{f:float(np.mean([r[role][f] for r in results])) for f in ('score','endpoint_pp','drop_error_pp','mape_percent')} for role in ('selected','control')}
    expand=wins>=4 and mean['selected']['score']<mean['control']['score'] and all(mean['selected'][f]<=mean['control'][f] for f in ('endpoint_pp','drop_error_pp'))
    save_json(root/'screening_decision.json',dict(fold_score_wins=wins,means=mean,expand_formal=expand,outer_loaded=False,goal_complete=False));print(json.dumps(dict(wins=wins,means=mean,expand_formal=expand)),flush=True)

if __name__=='__main__':main()
