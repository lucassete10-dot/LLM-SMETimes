"""Train-only whole-cell three-fold selection; old validation never loaded."""
import argparse,json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,joblib
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from battery_smetimes.neighbor_curve import fit as reference_fit
from battery_smetimes.tail_selection import fit as primary_fit
from battery_smetimes.curve_fidelity import selection_score
from scripts.train_tail_boost import load_cells
from scripts.train_neighbor_curve_nca_formal import components,evaluate
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-fold',type=int,required=True);parser.add_argument('--seed',type=int,required=True);args=parser.parse_args();assert 0<=args.parent_fold<5
    start=time.perf_counter();route=Path('experiments/tail_boost/20261005T034431_681399Z');source=json.loads((route/'config.json').read_text());plan=source['plans'][args.parent_fold];ids=plan['train'];seed=args.seed
    cells=load_cells(dict(plan,validation=[]));assert set(cells)==set(ids) and not set(cells)&set(plan['validation']+plan['test'])
    choice=dict(mode='tail',parameters=dict(model='hist_gradient_boosting',max_leaves=15,tail_weight=1,local=False))
    candidates=[dict(k=1,weight=0.)]+[dict(k=k,weight=w) for k in (1,3) for w in (.25,.5,1.)];splits=[]
    for i,(tr,va) in enumerate(KFold(n_splits=3,shuffle=True,random_state=20261005).split(ids)):
        splits.append(dict(name='inner'+str(i+1),train=[ids[j] for j in tr],validation=[ids[j] for j in va]))
    root=Path('experiments/neighbor_nested_formal_inner')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snapshot=root/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__),Path('docs/soh_neighbor_nested_formal_20261005.md'),Path('scripts/train_neighbor_curve_nca_formal.py'),*Path('battery_smetimes').glob('*.py'),*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','train_crossfit_drift_smoke','train_nca_curve_fidelity','train_zhu2022_nca','prepare_zhu2022_nca','prepare_same_type_forecast')]]
    for f in files:shutil.copy2(f,snapshot/f.name)
    save_json(root/'config.json',dict(parent_plan=plan,seed=seed,splits=splits,primary_choice=choice,candidates=candidates,source_route=str(route.resolve()),source_config_sha256=sha(route/'config.json'),source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:source['prepared_sha256'][k] for k in ids},original_validation_loaded=False,outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);pooled=[[] for _ in candidates];fold_rows=[]
    for split in splits:
        tr,va=split['train'],split['validation'];assert not set(tr)&set(va) and set(tr+va)==set(ids);folder=root/split['name'];folder.mkdir();parts=dict(analog=reference_fit(cells,tr),primary=primary_fit(cells,tr,choice,seed));joblib.dump(parts,folder/'components.joblib')
        training=components(parts,cells,tr,folder,'training',True);validation=components(parts,cells,va,folder,'validation',False);results=[]
        for index,params in enumerate(candidates):
            dest=folder/f'candidate_{index}';dest.mkdir();evaluate(training,tr,params,dest,'training');rows=evaluate(validation,va,params,dest,'validation');pooled[index].extend(rows)
            results.append(dict(candidate=index,score=selection_score(rows),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in rows])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in rows]))))
        save_json(folder/'candidate_results.json',results);fold_rows.append(results);logging.info('SAVED %s heldout=%s scores=%s',split['name'],len(va),[r['score'] for r in results])
    scores=[]
    for index,rows in enumerate(pooled):
        assert len(rows)==len(ids) and len({r['cell_id'] for r in rows})==len(ids)
        scores.append(dict(candidate=index,score=selection_score(rows),mape_percent=float(np.mean([r['mape_percent'] for r in rows])),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in rows])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in rows]))))
        save_json(root/f'pooled_candidate_{index}_metrics.json',rows)
    selected=min(scores,key=lambda r:(r['score'],r['candidate']));control=scores[0];wins=sum(r[selected['candidate']]['score']<r[0]['score'] for r in fold_rows);screen=selected['score']<control['score'] and selected['endpoint_pp']<=control['endpoint_pp'] and selected['drop_error_pp']<=control['drop_error_pp'] and wins>=2
    save_json(root/'selection.json',dict(selected=selected,control=control,candidates=scores,subfold_score_wins=wins,expand_all_outer_training_cv=screen,outer_loaded=False,goal_complete=False));save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,original_validation_loaded=False,outer_loaded=False,goal_complete=False));print(json.dumps(dict(selected=selected,control=control,wins=wins,expand=screen)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
