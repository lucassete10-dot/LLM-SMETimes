"""Preregistered functional-loss joint output heads; all diagnostics preserved."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.pls_joint import fit,predict,predicted_parameters
from battery_smetimes.tail_boost import labels,is_nca
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    torch.set_num_threads(2)
    source=Path('experiments/parameter_pls/20261005T073030_435711Z');old=json.loads((source/'config.json').read_text())
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    root=Path('experiments/pls_joint')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('pls_joint','parameter_pls','parameter_ridge','conditional_drift','crossfit_drift','tail_boost','research_baselines','forecast')],*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    candidates=[None]+[dict(components=n,decay=d) for n in (2,4) for d in (.01,.1)]
    encoders={p['name']:{str(n):str((source/p['name']/f'candidate_{i}'/'model.joblib').resolve()) for n,i in ((2,2),(4,4))} for p in old['plans']}
    save_json(root/'config.json',dict(plans=old['plans'],seed=old['seed'],candidates=candidates,epochs=2000,smoke=True,source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),prepared_sha256=old['prepared_sha256'],source_sha256={f.name:sha(f) for f in files},base_paths=old['base_paths'],base_sha256=old['base_sha256'],encoder_paths=encoders,encoder_sha256={p:{n:sha(f) for n,f in values.items()} for p,values in encoders.items()},preregistration_sha256=sha('docs/soh_pls_joint_stage_20261005.md')))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);reports=[];start=time.perf_counter()
    try:
        for plan in old['plans']:
            cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
            for index,params in enumerate(candidates):
                folder=root/plan['name']/f'candidate_{index}';folder.mkdir(parents=True);logging.info('FIT %s candidate=%s params=%s',plan['name'],index,params)
                if params is None:b=dict(mode='control',base=joblib.load(old['base_paths'][plan['name']]),train_ids=plan['train'])
                else:
                    encoder=joblib.load(encoders[plan['name']][str(params['components'])]);b=fit(cells,plan['train'],encoder,params['decay'],old['seed'])
                    save_json(folder/'loss_history.json',b['history'])
                joblib.dump(b,folder/'model.joblib')
                for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                    rows=[];arrays={};parameter_rows=[]
                    for key in ids:
                        cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x)
                        if params is not None:
                            assert np.all(np.diff(p)<=1e-12);parameter_rows.append(dict(cell_id=key,values=predicted_parameters(b,cc).tolist()))
                        if role=='validation':
                            if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                            else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                            np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                            try:predict(b,dict(cc,physical_family='different'),x)
                            except ValueError:pass
                            else:raise AssertionError('family isolation failed')
                        rows.append(dict(cell_id=key,**curve_metrics(x,y,p)));arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
                    save_json(folder/(role+'_metrics.json'),rows);np.savez_compressed(folder/(role+'_predictions.npz'),**arrays);save_json(folder/(role+'_predicted_parameters.json'),parameter_rows)
                    if role=='training':training_mape=float(np.mean([r['mape_percent'] for r in rows]))
                report=dict(dataset=plan['dataset'],fold=plan['name'],candidate=index,parameters=params,training_mape=training_mape,score=selection_score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')})
                reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
        save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_evaluated=False,goal_complete=False));print('OUTPUT='+str(root.resolve()))
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
