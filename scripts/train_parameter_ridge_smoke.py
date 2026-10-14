"""Fixed parameter transformation/regularization comparison, validation only."""
import json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.parameter_ridge import fit,predict
from battery_smetimes.warm_monotone import derive_prior
from battery_smetimes.tail_boost import labels,is_nca
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    source=Path('experiments/crossfit_drift/20261005T042311_372306Z');old=json.loads((source/'config.json').read_text())
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    root=Path('experiments/parameter_ridge')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('parameter_ridge','warm_monotone','monotone_network','conditional_drift','crossfit_drift','tail_boost','research_baselines','forecast')],*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    candidates=[None]+[dict(alpha=a,logarithmic=l) for a in (10,100) for l in (False,True)]
    save_json(root/'config.json',dict(plans=old['plans'],seed=old['seed'],candidates=candidates,smoke=True,source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),prepared_sha256=old['prepared_sha256'],source_sha256={f.name:sha(f) for f in files},base_sha256={p['name']:sha(source/p['name']/'base.joblib') for p in old['plans']},preregistration_sha256=sha('docs/soh_parameter_ridge_stage_20261005.md')))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    reports=[];start=time.perf_counter()
    for plan in old['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        logging.info('FIT training parameters %s',plan['name']);parameter_fits=derive_prior(cells,plan['train'])[1]
        save_json(root/(plan['name']+'_training_parameters.json'),parameter_fits)
        for index,params in enumerate(candidates):
            folder=root/plan['name']/f'candidate_{index}';folder.mkdir(parents=True)
            logging.info('FIT %s candidate=%s params=%s',plan['name'],index,params)
            b=dict(mode='control',base=joblib.load(source/plan['name']/'base.joblib'),train_ids=plan['train']) if params is None else fit(cells,plan['train'],seed=old['seed'],parameter_fits=parameter_fits,**params)
            joblib.dump(b,folder/'model.joblib');rows=[];arrays={}
            for key in plan['validation']:
                cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x)
                if params is not None:assert np.all(np.diff(p)<=1e-12)
                if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                try:predict(b,dict(cc,physical_family='different'),x)
                except ValueError:pass
                else:raise AssertionError('family isolation failed')
                rows.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
            save_json(folder/'validation_metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
            report=dict(dataset=plan['dataset'],fold=plan['name'],candidate=index,parameters=params,score=selection_score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')})
            reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
    save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_evaluated=False,goal_complete=False))
    print('OUTPUT='+str(root.resolve()))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
