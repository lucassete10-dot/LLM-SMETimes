"""Four fixed relative-anchor candidates, independent internal NASA folds."""
import json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.relative_anchor import fit,predict
from battery_smetimes.tail_boost import labels
from battery_smetimes.fade_rate import predict as fade
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    previous=Path('experiments/decoupled_anchor/20261005T091119_090689Z');pcfg=json.loads((previous/'config.json').read_text());assert json.loads((previous/'independent_verification.json').read_text())['passed']
    rate=Path(pcfg['rate_run']);root=Path('experiments/relative_anchor')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('relative_anchor','pooled_anchor','fade_rate','tail_boost','forecast','research_baselines')],*[Path('scripts')/(n+'.py') for n in ('train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    candidates=[dict(compact=c,alpha=a) for c in (False,True) for a in (10,100)]
    save_json(root/'config.json',dict(plans=pcfg['plans'],seed=pcfg['seed'],candidates=candidates,prepared_sha256=pcfg['prepared_sha256'],previous_run=str(previous.resolve()),previous_config_sha256=sha(previous/'config.json'),rate_run=str(rate.resolve()),rate_config_sha256=sha(rate/'config.json'),source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_relative_anchor_stage_20261005.md'),outer_loaded=False))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);reports=[];start=time.perf_counter()
    for plan in pcfg['plans']:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
        rate_file=rate/plan['name']/str(pcfg['seed'])/'inner'/'model.joblib';base=joblib.load(rate_file)
        for index,params in enumerate(candidates):
            folder=root/plan['name']/('candidate_'+str(index));folder.mkdir(parents=True);logging.info('FIT %s candidate=%s',plan['name'],index)
            b=fit(cells,plan['train'],seed=pcfg['seed'],rate=base,**params);joblib.dump(b,folder/'model.joblib');save_json(folder/'provenance.json',dict(rate_file=str(rate_file.resolve()),rate_sha256=sha(rate_file)));validation=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];arrays={}
                for key in ids:
                    c=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,c,x);d=fade(base,c,x);np.testing.assert_allclose(p[0]-p[-1],d[0]-d[-1],atol=1e-12,rtol=0)
                    poison=dict(c,cycle=np.r_[c['cycle'],101],raw=np.vstack([c['raw'],np.full((1,c['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf);np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                    try:predict(b,dict(c,physical_family='different'),x)
                    except ValueError:pass
                    else:raise AssertionError('family gate failed')
                    rows.append(dict(cell_id=key,**curve_metrics(x,y,p)));arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
                save_json(folder/(role+'_metrics.json'),rows);np.savez_compressed(folder/(role+'_predictions.npz'),**arrays)
                if role=='validation':validation=rows
            report=dict(fold=plan['name'],candidate=index,parameters=params,score=selection_score(validation),**{f:float(np.mean([r[f] for r in validation])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')});reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
    save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,goal_complete=False));print('OUTPUT='+str(root.resolve()),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
