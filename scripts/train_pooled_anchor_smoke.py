"""Fourfold NASA fixed pooled-anchor budget; never outer access."""
import json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.pooled_anchor import fit,predict,anchor
from battery_smetimes.fade_rate import predict as control_predict
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    source=Path('experiments/nasa_rate_formal/20261005T085515_209007Z');old=json.loads((source/'config.json').read_text());assert json.loads((source/'independent_verification.json').read_text())['passed']
    root=Path('experiments/pooled_anchor')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('pooled_anchor','fade_rate','conditional_drift','crossfit_drift','tail_boost','forecast','research_baselines')],*[Path('scripts')/(n+'.py') for n in ('train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    candidates=[None]+[dict(kind='ridge',alpha=a) for a in (1,10,100)]+[dict(kind='extra_trees',alpha=None)]
    save_json(root/'config.json',dict(plans=old['plans'],seed=20261005,candidates=candidates,prepared_sha256=old['prepared_sha256'],source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),source_sha256={f.name:sha(f) for f in files},control_sha256={p['name']:sha(source/p['name']/'20261005'/'inner'/'model.joblib') for p in old['plans']},preregistration_sha256=sha('docs/soh_nasa_pooled_anchor_stage_20261005.md'),outer_loaded=False))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);reports=[];start=time.perf_counter()
    for plan in old['plans']:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
        for idx,params in enumerate(candidates):
            folder=root/plan['name']/('candidate_'+str(idx));folder.mkdir(parents=True);logging.info('FIT %s candidate=%s',plan['name'],idx)
            b=joblib.load(source/plan['name']/'20261005'/'inner'/'model.joblib') if params is None else fit(cells,plan['train'],seed=20261005,**params)
            assert b['train_ids']==plan['train'];joblib.dump(b,folder/'model.joblib');fn=control_predict if params is None else predict;validation=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];arrays={}
                for key in ids:
                    c=clean(cells[key]);x,y=labels(cells[key]);p=fn(b,c,x);rows.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                    if params is not None:rows[-1]['predicted_anchor']=anchor(b,c)
                    arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
                    poison=dict(c,cycle=np.r_[c['cycle'],101],raw=np.vstack([c['raw'],np.full((1,c['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(p,fn(b,poison,x),atol=1e-12,rtol=0)
                    try:fn(b,dict(c,physical_family='different'),x)
                    except ValueError:pass
                    else:raise AssertionError('family gate failed')
                save_json(folder/(role+'_metrics.json'),rows);np.savez_compressed(folder/(role+'_predictions.npz'),**arrays)
                if role=='validation':validation=rows
            report=dict(dataset='NASA',fold=plan['name'],candidate=idx,parameters=params,score=selection_score(validation),**{f:float(np.mean([r[f] for r in validation])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')});reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
    save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False));print('OUTPUT='+str(root.resolve()),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
