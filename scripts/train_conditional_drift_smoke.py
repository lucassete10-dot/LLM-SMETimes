"""Frozen verified OOF source; no new outer read or baseline selection."""
import json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.conditional_drift import fit_head,predict
from battery_smetimes.tail_boost import labels,is_nca
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    source=Path('experiments/crossfit_drift/20261005T042311_372306Z')
    cfg=json.loads((source/'config.json').read_text());assert json.loads((source/'independent_verification.json').read_text())['passed']
    assert json.loads((source/'progress.json').read_text())['training_complete']
    root=Path('experiments/conditional_drift')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/conditional_drift.py'),Path('battery_smetimes/crossfit_drift.py'),
        Path('battery_smetimes/tail_boost.py'),Path('battery_smetimes/research_baselines.py'),
        Path('battery_smetimes/forecast.py'),Path('scripts/train_tail_boost.py'),
        Path('scripts/train_curve_fidelity.py'),Path('scripts/prepare_zhu2022_nca.py'),Path('scripts/train_zhu2022_nca.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    candidates=[None,dict(components=2,degree=1),dict(components=2,degree=2),dict(components=4,degree=2)]
    saved=dict(plans=cfg['plans'],seed=cfg['seed'],smoke=True,candidates=candidates,alpha=100,
        source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),
        source_verification_sha256=sha(source/'independent_verification.json'),prepared_sha256=cfg['prepared_sha256'],
        source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_conditional_drift_stage_20261005.md'),
        oof_sha256={p['name']:sha(source/p['name']/'oof_residuals.npz') for p in cfg['plans']},
        base_sha256={p['name']:sha(source/p['name']/'base.joblib') for p in cfg['plans']})
    save_json(root/'config.json',saved);start=time.perf_counter()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    reports=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert not set(cells)&set(plan['test']);base=joblib.load(source/plan['name']/'base.joblib')
        assert base['train_ids']==plan['train'];residuals={}
        with np.load(source/plan['name']/'oof_residuals.npz') as a:
            for key in plan['train']:residuals[key]={f:a[key+'__'+f] for f in ('cycle','truth','prediction')}
        for index,params in enumerate(candidates):
            dest=root/plan['name']/f'candidate_{index}';dest.mkdir(parents=True)
            logging.info('FIT %s candidate=%s params=%s',plan['name'],index,params)
            head=None if params is None else fit_head(cells,plan['train'],residuals,alpha=100,**params)
            b=dict(base=base,head=head,train_ids=plan['train']);joblib.dump(b,dest/'model.joblib');rows=[];arrays={}
            for key in plan['validation']:
                c=cells[key];x,y=labels(c);cc=clean(c);p=predict(b,cc,x)
                if is_nca(cc):
                    poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                else:
                    poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                try:predict(b,dict(cc,physical_family='different'),x)
                except ValueError:pass
                else:raise AssertionError('family gate failed')
                rows.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
            save_json(dest/'validation_metrics.json',rows);np.savez_compressed(dest/'validation_predictions.npz',**arrays)
            report=dict(dataset=plan['dataset'],fold=plan['name'],candidate=index,score=selection_score(rows),
                **{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')})
            reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
    save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_evaluated=False,goal_complete=False))
    print('OUTPUT='+str(root.resolve()))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
