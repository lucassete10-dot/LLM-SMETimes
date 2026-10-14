"""Reuse audited train-only components; save all fixed-weight full curves."""
import json,logging,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.rate_blend import make,predict
from battery_smetimes.fade_rate import predict as rate_predict
from battery_smetimes.tail_selection import choose,predict as primary_predict
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    source=Path('experiments/fade_rate_mit_fivefold/20261005T084219_387186Z');cfg=json.loads((source/'config.json').read_text())
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    route=Path('experiments/tail_boost/20261005T034441_870182Z');pcfg=json.loads((route/'config.json').read_text());assert cfg['plans']==pcfg['plans']
    root=Path('experiments/rate_blend_mit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('rate_blend','fade_rate','tail_selection','tail_boost','curve_fidelity','forecast','research_baselines')],*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','train_crossfit_drift_smoke')]]
    for f in files:shutil.copy2(f,snap/f.name)
    weights=[0,.25,.5,.75,1]
    save_json(root/'config.json',dict(plans=cfg['plans'],seed=cfg['seed'],weights=weights,source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),primary_run=str(route.resolve()),primary_config_sha256=sha(route/'config.json'),prepared_sha256=cfg['prepared_sha256'],source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_rate_blend_stage_20261005.md'),outer_loaded=False))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'execution.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();reports=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        decision=choose(route,pcfg,plan,cfg['seed'],cells);sel=decision['selected'];primary_file=Path(sel['selection_file']).parent/f"candidate_{sel['candidate']}"/'model.joblib'
        rate_file=source/plan['name']/'candidate_1'/'model.joblib';rate=joblib.load(rate_file);primary=dict(mode=sel['mode'],inner=joblib.load(primary_file),train_ids=plan['train'],family=rate['family'])
        inner=primary['inner'];assert inner['train_ids']==plan['train'] and rate['train_ids']==plan['train'] and rate['seed']==cfg['seed']
        folder=root/plan['name'];folder.mkdir();joblib.dump(dict(primary=primary,rate=rate),folder/'components.joblib')
        save_json(folder/'provenance.json',dict(decision=decision,primary_file=str(primary_file.resolve()),primary_sha256=sha(primary_file),rate_file=str(rate_file.resolve()),rate_sha256=sha(rate_file)))
        cache={}
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            cache[role]={}
            for key in ids:
                cell=clean(cells[key]);cycle,truth=labels(cells[key]);p=primary_predict(primary,cell,cycle);q=rate_predict(rate,cell,cycle);cache[role][key]=(cycle,truth,p,q)
                poison=dict(cell,cycle=np.r_[cell['cycle'],101],raw=np.vstack([cell['raw'],np.full((1,cell['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                np.testing.assert_allclose(p,primary_predict(primary,poison,cycle),atol=1e-12,rtol=0);np.testing.assert_allclose(q,rate_predict(rate,poison,cycle),atol=1e-12,rtol=0)
                try:predict(make(primary,rate,.5),dict(cell,physical_family='different'),cycle)
                except ValueError:pass
                else:raise AssertionError('family gate failed')
        for index,w in enumerate(weights):
            dest=folder/f'candidate_{index}';dest.mkdir();make(primary,rate,w)
            save_json(dest/'model.json',dict(weight=w,components_file=str((folder/'components.joblib').resolve()),components_sha256=sha(folder/'components.joblib'),train_ids=plan['train'],family=rate['family']))
            for role in ('training','validation'):
                rows=[];arrays={}
                for key,(cycle,truth,p,q) in cache[role].items():
                    y=(1-w)*p+w*q;rows.append(dict(cell_id=key,**curve_metrics(cycle,truth,y)))
                    arrays.update({key+'__100__'+f:a for f,a in [('cycle',cycle),('truth',truth),('prediction',y)]})
                save_json(dest/(role+'_metrics.json'),rows);np.savez_compressed(dest/(role+'_predictions.npz'),**arrays)
                if role=='validation':val=rows
            report=dict(fold=plan['name'],candidate=index,weight=w,score=selection_score(val),**{f:float(np.mean([r[f] for r in val])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')});reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
        selected=min([r for r in reports if r['fold']==plan['name']],key=lambda r:(r['score'],r['weight']));save_json(folder/'selection.json',dict(selected=selected,used_only_validation=True))
    save_json(root/'progress.json',dict(assembly_complete=True,new_models_trained=0,seconds=time.perf_counter()-start,goal_complete=False,outer_loaded=False));print('OUTPUT='+str(root.resolve()),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
