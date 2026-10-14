"""Assemble fixed global weights from frozen train-only models, never outer refits."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.joint_blend import make,predict,components
from battery_smetimes.tail_selection import choose
from battery_smetimes.tail_boost import labels,is_nca
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    source=Path('experiments/pls_joint/20261005T074128_082057Z');old=json.loads((source/'config.json').read_text())
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    root=Path('experiments/joint_blend')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('joint_blend','pls_joint','parameter_pls','parameter_ridge','tail_selection','conditional_drift','tail_boost','research_baselines','forecast')],*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    weights=[0,.25,.5,.75,1];joint_candidates={'MIT':3,'XJTU':4,'Zhu2022_NCA':3,'NASA':2}
    routes={'MIT':'20261005T034441_870182Z','XJTU':'20261005T034452_066902Z','Zhu2022_NCA':'20261005T034431_681399Z'}
    save_json(root/'config.json',dict(plans=old['plans'],seed=old['seed'],weights=weights,joint_candidates=joint_candidates,smoke=True,source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),prepared_sha256=old['prepared_sha256'],source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_joint_blend_stage_20261005.md')))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);reports=[];start=time.perf_counter()
    try:
        for plan in old['plans']:
            cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
            folder=root/plan['name'];folder.mkdir();joint_file=source/plan['name']/f"candidate_{joint_candidates[plan['dataset']]}"/'model.joblib';joint=joblib.load(joint_file)
            if plan['dataset']=='NASA':
                route=Path('experiments/nasa_signal_trajectory/ridge/20261004T165029_391663Z');cfg=json.loads((route/'config.json').read_text());p=next(p for p in cfg['plans'] if p['name']==plan['name'])
                for role in ('train','validation','test'):assert p[role]==plan[role]
                selection=route/plan['name']/str(old['seed'])/'selection.json';sel=json.loads(selection.read_text())
                idx=cfg['candidates'].index(sel['selected']['parameters']);model_file=selection.parent/f'candidate_{idx}'/'model.joblib';inner=joblib.load(model_file)
                assert inner['train_ids']==plan['train'] and inner['used_train_ids']==plan['train']
                primary=dict(kind='research_baseline',inner=inner,family=inner['physical_family'],train_ids=plan['train']);decision=sel
            else:
                route=Path('experiments/tail_boost')/routes[plan['dataset']];cfg=json.loads((route/'config.json').read_text())
                decision=choose(route,cfg,plan,old['seed'],cells);sel=decision['selected'];selection=Path(sel['selection_file']);model_file=selection.parent/f"candidate_{sel['candidate']}"/'model.joblib'
                primary=dict(kind='tail_primary',mode=sel['mode'],inner=joblib.load(model_file),family=joint['family'],train_ids=plan['train'])
                assert sha(model_file)==sel['model_sha256']
            make(primary,joint,0)
            provenance=dict(primary_config=str((route/'config.json').resolve()),primary_config_sha256=sha(route/'config.json'),selection_file=str(selection.resolve()),selection_sha256=sha(selection),primary_model_file=str(model_file.resolve()),primary_model_sha256=sha(model_file),joint_model_file=str(joint_file.resolve()),joint_model_sha256=sha(joint_file),decision=decision)
            save_json(folder/'provenance.json',provenance);joblib.dump(dict(primary=primary,joint=joint),folder/'components.joblib')
            for index,w in enumerate(weights):
                dest=folder/f'candidate_{index}';dest.mkdir();b=make(primary,joint,w);rows=[];arrays={};logging.info('ASSEMBLE %s weight=%s',plan['name'],w)
                save_json(dest/'model.json',dict(schema='global_joint_blend',weight=w,components_file=str((folder/'components.joblib').resolve()),components_sha256=sha(folder/'components.joblib'),train_ids=plan['train'],family=b['family']))
                for key in plan['validation']:
                    cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x);p0,p1=components(b,cc,x)
                    if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                    else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                    try:predict(b,dict(cc,physical_family='different'),x)
                    except ValueError:pass
                    else:raise AssertionError('family isolation failed')
                    rows.append(dict(cell_id=key,**curve_metrics(x,y,p)));arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p),('primary',p0),('joint',p1)]})
                save_json(dest/'validation_metrics.json',rows);np.savez_compressed(dest/'validation_predictions.npz',**arrays)
                report=dict(dataset=plan['dataset'],fold=plan['name'],candidate=index,weight=w,score=selection_score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')})
                reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
            eligible=[r for r in reports if r['fold']==plan['name']];save_json(folder/'selection.json',dict(selected=min(eligible,key=lambda r:(r['score'],r['weight'])),used_only_validation=True,one_weight_for_all_cells_and_times=True))
        save_json(root/'progress.json',dict(training_complete=True,new_gradient_models_trained=0,seconds=time.perf_counter()-start,outer_evaluated=False,goal_complete=False));print('OUTPUT='+str(root.resolve()))
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
