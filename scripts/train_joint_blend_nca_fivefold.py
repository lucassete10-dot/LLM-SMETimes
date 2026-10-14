"""Fixed NCA global mixture, no new search and no outer data."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.parameter_pls import fit as pls_fit
from battery_smetimes.pls_joint import fit as joint_fit
from battery_smetimes.warm_monotone import derive_prior
from battery_smetimes.joint_blend import make,predict,components
from battery_smetimes.tail_selection import choose
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    torch.set_num_threads(2)
    smoke=Path('experiments/joint_blend/20261005T075316_729076Z');assert json.loads((smoke/'independent_verification.json').read_text())['passed']
    source=Path('experiments/tail_boost/20261005T034431_681399Z');old=json.loads((source/'config.json').read_text());assert len(old['plans'])==5 and all(p['dataset']=='Zhu2022_NCA' for p in old['plans'])
    root=Path('experiments/joint_blend_nca_fivefold')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('joint_blend','pls_joint','parameter_pls','parameter_ridge','warm_monotone','tail_selection','conditional_drift','tail_boost','research_baselines','forecast')],*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plans=old['plans'],seed=20261005,weights=[0,.5],components=4,logarithmic=True,decay=.01,epochs=2000,smoke=True,source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),prepared_sha256=old['prepared_sha256'],source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_joint_blend_nca_fivefold_20261005.md')))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);reports=[];start=time.perf_counter()
    try:
        for plan in old['plans']:
            cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
            folder=root/plan['name'];folder.mkdir();logging.info('FIT %s train=%s validation=%s',plan['name'],len(plan['train']),len(plan['validation']))
            fits=derive_prior(cells,plan['train'])[1];save_json(folder/'training_parameter_fits.json',fits)
            encoder=pls_fit(cells,plan['train'],4,True,20261005,fits);joblib.dump(encoder,folder/'encoder.joblib')
            joint=joint_fit(cells,plan['train'],encoder,.01,20261005);joblib.dump(joint,folder/'joint.joblib');save_json(folder/'loss_history.json',joint['history'])
            decision=choose(source,old,plan,20261005,cells);sel=decision['selected'];selection=Path(sel['selection_file']);model_file=selection.parent/f"candidate_{sel['candidate']}"/'model.joblib';assert sha(model_file)==sel['model_sha256']
            primary=dict(kind='tail_primary',mode=sel['mode'],inner=joblib.load(model_file),family=joint['family'],train_ids=plan['train']);assert primary['inner']['train_ids']==plan['train']
            save_json(folder/'provenance.json',dict(primary_config=str((source/'config.json').resolve()),primary_config_sha256=sha(source/'config.json'),selection_file=str(selection.resolve()),selection_sha256=sha(selection),primary_model_file=str(model_file.resolve()),primary_model_sha256=sha(model_file),joint_model_file=str((folder/'joint.joblib').resolve()),joint_model_sha256=sha(folder/'joint.joblib'),decision=decision))
            joblib.dump(dict(primary=primary,joint=joint),folder/'components.joblib')
            for index,w in enumerate([0,.5]):
                dest=folder/f'candidate_{index}';dest.mkdir();b=make(primary,joint,w)
                save_json(dest/'model.json',dict(schema='global_joint_blend',weight=w,components_file=str((folder/'components.joblib').resolve()),components_sha256=sha(folder/'components.joblib'),train_ids=plan['train'],family=b['family']))
                for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                    rows=[];arrays={}
                    for key in ids:
                        cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x);p0,p1=components(b,cc,x)
                        if role=='validation':
                            poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                            np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                            try:predict(b,dict(cc,physical_family='different'),x)
                            except ValueError:pass
                            else:raise AssertionError('family isolation failed')
                        rows.append(dict(cell_id=key,**curve_metrics(x,y,p)));arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p),('primary',p0),('joint',p1)]})
                    save_json(dest/(role+'_metrics.json'),rows);np.savez_compressed(dest/(role+'_predictions.npz'),**arrays)
                    if role=='training':train_mape=float(np.mean([r['mape_percent'] for r in rows]))
                report=dict(dataset='Zhu2022_NCA',fold=plan['name'],candidate=index,weight=w,training_mape=train_mape,score=selection_score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')})
                reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
        save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_evaluated=False,goal_complete=False));print('OUTPUT='+str(root.resolve()))
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
