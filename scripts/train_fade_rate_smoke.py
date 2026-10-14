"""Fixed transformed-target smoke; training and validation only."""
import argparse,json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.fade_rate import fit,predict,anchor
from battery_smetimes.tail_boost import labels,is_nca
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mit-fivefold',action='store_true');args=parser.parse_args()
    source=Path('experiments/crossfit_drift/20261005T042311_372306Z');cfg=json.loads((source/'config.json').read_text())
    cache=Path('experiments/parameter_ridge/20261005T072119_035345Z')
    assert json.loads((cache/'independent_verification.json').read_text())['passed']
    if args.mit_fivefold:
        smoke=Path('experiments/fade_rate/20261005T083158_483446Z')
        assert json.loads((smoke/'independent_verification.json').read_text())['passed']
        source=Path('experiments/tail_boost/20261005T034441_870182Z');cfg=json.loads((source/'config.json').read_text());cfg['seed']=20261005
        assert len(cfg['plans'])==5 and all(p['dataset']=='MIT' for p in cfg['plans'])
    root=Path('experiments/fade_rate_mit_fivefold' if args.mit_fivefold else 'experiments/fade_rate')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    print('RUN='+str(root.resolve()),flush=True)
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('fade_rate','warm_monotone','monotone_network','conditional_drift','crossfit_drift','tail_boost','forecast','research_baselines')],*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    caches={p['name']:str((cache/(p['name']+'_training_parameters.json')).resolve()) for p in cfg['plans']}
    candidates=[None,dict(learned_anchor=False),dict(learned_anchor=True)]
    controls={p['name']:str((source/p['name']/'base.joblib').resolve()) for p in cfg['plans']}
    prereg='docs/soh_fade_rate_stage_20261005.md'
    if args.mit_fivefold:
        from battery_smetimes.warm_monotone import derive_prior
        candidates=[None,dict(learned_anchor=True)];prereg='docs/soh_fade_rate_mit_fivefold_20261005.md'
        controls={p['name']:str((source/p['name']/'20261005'/'candidate_0'/'model.joblib').resolve()) for p in cfg['plans']};caches={}
        for p in cfg['plans']:
            print('FIT TRAIN PARAMETER CACHE '+p['name'],flush=True)
            cells=load_cells(p);assert set(cells)==set(p['train']+p['validation']) and not set(cells)&set(p['test'])
            f=root/(p['name']+'_training_parameters.json');save_json(f,derive_prior(cells,p['train'])[1]);caches[p['name']]=str(f.resolve())
    save_json(root/'config.json',dict(plans=cfg['plans'],seed=cfg['seed'],candidates=candidates,fivefold_rate=args.mit_fivefold,source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),prepared_sha256=cfg['prepared_sha256'],source_sha256={f.name:sha(f) for f in files},parameter_files=caches,parameter_sha256={k:sha(v) for k,v in caches.items()},control_paths=controls,base_sha256={k:sha(v) for k,v in controls.items()},preregistration_file=prereg,preregistration_sha256=sha(prereg),smoke=True))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);reports=[];start=time.perf_counter()
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        parameters=json.loads(Path(caches[plan['name']]).read_text());assert [r['cell_id'] for r in parameters]==plan['train']
        for index,params in enumerate(candidates):
            folder=root/plan['name']/('candidate_'+str(index));folder.mkdir(parents=True);logging.info('FIT %s candidate=%s',plan['name'],index)
            b=dict(mode='control',base=joblib.load(controls[plan['name']]),train_ids=plan['train']) if params is None else fit(cells,plan['train'],seed=cfg['seed'],parameter_fits=parameters,**params)
            if params is None:
                assert b['base']['train_ids']==plan['train'] and b['base']['seed']==cfg['seed']
                assert b['base']['parameters']==dict(model='extra_trees',leaf=1,tail_weight=3,local=False)
            joblib.dump(b,folder/'model.joblib');rows=[];arrays={};train_rows=[];train_arrays={}
            for key in plan['train']+plan['validation']:
                cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x);item=dict(cell_id=key,**curve_metrics(x,y,p))
                target=rows if key in plan['validation'] else train_rows;target.append(item)
                target_arrays=arrays if key in plan['validation'] else train_arrays
                target_arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
                if key in plan['train'] and params is not None:
                    item['predicted_anchor']=anchor(b,cc);item['training_anchor']=float(b['training_anchors'][plan['train'].index(key)])
                if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                try:predict(b,dict(cc,physical_family='different'),x)
                except ValueError:pass
                else:raise AssertionError('family gate failed')
            for name,rr,aa in [('validation',rows,arrays),('training',train_rows,train_arrays)]:save_json(folder/(name+'_metrics.json'),rr);np.savez_compressed(folder/(name+'_predictions.npz'),**aa)
            report=dict(dataset=plan['dataset'],fold=plan['name'],candidate=index,parameters=params,score=selection_score(rows),training_mape=float(np.mean([r['mape_percent'] for r in train_rows])),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')})
            reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
    save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False));print('OUTPUT='+str(root.resolve()))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
