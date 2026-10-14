"""Four-cohort fixed-budget inner validation, preserving all future arrays."""
import argparse,json,logging,shutil,traceback
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,sklearn
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import fit,predict,labels,is_nca
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_curve_fidelity import AUDIT,load_subset
from scripts.train_nca_curve_fidelity import OLD
from scripts.prepare_same_type_forecast import save_json,sha

NCA=Path('data/interim/nca_curve_cells/20261005T031838_298841Z')
CANDIDATES=[dict(model='extra_trees',leaf=l,tail_weight=3,local=loc) for l in (1,4) for loc in (False,True)]+[
    dict(model='hist_gradient_boosting',max_leaves=n,tail_weight=t,local=False) for n in (15,31) for t in (1,3)]


def load_cells(plan):
    ids=plan['train']+plan['validation']
    if plan['dataset']!='Zhu2022_NCA':return load_subset(ids)
    index=json.loads((NCA/'index.json').read_text());cells={}
    for key in ids:
        entry=index['cells'][key];p=NCA/entry['file'];assert sha(p)==entry['sha256'];cells[key]=joblib.load(p)
    return cells


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');parser.add_argument('--dataset')
    args=parser.parse_args();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=Path('experiments/tail_boost')/stamp;out.mkdir(parents=True);snap=out/'source_snapshot';snap.mkdir()
    sources=[Path(__file__),Path('battery_smetimes/tail_boost.py'),Path('battery_smetimes/curve_fidelity.py'),
        Path('battery_smetimes/research_baselines.py'),Path('battery_smetimes/forecast.py'),
        Path('scripts/prepare_zhu2022_nca.py'),Path('scripts/train_zhu2022_nca.py'),Path('scripts/train_curve_fidelity.py')]
    for p in sources:shutil.copy2(p,snap/p.name)
    shutil.copy2('docs/soh_tail_boost_stage_20261005.md',out/'plan.md')
    plans=json.loads((AUDIT/'splits.json').read_text())['plans']
    plans+=[dict(p,dataset='Zhu2022_NCA') for p in json.loads((OLD/'config.json').read_text())['plans']]
    if not args.full:plans=[p for p in plans if p['name'].endswith('fold1')]
    if args.dataset:plans=[p for p in plans if p['dataset']==args.dataset]
    if not plans:raise ValueError('no matching plans')
    seeds=list(range(20261005,20261010)) if args.full else [20261005]
    registry=json.loads((AUDIT/'cell_registry.json').read_text())['cells'];index=json.loads((NCA/'index.json').read_text())
    members={k for p in plans for k in p['train']+p['validation']}
    references={}
    if args.full:
        outer_configs={'MIT':'experiments/curve_fidelity_outer/20261005T030425_342025Z/config.json',
            'XJTU':'experiments/curve_fidelity_outer/20261005T023843_153503Z/config.json',
            'NASA':'experiments/curve_fidelity_outer/20261005T032919_625507Z/config.json'}
        for dataset in {p['dataset'] for p in plans if p['dataset']!='Zhu2022_NCA'}:
            # Read ONLY frozen training-source routing from configuration, not
            # outer arrays/metrics. Reference choices remain inner selections.
            route=json.loads(Path(outer_configs[dataset]).read_text())
            for item in route['validation_sources']:
                ref=Path(item['run']);rcfg=json.loads((ref/'config.json').read_text())
                references[rcfg['plan']['name']]=dict(run=str(ref),config_sha256=sha(ref/'config.json'),layout='core')
        if any(p['dataset']=='Zhu2022_NCA' for p in plans):
            ref=Path('experiments/nca_curve_fidelity/20261005T032116_988981Z')
            for p in plans:
                if p['dataset']=='Zhu2022_NCA':references[p['name']]=dict(run=str(ref.resolve()),config_sha256=sha(ref/'config.json'),layout='nca')
        assert set(references)=={p['name'] for p in plans}
    save_json(out/'config.json',dict(plans=plans,seeds=seeds,candidates=CANDIDATES,smoke=not args.full,
        source_sha256={p.name:sha(p) for p in sources},sklearn_version=sklearn.__version__,
        prepared_sha256={k:(registry[k]['prepared_sha256'] if k in registry else index['cells'][k]['sha256']) for k in members},
        nca_index_sha256=sha(NCA/'index.json'),core_splits_sha256=sha(AUDIT/'splits.json'),
        nca_original_config_sha256=sha(OLD/'config.json'),outer_test_loaded=False,
        frozen_references=references,
        primary_outer_policy='minimum fixed validation score of new eight-candidate winner and frozen first-stage winner; ties keep original',
        selection='original curve-fidelity selection_score; no row-level early stopping'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    reports=[]
    try:
        for plan in plans:
            cells=load_cells(plan);assert not set(cells)&set(plan['test'])
            for seed in seeds:
                scores=[]
                for i,parameters in enumerate(CANDIDATES):
                    logging.info('TRAIN %s seed=%s candidate=%s %s',plan['name'],seed,i,parameters)
                    folder=out/plan['name']/str(seed)/f'candidate_{i}';folder.mkdir(parents=True)
                    b=fit(cells,plan['train'],parameters,seed);joblib.dump(b,folder/'model.joblib')
                    rows=[];arrays={}
                    for key in plan['validation']:
                        c=cells[key];x,y=labels(c)
                        remove=('cycle','truth') if is_nca(c) else ('soh','label_cycle','unfiltered_soh','reference_capacity_ah')
                        clean={k:v for k,v in c.items() if k not in remove};p=predict(b,clean,x)
                        if is_nca(c):
                            poison=dict(clean,early_cycle=np.r_[clean['early_cycle'],101,10000],
                                early_raw=np.vstack([clean['early_raw'],np.full((2,clean['early_raw'].shape[1]),np.inf)]),
                                prefix=np.full_like(clean['prefix'],np.inf),truth=np.inf,cycle_life=np.inf)
                        else:
                            poison=dict(clean,cycle=np.r_[clean['cycle'],101,10000],
                                raw=np.vstack([clean['raw'],np.full((2,clean['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                        np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                        try:predict(b,dict(clean,physical_family='different'),x)
                        except ValueError:pass
                        else:raise AssertionError('family gate failed')
                        rows.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                        arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
                    save_json(folder/'validation_metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                    summary=dict(fold=plan['name'],seed=seed,candidate=i,parameters=parameters,selection_score=selection_score(rows),
                        **{k:float(np.mean([r[k] for r in rows])) for k in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','joint_pass')})
                    reports.append(summary);scores.append(summary);save_json(out/'summary.json',reports)
                    logging.info('SAVED %s candidate=%s score=%.3f MAPE=%.3f endpoint=%.3f drop=%.3f',
                        plan['name'],i,summary['selection_score'],summary['mape_percent'],summary['endpoint_pp'],summary['drop_error_pp'])
                save_json(out/plan['name']/str(seed)/'selection.json',dict(selected=min(scores,key=lambda r:r['selection_score']),
                    candidates=scores,used_only_validation=True))
        save_json(out/'progress.json',dict(training_complete=True,outer_evaluation_complete=False,goal_complete=False,
            smoke=not args.full,label_free_and_future_poison_passed=True))
        print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
