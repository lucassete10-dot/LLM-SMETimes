"""Equal-tree-budget level/loss decomposition and condition-only diagnostic."""
from copy import deepcopy
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import time
import traceback
import joblib
import numpy as np
from battery_smetimes import continuous_length as point, level_loss_curve as split
from battery_smetimes import joint_shape_mixture as joint
from scripts.train_continuous_length_pilot import load_nominal, target, evaluate
from scripts.nca_full_label_loader import load as load_full
from scripts.prepare_same_type_forecast import sha, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['MIT','XJTU','HUST','Zhu2022_NCA'])
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--curve-mixture', action='store_true')
    args = parser.parse_args()
    source = Path('experiments/voltage_change_probe/20261008T065227_466334Z')
    cfg = json.loads((source/'config.json').read_text())
    coverage = json.loads((source/'coverage.json').read_text())
    plans = [p for p in cfg['plans'] if p['dataset'] != 'Zhu2022_NCA']
    nca_path = Path('experiments/nca_full_label_cv/20261008T065806_427489Z')
    assert json.loads((nca_path/'verification.json').read_text())['passed']
    nca = json.loads((nca_path/'config.json').read_text())
    plans += nca['plans']
    if args.dataset:
        plans = [p for p in plans if p['dataset'] == args.dataset]
    seed = args.seed
    planned_fits = 3*len(plans)
    candidates = (['point_VIT','continuous_meanparams','continuous_jointmixture'] if args.curve_mixture
                  else ['point_VIT','point_condition','level_loss'])
    root = Path('experiments/level_loss_probe')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    files = [Path(__file__), Path('battery_smetimes/level_loss_curve.py'),
             Path('battery_smetimes/continuous_length.py'), Path('scripts/nca_full_label_loader.py'),
             Path('scripts/train_continuous_length_pilot.py')]
    if args.curve_mixture:
        files.append(Path('battery_smetimes/joint_shape_mixture.py'))
    (root/'source_snapshot').mkdir()
    for p in files:
        shutil.copy2(p, root/'source_snapshot'/p.name)
    save_json(root/'config.json', dict(plans=plans, origin=100, seeds=[seed],
        source_sha256={str(p):sha(p) for p in files}, hust_index=cfg['hust_index'],
        hust_index_sha256=cfg['hust_index_sha256'], nca_label_index=nca['label_index'],
        nca_label_index_sha256=nca['label_index_sha256'], source_probe=str(source),
        candidates=candidates, trees_per_candidate=80,
        hypothesis=('Joint knee/rate donor mixtures avoid nonlinear parameter-averaging artifacts without changing signals or tree budgets.' if args.curve_mixture else
                    'Separating signal-derived initial health and future loss may reduce level/path confounding; condition-only measures individual-signal contribution.'),
        constraints='No prediction smoothing/clipping; separate health head never receives labels at inference.',
        independent_test=False, outer_loaded=False, goal_complete=False,
        limitation='One seed exploration; three existing development splits plus five NCA development folds. Not final selection.'))
    log = logging.getLogger('level_loss_probe'); log.setLevel(logging.INFO)
    fh = logging.FileHandler(root/'training.log', encoding='utf-8'); log.addHandler(fh)
    allrows, used_coverage, done = [], [], 0
    start = time.perf_counter()
    print('RUN='+str(root.resolve()), flush=True)
    try:
        for plan in plans:
            if plan['dataset'] == 'HUST':
                path = Path(cfg['hust_index']); assert sha(path) == cfg['hust_index_sha256']
                index = json.loads(path.read_text()); records = {r['cell_id']:r for r in index['cells']}
                cells = {}
                for key in plan['train']+plan['validation']:
                    record = records[key]; path = Path(record['path']); assert sha(path) == record['sha256']
                    with np.load(path, allow_pickle=False) as data:
                        cells[key] = {k:data[k].copy() for k in data.files}
                    cells[key]['physical_family'] = record['physical_family']
            elif plan['dataset'] == 'Zhu2022_NCA':
                assert sha(Path(nca['label_index'])) == nca['label_index_sha256']
                cells, _ = load_full(plan, nca['label_index'])
            else:
                cells, _ = load_nominal(plan)
            assert not set(cells)&set(plan['test'])
            assert not set(plan['train'])&set(plan['validation'])
            roles = {role:(plan[role] if plan['dataset']=='Zhu2022_NCA' else
                     next(r['used'] for r in coverage if r['fold']==plan['name'] and r['role']==role))
                     for role in ('train','validation')}
            targets = {k:target(v,100) for k,v in cells.items()}
            for key in roles['train']:
                split.reference_level(cells[key],100)
            used_coverage.append(dict(fold=plan['name'],**roles))
            save_json(root/'coverage.json', used_coverage)
            for name in candidates:
                begin = time.perf_counter()
                if name == 'continuous_jointmixture':
                    bundle = joint.fit(cells,roles['train'],targets,100,seed); module = joint
                elif name == 'continuous_meanparams':
                    bundle = point.fit(cells,roles['train'],100,'continuous','VIT',1,seed,targets); module = point
                elif name == 'level_loss':
                    bundle = split.fit(cells,roles['train'],targets,100,seed); module = split
                else:
                    bundle = point.fit(cells,roles['train'],100,'point',name.split('_')[1],1,seed,targets); module = point
                folder = root/plan['name']/name; folder.mkdir(parents=True)
                path = folder/'model.joblib'; joblib.dump(bundle,path); replay = joblib.load(path)
                arrays, rows = {}, []
                for key in roles['validation']:
                    c,y = targets[key]; p = module.predict(bundle,cells[key],c)
                    np.testing.assert_allclose(p,module.predict(replay,cells[key],c),rtol=0,atol=1e-12)
                    poison = deepcopy(cells[key])
                    if 'early_cycle' in poison:
                        poison['early_raw'][poison['early_cycle']>100] = np.inf
                        poison['truth'] = np.inf; poison['cycle'] = np.array([np.inf])
                    else:
                        poison['raw'][poison['cycle']>100] = np.inf
                        poison['soh'] = np.inf; poison['label_cycle'] = np.array([np.inf])
                        poison['reference_capacity_ah'] = np.inf
                    poison['cycle_life'] = np.inf
                    np.testing.assert_allclose(p,module.predict(bundle,poison,c),rtol=0,atol=1e-12)
                    try:
                        module.predict(bundle,dict(cells[key],physical_family='wrong'),c)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('physical family gate missing')
                    row = dict(dataset=plan['dataset'],fold=plan['name'],candidate=name,
                               origin=100,seed=seed,cell_id=key,**evaluate(c,y,p))
                    rows.append(row); allrows.append(row)
                    arrays.update({key+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                save_json(folder/'metrics.json',rows)
                np.savez_compressed(folder/'predictions.npz',**arrays)
                save_json(folder/'config.json',dict(train=roles['train'],validation=roles['validation'],
                    model_sha256=sha(path),seconds=time.perf_counter()-begin,poison_passed=True))
                done += 1; save_json(root/'all_rows.json',allrows)
                save_json(root/'progress.json',dict(completed_fits=done,planned_fits=planned_fits,terminal=False))
                message = f'{plan["name"]} {name}: MAPE={np.mean([r["mape_percent"] for r in rows]):.3f}, lateMAEpp={np.mean([r["late20_mae_pp"] for r in rows]):.3f}'
                log.info(message); print(message,flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,planned_fits=planned_fits,terminal=True,success=True,
            seconds=time.perf_counter()-start,goal_complete=False))
    except BaseException:
        log.exception('FAILED'); save_json(root/'failure.json',dict(traceback=traceback.format_exc()))
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False)); raise
    finally:
        fh.close(); log.removeHandler(fh)


if __name__ == '__main__':
    main()
