"""Recompute saved nested-CV evidence, including training-only preprocessing."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_zhu2022_nca import matrix, predict
from battery_smetimes.research_baselines import regression_metrics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def independent_matrix(cell, future):
    rows = []
    for cycle in future:
        h = (float(cycle)-100)/100
        rows.append(np.concatenate([cell['prefix'], [h, h*h, np.log1p(h)], cell['prefix']*h]))
    return np.asarray(rows)


def check_fit(bundle, cells, ids):
    assert bundle['train_ids'] == ids
    xs, weights = [], []
    for key in ids:
        c = cells[key]
        take = np.unique(np.linspace(0, len(c['truth'])-1, min(96, len(c['truth']))).round().astype(int))
        xs.append(independent_matrix(c, c['cycle'][take]))
        weights.append(np.full(len(take), 1/len(take)))
    x, w = np.concatenate(xs), np.concatenate(weights)
    w *= len(w)/w.sum()
    assert bundle['training_pairs'] == len(x)
    np.testing.assert_allclose(bundle['imputer'].statistics_, np.nanmedian(x, axis=0), rtol=0, atol=1e-12)
    x = bundle['imputer'].transform(x)
    if bundle['scaler'] is not None:
        offset = x[0]
        centered_mean = np.average(x-offset, axis=0, weights=w)
        var = np.average((x-offset-centered_mean)**2, axis=0, weights=w)
        np.testing.assert_allclose(bundle['scaler'].mean_, offset+centered_mean, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(bundle['scaler'].var_, var, rtol=1e-9, atol=1e-9)


def check_predictions(folder, bundle, cells, ids, validation=False):
    stem = 'validation_' if validation else ''
    reports = read(folder/(stem+'metrics.json'))
    assert [r['cell_id'] for r in reports] == ids
    rows = []
    with np.load(folder/(stem+'predictions.npz')) as arrays:
        assert set(arrays.files) == {key+'__100__'+f for key in ids for f in ('cycle', 'truth', 'prediction')}
        for report in reports:
            key = report['cell_id']; c = cells[key]; base = key+'__100__'
            x, y, p = [arrays[base+f] for f in ('cycle', 'truth', 'prediction')]
            np.testing.assert_array_equal(x, c['cycle'])
            np.testing.assert_array_equal(y, c['truth'])
            clean = {k:v for k,v in c.items() if k not in ('truth', 'cycle')}
            np.testing.assert_array_equal(matrix(clean, x), independent_matrix(c, x))
            np.testing.assert_allclose(predict(bundle, clean, x), p, rtol=0, atol=1e-12)
            poisoned = dict(clean, truth=np.full(len(y), np.inf), cycle=np.full(len(x), -999))
            np.testing.assert_array_equal(matrix(poisoned, x), matrix(clean, x))
            try:
                predict(bundle, dict(clean, physical_family='MIT_not_NCA'), x)
            except ValueError:
                pass
            else:
                raise AssertionError('physical family gate failed')
            for name, mask in [('metrics', np.ones(len(x), dtype=bool)), ('late_half', x >= (x[0]+x[-1])/2), ('last100', x > x[-1]-100)]:
                recalculated = regression_metrics(y[mask], p[mask])
                for metric, value in recalculated.items():
                    np.testing.assert_allclose(report[name][metric], value, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(report['endpoint_abs_soh_pp'], abs(y[-1]-p[-1])*100, rtol=0, atol=1e-12)
            assert report['future_targets'] == len(x)
            assert report['last100_observed_targets'] == int((x>x[-1]-100).sum())
            rows.append(report)
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); run = args.run; config = read(run/'config.json')
    assert not config['smoke'] and len(config['plans']) == 5 and len(config['seeds']) == 5
    prep = Path(config['prepared_directory']); audit = read(prep/'audit.json')
    assert sha(prep/'prepared_cells.joblib') == config['prepared_sha256']
    assert sha(prep/'audit.json') == config['audit_sha256']
    assert sha(run/'train_zhu2022_nca.py') == config['source_sha256']
    pc = read(prep/'config.json')
    assert sha(prep/'prepare_zhu2022_nca.py') == pc['source_sha256']
    raw = Path(audit['raw_directory']); intake = read(raw/'intake_manifest.json')
    assert sha(raw/intake['file']) == audit['raw_sha256'] == pc['raw_sha256']
    cells = joblib.load(prep/'prepared_cells.joblib'); ids = set(cells)
    assert len(ids) == audit['eligible_cells'] == 55
    assert audit['raw_cells'] == 66 and len(audit['ineligible_cells']) == 11
    for key, c in cells.items():
        cycle, a = c['early_cycle'], c['early_raw']
        assert 90 <= len(cycle) <= 100 and np.all(cycle<=100) and np.all(c['cycle']>100)
        first, last = np.median(a[:5], axis=0), np.median(a[-3:], axis=0)
        take = cycle >= cycle[-1]-9; t = cycle[take]-cycle[take].mean(); v = a[take]
        slope = np.sum(t[:,None]*(v-v.mean(axis=0)), axis=0)/np.sum(t*t)*100
        rebuilt = np.r_[first, last, last-first, slope, c['conditions'], cycle[0]/100, cycle[-1]/100, len(cycle)/100]
        np.testing.assert_allclose(c['prefix'], rebuilt, rtol=0, atol=1e-12)
    for seed in config['seeds']:
        tested = [key for p in config['plans'] for key in p['test']]
        assert len(tested) == len(set(tested)) == len(ids) and set(tested) == ids
    out = Path('experiments/zhu2022_nca_verification')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True); all_rows = []; artifacts = []; counts = dict(candidate_models=0, final_models=0)
    for model in config['candidates']:
        for plan in config['plans']:
            train, val, test = [plan[k] for k in ('train','validation','test')]
            assert not set(train)&set(val) and not set(train+val)&set(test)
            assert set(train+val+test) == ids
            for seed in config['seeds']:
                folder = run/model/plan['name']/str(seed); selection = read(folder/'selection.json'); scores = []
                for i, parameters in enumerate(config['candidates'][model]):
                    dest = folder/f'candidate_{i}'; bundle = joblib.load(dest/'model.joblib')
                    check_fit(bundle, cells, train); assert bundle['parameters'] == parameters
                    rows = check_predictions(dest, bundle, cells, val, True)
                    scores.append(float(np.mean([r['metrics']['mape_percent'] for r in rows])))
                    np.testing.assert_allclose(scores[-1], selection['candidates'][i]['validation_mape'], rtol=0, atol=1e-12)
                    counts['candidate_models'] += 1
                    artifacts.append(dict(path=str((dest/'model.joblib').resolve()), sha256=sha(dest/'model.joblib')))
                best = int(np.argmin(scores)); assert selection['selected'] == selection['candidates'][best]
                bundle = joblib.load(folder/'model.joblib'); check_fit(bundle, cells, train+val)
                assert bundle['parameters'] == config['candidates'][model][best]
                rows = check_predictions(folder, bundle, cells, test)
                all_rows.extend([dict(r, model=model, fold=plan['name'], seed=seed) for r in rows])
                counts['final_models'] += 1
                artifacts.append(dict(path=str((folder/'model.joblib').resolve()), sha256=sha(folder/'model.joblib')))
                print(model, plan['name'], seed, 'verified', flush=True)
    save_json(out/'by_cell.json', all_rows)
    summaries = {}
    for model in config['candidates']:
        rows = [r for r in all_rows if r['model']==model]; result = {}
        for section in ('metrics','late_half','last100'):
            result[section] = {}
            for metric in rows[0][section]:
                means = [np.mean([r[section][metric] for r in rows if r['seed']==seed]) for seed in config['seeds']]
                result[section][metric] = dict(mean=float(np.mean(means)), sd=float(np.std(means, ddof=1)))
        per_cell = {key:float(np.mean([r['metrics']['mape_percent'] for r in rows if r['cell_id']==key])) for key in sorted(ids)}
        result.update(cells=len(ids), per_cell_mape=per_cell, failing_cells={k:v for k,v in per_cell.items() if v>6},
            endpoint_abs_soh_pp=float(np.mean([r['endpoint_abs_soh_pp'] for r in rows])),
            deterministic_repetition=model=='ridge')
        summaries[model] = result
    save_json(out/'summary.json', summaries)
    save_json(out/'verification.json', dict(passed=True, run=str(run.resolve()), prepared=str(prep.resolve()),
        checks=['all whole-cell roles disjoint','train-only candidate/final imputer and scaler',
        'inner-only candidate selection','all test future labels preserved','all saved metrics recomputed',
        'capacity labels removed and poisoned without input change','physical family rejection',
        'cached prefixes rebuilt from <=100 source cycles','raw/prepared/source SHA256'],
        **counts, cell_seed_model_evaluations=len(all_rows), artifacts=artifacts))
    print('OUTPUT='+str(out.resolve()), flush=True)


if __name__=='__main__':
    main()
