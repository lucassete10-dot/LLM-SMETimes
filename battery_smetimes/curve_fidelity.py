"""Early-V/I-conditioned whole-curve models; no query-cell capacity input."""
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_smetimes.forecast import query_matrix
from battery_smetimes.research_baselines import signal_prefix, build_training_pairs, regression_metrics


def condition_key(cell):
    """Known experimental protocol, never lifespan or target-derived cluster."""
    meta = cell['metadata']
    return (meta.get('charge_protocol'), meta.get('discharge_rate_c'),
            meta.get('discharge_cutoff_v'))


def targets(cell):
    keep = np.isfinite(cell['soh']) & (cell['soh'] > 0) & (cell['label_cycle'] > 100)
    return cell['label_cycle'][keep].astype(float), cell['soh'][keep].astype(float)


def shape_parameters(cycle, truth, degree=2):
    """Fit supervision to one TRAIN cell, not an inference-time operation.

    A constrained linear/quadratic fade in absolute future source-cycle units.
    No normalized true lifespan is an input. No clipping to SOH=1 (activation
    can legitimately exceed the initial reference capacity).
    """
    h = (np.asarray(cycle, dtype=float)-100)/100
    y = np.asarray(truth, dtype=float)
    if len(y) < 2 or np.any(h <= 0) or not np.isfinite(y).all():
        raise ValueError('invalid training trajectory')
    basis = np.column_stack([np.ones(len(h)), -h] + ([-h*h] if degree == 2 else []))
    weights = np.sqrt(1+3*h/h.max())
    penalty = np.eye(basis.shape[1])*np.sqrt(.001)
    # Regularize towards a=1, b=c=0 without treating the reference as an input.
    anchor = np.zeros(basis.shape[1]); anchor[0] = 1
    fit = lsq_linear(np.vstack([basis*weights[:, None], penalty]),
        np.r_[y*weights, penalty@anchor], bounds=(np.r_[.2, np.zeros(basis.shape[1]-1)],
        np.r_[1.3, np.full(basis.shape[1]-1, 5.)]), tol=1e-10)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('training shape fit failed')
    return fit.x


def fit_model(cells, ids, seed, parameters):
    ids = list(ids)
    family = cells[ids[0]]['physical_family']
    if any(cells[k]['physical_family'] != family for k in ids):
        raise ValueError('different physical families')
    arm = parameters['arm']
    if arm == 'ridge':
        from battery_smetimes.research_baselines import fit_baseline
        base = fit_baseline(cells, ids, family, parameters['_registry'],
            dict(origins_source_cycles=[100]), 'ridge', dict(alpha=parameters['alpha']), seed)
        return dict(arm=arm, family=family, train_ids=ids, ridge=base)
    if arm == 'compact':
        from scripts.train_stm32_compact_trees import fit
        registry = parameters['_registry']
        args = {k: parameters[k] for k in ('trees', 'depth', 'leaf')}
        return dict(arm=arm, family=family, train_ids=ids, compact=fit(cells, ids, seed, args, registry))
    if arm == 'local_tree':
        global_params = dict(parameters, arm='full_tree')
        global_model = fit_model(cells, ids, seed, global_params)
        groups = {}
        for key in ids:
            groups.setdefault(condition_key(cells[key]), []).append(key)
        local = {key: fit_model(cells, members, seed, global_params)
                 for key, members in groups.items() if len(members) >= 6}
        return dict(arm=arm, family=family, train_ids=ids, global_model=global_model, local=local)
    if arm == 'shape':
        x = np.stack([signal_prefix(cells[k], 100) for k in ids])
        coefficient_targets = np.stack([shape_parameters(*targets(cells[k]), parameters['degree']) for k in ids])
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        z = imputer.fit_transform(x)
        scale = np.maximum(np.std(coefficient_targets, axis=0), .001)
        estimator = ExtraTreesRegressor(n_estimators=128, max_depth=None,
            min_samples_leaf=parameters['leaf'], random_state=seed, n_jobs=2)
        estimator.fit(z, coefficient_targets/scale)
        return dict(arm=arm, family=family, train_ids=ids, parameters=parameters,
            imputer=imputer, model=estimator, target_scale=scale,
            training_shape_parameters=coefficient_targets)
    if arm != 'full_tree':
        raise ValueError('unknown arm')
    x, y, weight, used = build_training_pairs(cells, ids, [100])
    # Extra weight on late observations is computed from TRAIN labels only.
    if parameters.get('tail_weight', 1) != 1:
        pos = 0
        for key in used:
            cycle, _ = targets(cells[key]); n = min(len(cycle), 96)
            factor = np.linspace(1, parameters['tail_weight'], n)
            factor /= factor.mean()
            weight[pos:pos+n] *= factor
            pos += n
        assert pos == len(y)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    z = imputer.fit_transform(x.astype(float))
    estimator = ExtraTreesRegressor(n_estimators=128, max_depth=parameters['depth'],
        min_samples_leaf=parameters['leaf'], random_state=seed, n_jobs=2)
    estimator.fit(z, y, sample_weight=weight)
    return dict(arm=arm, family=family, train_ids=ids, parameters=parameters,
        imputer=imputer, model=estimator, used_train_ids=used)


def predict(bundle, cell, future):
    if cell['physical_family'] != bundle['family']:
        raise ValueError('different battery physical family')
    # query_matrix validates queries and reads no test labels/lifespan.
    prefix = signal_prefix(cell, 100)
    query = query_matrix(prefix, 100, future)
    if bundle['arm'] == 'ridge':
        from battery_smetimes.research_baselines import predict_baseline
        result = predict_baseline(bundle['ridge'], cell, 100, future)
    elif bundle['arm'] == 'compact':
        from scripts.train_stm32_compact_trees import predict as old_predict
        result = old_predict(bundle['compact'], cell, 100, future)
    elif bundle['arm'] == 'local_tree':
        local = bundle['local'].get(condition_key(cell), bundle['global_model'])
        result = predict(local, cell, future)
    elif bundle['arm'] == 'shape':
        coefficients = bundle['model'].predict(bundle['imputer'].transform(prefix[None]))[0]*bundle['target_scale']
        h = (np.asarray(future)-100)/100
        result = coefficients[0]-coefficients[1]*h
        if len(coefficients) == 3:
            result -= coefficients[2]*h*h
    else:
        result = bundle['model'].predict(bundle['imputer'].transform(query.astype(float)))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite forecast')
    return result


def curve_metrics(cycle, y, prediction):
    p = np.asarray(prediction)
    cycle, y = np.asarray(cycle), np.asarray(y)
    late = cycle >= (cycle[0]+cycle[-1])/2
    last = cycle > cycle[-1]-100
    drop, predicted_drop = 100*(y[0]-y[-1]), 100*(p[0]-p[-1])
    result = dict(**regression_metrics(y, p),
        late_mape=float(100*np.mean(abs(p[late]-y[late])/y[late])),
        last100_mape=float(100*np.mean(abs(p[last]-y[last])/y[last])),
        endpoint_pp=float(100*abs(p[-1]-y[-1])),
        drop_error_pp=float(abs(drop-predicted_drop)), truth_drop_pp=float(drop),
        prediction_drop_pp=float(predicted_drop),
        severe_flat_failure=bool(drop >= 5 and predicted_drop < .3*drop),
        future_targets=len(y), future_span=float(cycle[-1]-100),
        short_coverage=bool(len(y) < 10 or cycle[-1]-100 < 100))
    result['joint_pass'] = bool(result['mape_percent'] <= 6 and result['late_mape'] <= 8
        and result['endpoint_pp'] <= 8 and result['drop_error_pp'] <= 5)
    return result


def evaluate(bundle, cells, ids):
    rows, arrays = [], {}
    for key in ids:
        c = cells[key]; cycle, y = targets(c)
        clean = {k: v for k, v in c.items() if k not in ('soh', 'label_cycle', 'unfiltered_soh')}
        p = predict(bundle, clean, cycle)
        poisoned = dict(clean, cycle=np.r_[clean['cycle'], 101, 10000],
            raw=np.vstack([clean['raw'], np.full((2, clean['raw'].shape[1]), np.inf)]),
            cycle_life=np.inf, future_soh=np.inf)
        np.testing.assert_allclose(p, predict(bundle, poisoned, cycle), atol=1e-12, rtol=0)
        bad = dict(clean, physical_family='different_unregistered_family')
        try:
            predict(bundle, bad, cycle)
        except ValueError:
            pass
        else:
            raise AssertionError('family isolation failed')
        rows.append(dict(cell_id=key, **curve_metrics(cycle, y, p)))
        arrays.update({key+'__100__'+f: a for f, a in [('cycle', cycle), ('truth', y), ('prediction', p)]})
    return rows, arrays


def selection_score(rows):
    scores = [(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]
    return float(np.mean(scores)+.25*np.quantile(scores, .9))
