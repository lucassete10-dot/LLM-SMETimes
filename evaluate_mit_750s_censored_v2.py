"""MIT 750 s prefix ridge with source-endpoint right-censor constraints.

An event cell supplies log(source EOL). A censored cell supplies only the
lower bound log(last observed cycle); it is never assigned a fabricated EOL.
The same fixed five observed-prefix features and whole-batch folds are used.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from audit_mit_exact80_rul_v2 import score
from battery_v2_features import load_manifest
from evaluate_mit_750s_multicycle_v2 import ALPHA, ORIGIN, extract_features
from train_soh_v2 import outer_splits


ROOT = Path('battery_models_v2/mit_qv_challenger')
BASE = ROOT / 'prefix_750s_multicycle_cv.json'
TARGETS = Path('battery_models_v2/exact80_target_audit.json')
OUT = ROOT / 'prefix_750s_censored_cv.json'


def fit_censored_ridge(event_x, event_life, censored_x, censor_cycle,
                       alpha=ALPHA):
    event_x = np.asarray(event_x, dtype=float)
    censored_x = np.asarray(censored_x, dtype=float)
    if event_x.ndim != 2 or censored_x.ndim != 2 or event_x.shape[1] != censored_x.shape[1]:
        raise ValueError('unaligned feature matrices')
    scaler = StandardScaler().fit(event_x)
    x_event = np.column_stack((np.ones(len(event_x)), scaler.transform(event_x)))
    x_cens = np.column_stack((np.ones(len(censored_x)), scaler.transform(censored_x)))
    y_event = np.log(np.asarray(event_life, dtype=float))
    y_cens = np.log(np.asarray(censor_cycle, dtype=float))
    if np.any(~np.isfinite(y_event)) or np.any(~np.isfinite(y_cens)):
        raise ValueError('nonfinite life labels')
    base = Ridge(alpha=alpha).fit(scaler.transform(event_x), y_event)
    initial = np.r_[base.intercept_, base.coef_]

    def objective(coef):
        error = x_event@coef-y_event
        violation = np.maximum(0., y_cens-x_cens@coef)
        loss = .5*(error@error+violation@violation+alpha*(coef[1:]@coef[1:]))
        gradient = x_event.T@error-x_cens.T@violation
        gradient[1:] += alpha*coef[1:]
        return float(loss), gradient

    fitted = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      options={'maxiter': 1000, 'ftol': 1e-12})
    if not fitted.success:
        raise RuntimeError(f'censored ridge fit failed: {fitted.message}')
    return scaler, fitted.x, {
        'training_event_cells': len(event_x),
        'training_censored_cells': len(censored_x),
        'censored_constraints_violated_after_fit': int(np.sum(
            y_cens-x_cens@fitted.x > 1e-8)),
    }


def predict_endpoint(bundle, features):
    scaler, coefficients, _ = bundle
    x = np.r_[1., scaler.transform(np.asarray(features)[None, :])[0]]
    return float(np.clip(np.exp(x@coefficients), ORIGIN+1, 2500))


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    by_id = {row['cell_id']: row for row in records}
    features = extract_features(records)
    baseline = {row['cell_id']: row for fold in json.loads(
        BASE.read_text(encoding='utf-8'))['folds'] for row in fold['by_cell']}
    events = {row['cell_id']: row['first_confirmed_exact80_cycle']
              for row in json.loads(TARGETS.read_text(encoding='utf-8'))[
                  'domains']['MIT']['by_cell']}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        observed = [key for key in train if key in features and
                    by_id[key]['endpoint_cycle'] is not None]
        censored = [key for key in train if key in features and
                    by_id[key]['endpoint_cycle'] is None]
        bundle = fit_censored_ridge(
            np.stack([features[key] for key in observed]),
            [by_id[key]['endpoint_cycle'] for key in observed],
            np.stack([features[key] for key in censored]),
            [by_id[key]['censor_cycle'] for key in censored])
        rows = []
        for key in validation:
            if key not in features or key not in baseline:
                continue
            actual = int(by_id[key]['endpoint_cycle'])
            predicted = predict_endpoint(bundle, features[key])
            before = baseline[key]['predicted_source_endpoint_cycle']
            event = events[key]
            rows.append({
                'cell_id': key, 'held_out_batch': group,
                'true_source_endpoint_cycle': actual,
                'baseline_predicted_source_endpoint_cycle': before,
                'predicted_source_endpoint_cycle': predicted,
                'baseline_source_ape_percent': float(100*abs(before-actual)/actual),
                'censored_source_ape_percent': float(100*abs(predicted-actual)/actual),
                'baseline_exact80_proxy_rul_ape_percent': score(event, before, ORIGIN)
                if event is not None and event > ORIGIN else None,
                'censored_exact80_proxy_rul_ape_percent': score(event, predicted, ORIGIN)
                if event is not None and event > ORIGIN else None,
            })
        folds.append({
            'held_out_batch': group, 'training': bundle[2], 'scored_cells': len(rows),
            'baseline_source_mape_percent': float(np.mean([
                row['baseline_source_ape_percent'] for row in rows])),
            'censored_source_mape_percent': float(np.mean([
                row['censored_source_ape_percent'] for row in rows])),
            'by_cell': rows,
        })
        print(group, 'censored train', len(censored),
              'baseline', folds[-1]['baseline_source_mape_percent'],
              'censored', folds[-1]['censored_source_mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    exact_rows = [row for row in rows if row['censored_exact80_proxy_rul_ape_percent']
                  is not None]
    report = {
        'status': 'unpromoted MIT whole-batch censored-prefix development ablation',
        'input': 'five cycle-10/50/100 observed 750 s Q(V) features',
        'target': 'source 0.88 Ah EOL events with source right-censor lower bounds',
        'exact80_warning': 'source EOL point prediction used unchanged as exact80 proxy; not coherent curve crossing',
        'source_scored_cells': len(rows),
        'baseline_source_mape_percent': float(np.mean([
            row['baseline_source_ape_percent'] for row in rows])),
        'censored_source_mape_percent': float(np.mean([
            row['censored_source_ape_percent'] for row in rows])),
        'exact80_scored_cells': len(exact_rows),
        'baseline_exact80_proxy_rul_mape_percent': float(np.mean([
            row['baseline_exact80_proxy_rul_ape_percent'] for row in exact_rows]))
        if exact_rows else None,
        'censored_exact80_proxy_rul_mape_percent': float(np.mean([
            row['censored_exact80_proxy_rul_ape_percent'] for row in exact_rows]))
        if exact_rows else None,
        'folds': folds,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', {key: value for key, value in report.items()
                      if key.endswith('mape_percent') or key.endswith('cells')})
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
