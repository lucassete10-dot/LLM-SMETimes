"""TRAIN/VAL scale, origin and rate-node diagnosis of preserved R0 models."""
import argparse
import copy
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from battery_smetimes.research_llm import load_adapter, cards_embedding
from battery_smetimes.physical_types import load_registry
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3
from scripts.train_llm_curve_v3 import create_network
from scripts.audit_llm_curve_repair import load_frozen, fingerprint
from scripts.prepare_same_type_forecast import save_json


def affine_rms_pp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = (x - x[0]) / max(float(x[-1] - x[0]), 1.)
    fitted = np.column_stack((np.ones(len(z)), z)) @ np.linalg.lstsq(
        np.column_stack((np.ones(len(z)), z)), y, rcond=None)[0]
    return float(100 * np.sqrt(np.mean((y - fitted) ** 2)))


def poisoned_visible_cell(cell, origin):
    contaminated = copy.deepcopy(cell)
    for field in ('soh', 'label_cycle', 'capacity', 'eol', 'lifetime', 'unfiltered_soh'):
        contaminated.pop(field, None)
    # External caches can retain all source cycles. Keep an ordered prefix,
    # then append an unseen row; appending into a full cache is not ordered.
    keep = contaminated['cycle'] <= origin
    contaminated['raw'] = np.r_[contaminated['raw'][keep], np.full((1, cell['raw'].shape[1]), np.inf)]
    contaminated['cycle'] = np.r_[contaminated['cycle'][keep], origin+100]
    return contaminated


def diagnose(run, out):
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    _, cells, plan, registry_path, _ = load_frozen(cfg)
    processor = joblib.load(run/'preprocessor.joblib')
    # Refit independently using TRAIN; no validation or test statistics.
    from battery_smetimes.research_sequences import HistoryPreprocessor
    refit = HistoryPreprocessor().fit(cells, cfg['train_ids'], [cfg['origin']], history_slots=cfg['history_slots'])
    for name in ('raw_scale', 'stats_scale', 'query_scale'):
        old, fresh = getattr(processor, name), getattr(refit, name)
        for field in ('mean_', 'scale_'):
            np.testing.assert_allclose(getattr(old.scaler, field), getattr(fresh.scaler, field), rtol=0, atol=1e-10)
        np.testing.assert_allclose(old.imputer.statistics_, fresh.imputer.statistics_, rtol=0, atol=1e-10)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    network = create_network(cfg, processor, cfg['architecture'], cfg['dropout'])
    load_adapter(network, torch.load(run/'model.pt', map_location='cpu', weights_only=True))
    network.eval()
    bundle = dict(network=network, processor=processor, device='cuda', registry=load_registry(registry_path),
                  physical_family=cells[cfg['train_ids'][0]]['physical_family'])
    rows = {}
    for role, ids in (('training', cfg['train_ids']), ('validation', plan['validation'])):
        records = {}
        with np.load(run/(role+'_audited_predictions.npz'), allow_pickle=False) as data:
            for cid in ids:
                cell = cells[cid]
                values = processor.model_input(cell, cfg['origin'])
                contaminated = poisoned_visible_cell(cell, cfg['origin'])
                isolated = processor.model_input(contaminated, cfg['origin'])
                for key in values:
                    np.testing.assert_array_equal(values[key], isolated[key])
                prefix = f'{cid}__{cfg["origin"]}__'
                x, y, p = [data[prefix+k] for k in ('cycle', 'truth', 'prediction')]
                metrics = curve_metrics_v3(x, y, p)
                count, width = values['length'], processor.raw_width
                available = values['history'][:count, width:2*width] > .5
                scaled = values['history'][:count, :width]
                raw = cell['raw'][cell['cycle'] <= cfg['origin']]
                with torch.inference_mode():
                    cards = cards_embedding(bundle, cell, cfg['origin'])[None].cuda()
                    encoding = network.encode(torch.as_tensor(values['history'][None], device='cuda'),
                                              torch.as_tensor([count], device='cuda'), cards)
                    condition = torch.as_tensor(values['conditions'][None], device='cuda')
                    params = network.trajectory_parameters(encoding, condition)
                    rates = params[1][0].cpu().numpy().astype(float)
                    local = network.local_parameters(encoding, condition)[0].cpu().numpy().astype(float)
                knots = network.rate_knots.detach().cpu().numpy()
                horizon = float((x[-1]-cfg['origin'])/100)
                # Include bracketing node for the actual observed horizon.
                end = min(len(knots), int(np.searchsorted(knots, horizon, side='right'))+1)
                active_rates = rates[:max(2, end)]
                local_end = min(len(local), int(np.ceil((x[-1]-cfg['origin'])/cfg['local_recovery_spacing_cycles']))+1)
                local_values = local[1:local_end]
                records[cid] = {
                    **metrics, 'visible_observations':int(count),
                    'missing_by_feature_fraction':np.mean(~np.isfinite(raw), axis=0).tolist(),
                    'observed_scaled_abs_gt5_fraction':float(np.mean(np.abs(scaled[available])>5)) if available.any() else None,
                    'conditions_scaled_max_abs':float(np.max(abs(values['conditions']))),
                    'truth_affine_rms_pp':affine_rms_pp(x, y), 'prediction_affine_rms_pp':affine_rms_pp(x, p),
                    'origin_prediction_soh':float(params[0][0].cpu()),
                    'first_future_truth_soh':float(y[0]),
                    'rates_soh_per_100_cycles':rates.tolist(), 'rate_cv_in_observed_span':float(np.std(active_rates)/max(np.mean(active_rates),1e-12)),
                    'rate_near_zero_fraction':float(np.mean(active_rates<1e-5)),
                    'local_bound_saturation_fraction':float(np.mean(abs(local_values)>=.95*cfg['local_recovery_max_soh'])) if len(local_values) else None,
                    'input_label_future_poison_passed':True,
                }
        rows[role] = records
    result = {'dataset':cfg['dataset'], 'fold':cfg['fold'], 'seed':cfg['seed'], 'run':str(run.resolve()),
              'roles':['training', 'validation'], 'test_used':False,
              'train_scalers_refit_equal':True, 'mit_current_units':cfg.get('mit_current_units'),
              'model_sha256':fingerprint(run/'model.pt'), 'by_role':rows,
              'predictions_sha256':{r:fingerprint(run/(r+'_audited_predictions.npz')) for r in rows},
              'diagnostic_source_sha256':fingerprint(Path(__file__)),
              'diagnostic_rules':{'shape_miss':'truth affine RMS>0.3pp AND prediction RMS<0.1pp',
                                  'start_bias':'first future absolute error>1pp; origin truth not used as inference anchor',
                                  'tail_miss':'late MAPE>3%; real linear truth not penalized'},
              'summary':{}}
    for role, records in rows.items():
        rr = list(records.values())
        result['summary'][role] = {
            'cells':len(rr), 'start_error_gt1pp':sum(r['start_error_pp']>1 for r in rr),
            'late_mape_gt3percent':sum(r['late_mape_percent']>3 for r in rr),
            'shape_miss_count':sum(r['truth_affine_rms_pp']>.3 and r['prediction_affine_rms_pp']<.1 for r in rr),
            'mean_rate_cv':float(np.mean([r['rate_cv_in_observed_span'] for r in rr])),
            'mean_local_saturation_fraction':float(np.mean([r['local_bound_saturation_fraction'] for r in rr if r['local_bound_saturation_fraction'] is not None])),
            'mean_mape_percent':float(np.mean([r['mape_percent'] for r in rr])),
        }
    save_json(out/'diagnosis.json', result)
    del bundle, network, encoding, cards, params
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = diagnose(args.run, args.out)
    print(json.dumps({'dataset':result['dataset'], 'summary':result['summary'], 'test_used':False}), flush=True)


if __name__ == '__main__': main()
