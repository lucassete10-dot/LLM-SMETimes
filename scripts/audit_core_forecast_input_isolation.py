"""All193 prepared cells and all965 frozen tree test forecasts: input isolation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from battery_smetimes.external_soh import load_context
from battery_smetimes.research_baselines import signal_prefix
from scripts.train_stm32_compact_trees import predict
from scripts.prepare_same_type_forecast import save_json, sha

RUN = Path('experiments/stm32_compact_trees/runs/20261004T074429_087457Z')
RELEASE = Path('deployment/stm32_release_trees/20261004T083706_476897Z')
WAVE = Path('experiments/stm32_waveform_replay/20261004T082559_705701Z')


def main():
    out = Path('experiments/core_input_isolation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/Path(__file__).name)
    rows = []
    try:
        context = load_context(RELEASE)
        cells = context['cells']
        cfg = json.loads((RUN/'config.json').read_text())
        plans = json.loads((RUN/'splits.json').read_text())['plans']
        assert plans == json.loads((RELEASE/'frozen_audit/splits.json').read_text())['plans']
        wave = json.loads((WAVE/'report.json').read_text())
        assert wave['status'] == 'passed' and wave['streaming'] and wave['all_cells']
        assert {r['cell_id'] for r in wave['reports']} == set(cells)
        prefix_checks = []
        prepared = {}
        for key, cell in sorted(cells.items()):
            assert np.all(cell['cycle'] <= 100)
            file = WAVE/(key+'.npz')
            assert sha(file) == wave['files_sha256'][file.name]
            with np.load(file) as arrays:
                raw = dict(zip(arrays['cycle'].tolist(), arrays['raw_c']))
                assert max(raw) <= 100
                for n, row in zip(cell['cycle'], cell['raw']):
                    np.testing.assert_allclose(raw[int(n)], row, rtol=1e-6, atol=1e-5, equal_nan=True)
            clean = {k: v for k, v in cell.items() if k not in ('soh', 'label_cycle', 'unfiltered_soh')}
            poison = dict(clean, cycle=np.r_[clean['cycle'], 101, 200, 10000],
                          raw=np.vstack([clean['raw'], np.full((3, clean['raw'].shape[1]), np.inf)]),
                          capacity=np.inf, cycle_life=np.inf, rul=np.inf)
            poison['metadata'] = dict(clean['metadata'], cycle_life=np.inf, future_soh=np.inf)
            a = signal_prefix(cell, 100)
            np.testing.assert_array_equal(a, signal_prefix(clean, 100))
            np.testing.assert_array_equal(a, signal_prefix(poison, 100))
            prepared[key] = (clean, poison)
            prefix_checks.append(dict(cell_id=key, dataset=cell['metadata']['dataset'],
                source_cycles=len(cell['cycle']), raw_cache_sha256=sha(file),
                early_raw_replay_matches=True, label_removal_prefix_exact=True,
                future_and_lifetime_poison_prefix_exact=True))
        for plan in plans:
            allowed = set(plan['train']) | set(plan['validation'])
            assert not allowed & set(plan['test'])
            for seed in cfg['seeds']:
                folder = RUN/plan['name']/str(seed)
                model = joblib.load(folder/'model.joblib')
                assert set(model['train_ids']) == allowed
                with np.load(folder/'test/predictions.npz') as arrays:
                    for key in plan['test']:
                        x = arrays[key+'__100__cycle']
                        p = arrays[key+'__100__prediction']
                        clean, poison = prepared[key]
                        for variant in (clean, poison):
                            np.testing.assert_allclose(p, predict(model, variant, 100, x), rtol=0, atol=1e-12)
                        try:
                            predict(model, dict(clean, physical_family='unregistered_mismatched_family'), 100, x)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError('Mismatched physical family accepted')
                        rows.append(dict(cell_id=key, dataset=plan['dataset'], fold=plan['name'],
                            seed=seed, future_queries=len(x), model_sha256=sha(folder/'model.joblib'),
                            label_removal_passed=True, future_and_lifetime_poison_passed=True,
                            mismatched_family_rejected=True))
            print('PASS '+plan['name'], flush=True)
        assert len(prefix_checks) == 193 and len(rows) == 965
        save_json(out/'verification.json', dict(passed=True, all193_prepared_inputs=prefix_checks,
            all965_cell_seed_test_forecasts=rows,
            waveform_source_report_sha256=sha(WAVE/'report.json'), waveform_run=str(WAVE.resolve()),
            frozen_splits_sha256=sha(RUN/'splits.json'), run=str(RUN.resolve()),
            source_sha256=sha(__file__),
            caution='Cached raw replay confirms declared prefixes; waveform reference is numeric input evidence, not new-cell accuracy. Capacity labels remain used solely for evaluation.'))
        print('OUTPUT='+str(out.resolve()), flush=True)
    except BaseException:
        save_json(out/'failure.json', dict(traceback=traceback.format_exc(), checked_test_rows=len(rows)))
        raise


if __name__ == '__main__':
    main()
