"""Inventory the current goal's valid and failed experiments without retraining.

This checks archive completeness/hash integrity, not scientific accuracy; the
independent model/input/curve audits remain separate evidence.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.prepare_same_type_forecast import save_json, sha

RUNS = [
    ('core_trees', 'experiments/stm32_compact_trees/runs/20261004T074429_087457Z', 280),
    ('nasa_ridge', 'experiments/nasa_signal_trajectory/ridge/20261004T165029_391663Z', 80),
    ('nasa_signal_failed', 'experiments/nasa_signal_trajectory/signal_trajectory/20261004T171958_017920Z', 140),
    ('nasa_bounded_failed', 'experiments/nasa_bounded_prefix/20261004T172001_284366Z', 200),
    ('nasa_interaction_inner_only', 'experiments/nasa_interaction_ridge/20261004T172145_185139Z', 60),
    ('nca', 'experiments/zhu2022_nca_forecast/20261004T161453_449950Z', 200),
    ('oxford_supplement', 'experiments/oxford_sparse_diagnostic/20261004T145450_603603Z', 100),
]


def main():
    out = Path('experiments/public_soh_archive_audit') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    records = []
    for name, directory, expected in RUNS:
        root = Path(directory)
        cfg = json.loads((root/'config.json').read_text())
        assert len(cfg['seeds']) >= 5
        assert (root/'training.log').stat().st_size > 0
        sources = cfg['source_sha256']
        if isinstance(sources, str):
            assert any(sha(p) == sources for p in root.glob('train_*.py'))
        else:
            for original, digest in sources.items():
                matches = list((root/'source_snapshot').rglob(Path(original.replace('\\', '/')).name))
                assert any(sha(p) == digest for p in matches), original
        for filename, digest in cfg.get('dependency_sha256', {}).items():
            assert sha(root/'source_snapshot'/filename) == digest
        weights = list(root.rglob('model.joblib'))
        assert len(weights) == expected, (name, len(weights), expected)
        for weight in weights:
            folder = weight.parent
            candidate = folder.name.startswith('candidate_')
            prediction = folder/('validation_predictions.npz' if candidate else 'predictions.npz')
            if not prediction.exists() and not candidate:
                prediction = folder/'test'/'predictions.npz'
            assert prediction.stat().st_size > 0, str(prediction)
            metrics = folder/('validation_metrics.json' if candidate else 'metrics.json')
            if not metrics.exists() and not candidate:
                metrics = folder/'test'/'metrics.json'
            assert metrics.stat().st_size > 0, str(metrics)
            json.loads(metrics.read_text())
        selections = list(root.rglob('selection.json'))
        units = len(cfg.get('plans', []))*len(cfg['seeds'])
        if name == 'core_trees':
            units = 70
        if name == 'nca':
            units *= 2
        assert len(selections) == units, (name, len(selections), units)
        if name == 'nca':
            prep = Path(cfg['prepared_directory'])
            assert sha(prep/'prepared_cells.joblib') == cfg['prepared_sha256']
            assert sha(prep/'audit.json') == cfg['audit_sha256']
        # Preserve original short historical logs; do not pretend they were rich
        # per-epoch traces. Later reproduction logs are separately identifiable.
        files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
        records.append(dict(name=name, run=str(root.resolve()), weights=len(weights),
            units=units, training_log_bytes=(root/'training.log').stat().st_size,
            files_sha256=files, validation_only=cfg.get('validation_only', False)))
        print(f'PASS {name}: {len(weights)} weights, {len(files)} archived files', flush=True)
    assert sum(r['weights'] for r in records) == 1060
    save_json(out/'verification.json', dict(passed=True, scope='current public SOH goal only',
        total_weights=1060, runs=records, source_sha256=sha(__file__),
        caution='Archive integrity is not an independent accuracy test. Original historical log/source gaps remain disclosed; instrumentation repeats are not independent experiments.'))
    print('OUTPUT='+str(out.resolve()), flush=True)


if __name__ == '__main__':
    main()
