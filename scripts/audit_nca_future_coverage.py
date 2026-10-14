"""Describe saved forecast coverage without excluding cells or selecting models."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
from scripts.prepare_same_type_forecast import save_json, sha

RUN = Path('experiments/zhu2022_nca_forecast/20261004T161453_449950Z')


def coverage(x):
    x = np.asarray(x)
    if x.ndim != 1 or not len(x) or not np.all(np.diff(x) > 0) or np.any(x <= 100):
        raise ValueError('Expected increasing, nonempty future source cycles >100')
    return dict(target_count=int(len(x)), first_source_cycle=int(x[0]),
                last_source_cycle=int(x[-1]),
                maximum_horizon_from_origin=int(x[-1] - 100),
                observed_source_span=int(x[-1] - x[0] + 1))


def main():
    cfg = json.loads((RUN/'config.json').read_text(encoding='utf-8'))
    rows = []
    hashes = {}
    for plan in cfg['plans']:
        first = {}
        for model in cfg['candidates']:
            for seed in cfg['seeds']:
                path = RUN/model/plan['name']/str(seed)/'predictions.npz'
                hashes[str(path)] = sha(path)
                with np.load(path) as arrays:
                    for key in plan['test']:
                        x = arrays[key+'__100__cycle']
                        if key in first:
                            np.testing.assert_array_equal(first[key], x)
                        else:
                            first[key] = x.copy()
        rows.extend(dict(cell_id=key, fold=plan['name'], **coverage(x))
                    for key, x in first.items())
    assert len(rows) == 55 and len({r['cell_id'] for r in rows}) == 55
    rows.sort(key=lambda r: r['cell_id'])
    short = [r for r in rows if r['maximum_horizon_from_origin'] < 100]
    summary = dict(cells=len(rows), role='Post-evaluation descriptive coverage only',
                   all_cells_retained=True, unchanged_predictions=True,
                   exclusion_or_model_selection=False, short_horizon_cells=short,
                   caution='Small MAPE over8 or9 future points is not long-horizon evidence.')
    for field in ('target_count', 'maximum_horizon_from_origin', 'observed_source_span'):
        values = [r[field] for r in rows]
        summary[field] = dict(min=min(values), median=float(np.median(values)), max=max(values))
    summary['horizon_groups'] = {
        'under100': len(short),
        '100to299': sum(100 <= r['maximum_horizon_from_origin'] < 300 for r in rows),
        '300or_more': sum(r['maximum_horizon_from_origin'] >= 300 for r in rows)}
    out = Path('experiments/nca_future_coverage')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/Path(__file__).name)
    save_json(out/'by_cell.json', rows)
    save_json(out/'summary.json', summary)
    save_json(out/'config.json', dict(run=str(RUN.resolve()), source_sha256=sha(__file__),
                                    prediction_sha256=hashes, origin=100))
    print(json.dumps(dict(output=str(out.resolve()), summary=summary), ensure_ascii=False))


if __name__ == '__main__':
    main()
