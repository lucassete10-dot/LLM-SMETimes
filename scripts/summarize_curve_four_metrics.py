"""Companion four-metric macro table; preserves original experiment files."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run
    assert json.loads((root/'progress.json').read_text())['training_complete']
    rows=json.loads((root/'by_cell.json').read_text())
    seeds=sorted({r['seed'] for r in rows})
    keys=sorted({r['cell_id'] for r in rows})
    for key in keys:
        selected=[r for r in rows if r['cell_id']==key]
        assert len(selected)==len(seeds) and {r['seed'] for r in selected}==set(seeds)
    fields=('mae_soh','rmse_soh','mape_percent','r2')
    report={}
    for field in fields:
        seed_means=[]
        for seed in seeds:
            valid=[r[field] for r in rows if r['seed']==seed and r[field] is not None and np.isfinite(r[field])]
            seed_means.append(float(np.mean(valid)) if valid else None)
        finite=[v for v in seed_means if v is not None]
        report[field]=dict(mean=float(np.mean(finite)) if finite else None,
            seed_sd=float(np.std(finite,ddof=1)) if len(finite)>1 else None,
            valid_cell_seed_rows=int(sum(r[field] is not None and np.isfinite(r[field]) for r in rows)),
            seed_means=seed_means)
    output=dict(metrics=report,cells=len(keys),seeds=seeds,rows=len(rows),
        r2_negative_rows=sum(r['r2'] is not None and r['r2']<0 for r in rows),
        note='Cell-equal macro metrics, sample standard deviation across seeds, not confidence intervals. Undefined R2 omitted with explicit counts; raw cell metrics retained.',
        source_rows_sha256=sha(root/'by_cell.json'),source_sha256=sha(__file__))
    save_json(root/'four_metric_summary.json',output)
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':
    main()
