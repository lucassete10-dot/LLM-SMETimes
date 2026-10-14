"""Supplementary four-metric tables and failure counts; never selects models."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    assert json.loads((root/'independent_verification.json').read_text())['passed']
    rows=[r for r in json.loads((root/'comprehensive_curve_metrics.json').read_text())['rows'] if r['role']=='test'];assert len(rows)==275
    seeds=sorted({r['seed'] for r in rows});assert len(seeds)==5
    fields=('mae_soh','rmse_soh','mape_percent','r2','late_mape','endpoint_pp','drop_error_pp','start_error_pp')
    table=[]
    for key in sorted({r['cell_id'] for r in rows}):
        rr=[r for r in rows if r['cell_id']==key];assert len(rr)==5 and {r['seed'] for r in rr}==set(seeds)
        stats={}
        for f in fields:
            values=[r[f] for r in rr if r[f] is not None]
            stats[f]=dict(mean=float(np.mean(values)) if values else None,seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,defined_seeds=len(values))
        table.append(dict(cell_id=key,metrics=stats,future_points=rr[0]['points'],future_span=rr[0]['future_observation_span'],short_coverage=rr[0]['short_coverage']))
    failures=json.loads((root/'failed_cells.json').read_text());thresholds={'mape_percent':6,'late_mape':8,'endpoint_pp':8,'drop_error_pp':5}
    counts={f:sum(r[f]>t for r in failures) for f,t in thresholds.items()}
    save_json(root/'four_metric_by_cell.json',dict(cells=table,seeds=seeds,meaning='mean and sample SD across five random seeds; not confidence intervals',units='SOH 0-1; mape percent, endpoint/drop/start SOH pp',source_sha256=sha(__file__)))
    save_json(root/'failure_field_counts.json',dict(failed_cells=len(failures),overlapping_failure_counts=counts,diagnostics_only=True,no_new_exclusions=True,no_model_selection=True,source_sha256=sha(__file__)))
    print(json.dumps(dict(cells=len(table),failed_cells=len(failures),field_counts=counts)),flush=True)

if __name__=='__main__':main()
