"""Apply new curve metrics to the frozen NCA baseline; no model selection."""
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes.curve_fidelity import curve_metrics
from scripts.finalize_curve_fidelity import summarize
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    run=Path('experiments/zhu2022_nca_forecast/20261004T161453_449950Z')
    config=json.loads((run/'config.json').read_text())
    prepared=Path(config['prepared_directory'])/'prepared_cells.joblib'
    assert sha(prepared)==config['prepared_sha256']
    cells=joblib.load(prepared)
    out=Path('experiments/nca_curve_goal_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    results={}
    for model_name in ('extra_trees','ridge'):
        rows=[]
        for plan in config['plans']:
            assert not set(plan['train']+plan['validation'])&set(plan['test'])
            for seed in config['seeds']:
                source=run/model_name/plan['name']/str(seed)/'predictions.npz'
                with np.load(source) as a:
                    for key in plan['test']:
                        x,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')]
                        np.testing.assert_array_equal(x,cells[key]['cycle'])
                        np.testing.assert_array_equal(y,cells[key]['truth'])
                        rows.append(dict(cell_id=key,seed=seed,fold=plan['name'],
                            prediction_sha256=sha(source),**curve_metrics(x,y,p)))
        summary,by_cell=summarize(rows)
        gates=dict(full_mape=summary['mape_percent']['mean']<=6,
            late_mape=summary['late_mape']['mean']<=6,rmse=summary['rmse_soh']['mean']<=.04,
            endpoint_mean=summary['endpoint_pp']['mean']<=5,drop_mean=summary['drop_error_pp']['mean']<=4,
            joint_fraction=summary['joint_pass_fraction']>=.9,severe_fraction=summary['severe_flat_fraction']<=.1)
        results[model_name]=dict(summary=summary,absolute_gates=gates,
            failed_cells=[r['cell_id'] for r in by_cell if not r['joint_pass']],
            note='Frozen baseline diagnostic, not new training; relative P90 improvement not evaluated.')
        save_json(out/(model_name+'_by_cell.json'),rows)
        save_json(out/(model_name+'_cell_seed_means.json'),by_cell)
        print(model_name,json.dumps(results[model_name],ensure_ascii=False))
    save_json(out/'audit.json',dict(results=results,baseline_run=str(run.resolve()),
        baseline_config_sha256=sha(run/'config.json'),prepared_sha256=sha(prepared),
        source_sha256=sha(__file__),goal_complete=False,new_training=False))
    print('OUTPUT='+str(out.resolve()))


if __name__=='__main__':
    main()
