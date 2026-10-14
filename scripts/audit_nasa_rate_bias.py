"""Evaluation-only offset versus curve-shape error decomposition; no calibration."""
import json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    root=Path('experiments/nasa_rate_formal/20261005T085515_209007Z');cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'independent_verification.json').read_text())['passed'];rows=[]
    for plan in cfg['plans']:
        for seed in cfg['seeds']:
            file=root/plan['name']/str(seed)/'outer'/'test_predictions.npz'
            with np.load(file) as a:
                for key in plan['test']:
                    y,p=[a[key+'__100__'+f] for f in ('truth','prediction')];e=p-y;bias=float(e.mean());variance=float(e.var());mse=float((e*e).mean())
                    np.testing.assert_allclose(mse,bias*bias+variance,atol=1e-14,rtol=0)
                    rows.append(dict(cell_id=key,seed=seed,first_signed_error_pp=float(100*e[0]),last_signed_error_pp=float(100*e[-1]),mean_signed_error_pp=100*bias,error_sd_pp=float(100*np.sqrt(variance)),bias_fraction_of_mse=bias*bias/mse if mse>0 else 0.,source_file=str(file.resolve()),source_sha256=sha(file)))
    fields=('first_signed_error_pp','last_signed_error_pp','mean_signed_error_pp','error_sd_pp','bias_fraction_of_mse')
    cells=[dict(cell_id=key,**{f:float(np.mean([r[f] for r in rows if r['cell_id']==key])) for f in fields}) for key in sorted({r['cell_id'] for r in rows})]
    save_json(root/'offset_shape_error_diagnostic.json',dict(rows=rows,cells=cells,source_sha256=sha(__file__),interpretation='Evaluation decomposition only. No query-label-derived correction is deployed or claimed as an improved forecast. Do not tune to outer residuals.',goal_complete=False))
    print(json.dumps(cells),flush=True)


if __name__=='__main__':main()
