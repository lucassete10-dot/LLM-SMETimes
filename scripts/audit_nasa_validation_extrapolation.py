"""Inspect frozen inner-validation standardized inputs; no outer evaluation."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from battery_smetimes.external_soh import load_context
from battery_smetimes.research_baselines import signal_prefix
from battery_smetimes.forecast import query_matrix
from scripts.prepare_same_type_forecast import save_json,sha

RUN=Path('experiments/nasa_signal_trajectory/ridge/20261004T143636_110096Z')

def main():
    out=Path('experiments/nasa_validation_extrapolation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    cfg=json.loads((RUN/'config.json').read_text());cells=load_context(Path(cfg['reference_release']))['cells'];rows=[]
    for plan in cfg['plans']:
        folder=RUN/plan['name']/str(cfg['seeds'][0]);selection=json.loads((folder/'selection.json').read_text())
        selected=selection['candidates'].index(selection['selected'])
        bundle=joblib.load(folder/f'candidate_{selected}'/'model.joblib')
        assert bundle['train_ids']==plan['train']
        for key in plan['validation']:
            assert key not in bundle['train_ids'] and key not in plan['test']
            c=cells[key];future=c['label_cycle'][np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)]
            x=query_matrix(signal_prefix(c,100),100,future)
            z=bundle['scaler'].transform(bundle['imputer'].transform(x.astype(float)))
            extreme=np.max(np.abs(z[:,:-3]),axis=0)
            cols=np.argsort(extreme)[-10:][::-1]
            rows.append(dict(fold=plan['name'],validation_cell=key,train_ids=plan['train'],
                selected_alpha=selection['selected']['parameters']['alpha'],
                validation_mape=selection['selected']['validation_mape'],
                prefix_columns_over_abs_z6=int(np.sum(extreme>6)),prefix_columns_over_abs_z3=int(np.sum(extreme>3)),
                maximum_abs_prefix_z=float(extreme.max()),
                strongest_columns=[dict(column=int(i),abs_z=float(extreme[i]),
                    train_mean=float(bundle['scaler'].mean_[i]),train_scale=float(bundle['scaler'].scale_[i])) for i in cols]))
    save_json(out/'audit.json',dict(complete=True,outer_test_inputs_examined=False,
        run=str(RUN.resolve()),config_sha256=sha(RUN/'config.json'),rows=rows))
    print('OUTPUT='+str(out.resolve()),flush=True)
    for row in rows:print(row['fold'],row['validation_cell'],'max_prefix_z',row['maximum_abs_prefix_z'],flush=True)

if __name__=='__main__':main()
