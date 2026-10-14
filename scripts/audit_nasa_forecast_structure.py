"""Read-only diagnosis on development inputs, no outer-score model selection."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from battery_smetimes.external_soh import load_context
from battery_smetimes.research_baselines import predict_baseline
from scripts.prepare_same_type_forecast import save_json,sha

RUN=Path('experiments/nasa_signal_trajectory/ridge/20261004T143636_110096Z')

def main():
    out=Path('experiments/nasa_structure_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    cfg=json.loads((RUN/'config.json').read_text());cells=load_context(Path(cfg['reference_release']))['cells']
    rows=[];future=np.array([101,110,120,130],dtype=float)
    for plan in cfg['plans']:
        bundle=joblib.load(RUN/plan['name']/str(cfg['seeds'][0])/'model.joblib')
        development=plan['train']+plan['validation'];a,b=development[:2]
        assert a not in plan['test'] and b not in plan['test']
        pa=predict_baseline(bundle,cells[a],100,future);pb=predict_baseline(bundle,cells[b],100,future)
        shift=pa-pb
        spread=float(np.ptp(shift))
        assert spread<1e-10
        cutoffs=[cells[k]['metadata']['discharge_cutoff_v'] for k in development]
        rows.append(dict(fold=plan['name'],development_comparison=[a,b],requested_cycles=future.tolist(),
            prediction_difference=shift.tolist(),difference_range=spread,
            training_cutoff_v_range=[min(cutoffs),max(cutoffs)],
            test_cutoff_v=[cells[k]['metadata']['discharge_cutoff_v'] for k in plan['test']],
            additive_ridge=True,no_signal_horizon_interactions=True))
    save_json(out/'structure_audit.json',dict(passed=True,run=str(RUN.resolve()),
        config_sha256=sha(RUN/'config.json'),rows=rows,
        conclusion='Within a fold, additive Ridge allows prefix-dependent offsets but the same future shape/slope for every query cell; verified on development cells, not selected by test errors.',
        next_experiment='Preregister prefix-by-horizon interactions and regularization, choose using frozen inner validation only; retain both arms regardless of outer results.'))
    print('OUTPUT='+str(out.resolve()),flush=True)

if __name__=='__main__':main()
