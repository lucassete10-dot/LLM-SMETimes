"""Training-side residual decomposition only; not a forecast or selection rule."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert not cfg['outer_loaded'] and json.loads((root/'independent_verification.json').read_text())['passed'];records=[]
    for plan in cfg['plans']:
        selection=json.loads((root/plan['name']/'selection.json').read_text())
        for held in plan['train']+plan['validation']:
            for arm in ('selected','control'):
                candidate=selection[arm]['candidate'];file=root/plan['name']/('hold_'+held)/f'candidate_{candidate}/validation_predictions.npz'
                with np.load(file) as a:
                    y,p,c=[a[held+'__100__'+f] for f in ('truth','prediction','cycle')];e=p-y;centered=e-e[0]
                    records.append(dict(fold=plan['name'],heldout=held,arm=arm,candidate=candidate,initial_signed_error_pp=float(100*e[0]),initial_abs_error_pp=float(100*abs(e[0])),final_signed_error_pp=float(100*e[-1]),shape_end_abs_pp=float(100*abs(e[-1]-e[0])),mape_percent=float(100*np.mean(abs(e)/y)),initial_offset_removed_residual_mape_percent=float(100*np.mean(abs(centered)/y)),future_span=float(c[-1]-c[0]),source_file=str(file.resolve()),source_sha256=sha(file)))
    fields=('initial_abs_error_pp','shape_end_abs_pp','mape_percent','initial_offset_removed_residual_mape_percent');summary={arm:{f:float(np.mean([r[f] for r in records if r['arm']==arm])) for f in fields} for arm in ('selected','control')};save_json(root/'inner_residual_decomposition.json',dict(rows=records,summary=summary,outer_loaded=False,evaluation_only=True,not_model_selection=True,interpretation='Residual minus initial residual uses heldout inner labels only for diagnosis. Not a label-free corrected forecast and not official accuracy. 12 overlapping parent-fold/inner-cell records per arm, not12 independent cells.',source_sha256=sha(__file__),goal_complete=False));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
