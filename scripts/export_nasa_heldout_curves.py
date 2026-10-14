"""Export saved predictions, never refit or select using outer errors."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import matplotlib
import numpy as np
from scripts.save_release_soh_diagnostics import curve, audit
from scripts.prepare_same_type_forecast import save_json, sha

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--figure-tools',type=Path,required=True)
    parser.add_argument('--verification',type=Path,default=Path('experiments/nasa_ridge_verification/20261004T144551_758144Z/verification.json'))
    parser.add_argument('--dataset',default='NASA')
    parser.add_argument('--contract',type=Path,default=Path('docs/nasa_heldout_curve_contract.md'))
    parser.add_argument('--model-label',default='Ridge future forecast')
    args=parser.parse_args()
    sys.path.insert(0,str(args.figure_tools.resolve()))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.right':False,'axes.spines.top':False})
    source=args.verification
    verified=json.loads(source.read_text()); assert verified['passed']
    run=Path(verified['run']);cfg=json.loads((run/'config.json').read_text())
    out=Path('results')/(args.dataset.lower()+'_heldout_curves')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    shutil.copy2('scripts/save_release_soh_diagnostics.py',out/'plot_helper.py')
    shutil.copy2(args.contract,out/'figure_contract.md')
    preflight=subprocess.run([sys.executable,str(args.figure_tools/'validate_figure.py'),
        'scripts/save_release_soh_diagnostics.py','--json'],capture_output=True,text=True)
    save_json(out/'source_preflight.json',dict(exit_code=preflight.returncode,stdout=preflight.stdout,stderr=preflight.stderr))
    if preflight.returncode:raise RuntimeError('figure source preflight failed')
    manifest=[]
    for plan in cfg['plans']:
        reference=None
        for seed in cfg['seeds']:
            path=run/plan['name']/str(seed)/'predictions.npz'
            with np.load(path,allow_pickle=False) as archive:
                arrays={k:archive[k] for k in archive.files}
            if reference is None:reference=arrays
            else:
                assert set(reference)==set(arrays)
                for key in arrays:np.testing.assert_array_equal(reference[key],arrays[key])
        for cell_id in plan['test']:
            row=next(r for r in verified['rows'] if r['fold']==plan['name'] and r['cell_id']==cell_id and r['seed']==cfg['seeds'][0])
            row=dict(row,dataset=args.dataset,model_label=args.model_label,
                evaluation_label=('Unseen-cell LOBO — public development evaluation' if args.dataset=='NASA'
                    else 'Unseen-cell 5-fold — sparse diagnostics / known constant I'))
            if args.dataset!='NASA':
                row['input_axis_label']='Future source cycle (input: diagnostics at cycles 0 and 100)'
            base=curve(row,reference,out,require_matplotlib_panel_alignment)
            audits=audit(base,args.figure_tools)
            manifest.append(dict(cell_id=row['cell_id'],fold=plan['name'],seeds=cfg['seeds'],
                identical_repetitions=True,figure=str(base.resolve()),
                png_sha256=sha(Path(str(base)+'.png')),metrics=row,audits=audits))
    save_json(out/'manifest.json',dict(complete=True,figures=manifest,
        verification_sha256=sha(source),all_cells_included=True,visual_review='pending'))
    print('OUTPUT='+str(out.resolve()),flush=True)

if __name__=='__main__':main()
