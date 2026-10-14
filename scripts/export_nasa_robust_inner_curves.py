"""Structural curve-export adaptation; all four cells, no fitting/selection."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.export_tail_curve_fidelity import curves
from scripts.save_release_soh_diagnostics import audit
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--figure-tools',type=Path,required=True);args=ap.parse_args();root=args.run;tools=args.figure_tools;cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'independent_verification.json').read_text())['passed'];old=Path(cfg['baseline']);assert sha(old/'config.json')==cfg['baseline_config_sha256']
    out=Path('results/nasa_robust_inner_curves')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True);print('OUTPUT='+str(out.resolve()),flush=True);shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2('docs/nasa_robust_inner_figure_contract.md',out/'figure_contract.md')
    check=subprocess.run([sys.executable,str(tools/'validate_figure.py'),__file__,'--json'],capture_output=True,text=True);save_json(out/'source_preflight.json',dict(exit_code=check.returncode,stdout=check.stdout,stderr=check.stderr));assert check.returncode==0,check.stdout;sys.path.insert(0,str(tools.resolve()));from audit_panel_alignment import require_matplotlib_panel_alignment
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False})
    bycell={r['cell_id']:r for r in json.loads((root/'by_cell.json').read_text())};manifest=[]
    for plan in cfg['plans']:
        for key in plan['test']:
            x,y,p,sd,new_hashes,ps=curves(root,plan['name'],cfg['seeds'],key,'test_predictions.npz');bx,by,b,bsd,old_hashes,bps=curves(old,plan['name'],range(20261004,20261009),key,'predictions.npz');np.testing.assert_array_equal(x,bx);np.testing.assert_array_equal(y,by);assert all(np.array_equal(v,ps[0]) for v in ps);assert all(np.array_equal(v,bps[0]) for v in bps);assert np.all(np.diff(x)>0);r=bycell[key];base=out/key;np.savez_compressed(str(base)+'.source.npz',cycle=x,truth=y,new_mean=p,new_seed_sd=sd,old_mean=b,old_seed_sd=bsd)
            fig,ax=plt.subplots(figsize=(183/25.4,4.3));fig.subplots_adjust(left=.11,right=.97,bottom=.20,top=.77);ax.plot(x,y,color='#252525',linewidth=1.5,label='Observed SOH');ax.plot(x,b,color='#888888',linewidth=1.2,linestyle='--',label='Frozen Ridge baseline');ax.plot(x,p,color='#2468A0',linewidth=1.5,label='Robust-inner candidate');ax.set_xlabel('Future source cycle (input: first 100 cycles)');ax.set_ylabel('SOH (0–1)');ax.grid(alpha=.14);fig.text(.11,.94,'NASA | '+key,fontsize=12);fig.text(.11,.895,'Held-out-cell development CV; 5 deterministic repeats, not a CI',fontsize=8);fig.text(.11,.85,f"MAPE {r['mape_percent']:.2f}% | Endpoint {r['endpoint_pp']:.2f} pp | Drop error {r['drop_error_pp']:.2f} pp",fontsize=8);handles,names=ax.get_legend_handles_labels();fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.54,.005),ncol=3,fontsize=8)
            if len(x)<10 or x[-1]-x[0]<100:fig.text(.11,.805,f'Short future coverage: {len(x)} points over {x[-1]-x[0]:.0f} source cycles',fontsize=8)
            require_matplotlib_panel_alignment(fig,json_out=str(base)+'.alignment.json',overlay_svg=str(base)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
            fig.savefig(str(base)+'.pdf');fig.savefig(str(base)+'.svg');fig.savefig(str(base)+'.png',dpi=300);plt.close(fig);checks=audit(base,tools);manifest.append(dict(cell_id=key,fold=plan['name'],metrics=r,figure=str(base.resolve()),source_npz_sha256=sha(Path(str(base)+'.source.npz')),new_prediction_sha256=new_hashes,old_prediction_sha256=old_hashes,audits=checks,short_coverage=bool(len(x)<10 or x[-1]-x[0]<100)));print('EXPORTED '+key,flush=True)
    assert len(manifest)==4;save_json(out/'manifest.json',dict(complete=True,all_cells_included=True,figures=manifest,verification_sha256=sha(root/'independent_verification.json'),visual_review='pending',goal_complete=False))


if __name__=='__main__':main()
