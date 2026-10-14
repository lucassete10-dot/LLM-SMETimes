"""All 55 cells; no fitting, interpolation, smoothing or selective display."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scripts.export_tail_curve_fidelity import curves
from scripts.save_release_soh_diagnostics import audit
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--figure-tools',type=Path,required=True);args=ap.parse_args();root=args.run;tools=args.figure_tools
    cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'independent_verification.json').read_text())['passed'];old=Path(cfg['current_primary']);assert sha(old/'config.json')==cfg['current_primary_config_sha256']
    out=Path('results/xjtu_segment_curves')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True);print('OUTPUT='+str(out.resolve()),flush=True)
    shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2('docs/xjtu_segment_figure_contract.md',out/'figure_contract.md')
    preflight=subprocess.run([sys.executable,str(tools/'validate_figure.py'),__file__,'--json'],capture_output=True,text=True);save_json(out/'source_preflight.json',dict(exit_code=preflight.returncode,stdout=preflight.stdout,stderr=preflight.stderr));assert preflight.returncode==0,preflight.stdout
    sys.path.insert(0,str(tools.resolve()));from audit_panel_alignment import require_matplotlib_panel_alignment
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False})
    metrics={r['cell_id']:r for r in json.loads((root/'by_cell.json').read_text())};oldmetrics={r['cell_id']:r for r in json.loads((old/'by_cell.json').read_text())};manifest=[]
    with PdfPages(out/'all_cells.pdf') as pdf:
        for plan in cfg['plans']:
            for key in plan['test']:
                x,y,p,sd,nh,stack=curves(root,plan['name'],cfg['seeds'],key,'test_predictions.npz');bx,by,b,bsd,oh,b_stack=curves(old,plan['name'],cfg['seeds'],key,'test_predictions.npz')
                np.testing.assert_array_equal(x,bx);np.testing.assert_array_equal(y,by);assert np.all(np.diff(x)>0);m=metrics[key];om=oldmetrics[key];base=out/key
                np.savez_compressed(str(base)+'.source.npz',cycle=x,truth=y,new_seed_predictions=np.stack(stack),old_seed_predictions=np.stack(b_stack),new_mean=p,new_seed_sd=sd,old_mean=b,old_seed_sd=bsd)
                fig,ax=plt.subplots(figsize=(183/25.4,110/25.4));fig.subplots_adjust(left=.105,right=.97,bottom=.22,top=.73)
                ax.fill_between(x,b-bsd,b+bsd,color='#888888',alpha=.12,linewidth=0);ax.fill_between(x,p-sd,p+sd,color='#2468A0',alpha=.16,linewidth=0)
                ax.plot(x,y,color='#252525',linewidth=1.5,label='Observed SOH');ax.plot(x,b,color='#888888',linewidth=1.2,linestyle='--',label='Previous tail model');ax.plot(x,p,color='#2468A0',linewidth=1.4,label='Feature-selected candidate')
                ax.set_xlabel('Future source cycle (input: first 100 source cycles)');ax.set_ylabel('SOH (0–1)');ax.grid(alpha=.14)
                fig.text(.105,.945,'XJTU | '+key,fontsize=11);fig.text(.105,.895,'Held-out-cell development CV | Mean ± 1 seed SD (n=5; not a CI)',fontsize=8)
                fig.text(.105,.845,f"Seed-mean metrics: MAPE {om['mape_percent']:.2f}% → {m['mape_percent']:.2f}% | Endpoint {m['endpoint_pp']:.2f} pp | Drop {m['drop_error_pp']:.2f} pp",fontsize=8)
                short=bool(len(x)<10 or x[-1]-x[0]<100);status='Joint criteria passed' if m['joint_pass'] else 'Joint criteria NOT passed'
                if short:status+=f' | Short future: {len(x)} points, span {x[-1]-x[0]:.0f} cycles'
                fig.text(.105,.795,status,fontsize=8)
                handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.105,.015),ncol=3,fontsize=8)
                require_matplotlib_panel_alignment(fig,json_out=str(base)+'.alignment.json',overlay_svg=str(base)+'.alignment.svg',strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
                fig.savefig(str(base)+'.pdf');fig.savefig(str(base)+'.svg');fig.savefig(str(base)+'.png',dpi=300);pdf.savefig(fig);plt.close(fig)
                checks=audit(base,tools);manifest.append(dict(cell_id=key,fold=plan['name'],figure=str(base.resolve()),metrics=m,source_npz_sha256=sha(Path(str(base)+'.source.npz')),new_prediction_sha256=nh,old_prediction_sha256=oh,audits=checks,short_coverage=short))
                save_json(out/'progress.json',dict(exported=len(manifest),total=55,goal_complete=False));print('EXPORTED '+key,flush=True)
    assert len(manifest)==len({r['cell_id'] for r in manifest})==55
    combined=audit(out/'all_cells',tools)
    save_json(out/'manifest.json',dict(complete=True,all_cells_included=True,figures=manifest,combined_pdf_audits=combined,outer_verification_sha256=sha(root/'independent_verification.json'),visual_review='pending',goal_complete=False))

if __name__=='__main__':main()
