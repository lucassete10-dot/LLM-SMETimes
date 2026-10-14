"""All-cell raw figures and role-separated summaries for a completed dataset."""
import argparse,json,shutil,sys,subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint
from battery_smetimes.llm_reporting import summarize_curve_rows,finite_metric
from scripts.llm_dataset_report_contract import dataset_contract,write_readable_summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue',type=Path,required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--variant',choices=['baseline','optimized'],default='baseline')
    ap.add_argument('--progress-only',action='store_true');ap.add_argument('--figure-roles',nargs='+',choices=['training','validation','test'])
    ap.add_argument('--output-dir',type=Path);ap.add_argument('--quality-v3',action='store_true')
    ap.add_argument('--available-roles',nargs='+',choices=['training','validation','test'],default=['training','validation','test']);args=ap.parse_args()
    if args.figure_roles and not set(args.figure_roles)<=set(args.available_roles):raise ValueError('figure role was not included in available report roles')
    queue=json.loads((args.queue/'queue.json').read_text(encoding='utf-8'));tasks=[t for t in queue['tasks'] if t['dataset']==args.dataset]
    if args.variant=='optimized':tasks=json.loads((args.queue/'optimization'/args.dataset/'state.json').read_text(encoding='utf-8'))['formal_tasks']
    planned_tasks=list(tasks)
    if args.progress_only:tasks=[t for t in tasks if t['state']=='complete']
    if not tasks or any(t['state']!='complete' for t in tasks):raise ValueError('dataset queue is not completely trained and verified')
    out=args.queue/'reports'/args.dataset;out.mkdir(parents=True,exist_ok=True)
    if args.variant=='optimized':out=out/'optimized';out.mkdir(exist_ok=True)
    if args.progress_only:
        out=out/'progress'/f'completed_{len(tasks):03d}';out.mkdir(parents=True,exist_ok=True)
    practical_path=Path('configs/llm_curve_reporting_practical_v1.json')
    practical=json.loads(practical_path.read_text(encoding='utf-8'))
    if args.progress_only and practical['numeric_thresholds']['mape_percent']!=5:
        out=out.with_name(out.name+f"_mape{practical['numeric_thresholds']['mape_percent']:g}");out.mkdir(parents=True,exist_ok=True)
    if args.output_dir:
        out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    archive={};per_role={r:[] for r in args.available_roles};source_hashes={};coverage=[];selection_versions=set()
    for task in tasks:
        run=Path(task['run']);report=json.loads((run/'strict_acceptance.json').read_text(encoding='utf-8'))
        selection_versions.add(json.loads((run/'config.json').read_text(encoding='utf-8')).get('selection_version','legacy_strict_joint_first'))
        for role in per_role:
            record=report[role];coverage.append({'fold':task['fold'],'seed':task['seed'],'role':role,
                'requested':record['cells_requested'],'evaluated':len(record['by_cell']),'skipped':record['skipped']})
            for cid,row in record['by_cell'].items():
                numeric_pass=bool(all(finite_metric(row,k) and row[k]<=v for k,v in practical['numeric_thresholds'].items()))
                per_role[role].append({'cell_id':cid,'fold':task['fold'],'seed':task['seed'],**row,'practical_numeric_pass':numeric_pass})
            source=run/(role+'_audited_predictions.npz');source_hashes[str(source)]=fingerprint(source)
            with np.load(source,allow_pickle=False) as data:
                for cid in record['by_cell']:
                    prefix=f'{cid}__{queue["origin"]}__';x,y,p=[data[prefix+k].copy() for k in ('cycle','truth','prediction')]
                    archive.setdefault(cid,{}).setdefault(role,[]).append({'fold':task['fold'],'seed':task['seed'],'x':x,'y':y,'p':p})
    if args.quality_v3:
        from battery_smetimes.llm_v3_reporting import enrich_quality_rows
        for role,rows in per_role.items():enrich_quality_rows(rows,archive,role)
    statistics={}
    for role,rows in per_role.items():
        statistics[role]=summarize_curve_rows(rows,sum(c['requested'] for c in coverage if c['role']==role),practical['numeric_thresholds'])
        statistics[role]['strict_reference_failures']=[{k:r.get(k) for k in ('cell_id','fold','seed','numeric_gates','knee_pass','reason')} for r in rows if not r['strict_joint_pass']]
    contract=dataset_contract(json.loads((Path(tasks[0]['run'])/'config.json').read_text(encoding='utf-8')),args.dataset)
    summary={'dataset':args.dataset,'runs':len(tasks),'planned_runs':len(planned_tasks),'dataset_complete':not args.progress_only,
        'planned_role_instances':{r:sum(t['role_counts']['train' if r=='training' else r] for t in planned_tasks) for r in per_role},'seeds':queue['seeds'],'roles':statistics,
        'dataset_contract':contract,'primary_reporting_standard':practical,'reporting_standard_sha256':fingerprint(practical_path),'coverage':coverage,'source_predictions_sha256':source_hashes,'independent_confirmation_certified':False,'goal_complete':False}
    save_json(out/'summary.json',summary);write_readable_summary(out,summary)
    if args.quality_v3:
        from battery_smetimes.llm_v3_reporting import quality_summary,write_quality_readme
        summary['v3_reference_quality']={role:quality_summary(rows,statistics[role]['requested_cell_seed_instances']) for role,rows in per_role.items()}
        summary['model_selection_versions']=sorted(selection_versions)
        helper_sources=[Path('battery_smetimes/llm_v3_reporting.py'),Path('battery_smetimes/llm_curve_quality_v3.py'),Path('battery_smetimes/llm_v3_workflow.py')]
        summary['v3_reporting_sources_sha256']={str(p):fingerprint(p) for p in helper_sources}
        for source in helper_sources:
            target=out/'report_source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        save_json(out/'summary.json',summary);write_quality_readme(out,summary)
    save_json(out/'per_cell_seed_metrics.json',per_role)
    figure_dir=out/'figures';figure_dir.mkdir(exist_ok=True)
    tools=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts');sys.path.insert(0,str(tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    figure_rows=[]
    with PdfPages(out/'all_cells.pdf') as pdf:
        for cid,roles in sorted(archive.items()):
            eligible=[r for r in ('test','validation','training') if roles.get(r) and (args.figure_roles is None or r in args.figure_roles)]
            if not eligible:continue
            role=eligible[0];entries=roles[role]
            # A held-out role has priority. Select its first fold BEFORE looking
            # at errors; do not mix predictions fitted on a cell into its test mean.
            fold=entries[0]['fold'];entries=[r for r in entries if r['fold']==fold]
            if len({r['seed'] for r in entries})!=len(entries):raise ValueError('duplicate seed within plotting fold')
            x,y=entries[0]['x'],entries[0]['y']
            for r in entries:np.testing.assert_array_equal(r['x'],x);np.testing.assert_array_equal(r['y'],y)
            predictions=np.stack([r['p'] for r in entries]);mean=predictions.mean(0);sd=predictions.std(0,ddof=1) if len(entries)>1 else np.zeros_like(mean)
            metrics=[r for r in per_role[role] if r['cell_id']==cid and r['fold']==fold];strict=sum(r['strict_joint_pass'] for r in metrics)
            fig=plt.figure(figsize=(8,5));ax=fig.add_axes([.12,.20,.85,.55])
            sparse_rpt=args.dataset=='UConn_ISU_ILCC'
            ax.plot(x,y,color='#222222',lw=1.5,label='Observed SOH',marker='o' if sparse_rpt else None,markersize=3)
            ax.plot(x,mean,color='#2673A6',lw=1.5,label='LLM + SMETimes forecast')
            if len(entries)>1:ax.fill_between(x,mean-sd,mean+sd,color='#2673A6',alpha=.15,lw=0)
            ax.set(xlabel=f'Future source cycle (input: first {queue["origin"]} cycles)',ylabel='SOH (initial capacity reference)');ax.grid(alpha=.15)
            fig.text(.12,.95,args.dataset+' | '+cid,fontsize=12)
            seed_text=f'Mean +/- 1 seed SD (n={len(entries)}; not a CI)' if len(entries)>1 else 'Single seed (n=1; no SD band)'
            if sparse_rpt:seed_text+=' | Sparse RPT points'
            fig.text(.12,.89,f'{fold} | {role} | {seed_text}',fontsize=8)
            late=x>=x[0]+.8*(x[-1]-x[0])
            late_mape=100*np.mean(np.abs(predictions[:,late]-y[None,late])/y[None,late])
            available_stage=[r['stage_drop_max_pp'] for r in metrics if finite_metric(r,'stage_drop_max_pp')]
            stage_text=f'{np.mean(available_stage):.3f} pp' if len(available_stage)==len(metrics) else 'unavailable'
            fig.text(.12,.83,f'Seed-mean MAPE {np.mean([r["mape_percent"] for r in metrics]):.3f}% | Late {late_mape:.3f}% | Endpoint {np.mean([r["endpoint_error_pp"] for r in metrics]):.3f} pp | Stage {stage_text}',fontsize=8)
            mape_passes=sum(finite_metric(r,'mape_percent') and r['mape_percent']<=practical['numeric_thresholds']['mape_percent'] for r in metrics)
            quality_text=(f' | Reference quality: {sum(r["reference_numeric_pass"] for r in metrics)}/{len(entries)}' if args.quality_v3 else f' | Strict reference: {strict}/{len(entries)}')
            fig.text(.12,.78,f'MAPE <= {practical["numeric_thresholds"]["mape_percent"]:g}%: {mape_passes}/{len(entries)} | Practical joint: {sum(r["practical_numeric_pass"] for r in metrics)}/{len(entries)}'+quality_text,fontsize=8)
            fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.105,.03),ncol=2,fontsize=9,frameon=False)
            stem=figure_dir/cid.replace('#','_');require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
            fig.savefig(str(stem)+'.png',dpi=300);fig.savefig(str(stem)+'.svg');pdf.savefig(fig);plt.close(fig)
            figure_rows.append({'cell_id':cid,'role':role,'fold':fold,'seeds':[r['seed'] for r in entries],'png':str(stem)+'.png'})
    shutil.copy2(__file__,out/'plot_source.py')
    qa=[]
    for tool,extra in [('audit_pdf_text.py',['--min-pt','5']),('audit_figure_collisions.py',['--json-out',str(out/'collision_audit.json')])]:
        result=subprocess.run([sys.executable,str(tools/tool),str(out/'all_cells.pdf'),*extra],capture_output=True,text=True,encoding='utf-8')
        qa.append({'tool':tool,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    save_json(out/'figure_manifest.json',{'figures':figure_rows,'raw_truth_unchanged':True,'mean_not_used_for_per_seed_acceptance':True,
        'progress_only':args.progress_only,'figure_role_filter':args.figure_roles,
        'missing_or_unevaluable_cells_remain_in_coverage':True,'qa':qa,'visual_review':'pending','goal_complete':False})
    if any(row['exit_code'] for row in qa):raise ValueError('figure QA failed; raw figures and diagnostics retained')
    print(json.dumps({'dataset':args.dataset,'runs':len(tasks),'all_cell_figures':len(figure_rows),
        'roles':{role:{k:stats[k] for k in ('requested_cell_seed_instances','mape_target_passes','practical_numeric_passes','mape_percent','mae_pp')} for role,stats in statistics.items()}},indent=2),flush=True)


if __name__=='__main__':main()
