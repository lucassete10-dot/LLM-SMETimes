"""Deliver all held-out curves of the validation-selected LLM-only MVP."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha

SKILL=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=Path(args.run);cfg=json.loads((run/'config.json').read_text())
    results=json.loads((run/'results.json').read_text())
    selected=min(results,key=lambda r:tuple(r['validation_score']))['name']
    root=Path('deliveries')/('MIT_LLM_SMETimes_MVP_'+run.name)
    root.mkdir(exist_ok=True);folder=root/'MIT/input_100/MIT_fold1';folder.mkdir(parents=True,exist_ok=True)
    qa=root/'figure_sources';qa.mkdir(exist_ok=True)
    shutil.copy2('docs/llm_mvp_figure_contract.md',qa/'contract.md')
    shutil.copytree(run,root/'saved_results',dirs_exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    metrics=json.loads((run/selected/'heldout_metrics.json').read_text());figures=[]
    with np.load(run/selected/'heldout_predictions.npz') as data:
        for cid,row in sorted(metrics.items()):
            prefix=f'{cid}__100__';x,y,p=[data[prefix+k] for k in ('cycle','truth','prediction')]
            assert np.all(np.diff(x)>0) and np.isfinite(np.c_[x,y,p]).all()
            fig=plt.figure(figsize=(8,5));ax=fig.add_axes([.12,.19,.85,.56])
            ax.plot(x,y,color='#222222',lw=1.6,label='Observed SOH')
            ax.plot(x,p,color='#2673A6',lw=1.6,label='LLM + SMETimes forecast')
            ax.set(xlabel='Future source cycle | input: first 100 source cycles',ylabel='SOH (initial capacity reference)')
            ax.grid(alpha=.15);ax.margins(x=.015,y=.07)
            fig.text(.12,.96,'MIT | '+cid,fontsize=12,va='top')
            fig.text(.12,.90,f'{selected} | MIT_fold1 | Single seed | Development holdout',fontsize=8,va='top')
            fig.text(.12,.85,f'MAPE {row["mape_percent"]:.2f}% | MAE {row["mae_pp"]:.2f} pp | Tail MAE {row["tail_mae_pp"]:.2f} pp',fontsize=8,va='top')
            fig.text(.12,.80,f'Stage drop error {row["stage_drop_max_pp"]:.2f} pp | Joint criteria: '+('Passed' if row['joint_pass'] else 'Not passed'),fontsize=8,va='top')
            fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.105,.025),ncol=2,fontsize=8,frameon=False)
            stem=qa/cid;require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
            fig.savefig(folder/(cid+'.png'),dpi=300);fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.svg');plt.close(fig)
            for script,extra in [('audit_pdf_text.py',['--min-pt','5']),('audit_figure_collisions.py',['--json-out',str(stem)+'.collision.json'])]:
                subprocess.run([sys.executable,str(SKILL/script),str(stem)+'.pdf',*extra],check=True,stdout=subprocess.DEVNULL)
            figures.append({'cell_id':cid,'file':str((folder/(cid+'.png')).relative_to(root)),
                'sha256':sha(folder/(cid+'.png')),'automated_pdf_checks':True,'visual_review':False})
            print('EXPORTED',cid,flush=True)
    summary={}
    for result in results:
        d=list(json.loads((run/result['name']/'heldout_metrics.json').read_text()).values())
        summary[result['name']]={'cells':len(d),'joint_pass':sum(r['joint_pass'] for r in d),
            **{k:float(np.mean([r[k] for r in d])) for k in ('mape_percent','mae_pp','tail_mae_pp','stage_drop_max_pp')}}
    save_json(root/'metrics.json',{'summary':summary,'selected':selected,'per_cell':metrics})
    save_json(root/'evaluation_manifest.json',{'selected_by_validation_only':selected,'figures':figures,
        'planned_cells':27,'completed_cells':len(figures),'seed':cfg['seed'],'input_cycles':100,
        'independent_confirmation':False,'full_goal_complete':False,'all_individual_visual_review':False})
    text='# MIT LLM＋SMETimes MVP 阶段结果\n\n'
    text+='仅一个开发划分、一个种子、输入前100次源循环。两组均真实使用预训练 LLM，未训练树模型。\n\n'
    text+='|模型|MAPE|MAE(pp)|尾段MAE(pp)|阶段下降量误差(pp)|联合达标|\n|---|---:|---:|---:|---:|---:|\n'
    for n,r in summary.items():
        text+=f'|{n}|{r["mape_percent"]:.2f}%|{r["mae_pp"]:.2f}|{r["tail_mae_pp"]:.2f}|{r["stage_drop_max_pp"]:.2f}|{r["joint_pass"]}/{r["cells"]}|\n'
    text+='\n验证选中曲线监督候选，但留出尾段平均误差反而增加。阶段误差改善不代表整条曲线成功，两组均未达联合标准。保留全部失败图。下一步应检查 LLM 表示尺度、专家门控及连续轨迹头，不能只增加训练轮数。\n\n'
    text+='图片包含全部27颗留出电芯，没有每颗单独挑模型，没有平滑、裁剪或修改标签。源记录最后位置只用于评分。标签保留历史初始容量参考，未与额定容量版本混比。\n\n'
    text+='复现命令：`.venv/Scripts/python.exe -X utf8 -m scripts.train_llm_soh_mvp --config configs/llm_soh_mvp.json`。完整配置、模型外围权重、预处理器、训练日志、逐点预测及代码快照见 saved_results。冻结骨干按哈希恢复，未重复复制2.47GB权重。\n'
    (root/'stage_report.md').write_text(text,encoding='utf-8')
    print('DELIVERY',root.resolve(),flush=True)


if __name__=='__main__':main()
