"""Write a paired, reproducible recovery report from all saved diagnostic runs."""
import hashlib
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.diagnose_soh_curve_small import save
from battery_smetimes.physical_types import load_registry


def main():
    root=Path(sys.argv[1]);config=json.loads((root/'config.json').read_text())
    if not json.loads((root/'completion.json').read_text())['complete']:raise ValueError('incomplete experiment')
    previous=Path(config['previous']);old=json.loads((previous/'summary.json').read_text())
    rows=[r for r in old if r['variant']=='original_head']+json.loads((root/'summary.json').read_text())
    with (root/'summary.csv').open('w',newline='',encoding='utf-8') as handle:
        fields=sorted(set().union(*(r.keys() for r in rows)))
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    variants=['original_head']+config['variants']
    keys=['mape','late_mape','endpoint_error_pp','mae_pp','rmse_pp','r2','train_mape']
    aggregate={v:{k:dict(mean=float(np.mean([r[k] for r in rows if r['variant']==v])),
                        sd=float(np.std([r[k] for r in rows if r['variant']==v],ddof=1))) for k in keys} for v in variants}
    save(root/'aggregate.json',aggregate)
    sources={v:(previous if v=='original_head' else root) for v in variants}
    artifacts={v:[np.load(sources[v]/v/str(s)/'predictions.npz') for s in config['seeds']] for v in variants}
    ids=config['diagnostic_validation'];differences=[]
    for key in ids:
        reference=artifacts['tree_large'][0]
        for v in variants:
            for a in artifacts[v]:
                np.testing.assert_allclose(a[key+'__truth'],reference[key+'__truth'],rtol=0,atol=1e-7)
                np.testing.assert_allclose(a[key+'__cycle'],reference[key+'__cycle'],rtol=0,atol=.001)
        per_variant={v:np.mean([a[key+'__prediction'] for a in artifacts[v]],axis=0) for v in variants}
        y=reference[key+'__truth']
        differences.append(dict(cell_id=key,**{v:float(np.mean(abs(p-y)/y)*100) for v,p in per_variant.items()}))
    save(root/'paired_mean_prediction_cell_metrics.json',differences)
    historical=json.loads((root/'historical_same_cells.json').read_text())
    historical_sources=historical['source_sha256']
    for hashes in historical_sources.values():
        for path,digest in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('historical artifact changed')
    names={'original_head':'Small MLP (24 train)','tree_small':'Tree (24 train)',
           'tree_small_provenance_features':'Tree + provenance (24)',
           'tree_large':'Restored tree (76 train)','mlp_large':'Small MLP (76 train)'}
    lines=['# SOH 曲线版本回溯与恢复','',
           '本报告追溯历史版本，并用同一固定开发划分重训。历史结果比较不参与新训练选参。',
           '', '## 已确认的差异','',
           '20260930T064130_235459Z 的 v4 未来 SOH 树模型：MIT N100 MAPE 1.5707%，三次开发留出、72 个去重测试电芯、单种子；XJTU 1.9085%，NASA 8.5579%。旧版以批次/倍率方向作类别。',
           '当前 v5 仍保留树模型，改为真实物理家族隔离、已知工况输入。之前发送的近直线图来自 LLM 专家数 1 的某折结果；不能把它当成旧树模型升级后的统一预测。',
           '同 MIT_fold3、seed20260930、同 27 颗电芯、同 N100 未来网格的保存结果逐点核验：','',
           '| 历史同电芯对照 | MAPE % | 后半程 MAPE % | 末点误差 pp | RMSE pp |',
           '|---|---:|---:|---:|---:|']
    for v in ('tree','llm'):
        a=historical['aggregate'][v];lines.append('| '+v+' | '+' | '.join(f'{a[k]:.3f}' for k in ('mape','late_mape','endpoint_error_pp','rmse_pp'))+' |')
    lines += ['', '这 27 颗结果仅作已完成历史审计，单折单种子不能代替全研究结果。树模型在全程/末期更好，但本折平均 R² 低于 LLM；不能声称所有指标都胜出。',
              '', '## 相同开发划分的重训','',
              f"原小实验：24 颗训练，6 颗内层验证，25 颗诊断验证。扩大实验：{len(config['train_large'])} 颗训练，原 6 颗内层验证，原 25 颗诊断验证。全部来自 MIT 同物理家族。原测试电芯未进入新训练/选参。",
              '固定前100循环 V/I/T、已知工况和请求循环；不输入容量/SOH/真实总寿命。不按预测误差删除电芯。树模型3个叶节点候选只由6颗内层验证选择；神经网络轮数也只由这些电芯选择。',
              '五个种子；均值±样本SD是训练种子变异，不是五折或置信区间。诊断验证已多次查看，需要独立新测试验证最终方案。树模型3个结构候选、MLP固定结构选轮数，预算不同，本实验用于故障定位，不作为公平全模型论文主比较。','',
              '| 方法 | MAPE % | 后半程 MAPE % | 末点误差 pp | MAE pp | RMSE pp | R² |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        lines.append('| '+names[v]+' | '+' | '.join(f"{aggregate[v][k]['mean']:.3f} ± {aggregate[v][k]['sd']:.3f}" for k in keys[:6])+' |')
    lines += ['', '## 处理结果','',
              '- 较好的旧树模型能力没有丢失。当前 LLM 和小型神经网络需要独立与它比较，不能凭架构更新替换已有较好曲线结果。',
              '- 增加训练电芯的对照量化样本数影响；batch/倍率方向只作为工况或来源特征，物理家族始终固定，MIT 与 NASA 不合并。',
              '- 先前强制单调加速曲线头在训练集更好、验证集更差的失败结果仍保留，不采用它作为恢复方案。',
              '- 恢复的五种子树预测直接保留模型输出，无真实寿命对齐、平滑、裁剪或事后拟合真实曲线。末端仍有偏差，MAPE 小不等于拐点精确。',
              '- `restored_tree_release.json` 绑定这次76颗训练的五个树工件及哈希，用于开发推理；不是公开数据之外的性能承诺。',
              '', '复现：`.venv/Scripts/python.exe -m scripts.restore_soh_curve_comparison`，报告：`.venv/Scripts/python.exe -m scripts.report_soh_curve_restoration <运行目录>`。所有候选、权重、内层选择、逐电芯结果和重放检查已保存。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    release=dict(schema='soh_curve_restoration_tree_ensemble_v1',scope='development; independent evaluation pending',
                 physical_family='A123_APR18650M1A_LFP_1p1Ah',origin=100,
                 physical_registry=load_registry(),
                 input_sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path('battery_smetimes/forecast.py'),Path('battery_smetimes/research_baselines.py'),Path('scripts/restore_soh_curve_comparison.py'))},
                 train_ids=config['train_large'],inner_validation_ids=config['inner_validation'],
                 model_selection='tree leaf chosen on inner validation per seed; no outer diagnostic score selection',
                 signal_schema='MIT 51 raw features, same frozen extractor; not arbitrary raw sample columns',
                 models=[dict(path=str((root/'tree_large'/str(s)/'model.joblib').resolve()),
                              sha256=hashlib.sha256((root/'tree_large'/str(s)/'model.joblib').read_bytes()).hexdigest()) for s in config['seeds']])
    save(root/'restored_tree_release.json',release)
    display=['original_head','tree_small','tree_large','mlp_large'];colors=['#0072B2','#999999','#009E73','#D55E00']
    fixed=[next(k for k in ids if k.startswith(b)) for b in ('b1','b2','b3')]
    for filename,cell_ids,columns,size in [('comparison_examples',fixed,3,(14,4.5)),('all_validation_curves',ids,5,(19,16))]:
        fig,axes=plt.subplots(int(np.ceil(len(cell_ids)/columns)),columns,figsize=size,squeeze=False)
        for ax,key in zip(axes.flat,cell_ids):
            ref=artifacts['tree_large'][0];x=ref[key+'__cycle'];y=ref[key+'__truth']
            ax.plot(x,y,color='black',lw=1.8,label='Observed')
            for v,color in zip(display,colors):
                p=np.array([a[key+'__prediction'] for a in artifacts[v]]);mean=p.mean(0);sd=p.std(0,ddof=1)
                ax.plot(x,mean,color=color,lw=1.2,label=names[v]);ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.1,linewidth=0)
            ax.set(title=key,xlabel='Future source cycle (origin 100)',ylabel='SOH');ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(cell_ids):]:ax.set_visible(False)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.90 if len(cell_ids)==3 else .96))
        fig.savefig(root/(filename+'.png'),dpi=160);fig.savefig(root/(filename+'.pdf'));plt.close(fig)
    # Exact historical LLM chart cells: show the tree that was already retained.
    base=Path('experiments/soh_research_v5/runs');tp=base/'20260930T080008_026015Z/MIT_fold3/20260930/extra_trees/test/predictions.npz'
    lp=base/'20260930T090318_035370Z/MIT_fold3/20260930/llm_experts1/test/predictions.npz'
    with np.load(tp) as tree,np.load(lp) as llm:
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,key in zip(axes,('b1c12','b2c24','b3c28')):
            p=key+'__100__';x=tree[p+'cycle']
            ax.plot(x,tree[p+'truth'],color='black',label='Observed');ax.plot(x,llm[p+'prediction'],color='#0072B2',label='LLM (shown earlier)')
            ax.plot(x,tree[p+'prediction'],color='#009E73',label='Retained tree')
            ax.set(title=key,xlabel='Future source cycle (origin 100)',ylabel='SOH');ax.grid(alpha=.2)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False);fig.tight_layout(rect=(0,0,1,.9))
        fig.savefig(root/'historical_same_cells.png',dpi=160);fig.savefig(root/'historical_same_cells.pdf');plt.close(fig)
    for values in artifacts.values():
        for a in values:a.close()
    save(root/'report_checks.json',dict(all_25_cells_all_seeds_included=True,paired_truth_max_tolerance=1e-7,
                                      paired_cycle_max_tolerance=.001,historical_files_hashes_verified=True))
    print(root/'report.md')


if __name__=='__main__':main()
