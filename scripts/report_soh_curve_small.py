"""Report every diagnostic variant, including unsuccessful changes."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(sys.argv[1])
    config=json.loads((root/'config.json').read_text())
    rows=json.loads((root/'summary.json').read_text())
    keys=['mape','late_mape','endpoint_error_pp','train_mape','permuted_mape','rmse_pp','mae_pp','r2']
    aggregate={v:{k:{'mean':float(np.mean([r[k] for r in rows if r['variant']==v])),
                        'sd':float(np.std([r[k] for r in rows if r['variant']==v],ddof=1))}
                     for k in keys} for v in config['variants']}
    (root/'aggregate.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    lines=['# MIT 曲线形状小规模诊断','',
           '这是开发诊断，不是新的盲测，也不是原 LLM 的重训结果。使用相同早期信号处理器和小型统计编码器，比较输出头和损失。',
           '24 颗训练、6 颗内层验证选择 checkpoint、25 颗原验证电芯用于诊断。原 MIT_fold3 测试电芯未加载。固定前 100 个源循环，预测其后全部有效观测点。',
           '五个种子；表中先逐电芯平均，再报告种子均值 ± 样本标准差。这不等于五折验证，也不代表划分不确定性。',
           '', '| 设置 | 全未来 MAPE % | 后半程 MAPE % | 末点绝对误差 pp | 训练 MAPE % |',
           '|---|---:|---:|---:|---:|']
    for v in config['variants']:
        lines.append('| '+v+' | '+' | '.join(f"{aggregate[v][k]['mean']:.3f} ± {aggregate[v][k]['sd']:.3f}" for k in keys[:4])+' |')
    lines += ['', '## 结论与限制','',
              '- 原头允许非线性，不存在强制直线输出。当前基线仍有约 8 个 SOH 百分点的平均末点误差，平均 MAPE 掩盖了末期问题。',
              '- 去掉信号只保留周期/已知工况会退化；打乱整套早期输入也会退化。说明特征有用，不能据此认为模型完全忽略了电压电流。置换同时打乱了工况，只作联合输入敏感性诊断。',
              '- 当前单调加速曲线头降低训练误差却提高验证误差，不应替换主模型。它可表达弯曲，但不足以可靠定位新电芯拐点；本次固定形状先验也可能不合适。',
              '- 后期加权略改善末点误差，但全程与后半程 MAPE 未改善；不能将它包装成成功方案。',
              '- 不能仅凭本实验断言原 LLM 的具体故障已定位；这里验证的是输出头/损失方向的可行性，未改变 LLM backbone。',
              '- 下一阶段优先在训练/内层验证上检查拐点相关特征、模型容量与正则化，以及训练样本数学习曲线；再对真实 LLM 做冻结编码的输出头对照。保留独立新测试集验证最终方案。',
              '- 不依据预测误差删除电芯；本实验仅沿用既有数据完整性排除规则。未来总寿命没有作为输入。','',
              '## 文件','',
              '`config.json` 固定方案与划分；`data_provenance.json` 数据哈希；`checks.json` 隔离检查；每种方法/种子下保存模型、训练日志、逐电芯指标和完整预测。',
              '`validation_curves.png` 展示全部 25 颗诊断电芯，五种方法均保留；`example_curves.png` 展示按固定名单每批次首颗电芯（非按效果筛选）。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    names={'original_head':'Original head','original_head_tail':'Original + tail loss',
           'curve_head':'Curve head','curve_head_tail':'Curve + tail loss','time_conditions_only':'Time / conditions only'}
    colors=['#0072B2','#009E73','#D55E00','#CC79A7','#999999']
    loaded={v:[np.load(root/v/str(s)/'predictions.npz') for s in config['seeds']] for v in config['variants']}
    for name,ids,ncols,size in [('validation_curves',config['diagnostic_validation'],5,(19,16)),
            ('example_curves',[next(k for k in config['diagnostic_validation'] if k.startswith(b)) for b in ('b1','b2','b3')],3,(14,4.6))]:
        fig,axes=plt.subplots(int(np.ceil(len(ids)/ncols)),ncols,figsize=size,squeeze=False)
        for ax,key in zip(axes.flat,ids):
            first=loaded['original_head'][0]
            x=first[key+'__cycle']; truth=first[key+'__truth']
            ax.plot(x,truth,color='black',lw=1.7,label='Observed')
            for v,color in zip(config['variants'],colors):
                values=np.array([a[key+'__prediction'] for a in loaded[v]])
                mean=values.mean(0);sd=values.std(0,ddof=1)
                ax.plot(x,mean,color=color,lw=1.2,label=names[v])
                ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.08,linewidth=0)
            ax.set(title=key,xlabel='Source cycle (forecast after 100)',ylabel='SOH')
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(ids):]: ax.set_visible(False)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.90 if len(ids)==3 else .96))
        fig.savefig(root/(name+'.png'),dpi=160)
        fig.savefig(root/(name+'.pdf'))
        plt.close(fig)
    for files in loaded.values():
        for f in files:f.close()
    print(root/'report.md')


if __name__=='__main__':main()
