"""Fixed-cell model panels from completed native prediction artifacts only."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.diagnose_soh_curve_small import save


RUNS=['20260930T080008_026015Z','20260930T085425_091035Z','20260930T085533_188150Z','20260930T090318_035370Z']


def main():
    out=Path('results/soh_all_model_examples')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    base=Path('experiments/soh_research_v5/runs');units={};snapshots={}
    for run in RUNS:
        progress=json.loads((base/run/'progress.json').read_text())
        snapshots[run]=dict(complete=progress['complete'],completed_units=len(progress['completed']))
        for u in progress['completed']:
            if u['fold'] not in ('MIT_fold1','XJTU_fold1','NASA_fold1'):continue
            key=(u['fold'],u['model'],u['seed'])
            if key in units:raise ValueError('duplicate completed model unit')
            units[key]=base/run/u['fold']/str(u['seed'])/u['model']/'test'
    seeds=list(range(20260930,20260935))
    baseline=['ridge','extra_trees','elastic_net','gpr','lightgbm','cycle_conditions_only','physical_family_mean',
              'gru','lstm','tcn','transformer','statistics_mlp']
    llm=['llm_e1','llm_e2','llm_e4','llm_experts1','llm_experts8','llm_matched_plain']
    splits=json.loads((base/RUNS[0]/'frozen_audit/splits.json').read_text())['plans']
    index=[];source_hashes={};numerical=[]
    for dataset in ('MIT','XJTU','NASA'):
        fold=dataset+'_fold1';plan=next(p for p in splits if p['name']==fold)
        cell=sorted(plan['test'])[0]
        assert cell not in plan['train']+plan['validation']
        for group,models in [('baseline_and_sequence',baseline),('llm_variants',llm)]:
            available=[m for m in models if all((fold,m,s) in units for s in seeds)]
            if not available:
                index.append(dict(dataset=dataset,cell_id=cell,group=group,status='not yet five-seed-complete in this fold',missing=models));continue
            reference=None;prepared=[]
            for model in available:
                values=[];scores=[]
                for seed in seeds:
                    folder=units[(fold,model,seed)];path=folder/'predictions.npz';prefix=cell+'__100__'
                    digest=hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path) as d:x,y,p=[d[prefix+k].copy() for k in ('cycle','truth','prediction')]
                    if not (np.isfinite(p).all() and np.isfinite(y).all()):raise ValueError('invalid saved prediction')
                    if reference is None:reference=(x,y)
                    else:
                        assert np.array_equal(x,reference[0]);assert np.array_equal(y,reference[1])
                    if digest!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('prediction changed while reading')
                    report=json.loads((folder/'metrics.json').read_text())
                    row=next(r for r in report['by_cell'] if r['cell_id']==cell and r['origin']==100)
                    mape=float(np.mean(abs(p-y)/y)*100)
                    np.testing.assert_allclose(mape,row['metrics']['mape_percent'],rtol=0,atol=1e-9)
                    scores.append(mape);values.append(p);source_hashes[str(path)]=digest
                    numerical.append(dict(dataset=dataset,cell_id=cell,model=model,seed=seed,mape=mape))
                values=np.array(values);prepared.append((model,values.mean(0),values.std(0,ddof=1),np.mean(scores),np.std(scores,ddof=1)))
            cols=3;rows=int(np.ceil(len(prepared)/cols));fig,axes=plt.subplots(rows,cols,figsize=(14,3.4*rows),sharex=True,sharey=True,squeeze=False)
            x,y=reference
            for ax,(model,mean,sd,score,score_sd) in zip(axes.flat,prepared):
                ax.plot(x,y,color='black',lw=1.5,label='Observed')
                ax.plot(x,mean,color='#0072B2',lw=1.2,label='Forecast: five-seed mean')
                ax.fill_between(x,mean-sd,mean+sd,color='#0072B2',alpha=.15,linewidth=0,label='Seed SD')
                ax.set(title=f'{model}\nMAPE {score:.2f} +/- {score_sd:.2f}%',xlabel='Future source cycle',ylabel='SOH');ax.grid(alpha=.2)
            for ax in list(axes.flat)[len(prepared):]:ax.set_visible(False)
            subtitle=f'{dataset} / {cell} / {fold} / observed first 100 cycles'
            fig.suptitle(subtitle,fontsize=14,y=.998)
            fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.977),ncol=3,frameon=False,fontsize=9)
            fig.tight_layout(rect=(0,.025,1,.94 if rows>2 else .91))
            fig.text(.5,.008,'Fixed first test cell, not selected by accuracy. One fold/cell only; seed SD is not a confidence interval.',ha='center',fontsize=9)
            stem=dataset+'_'+group
            fig.savefig(out/(stem+'.png'),dpi=160);fig.savefig(out/(stem+'.pdf'));plt.close(fig)
            np.savez_compressed(out/(stem+'_curves.npz'),cycle=x,truth=y,**{m+'__mean':p for m,p,sd,sc,ss in prepared},**{m+'__sd':sd for m,p,sd,sc,ss in prepared})
            index.append(dict(dataset=dataset,cell_id=cell,group=group,status='plotted',models=available,
                              missing=[m for m in models if m not in available],file=str((out/(stem+'.png')).resolve())))
    save(out/'index.json',dict(scope='fixed first test cell in first fold, five seeds; diagnostic examples, not main aggregate',
                              run_snapshots=snapshots,entries=index,source_prediction_sha256=source_hashes))
    save(out/'numbers.json',numerical)
    shutil.copy2(Path(__file__),out/'plot_source.py')
    lines=['# 各模型 SOH 曲线图','',
           '每个数据集固定选第一折名单排序首颗测试电芯，展示各模型的五种子均值和SD；同一数据集内真值、未来循环和前100循环起点完全相同。未按效果挑例，单颗曲线不代表整体性能。',
           'MIT和XJTU属于不同物理家族；NASA只有4颗且规格未完整核验。LLM未完成的数据集明确保留待完成项。','']
    for row in index:
        if row['status']=='plotted':lines.append(f"- {row['dataset']} / {row['cell_id']} / {row['group']}: [曲线图]({Path(row['file']).as_posix()})")
        else:lines.append(f"- {row['dataset']} / {row['group']}: 本折尚无完整五种子结果。")
    lines+=['','新恢复模型另在 `experiments/soh_curve_restoration/20261001T030316_192868Z`，使用开发诊断划分；与这些原外折曲线分别标记。']
    (out/'index.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('OUTPUT='+str(out.resolve()))


if __name__=='__main__':main()
