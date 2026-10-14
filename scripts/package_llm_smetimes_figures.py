"""Export actual pretrained SMETimes-style E1 forecasts, never tree proxies."""
import json,hashlib,shutil,subprocess,sys,zipfile
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SKILL=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources=sorted(Path('experiments/soh_research_v5/runs').glob('*/**/llm_e1/test/predictions.npz'))
    assert sources
    root=Path('D:/LLM')/('SOH_LLM_SMETimes_figures_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    root.mkdir(exist_ok=False);qa=Path('deliveries')/(root.name+'_source');qa.mkdir(exist_ok=False)
    shutil.copy2('docs/llm_smetimes_export_contract_20261008.md',qa/'figure_contract.md')
    groups=defaultdict(list);provenance=[]
    for path in sources:
        unit=path.parent.parent;cfg=json.loads((unit/'config.json').read_text());llm=json.loads((unit/'llm.json').read_text())
        assert cfg['model']=='llm_e1' and llm['variant']['pretrained'] and llm['variant']['use_statistics']
        assert llm['variant']['experts']==4
        plan=cfg['plan'];dataset=plan['dataset'];seed=cfg['seed']
        assert not set(plan['test'])&set(plan['train']+plan['validation'])
        with np.load(path,allow_pickle=False) as d:
            for key in d.files:
                if not key.endswith('__prediction'):continue
                cid,n,_=key.split('__');n=int(n);assert cid in plan['test']
                c,y,p=[d[f'{cid}__{n}__{k}'].copy() for k in ('cycle','truth','prediction')]
                assert np.isfinite(np.c_[c,y,p]).all() and np.all(y>0) and np.all(c>n) and np.all(np.diff(c)>0)
                groups[(dataset,plan['name'],cid,n)].append(dict(seed=seed,cycle=c,truth=y,prediction=p,smoke=cfg['smoke']))
        dest=root/'saved_results'/dataset/plan['name']/str(seed);dest.mkdir(parents=True)
        for filename in ['config.json','llm.json','selected.json','resources.json','refit_history.json']:
            if (unit/filename).exists():shutil.copy2(unit/filename,dest/filename)
        shutil.copy2(path,dest/'predictions.npz');shutil.copy2(path.parent/'metrics.json',dest/'metrics.json')
        provenance.append(dict(source=str(path.resolve()),sha256=digest(path),dataset=dataset,fold=plan['name'],seed=seed,smoke=cfg['smoke'],llm=llm))
    print('OUTPUT='+str(root),'FIGURES='+str(len(groups)),flush=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    for (dataset,fold,cid,n),records in sorted(groups.items()):
        records=sorted(records,key=lambda r:r['seed']);assert len({r['seed'] for r in records})==len(records)
        c,y=records[0]['cycle'],records[0]['truth']
        for r in records:
            np.testing.assert_array_equal(c,r['cycle']);np.testing.assert_array_equal(y,r['truth'])
        stack=np.stack([r['prediction'] for r in records]);mean=stack.mean(0)
        sd=stack.std(0,ddof=1) if len(records)>1 else np.zeros(len(c))
        mape=np.mean(np.abs((stack-y)/y),axis=1)*100
        tail=np.arange(len(c))>=int(.8*len(c));late=np.abs(mean[tail]-y[tail]).mean()*100
        smoke=any(r['smoke'] for r in records)
        folder=root/dataset/f'input_{n:03d}'/fold;folder.mkdir(parents=True,exist_ok=True)
        qfolder=qa/dataset/f'input_{n:03d}'/fold;qfolder.mkdir(parents=True,exist_ok=True)
        fig=plt.figure(figsize=(8,5));ax=fig.add_axes([.12,.19,.85,.56])
        ax.plot(c,y,color='#222222',lw=1.6,label='Observed SOH')
        ax.plot(c,mean,color='#2673A6',lw=1.6,label='Forecast mean' if len(records)>1 else 'Forecast')
        if len(records)>1:ax.fill_between(c,mean-sd,mean+sd,color='#2673A6',alpha=.15,linewidth=0,label='One seed SD (not a CI)')
        ax.set(xlabel=f'Future source cycle | input: first {n} source cycles',ylabel='SOH (saved capacity reference)')
        ax.grid(alpha=.15);ax.margins(x=.015,y=.07)
        fig.text(.12,.96,dataset+' | '+cid,fontsize=12,va='top')
        fig.text(.12,.90,f'LLM + SMETimes-style E1 | {fold} | '+('SMOKE ONLY' if smoke else 'Historical development holdout'),fontsize=8,va='top')
        fig.text(.12,.85,f'Seed-mean MAPE {mape.mean():.2f}% | Mean-curve late MAE {late:.2f} pp | Seeds n={len(records)}',fontsize=8,va='top')
        fig.text(.12,.80,'No independent-test or curve-shape success claim | Actual completed runs only',fontsize=8,va='top')
        fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.105,.025),ncol=3,fontsize=8,frameon=False)
        stem=qfolder/cid;require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
        fig.savefig(folder/(cid+'.png'),dpi=300);fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.svg');plt.close(fig)
        for script,extra in [('audit_pdf_text.py',['--min-pt','5']),('audit_figure_collisions.py',['--json-out',str(stem)+'.collision.json'])]:
            subprocess.run([sys.executable,str(SKILL/script),str(stem)+'.pdf',*extra],check=True,stdout=subprocess.DEVNULL)
        rel=(folder/(cid+'.png')).relative_to(root).as_posix()
        figures.append(dict(dataset=dataset,fold=fold,cell_id=cid,input_cycles=n,image=rel,sha256=digest(root/rel),
            seeds=[r['seed'] for r in records],seed_mean_mape=float(mape.mean()),seed_sd_mape=float(mape.std(ddof=1)) if len(records)>1 else None,
            mean_curve_late_mae_pp=float(late),smoke_only=smoke,automated_pdf_checks=True,individual_visual_review=False))
        print('EXPORTED',len(figures),'/',len(groups),flush=True)
    counts={d:sum(r['dataset']==d for r in figures) for d in ['MIT','XJTU']}
    unique={d:len({r['cell_id'] for r in figures if r['dataset']==d}) for d in counts}
    manifest=dict(model='Actual Llama-3.2-1B frozen backbone + statistics cards + four experts; E1 only',figures=figures,
        sources=provenance,png_count=len(figures),counts=counts,unique_cells=unique,
        independent_test=False,full_training_complete=False,full_visual_review=False,
        missing_datasets=['HUST','UConn_ISU_ILCC','Zhu2022_NCA'],
        limitations=['XJTU is smoke only','MIT folds have unequal completed seed counts','Old saved SOH reference retained, not renormalized','Across folds the same cell can recur; image count is not unique cell count'])
    (root/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (root/'README.txt').write_text('LLM + SMETimes 风格 E1：真实预训练 Llama 3.2 1B + 统计提示 + 四专家。不是树模型。\n'
        '这是已有历史结果导出，本次没有重新训练。按数据集、输入循环数20/50/100、折分别保存单电芯图片。\n'
        f'图片共{len(figures)}张；分数据集{counts}；实际不同电芯数{unique}。折/长度重复不能当作新电池。\n'
        'MIT 各折只汇总实际完成种子，第三折尚未跑齐；XJTU 仅为单种子冒烟结果。\n'
        'HUST、UConn–ISU–ILCC、Zhu2022_NCA 未找到此 E1 架构预测，未用树模型或其他模型填充。\n'
        '保留原始容量参考，不能直接与最新额定容量归一化结果比较 SOH 纵轴。阴影为种子标准差，不是置信区间。\n'
        '这是开发结果，不是独立确认；MAPE低不代表曲线拟合达标。没有挑图、改数据或绘图平滑。\n'
        'PDF 字体/碰撞自动检查执行，逐张人工审查尚未完成。saved_results 包含每个种子的原始预测与配置。\n',encoding='utf-8-sig')
    archive=root.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and sum(n.endswith('.png') for n in z.namelist())==len(figures)
    print(json.dumps(dict(zip=str(archive),png_count=len(figures),counts=counts,unique_cells=unique,MB=round(archive.stat().st_size/1024**2,2))),flush=True)


if __name__=='__main__':main()
