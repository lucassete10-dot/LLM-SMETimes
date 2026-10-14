"""Export verified XJTU N100 development curves into a flat image-only ZIP."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SKILL=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    source=Path('experiments/core_length_cv/20261008T083003_121461Z').resolve()
    proof=json.loads((source/'verification.json').read_text())
    assert proof['passed'] and proof['validated_development_cells']==44
    cfg=json.loads((source/'config.json').read_text())
    metrics=[r for r in json.loads((source/'all_rows.json').read_text())
        if r['origin']==100 and r['arm']=='available' and r['horizon']=='operational']
    bycell={r['cell_id']:r for r in metrics}
    assert len(metrics)==len(bycell)==44
    root=Path('D:/LLM')/('SOH_XJTU_test_figures_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    folder=root/'XJTU';folder.mkdir(parents=True,exist_ok=False)
    qa=Path('deliveries')/(root.name+'_source');qa.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    images=[];hashes={}
    for plan in cfg['plans']:
        with np.load(source/plan['name']/'100/available/predictions.npz',allow_pickle=False) as a:
            for cid in plan['validation']:
                c,y,p=[a[cid+'__'+k] for k in ('cycle','truth','prediction')]
                assert np.all(np.diff(c)>0) and np.all(c>100)
                assert np.isfinite(y).all() and np.isfinite(p).all()
                r=bycell[cid]
                fig=plt.figure(figsize=(8,5));ax=fig.add_axes([.12,.19,.85,.56])
                ax.plot(c,y,color='#222222',lw=1.6,label='Observed SOH')
                ax.plot(c,p,color='#2673A6',lw=1.6,label='Forecast')
                ax.set(xlabel='Future source cycle | input: first 100 source cycles',
                    ylabel='SOH (capacity / rated capacity)')
                ax.grid(alpha=.15);ax.margins(x=.015,y=.07)
                fig.text(.12,.96,'XJTU | '+cid,fontsize=12,va='top')
                fig.text(.12,.90,'Held-out development cell | Five folds, one seed | Core point model',fontsize=8,va='top')
                fig.text(.12,.85,f"MAPE {r['mape_percent']:.2f}% | Late MAE {r['late20_mae_pp']:.2f} pp"
                    f" | Stage error {r['stage_error_pp']:.2f} pp",fontsize=8,va='top')
                fig.text(.12,.80,'Curve-shape success not established | No independent-test claim',fontsize=8,va='top')
                fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.105,.025),ncol=2,fontsize=8,frameon=False)
                stem=qa/cid
                require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
                fig.savefig(folder/(cid+'.png'),dpi=300)
                fig.savefig(str(stem)+'.pdf')
                fig.savefig(str(stem)+'.svg');plt.close(fig)
                for script,extra in [('audit_pdf_text.py',['--min-pt','5']),
                    ('audit_figure_collisions.py',['--json-out',str(stem)+'.collision.json'])]:
                    subprocess.run([sys.executable,str(SKILL/script),str(stem)+'.pdf',*extra],check=True,stdout=subprocess.DEVNULL)
                relative='XJTU/'+cid+'.png';digest=hashlib.sha256((folder/(cid+'.png')).read_bytes()).hexdigest()
                hashes[relative]=digest
                images.append(dict(dataset='XJTU',cell_id=cid,fold=plan['name'],figure=relative,
                    metrics=r,automated_pdf_qa=True,visual_review='pending',input_cycles=100))
                print(f'Exported {len(images)}/44: {cid}',flush=True)
    assert len(images)==len({r['cell_id'] for r in images})==44
    (root/'image_inventory.json').write_text(json.dumps(dict(png_count=44,
        counts_including_overviews={'XJTU':44},overview_count=0,sha256=hashes),indent=2),encoding='utf-8')
    (root/'evaluation_manifest.json').write_text(json.dumps(dict(figures=images,seeds=cfg['seeds'],
        source_run=str(source),source_verification_sha256=hashlib.sha256((source/'verification.json').read_bytes()).hexdigest(),
        independent_test=False,goal_complete=False,full_visual_review=False,
        input_selection_status='N100 provisional, not reliably optimal'),indent=2),encoding='utf-8')
    (root/'README.txt').write_text('XJTU当前五折留出开发验证：44颗，每颗一张主图，输入100次源循环。\n'
        '只含最新已复核core_length_cv结果，没有混用旧条件模型图片。\n'
        '一个随机种子20261008，不显示种子波动或置信区间。非独立测试，不代表曲线形状均已达标。\n'
        '100次为阶段输入选项，尚未确定可靠最优输入长度。\n'
        '全部图执行自动字体和碰撞检查，人工逐张审查尚未完成。无overview.png。\n',encoding='utf-8-sig')
    archive=root.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(root.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and sum(n.endswith('.png') for n in z.namelist())==44
    print(json.dumps(dict(zip=str(archive),folder=str(root),images=44,megabytes=round(archive.stat().st_size/1024**2,2),crc_passed=True)),flush=True)


if __name__=='__main__':main()
