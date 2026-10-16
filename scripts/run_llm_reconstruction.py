"""Serial five-dataset TRAIN reconstruction and direct Markdown/PDF delivery."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback
from battery_smetimes.llm_full_life import read
from scripts.train_llm_full_life import save
from scripts.audit_llm_curve_repair import fingerprint

DATASETS=['MIT','XJTU','HUST','UConn_ISU_ILCC','Zhu2022_NCA']


def summary(root,queue):
    lines=['# TRAIN完整寿命曲线重建','',
        '本轮输入完整寿命TRAIN信号，重建同一时段SOH；明确不是前100周期预测。固定首折、单种子，VAL/TEST不参与训练或模型选择。',
        '', '| 数据集 | 状态 | TRAIN平均MAPE | TRAIN平均MAE（pp） | 图集 | 张数 |',
        '|---|---|---:|---:|---|---:|']
    combined=[]
    for task in queue['tasks']:
        ds=task['dataset'];run=root/'runs'/ds;out=root/'reports'/ds
        result=read(run/'result.json') if (run/'result.json').exists() else None
        link=f"[TRAIN重建 PDF]({(out/'all_cells.pdf').resolve().as_posix()})" if (out/'figure_manifest.json').exists() else '待完成'
        mape=f"{result['TRAIN_mape_percent']:.3f}%" if result else '—'
        mae=f"{result['TRAIN_mae_pp']:.3f}" if result else '—'
        count=str(result['TRAIN_cells']) if result else '—'
        lines.append(f"| {ds} | {task['state']} | {mape} | {mae} | {link} | {count} |")
        if (out/'metrics.csv').exists():
            with (out/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:combined.extend(csv.DictReader(f))
    lines+=['',f"[逐电芯指标]({(root/'all_TRAIN_metrics.csv').resolve().as_posix()}) · [执行方法]({Path('docs/LLM_SMETimes_TRAIN_reconstruction_20261016.md').resolve().as_posix()})",'',
        '先12轮联合学习数值适配/融合/专家/重建头，再固定已学习编码器，最多200轮细化共享重建头。按TRAIN宏平均MAE选模型；不使用电芯ID查表、不读SOH作输入、不平滑原始曲线。',
        '', '重建误差小只能证明完整输入下的训练拟合能力，不能证明只看前100周期的新电芯预测能力。训练完成不等于每颗电芯都达到参考图。',
        '', '图集自动核对PDF文本及布局；实际逐页视觉检查状态记录在各图集的figure_manifest.json中。']
    (root/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if combined:
        with (root/'all_TRAIN_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    tiny=read(root/'tiny_XJTU_v1/result.json')
    if tiny['TRAIN_mae_pp']>1:raise ValueError('Tiny reconstruction diagnosis failed; investigate before full runs')
    sources=[Path('battery_smetimes/llm_reconstruction.py'),Path('scripts/train_llm_reconstruction.py'),Path(__file__),
        Path('scripts/report_llm_reconstruction.py'),Path('docs/LLM_SMETimes_TRAIN_reconstruction_20261016.md')]
    registration={str(p.resolve()):fingerprint(p) for p in sources}
    if (root/'registration.json').exists():
        if read(root/'registration.json')!=registration:raise ValueError('Registered execution source changed')
    else:save(root/'registration.json',registration)
    queue=read(root/'queue.json') if (root/'queue.json').exists() else dict(tasks=[dict(dataset=d,state='pending') for d in DATASETS],complete=False)
    lock=Path('experiments/llm_curve_full/gpu_queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    import msvcrt
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        for task in queue['tasks']:
            ds=task['dataset'];run=root/'runs'/ds;out=root/'reports'/ds
            if task['state']=='complete':continue
            for module,arguments,done in [
                ('scripts.train_llm_reconstruction',['--index',f'data/interim/llm_full_life/20261015_full_life/{ds}/index.json','--out',str(run)],run/'result.json'),
                ('scripts.report_llm_reconstruction',['--run',str(run),'--out',str(out)],out/'figure_manifest.json')]:
                if done.exists():continue
                task['state']='training' if 'train_' in module else 'exporting';save(root/'queue.json',queue);summary(root,queue)
                if 'train_' in module and (run/'config.json').exists():arguments+=['--resume']
                log=root/'logs'/f"{ds}_{module.split('.')[-1]}_{datetime.now(timezone.utc).strftime('%H%M%S')}.log"
                with log.open('w',encoding='utf-8') as f:
                    process=subprocess.Popen([sys.executable,'-m',module,*arguments],stdout=f,stderr=subprocess.STDOUT)
                    save(root/'active_process.json',dict(pid=process.pid,dataset=ds,module=module,log=str(log.resolve())))
                    if process.wait():raise RuntimeError('Task failed; inspect '+str(log))
            task['state']='complete';save(root/'queue.json',queue);summary(root,queue)
        queue['complete']=True;save(root/'queue.json',queue);summary(root,queue)
    except BaseException:
        save(root/'failure.json',dict(traceback=traceback.format_exc()));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
