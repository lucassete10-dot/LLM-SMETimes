"""Verify adopted TRAIN models and write direct per-dataset PDF links."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from battery_smetimes.llm_full_life import read,load_full_life
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3
from scripts.audit_llm_curve_repair import fingerprint
from scripts.train_llm_full_life import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);root=ap.parse_args().root
    if not read(root/'delivery_audit.json')['passed']:raise ValueError('First-pass delivery audit required')
    refinement=read(root/'refinement/result.json')
    if not refinement['complete']:raise ValueError('Refinement incomplete')
    for path,sha in read(root/'refinement/registration.json').items():
        if fingerprint(path)!=sha:raise ValueError('Refinement source changed')
        if fingerprint(root/'refinement/source_snapshot'/Path(path).relative_to(Path.cwd()))!=sha:raise ValueError('Refinement snapshot differs')
    rows=[];chosen=[]
    lines=['# TRAIN完整曲线重建结果','',
        '固定首折五数据集全部240颗TRAIN，真实预训练LLM＋SMETimes，单种子。完整寿命信号输入、同一时段SOH监督，VAL/TEST不参与训练或模型选择。以下全部是TRAIN重建，不是前100周期预测。',
        '', '| 数据集 | 平均MAPE 首轮→采用 | 平均MAE pp 首轮→采用 | 采用模型 | TRAIN图集 | 张数 |',
        '|---|---:|---:|---|---|---:|']
    for task in read(root/'queue.json')['tasks']:
        ds=task['dataset'];parent=root/'runs'/ds;base=read(parent/'result.json')
        decision=next(r for r in refinement['datasets'] if r['dataset']==ds)
        adopted=decision['adopted'];run=root/'refinement/runs'/ds if adopted else parent
        report=root/'refinement/reports'/ds if adopted else root/'reports'/ds
        cfg=read(run/'config.json');result=read(run/'result.json');proof=read(run/'replay_audit.json')
        if not proof['passed'] or fingerprint(run/'model.pt')!=proof['model_sha256']:raise ValueError('Adopted model proof differs')
        if adopted:
            if decision['parent_model_sha256']!=fingerprint(parent/'model.pt') or not result['TRAIN_mae_pp']<base['TRAIN_mae_pp']:
                raise ValueError('Refinement selection differs')
            before=torch.load(parent/'model.pt',map_location='cpu',weights_only=True)
            after=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
            for key in before:
                if not key.startswith(('reconstruction_','slot_embedding.')):torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
        manifest=read(report/'figure_manifest.json');review=read(report/'visual_review.json')
        if not isinstance(manifest['visual_review'],dict) or manifest['visual_review']['state']!='completed' or review['pending_figures']:
            raise ValueError('Adopted actual figure review incomplete')
        if fingerprint(report/'all_cells.pdf')!=manifest['pdf_sha256'] or any(r['exit_code'] for r in manifest['qa']):
            raise ValueError('Adopted PDF evidence differs')
        for picture in review['figures']:
            if fingerprint(picture['png'])!=picture['sha256']:raise ValueError('Reviewed adopted plot changed')
        _,cells,plan,_=load_full_life(cfg['index'])
        if {r['cell_id'] for r in review['figures']}!=set(plan['train']):raise ValueError('Adopted cell coverage differs')
        metrics=read(run/'metrics.json')
        with np.load(run/'predictions.npz',allow_pickle=False) as arrays:
            for cid in plan['train']:
                x,y,p=(arrays[cid+'__'+k] for k in ('cycle','truth','prediction'));c=cells[cid]
                take=np.isfinite(c['soh'])&(c['soh']>0)
                np.testing.assert_array_equal(x,c['label_cycle'][take]);np.testing.assert_array_equal(y,c['soh'][take])
                actual=curve_metrics_v3(x,y,p)
                for key in ('mape_percent','mae_pp','tail_mae_pp','endpoint_error_pp'):
                    np.testing.assert_allclose(actual[key],metrics[cid][key],rtol=0,atol=1e-9)
                rows.append(dict(dataset=ds,cell_id=cid,**actual,model_sha256=proof['model_sha256'],adopted_head_refinement=adopted))
        chosen.append(dict(dataset=ds,run=str(run.resolve()),report=str(report.resolve()),refined=adopted,
            model_sha256=proof['model_sha256'],mean_mape_percent=result['TRAIN_mape_percent'],mean_mae_pp=result['TRAIN_mae_pp']))
        version='稳定优化重建头' if adopted else '保留首轮'
        lines.append(f"| {ds} | {base['TRAIN_mape_percent']:.3f}%→{result['TRAIN_mape_percent']:.3f}% | {base['TRAIN_mae_pp']:.3f}→{result['TRAIN_mae_pp']:.3f} | {version} | [TRAIN重建PDF]({(report/'all_cells.pdf').resolve().as_posix()}) | {len(plan['train'])} |")
    if len(rows)!=240:raise ValueError('240 TRAIN cells required')
    keys=['dataset','cell_id','mape_percent','mae_pp','tail_mae_pp','endpoint_error_pp','stage_drop_max_pp','model_sha256','adopted_head_refinement']
    with (root/'selected_TRAIN_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines+=['',f"[全部逐电芯指标]({(root/'selected_TRAIN_metrics.csv').resolve().as_posix()}) · [采用模型核验]({(root/'selected_delivery_audit.json').resolve().as_posix()})",'',
        '每个数据集先12轮联合学习，再固定已学习的编码器，最多200轮共享重建头细化；对仍有明显欠拟合的集合，再做最多80步确定性全量梯度L-BFGS。全程按TRAIN宏平均MAE选模型，保留旧版本和全部失败样本。',
        '', '完整输入下的重建允许读取目标时段的真实信号，不能与旧的前100周期预测直接当成同条件算法对照。本次没有证明新电芯预测能力，也没有声称每个噪声点完全重合。',
        '', '| 数据集 | MAPE≤1%电芯 | MAPE≤3%电芯 | MAE≤0.5pp电芯 | 终点误差>3pp电芯 |',
        '|---|---:|---:|---:|---:|']
    for item in chosen:
        ds=item['dataset'];rr=[r for r in rows if r['dataset']==ds];n=len(rr)
        counts=[sum(r[k]<=cut for r in rr) for k,cut in [('mape_percent',1),('mape_percent',3),('mae_pp',.5)]]
        end=sum(r['endpoint_error_pp']>3 for r in rr)
        lines.append(f'| {ds} | {counts[0]}/{n} | {counts[1]}/{n} | {counts[2]}/{n} | {end}/{n} |')
    save(root/'selected_delivery_audit.json',dict(passed=True,TRAIN_cells=240,chosen=chosen,
        original_audit_sha256=fingerprint(root/'delivery_audit.json'),metrics_csv_sha256=fingerprint(root/'selected_TRAIN_metrics.csv'),
        finalizer_sha256=fingerprint(__file__),all_adopted_plots_actually_reviewed=True,reconstruction_not_forecasting=True,
        raw_labels_exact=True,predictive_ability_proven=False))
    (root/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dict(passed=True,TRAIN_cells=240,refined=sum(r['refined'] for r in chosen)))


if __name__=='__main__':main()
