"""One-time, lossless scheduling migration after stopping only the controller."""
import json, shutil
from datetime import datetime, timezone
from pathlib import Path
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint

def main():
    root=Path('experiments/llm_curve_full/20261011_MIT_then_other_datasets')
    path=root/'queue.json';queue=json.loads(path.read_text(encoding='utf-8'))
    tasks=queue['tasks'];done=[t for t in tasks if t['state']=='complete']
    active=[t for t in tasks if t['state'] not in ('pending','complete')]
    if len(active)!=1:raise ValueError('Expected exactly one attached training task')
    pending=[t for t in tasks if t['state']=='pending'];mvp=[]
    for dataset in queue['sequence']:
        if any(t['dataset']==dataset for t in done+active):continue
        task=next(t for t in pending if t['dataset']==dataset)
        mvp.append(task);pending.remove(task)
    ordered=done+active+mvp+pending
    key=lambda t:(t['dataset'],t['fold'],t['seed'])
    assert len(ordered)==len(tasks)==len(set(map(key,ordered)))
    assert set(map(key,ordered))==set(map(key,tasks))
    assert all(t in ordered for t in tasks)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    shutil.copy2(path,root/('queue_before_mvp_'+stamp+'.json'))
    queue.update(tasks=ordered,schedule_mode='mvp_first',mvp_reports={},
        schedule_update={'utc':stamp,'user_request':'先出其他数据集最小MVP，再补轮数和算法优化',
            'mvp_definition':'One original fold and seed; unchanged full TRAIN/VAL/TEST cell lists and training budget',
            'controller_sha256':fingerprint(Path('scripts/run_llm_full_dataset_queue.py')),
            'baseline_runs_preserved':len(tasks),'active_run_preserved':active[0]['run'],
            'next_mvp_tasks':[key(t) for t in mvp]})
    save_json(path,queue)
    doc=Path('docs/LLM_SMETimes_training_optimization_goal_20261012.md')
    content=doc.read_text(encoding='utf-8')
    content=content.replace('先完成 MIT 的训练和优化，再依次进入其他数据集。','按最新指示，先让所有数据集产出最小 MVP，再补齐各数据集折次、种子和算法优化。')
    content=content.replace('在该数据集基线跑完之后、进入下一数据集之前执行；','所有数据集 MVP 完成后，在该数据集完整基线跑完之后执行；')
    content=content.replace('按 MIT → XJTU → HUST → UConn → NCA 顺序完成基线、优化和报告。','优先保留当前 MIT 训练，随后 XJTU → HUST → UConn → NCA 各跑首折 seed 20260930 并产出 MVP 报告；之后按原数据集顺序补齐基线和优化。')
    content+='\n\n## 最新调度：先交付各数据集 MVP\n\n用户最新指示优先于此前“MIT 全部完成后再进入其他数据集”的顺序。已有 MIT 六次完成结果保留，当前第七次不打断。其他四个数据集各先跑一折、一种子；仍用原冻结划分、全部训练电芯、最多 300 epoch 与原验证提前停止，不缩减成几颗电芯的拟合演示。每次完成后输出原始测试曲线与 MAPE≤5%通过比例、失败和覆盖情况。单种子不画跨种子 SD，不称为全数据集完成或稳定性结论。MVP 覆盖的是该折留出测试名单，不是全部源电芯的交叉验证测试。\n\n完整基线仍为 85 次，算法优化仍需执行；只是将四次其他数据集基线提前，不增加重复训练。全部 MVP 产出后自动继续补轮数和优化。GPU 训练始终串行。\n'
    doc.write_text(content,encoding='utf-8')
    print(json.dumps(queue['schedule_update'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
