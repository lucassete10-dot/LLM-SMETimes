"""User-authorized three-seed migration; original completed extras stay archived."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint

def main():
    root=Path('experiments/llm_curve_full/20261011_MIT_then_other_datasets');read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    seeds=[20260930,20260931,20260932];path=root/'queue.json';q=read(path)
    if q.get('seed_count')==3 and q['seeds']==seeds and len(q['tasks'])==51:
        print('Three-seed migration already applied; archives unchanged');return
    for state in (root/'optimization').glob('*/state.json'):
        if read(state).get('screening'):raise ValueError('Optimization already started; registration must not be replaced')
    old=q['tasks'];active=[t for t in old if t['state'] not in ('pending','complete')]
    assert len(active)==1 and active[0]['seed'] in seeds
    extras=[t for t in old if t['seed'] not in seeds];assert all(t['state'] in ('complete','pending') for t in extras)
    kept=[t for t in old if t['seed'] in seeds];assert len(kept)==51
    assert len({(t['dataset'],t['fold'],t['seed']) for t in kept})==51
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');shutil.copy2(path,root/f'queue_before_three_seeds_{stamp}.json')
    save_json(root/'historical_extra_seed_tasks.json',{'completed_extras':[t for t in extras if t['state']=='complete'],'cancelled_pending':[t for t in extras if t['state']=='pending'],'reason':'User reduced required seeds to3; no artifacts deleted','excluded_from_current_formal_statistics':True})
    q.update(tasks=kept,seeds=seeds,seed_count=3,historical_extra_seed_archive='historical_extra_seed_tasks.json',
        seed_update={'utc':stamp,'authorization':'用户：我把种子数改成了3，然后mape值改成了3%','baseline_runs':51,'completed_extras_preserved':sum(t['state']=='complete' for t in extras),'active_run_preserved':active[0]['run']})
    save_json(path,q)
    base=Path('configs/llm_curve_full_v1.json');cfg=read(base);cfg['run_seeds']=seeds;save_json(base,cfg)
    plan=Path(q['optimization_plan']);shutil.copy2(plan,plan.with_name(plan.stem+'_five_seed_archive.json'))
    cfg=read(plan);cfg.update(version='20261012_three_seeds_mape3',baseline_runs=51,maximum_extra_runs=66,full_replication='Confirmed winner replicated on every original fold and all three seeds, reusing three confirmation fits.');cfg['numeric_thresholds']['mape_percent']=3.;save_json(plan,cfg)
    for registration in (root/'optimization').glob('*/registration.json'):
        shutil.copy2(registration,registration.with_name('registration_before_three_seeds.json'))
        cfg=read(registration);cfg.update(plan_sha256=fingerprint(plan),required_seeds=seeds,primary_mape_target_percent=3.,execution='After all MVPs and the complete three-seed baseline of this dataset; validation only');save_json(registration,cfg)
    doc=Path('docs/LLM_SMETimes_training_optimization_goal_20261012.md');s=doc.read_text(encoding='utf-8')
    for a,b in [('5 折 × 5 种子','5 折 × 3 种子'),('共 25 次','共 15 次'),('× 5 种子','× 3 种子'),('全部折次和五种子','全部折次和三种子'),('五种子均值','三种子均值'),('全折全种子','全折全种子'),('基线合计 85 次','基线合计 51 次'),('最多 185 次','最多 117 次'),('最多新增 100 次','最多新增 66 次'),('完整基线仍为 85 次','完整基线仍为 51 次'),('MAPE≤5%','MAPE≤3%'),('MAPE 阈值改为 5%','MAPE 阈值改为 3%')]:s=s.replace(a,b)
    s=s.replace('| 25 |','| 15 |').replace('固定划分 × 3 种子 | 5 |','固定划分 × 3 种子 | 3 |')
    s='\n'.join(line for line in s.splitlines() if not line.startswith('最新目标附件 8f0c7452'))
    s+='\n\n最新用户指示统一采用三个种子（20260930–20260932）、MAPE≤3%，验证通过数保护条件也同步为3%。算法对照尚未开始，更新前的登记与队列已归档。完整基线为51次，最多新增66次有界优化；已完成的四次额外MIT种子保留为历史结果，排除于当前正式三种子统计，当前正在训练的运行继续。总执行次数上限另加这四次历史训练。原冻结电芯名单、折次、前100源循环预算及全部五个数据集不变。\n'
    doc.write_text(s,encoding='utf-8')
    status=Path('experiments/llm_curve_repair/goal_status.json');cfg=read(status);cfg.update(required_seed_count=3,required_seeds=seeds,primary_mape_goal_percent=3.,baseline_runs=51,policy='Three seeds and MAPE3 throughout; old extra seeds remain historical; all five datasets and registered comparisons still required');save_json(status,cfg)
    print(json.dumps(q['seed_update'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
