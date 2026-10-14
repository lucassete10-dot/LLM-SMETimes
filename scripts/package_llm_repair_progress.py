"""Package saved P1 evidence without re-evaluating or selecting heldout cells."""
import json,re
from pathlib import Path
from scripts.audit_llm_curve_repair import fingerprint
from scripts.prepare_same_type_forecast import save_json


def main():
    root=Path('experiments/llm_curve_repair');rows=[]
    for file in sorted(root.glob('tiny_*/result.json')):
        result=json.loads(file.read_text());cfg=json.loads((file.parent/'config.json').read_text())
        verify=file.parent/'repair_verification.json'
        rows.append({'run':file.parent.name,'configuration':cfg,'result':result,
            'result_sha256':fingerprint(file),'checkpoint_hash_matches':fingerprint(file.parent/'model.pt')==result['model_sha256'],
            'real_verification_passed':json.loads(verify.read_text()).get('passed') if verify.exists() else None})
    save_json(root/'experiment_ledger.json',{'scope':'TRAIN in-sample P1 only, not generalization accuracy',
        'runs':rows,'P1_all_cells_pass':any(r['result']['tiny_all_pass'] for r in rows),'goal_complete':False})
    running=[str(p.parent.resolve()) for p in root.glob('tiny_*/status.json')
        if json.loads(p.read_text()).get('state')=='training' and not (p.parent/'result.json').exists()]
    status=json.loads((root/'goal_status.json').read_text());status.update(
        latest_P0=json.loads((root/'latest_P0.json').read_text())['folder'],completed_P1_runs=len(rows),
        current_diagnostic_run=running or None,goal_complete=False,
        P1_all_tiny_cells_pass=any(r['result']['tiny_all_pass'] for r in rows),
        next_required='Resolve remaining TRAIN fit and raw-noise constraints before P2; certify independent confirmation provenance')
    save_json(root/'goal_status.json',status)
    sources=sorted(set(Path('scripts').glob('*llm*repair*.py'))|set(Path('scripts').glob('diagnose_llm_*.py'))|
        set(Path('battery_smetimes').glob('llm_*.py'))|
        set(Path('configs').glob('llm_curve_repair*.json'))|set(Path('tests').glob('test_llm_*.py'))|
        {Path('battery_smetimes/mit_units.py'),Path('battery_smetimes/research_llm.py'),Path('tests/test_mit_current_units.py'),Path('scripts/audit_uconn_reserved_test.py')})
    save_json(root/'implementation_manifest.json',{'source_sha256':{str(p):fingerprint(p) for p in sources},'goal_complete':False})
    table='| 运行目录 | 选中 epoch | 平均 MAE / pp | 平均尾段 / pp | 平均阶段 / pp | P1 通过 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in rows:
        d=r['result'];m=d['TRAIN_summary']
        table+=f"| {r['run']} | {d['selected_epoch']} | {m['mae_pp']:.3f} | {m['tail_mae_pp']:.3f} | {m['stage_drop_max_pp']:.3f} | {sum(d['tiny_per_cell_pass'].values())}/12 |\n"
    path=Path('docs/LLM_SMETimes_curve_repair_execution_20261011.md');doc=path.read_text(encoding='utf-8')
    begin=doc.index('## 已完成试验');end=doc.index('\n七参数旧头',begin)
    doc=doc[:begin]+f'## 已完成试验（{len(rows)} 轮）\n\n'+table+doc[end:]
    doc=doc.replace('五轮已完成试验都只使用这些 TRAIN 电芯，未查询本轮验证或留出电芯预测。',
        f'{len(rows)} 轮拟合试验只使用这些 TRAIN 电芯；独立输入依赖审计另含 24 颗 validation 的可见历史干预，不计算其准确率，不用于选候选。')
    doc=doc.replace('34 项模型、积分、输入、线性规划下界与单位转换测试通过。','41 项模型、积分、局部残差、归一化路由、输入、线性规划下界与单位转换测试通过。')
    doc=re.sub(r'\d+ 轮拟合试验只使用',f'{len(rows)} 轮拟合试验只使用',doc)
    doc=re.sub(r'\d+ 项模型、积分、[^。]+测试通过。','47 项模型、积分、局部残差、向量节点读出、分组优化器、路由、输入与单位转换测试通过。',doc)
    doc=doc.replace('当前新头在 TRAIN 上满足原旧联合条件 12/12，但加上本 Goal 的全部严格条件后为 11/12；b1c1 的可信拐点区间条件失败。',
        '原 14 节点查询速率模型在 TRAIN 上满足原旧联合条件 12/12，但加上本 Goal 的全部严格条件后为 11/12；b1c1 的可信拐点区间条件失败。后续模型按各自独立验收文件报告。')
    marker='\n## 新增诊断与算法记录\n'
    if marker in doc:doc=doc[:doc.index(marker)]
    doc+=marker+'''\n在相同真实 LLM 路径中加入可训练的固定物理节点 query；随后仅取消 TRAIN 标签 96 点抽样，再单独增加每 5 循环一个节点、绝对幅度 ≤1 pp、起点严格为零的局部恢复分支。恢复分支同样读取 LLM+专家历史 token，不输入电芯 ID、真实终点、未来标签或答案拟合系数。自由答案诊断没有成为教师。三个改变均有独立配置、源码快照与最多 300 epoch 的新运行。

节点 query 模型的 TRAIN 路由审计显示平均专家利用率约 `[0.00000028, 0.00000048, 0.99999475, 0.00000452]`，熵约 `0.00005084 nats`，所有有效 token 的最大路由概率均超过 0.999。可确认该检查点路由饱和，不能据此断言它是全部预测失败的唯一原因。单独路由归一化实验保留失败或改善的实际结果，不凭更均匀的利用率宣布泛化收益。

`tiny_query_rate_20261011T065232_833163Z/visible_input_interventions.json` 对 TRAIN 12 与 validation 24 的固定未来查询开展历史顺序、历史替换、已知工况替换；平均最大输出变化分别约 6.70、17.07、24.74 pp。查询不取真实终点，不计算目标误差。底层旧加载器仍读取标签用于完整性；标签在推理前删除，不能将这项检查表述为整个进程从不读取标签。输出变化只证明依赖，部分替换属于分布外组合，不能当作准确率或因果收益。

局部恢复模型 `tiny_query_rate_local_20261011T070355_897333Z`：平均 TRAIN MAE 0.15329 pp、尾段 0.21237 pp、阶段 0.22601 pp，P1 10/12。相同父配置无局部恢复的完整标签模型平均 MAE 0.25706 pp，P1 3/12。两者均是固定 12 颗 TRAIN 同样本拟合。剩余 b1c0 MAE 0.28445 pp；b3c43 阶段误差 0.67793 pp。新分支解除旧累计恢复预算的数学限制，但未证明能从早期输入预测未来随机尖峰。

局部恢复模型已通过真实预训练主干重载、单点/子集/乱序、未来污染与标签删除检查。全部 12 颗原始预测图已导出 PNG/PDF/SVG 与合集；逐点原始真值不平滑。P1、旧联合通过与最终严格条件分别统计，禁止混用。机器账本见 `experiment_ledger.json`，当前状态见 `goal_status.json`。
'''
    doc+='\n后续单因素对照（均按预注册 P1 最大阈值比选中检查点，不按最低平均误差临时改选）：\n\n'
    for r in rows:
        c=r['configuration'];d=r['result'];m=d['TRAIN_summary']
        if c['architecture']=='query_rate_local' and c.get('normalize_router',False):
            doc+=f"- `{r['run']}`：局部读出={c.get('local_readout','query')}，局部 LR 倍数={c.get('local_decoder_lr_scale',1.)}，路由归一化={c.get('normalize_router',False)}，学习率={c.get('learning_rate_schedule','constant')}，逐点 Huber 转折={c.get('point_huber_beta_pp',1.)} pp；选中 {d['selected_epoch']} epoch；平均 MAE/尾段/阶段 {m['mae_pp']:.5f}/{m['tail_mae_pp']:.5f}/{m['stage_drop_max_pp']:.5f} pp；P1 {sum(d['tiny_per_cell_pass'].values())}/12；失败电芯 {', '.join(k for k,v in d['tiny_per_cell_pass'].items() if not v)}。\n"
    doc+='\n当前 14 个速率节点＋5 循环局部残差、1 pp 幅度的自由系数 TRAIN b1c0 拟合可达到 0.19686 pp（全局饱和恢复固定为零）。这证明存在一个可行的表达，但不证明端到端能学到或早期信号能预测尖峰；该结果没有进入模型输入、教师或权重。证据保存在 `P0_20261011T070847_841394Z/TRAIN_b1c0_configured_local_capacity_NOT_FORECAST.json`。\n'
    doc+='\nP1 入口也已强制最多 300 epoch；实际检查 `--epochs 301` 在创建模型前拒绝。全部完成状态继续严格按原 P0–P4 Goal 判定。\n'
    for r in rows:
        acceptance=root/r['run']/'strict_acceptance.json'
        if r['configuration'].get('point_huber_beta_pp')==.1 and acceptance.exists():
            accepted=json.loads(acceptance.read_text())['training']['by_cell'];count=sum(v['strict_joint_pass'] for v in accepted.values())
            doc+=f"\n`{r['run']}` 在 {len(accepted)} 颗 TRAIN 同样本上通过最终阈值口径 {count}/{len(accepted)}，实际重放检查={r['real_verification_passed']}，P1={sum(r['result']['tiny_per_cell_pass'].values())}/12。两种门槛不同；这不是 P4 全名单成功，也不证明未见电芯精度。\n"
    doc+='\n局部参数审计采用固定 0–3200 循环未来节点，不根据测试终点选择网格；全网格饱和比例包含超出某些电芯实际标签覆盖的外推，不能直接当作其尾段饱和证据。b1c0 的短于 20 循环频谱功率占比从查询读出的约 0.35% 增加到向量读出的约 8.08%，MAE 从 0.26870 降到 0.24508 pp。这是表达改变与 TRAIN 拟合证据，尚未达到 P1，也未证明早期预测随机尖峰的能力。每个节点系数由同一真实 LLM 历史注意力读取结果输出，不包含电芯 ID 或自由答案教师。\n'
    doc+='\nUConn 保留名单补充审计：`P0_20261011T072421_376067Z/UConn_reserved_membership_audit.json` 核对旧 sparse probe 的 16 个本地模型训练成员与 16 个预测存档键；11 颗预留 test 均未进入这些拟合和预测。检查没有运行树模型推理、提取教师或读取预留目标曲线。它证明该次存档的排除关系，不能证明全部外部历史均未查看；旧配置还显式写 independent_test=false。因此独立确认仍未认证，不能把名单直接当盲测。\n'
    doc+='\n向量局部头 LR×5、5 循环节点的诊断 `tiny_query_rate_local_20261011T072510_043572Z` 达到 P1 11/12，平均 MAE/尾段/阶段为 0.09001/0.09432/0.16549 pp，剩余 b1c0 MAE 0.24618 pp。相对其父配置，唯一变化是局部头学习率，主干与其余参数学习率不变。分组优化器重载保持倍数，日志保存各组学习率。\n'
    doc+='\n2.5 循环节点的自由系数 b1c0 表达能力可到约 0.14565 pp，但其实际端到端检查点 `tiny_query_rate_local_20261011T072817_994325Z` 的 P1 为 9/12，平均 MAE/尾段/阶段为 0.11638/0.19929/0.26450 pp。没有因为单芯误差降低就把它视为胜出配置。下一步应区分局部系数的优化瓶颈与可见表示的个体分辨能力，保持 ≤300 epoch、新运行、原始标签与全分母；不靠无限训练或放宽 P1 推进。\n'
    doc+='\n新导出的 `figures_v2` 补齐真实 fold、seed 和 MAPE，保留原图目录；PNG/PDF/SVG 和全部 12 颗合集均经过对齐、PDF 文字与碰撞检查，并审阅全范围 contact sheet。没有对齐答案、平滑原始容量尖峰或删失败尾段。\n'
    path.write_text(doc,encoding='utf-8');(root/'README.md').write_text(doc,encoding='utf-8')
    print(json.dumps({'completed_runs':len(rows),'checkpoint_hashes_match':all(r['checkpoint_hash_matches'] for r in rows),
        'P1_passed':status['P1_all_tiny_cells_pass'],'goal_complete':False},indent=2))


if __name__=='__main__':main()
