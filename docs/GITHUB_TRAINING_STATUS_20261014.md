# GitHub上传时的训练状态快照

记录时间：2026-10-14T19:40:35+08:00（Asia/Shanghai）。这是上传时快照，不是GitHub端实时训练服务。

当前队列阶段：screen；当前任务：XJTU_K_XJTU_fold1_20260930_120。
已完成新模型：1/10个已登记任务。队列可能按验证结果增加三种子确认和其他折。

五数据集基线诊断已完成；v4行为测试10项通过。原有v3有限训练/评估已完成，但精度只部分达到。

| 数据集 | 候选 | 所属阶段 | 完成状态 | 选中轮数 | VAL评分 |
|---|---|---|---|---:|---:|
| MIT | K | screen | complete | 80 | 1.86503 |
| XJTU | K | screen | training | — | 待完成 |
| HUST | K | screen | pending | — | 待完成 |
| UConn_ISU_ILCC | K | screen | pending | — | 待完成 |
| Zhu2022_NCA | K | screen | pending | — | 待完成 |
| MIT | L | screen | pending | — | 待完成 |
| XJTU | L | screen | pending | — | 待完成 |
| HUST | L | screen | pending | — | 待完成 |
| UConn_ISU_ILCC | L | screen | pending | — | 待完成 |
| Zhu2022_NCA | L | screen | pending | — | 待完成 |

K仅增加固定物理速率节点分辨率；L仅提高尾部/阶段监督权重。只用VAL选择候选；有效候选才补齐3种子。单种子训练完成不代表正式三种子验收通过。

最新参考为XJTU b1_2C_battery-4图。主验收仍是MAPE≤3%，进一步追赶参考图的全程、尾部、转折和终点质量。不能宣称当前每颗电芯都已达到该图片水平。

[代码阅读入口](GITHUB_PROJECT_GUIDE.md) / [持续优化Goal](LLM_SMETimes_reference_tracking_goal_20261014.md)
