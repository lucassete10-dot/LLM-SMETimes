# LLM + SMETimes 已保存结果导出合同

目的：让用户直接查看真实预训练 LLM 主模型 E1 的已保存未来 SOH 预测，不混入树模型或数值 Transformer。

来源：research_v5/runs 下 llm_e1/test/predictions.npz，要求 llm.json 证明 pretrained=true、use_statistics=true、experts=4。MIT 是历史开发留出结果，部分种子未完成；XJTU 是单种子冒烟实验，不能称正式完整训练。没有 HUST、UConn 或 NCA 的同架构结果，不用其他模型补齐。

结构：单电芯、单输入长度、单折一张图。只在同一折内平均实际已保存的种子，标注实际 n；不跨折混合、不挑最好种子。全部失败案例保留。源点、真实值、预测不做平滑、裁剪或重新归一化。沿用保存的容量参考，不与新 nominal-reference 模型混为同一结果。

证据：黑线原始真实 SOH、蓝线种子预测均值，多个种子时显示 ±1 seed SD，明确不是置信区间。标注 E1、N、折和开发/冒烟身份，平均 MAPE 不代表曲线通过。导出 PNG 300 dpi、8×5 inch；PDF/SVG 留作 QA，不宣称期刊生产稿。

采用此前已确认的 Python/matplotlib 后端和已有单图布局。每张执行单面板对齐适用性记录、PDF 字体及碰撞检查；整体逐张人工检查未完成时如实标注。仅导出已有结果，不重新训练，不改变目标状态。ZIP 校验 CRC、PNG 数目与清单。
