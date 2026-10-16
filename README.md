# LLM + SMETimes — Battery SOH Forecasting

本项目研究如何使用电池前100个实际源周期的可见信号，经预训练LLM与SMETimes融合/专家模块，预测后续SOH曲线。源码包含五数据集训练、验证选择、曲线解码、输入隔离核验和实验报告。

**给 ChatGPT 或首次阅读本项目的人：请先读 [项目代码阅读指南](docs/GITHUB_PROJECT_GUIDE.md)。**

**2026-10-16 新项目：[完整信号 SOH 估计 v0.1](projects/full_signal_soh/README.md)**。包含五数据集240颗TRAIN重建、5颗未训练验证电芯的曲线和便携推理入口。适配权重与PDF图集见[Release](https://github.com/lucassete10-dot/LLM-SMETimes/releases/tag/full-signal-soh-v0.1)。这一版是同一时段SOH估计，不能视为前100周期预测未来已经成功。

- [详细项目概况](docs/项目概况_供ChatGPT学习指导.md)
- [当前持续优化 Goal](docs/LLM_SMETimes_reference_tracking_goal_20261014.md)
- [当前训练状态快照](docs/GITHUB_TRAINING_STATUS_20261014.md)
- [最新已完成 v3 五数据集报告](docs/LLM_SMETimes_v3_five_dataset_results_20261013.md)

公开仓库包含源码、配置、测试和说明；大型原始数据、LLM/训练权重和实验缓存保存在原工作区。报告中本地绝对路径指向研究记录，不能作为公开下载链接。当前曲线精度只部分达到目标，v4优化正在进行。

## Windows Run Guide

This project can run on Windows after generating the timestamp embedding `.pt`
files and providing a Hugging Face language model.

## Python Environment

The environment lives inside the project at `.venv` (Python 3.12):

```powershell
.\.venv\Scripts\python.exe
```

To recreate it from scratch:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**GPU note (important on RTX 50-series / Blackwell):** PyTorch must be installed
from the CUDA 12.8 index. Wheels built against older CUDA (cu121/cu124) have no
`sm_120` kernels and fail at runtime with
`no kernel image is available for execution on the device`:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.9.1+cu128" `
  --index-url https://download.pytorch.org/whl/cu128
```

Verify the GPU is usable before training:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Free Model Choices

Recommended Llama-compatible free model:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Other supported free choices:

```text
openai-community/gpt2-large
facebook/opt-1.3b
```

Download TinyLlama to a local folder:

```powershell
.\.venv\Scripts\python.exe .\download_model.py `
  --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --local_dir .\models\llm\tinyllama-1.1b-chat
```

You can also pass the Hugging Face repo id directly to `--llm_ckp_dir`; Transformers
will download it to the Hugging Face cache automatically.

## Quick Smoke Test

This uses a tiny random Llama model to verify the Windows/Python pipeline quickly.
It is only for code validation, not for meaningful forecasting quality.

```powershell
.\scripts\long_time_series_forecasting\SMETimes_ETTh1_windows.ps1 -Smoke
```

Smoke mode runs the whole pipeline against a scratch copy of the CSV in
`dataset\_smoke\`, because the tiny model's hidden size (32) does not match the
real embeddings (2048) and `preprocess.py` writes to a fixed filename. The real
`dataset\ETT-small\ETTh1.pt` is never touched. Delete `dataset\_smoke\` when done.

The same hazard applies to any manual run: `preprocess.py` refuses to overwrite
an existing `.pt` unless you pass `--force`, and `--root_path` redirects the
output elsewhere.

```powershell
# normal re-run is now safe by default -- it stops with a message instead of clobbering
.\.venv\Scripts\python.exe .\preprocess.py --dataset ETTh1 --llm_ckp_dir .\models\llm\llama-3.2-1b-instruct

# deliberate re-generation
.\.venv\Scripts\python.exe .\preprocess.py --dataset ETTh1 --llm_ckp_dir .\models\llm\llama-3.2-1b-instruct --force
```

## Run ETTh1 With TinyLlama

Generate timestamp embeddings first. The `.pt` file is saved next to the CSV:

```powershell
.\.venv\Scripts\python.exe .\preprocess.py `
  --dataset ETTh1 `
  --llm_ckp_dir .\models\llm\tinyllama-1.1b-chat `
  --device auto
```

Train and test:

```powershell
.\.venv\Scripts\python.exe .\run.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path .\dataset\ETT-small\ `
  --data_path ETTh1.csv `
  --model_id ETTh1_672_96 `
  --model SMETimes_Llama `
  --data ETTh1 `
  --seq_len 672 `
  --label_len 576 `
  --token_len 96 `
  --test_seq_len 672 `
  --test_label_len 576 `
  --test_pred_len 96 `
  --batch_size 64 `
  --learning_rate 0.001 `
  --mlp_hidden_layers 0 `
  --train_epochs 2 `
  --gpu 0 `
  --device auto `
  --llm_ckp_dir .\models\llm\tinyllama-1.1b-chat `
  --num_workers 0 `
  --cosine `
  --tmax 10 `
  --mix_embeds
```

## Notes

- If you change `--llm_ckp_dir`, rerun the matching preprocess script so the
  `.pt` embedding dimension matches the training model.
- Use `preprocess_gpt.py` with `SMETimes_Gpt2`, and `preprocess_opt.py` with
  `SMETimes_Opt_1b`.
- On Windows, `--num_workers 0` is the safest DataLoader setting.
- The original bash scripts remain for Linux; use the `.ps1` script on Windows.
- With 8 GB of VRAM, keep `--batch_size` low (1-4 depending on model size).
