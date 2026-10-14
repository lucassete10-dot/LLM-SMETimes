# Windows 下载和运行说明

本文说明国内用户如何从魔搭社区下载 1B/3B 模型，并在本项目中运行。

## 0. Python 环境

本项目自带虚拟环境，位于项目根目录下的 `.venv`（Python 3.12）：

```text
D:\LLM\SMETimes-main-WINDOWS\.venv\Scripts\python.exe
```

重建方式：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**注意：RTX 50 系（Blackwell）显卡必须装 cu128 版本的 torch。**
老版本（cu121/cu124）没有 `sm_120` 内核，运行时会有
`no kernel image is available for execution on the device` 的报错：

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.9.1+cu128" `
  --index-url https://download.pytorch.org/whl/cu128
```

## 1. 先安装 ModelScope

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install modelscope
```

## 2. 下载 1B 模型

推荐先用 1B 跑通代码。1B 对 8GB 显存更友好。

魔搭模型：

```text
LLM-Research/Llama-3.2-1B-Instruct
```

下载命令：

```powershell
.\.venv\Scripts\python.exe -c "from modelscope import snapshot_download; snapshot_download('LLM-Research/Llama-3.2-1B-Instruct', local_dir=r'.\models\llm\llama-3.2-1b-instruct')"
```

下载后本地目录应为：

```text
D:\LLM\SMETimes-main-WINDOWS\models\llm\llama-3.2-1b-instruct
```

## 3. 下载 3B 模型

如果要更接近论文设置，可以下载 3B。3B 更吃显存，8GB 显存基本会 OOM，
建议 12GB 显存起步。

魔搭模型：

```text
LLM-Research/Llama-3.2-3B-Instruct
```

下载命令：

```powershell
.\.venv\Scripts\python.exe -c "from modelscope import snapshot_download; snapshot_download('LLM-Research/Llama-3.2-3B-Instruct', local_dir=r'.\models\llm\llama-3.2-3b-instruct')"
```

下载后本地目录应为：

```text
D:\LLM\SMETimes-main-WINDOWS\models\llm\llama-3.2-3b-instruct
```

## 4. 检查模型是否下载成功

检查 1B：

```powershell
.\.venv\Scripts\python.exe -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained(r'.\models\llm\llama-3.2-1b-instruct'); print(c.model_type, c.hidden_size)"
```

检查 3B：

```powershell
.\.venv\Scripts\python.exe -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained(r'.\models\llm\llama-3.2-3b-instruct'); print(c.model_type, c.hidden_size)"
```

能输出 `llama 2048` 或 `llama 3072` 就说明模型目录可用。

## 5. 预处理

用 1B：

```powershell
.\.venv\Scripts\python.exe .\preprocess.py `
  --dataset ETTh1 `
  --llm_ckp_dir .\models\llm\llama-3.2-1b-instruct `
  --device auto `
  --batch_size 1 `
  --num_workers 0
```

用 3B：

```powershell
.\.venv\Scripts\python.exe .\preprocess.py `
  --dataset ETTh1 `
  --llm_ckp_dir .\models\llm\llama-3.2-3b-instruct `
  --device auto `
  --batch_size 1 `
  --num_workers 0
```

预处理会生成：

```text
dataset\ETT-small\ETTh1.pt
```

**换模型后必须重新预处理**，否则时间戳嵌入维度和训练模型对不上。

> 注：仓库里已有的 `ETTh1.pt` 形状为 `[17420, 2048]`（fp16），
> 正是 1B 模型（hidden_size 2048）生成的。所以只要用
> `llama-3.2-1b-instruct`，这一份可以直接复用，不必重跑预处理。

**覆盖保护**：如果目标 `.pt` 已存在，`preprocess.py` 会拒绝覆盖并打印现有
形状，需要用 `--force` 才会重写。这是防止用不同模型（尤其是 smoke test 用的
tiny 模型，hidden_size 只有 32）把宝贵的嵌入文件覆盖掉。
若确实要重跑，加 `--force`；若只想写到别处，用 `--root_path`。

## 6. 训练

用 1B：

```powershell
.\scripts\long_time_series_forecasting\SMETimes_ETTh1_windows.ps1 `
  -LlmCkpDir .\models\llm\llama-3.2-1b-instruct `
  -BatchSize 4 `
  -TrainEpochs 2
```

用 3B：

```powershell
.\scripts\long_time_series_forecasting\SMETimes_ETTh1_windows.ps1 `
  -LlmCkpDir .\models\llm\llama-3.2-3b-instruct `
  -BatchSize 1 `
  -TrainEpochs 2
```

如果显存不足，把 `-BatchSize` 改成 `1`。

`SMETimes_ETTh1_windows.ps1` 默认会自动使用项目内的 `.venv`，
也可以用 `-Python` 显式指定解释器。

## 7. 模型选择建议

- 只是想跑通代码：下载 1B。
- 8GB 显存（如 RTX 5060 Laptop）：优先 1B，`-BatchSize` 从 1~4 试。
- 想尽量复现论文：下载 3B。
- 3B 建议 12GB 或更高显存。

魔搭社区链接：

- 1B：https://modelscope.cn/models/LLM-Research/Llama-3.2-1B-Instruct
- 3B：https://modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct
