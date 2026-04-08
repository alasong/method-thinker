# MethodThinker 训练快速启动指南

> 更新日期: 2026-04-08 | 适用于 QLoRA 训练模式

---

## 快速开始

### Colab 一键训练

```python
# === Cell 1: 环境准备 ===
!git clone https://github.com/alasong/method-thinker.git
%cd method-thinker
!pip install -q transformers accelerate peft datasets bitsandbytes trl

# === Cell 2: 检查GPU ===
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# === Cell 3: 开始训练 ===
!python scripts/train_sft.py --use-lora --epochs 3 --batch-size 4
```

### 本地 GPU 训练

```bash
# 安装依赖
pip install transformers accelerate peft datasets bitsandbytes trl

# 开始训练
python scripts/train_sft.py --use-lora --epochs 3 --batch-size 4
```

---

## 训练参数

### 推荐配置

| GPU | batch_size | max_length | lora_r | 预计时间 |
|-----|------------|------------|--------|---------|
| T4 16GB | 4 | 2048 | 16 | 10-30分钟 |
| RTX 3090 24GB | 8 | 4096 | 32 | 5-15分钟 |
| A100 40GB | 16 | 4096 | 64 | 3-10分钟 |

### 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--use-lora` | 必须启用 | QLoRA模式，节省显存 |
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 4 | 批大小 |
| `--learning-rate` | 5e-5 | 学习率 |
| `--max-length` | 2048 | 最大序列长度 |
| `--lora-r` | 16 | LoRA秩 |
| `--output-dir` | outputs/checkpoints | 输出目录 |

### 训练模式

```bash
# 方法论注入训练（推荐首选）
python scripts/train_sft.py --mode methodology-injection --use-lora

# 多样性训练
python scripts/train_sft.py --mode diversity --methods-per-problem 4 --use-lora

# 反思强化训练
python scripts/train_sft.py --mode reflection --use-lora

# 完整训练流程（三阶段依次执行）
python scripts/train_sft.py --mode full --use-lora
```

---

## 显存优化

### 当前方案：QLoRA (4-bit量化)

代码已内置4-bit量化，自动启用：
- BitsAndBytes 4-bit量化
- LoRA微调（仅训练q_proj/v_proj）
- 自动GPU检测和配置

### OOM 解决方案

```bash
# 方案1: 减小batch_size
--batch-size 2

# 方案2: 减小序列长度
--max-length 1024

# 方案3: 减小LoRA秩
--lora-r 8
```

---

## 模型保存

### Colab 保存到 Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# 保存模型
!cp -r outputs/checkpoints/final /content/drive/MyDrive/method-thinker-models/v1
```

### 上传到 HuggingFace Hub

```python
from huggingface_hub import HfApi, login

login(token="your-hf-token")

api = HfApi()
api.upload_folder(
    folder_path="outputs/checkpoints/final",
    repo_id="your-username/method-thinker-v1",
    repo_type="model"
)
```

---

## 常见问题

### Q1: CUDA out of memory

**解决**: 使用QLoRA模式，已内置4-bit量化。如果仍OOM：
```bash
--batch-size 2 --max-length 1024
```

### Q2: 训练很慢/CPU训练

**检查GPU是否连接**:
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
```

Colab: 运行时 → 更改运行时类型 → T4 GPU

### Q3: 模型下载慢

**使用镜像**:
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### Q4: warmup_ratio deprecated 警告

已修复，使用 `warmup_steps` 替代。

### Q5: 类型比较错误 (float/str)

已修复，所有配置参数强制类型转换。

---

## 训练流程检查清单

- [ ] GPU已连接（`torch.cuda.is_available()` 返回True）
- [ ] 依赖已安装（transformers, peft, trl, bitsandbytes）
- [ ] 训练数据存在（`data/train_data/train.json`）
- [ ] 使用 `--use-lora` 参数
- [ ] batch_size适合GPU显存

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [cloud_training_guide.md](cloud_training_guide.md) | 详细云端训练指南 |
| [colab_automation_guide.md](colab_automation_guide.md) | Colab自动化脚本 |
| [training_config.yaml](../configs/training_config.yaml) | 训练配置文件 |

---

## 下一步

训练完成后：

1. **验证模型**: `python scripts/run_validation.py --model outputs/checkpoints/final`
2. **基准测试**: `python scripts/run_evaluation.py --benchmark aime25`
3. **迭代训练**: 运行多轮迭代提升性能

详见: [项目总结](project_summary.md)