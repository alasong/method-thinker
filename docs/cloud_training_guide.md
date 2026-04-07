# MethodThinker 云端GPU训练指南

> 本指南详细介绍如何在云端GPU平台（Google Colab、Lambda Labs、RunPod）上进行MethodThinker模型训练。
> 
> 更新日期：2026-04-07

---

## 目录

1. [Google Colab训练指南](#1-google-colab训练指南)
2. [Lambda Labs训练指南](#2-lambda-labs训练指南)
3. [RunPod训练指南](#3-runpod训练指南)
4. [通用环境设置命令](#4-通用环境设置命令)
5. [训练启动命令](#5-训练启动命令)
6. [模型下载与保存](#6-模型下载与保存)
7. [常见问题与解决方案](#7-常见问题与解决方案)

---

## 1. Google Colab训练指南

### 1.1 为什么选择Colab

| 特性 | 说明 |
|-----|------|
| **免费T4 GPU** | 16GB显存，适合1.5B-7B模型训练 |
| **便捷访问** | 无需本地GPU，浏览器即可操作 |
| **预装环境** | PyTorch、CUDA已预装 |
| **Google Drive集成** | 模型保存便捷 |

### 1.2 Colab使用步骤

#### Step 1: 创建新笔记本

1. 访问 [Google Colab](https://colab.research.google.com/)
2. 点击 "新建笔记本" (New Notebook)
3. 或直接使用项目提供的模板：`notebooks/train_colab.ipynb`

#### Step 2: 启用GPU

```python
# 在第一个Cell中运行，检查GPU状态
!nvidia-smi
```

如果未显示GPU信息：
1. 点击菜单 `运行时` → `更改运行时类型`
2. 选择 `T4 GPU`（免费）或 `A100 GPU`（付费）
3. 点击 `保存`

#### Step 3: 验证GPU规格

```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU型号: {torch.cuda.get_device_name(0)}")
print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"支持BF16: {torch.cuda.is_bf16_supported()}")
```

### 1.3 Colab环境设置

#### 方式A: 快速设置（推荐）

```python
# === Cell 1: 克隆项目 ===
!git clone https://github.com/your-repo/method-thinker.git
%cd method-thinker

# === Cell 2: 安装依赖 ===
!pip install -q torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers>=4.40.0 accelerate>=0.25.0 peft>=0.7.0 trl>=0.7.0 datasets>=2.14.0
!pip install -q pyyaml pydantic numpy scipy sympy pandas tqdm rich python-dotenv

# === Cell 3: 验证安装 ===
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print("环境验证完成!")
```

#### 方式B: 从requirements.txt安装

```python
!git clone https://github.com/your-repo/method-thinker.git
%cd method-thinker
!pip install -r requirements.txt
```

### 1.4 Colab训练命令

#### 方法论注入训练（T4优化）

```python
# 使用Colab优化配置
!python scripts/train_sft.py \
    --config configs/training_config_colab.yaml \
    --mode methodology-injection \
    --kb data/methodology_kb/v0/math_methods.yaml \
    --use-lora --lora-r 16
```

#### 参数说明（T4 16GB优化）

| 参数 | Colab推荐值 | 说明 |
|-----|-------------|------|
| `--batch-size` | 4-8 | T4显存限制 |
| `--max-length` | 2048-4096 | 根据显存调整 |
| `--gradient-accumulation-steps` | 4-8 | 补偿小batch |
| `--use-lora` | 必须启用 | 节省显存 |
| `--lora-r` | 16-32 | LoRA秩 |
| `bf16` | True | T4支持BF16 |

### 1.5 Colab模型保存

#### 保存到Google Drive

```python
# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 保存模型到Drive
!cp -r outputs/checkpoints/final /content/drive/MyDrive/method-thinker-model/
```

#### 保存到HuggingFace Hub

```python
from huggingface_hub import HfApi

# 登录HF Hub（需要token）
from huggingface_hub import login
login(token="your-hf-token")  # 或使用 !huggingface-cli login

# 上传模型
api = HfApi()
api.upload_folder(
    folder_path="outputs/checkpoints/final",
    repo_id="your-username/method-thinker-v1",
    repo_type="model"
)
```

### 1.6 Colab注意事项

| 问题 | 解决方案 |
|-----|----------|
| **会话超时** | 12小时后会断开，定期保存检查点 |
| **显存不足** | 减小batch_size或启用gradient_checkpointing |
| **下载慢** | 使用 `HF_ENDPOINT=https://hf-mirror.com` 镜像 |
| **连接断开** | 使用脚本自动重连（见FAQ） |

---

## 2. Lambda Labs训练指南

### 2.1 为什么选择Lambda Labs

| 特性 | 说明 |
|-----|------|
| **高性能GPU** | A100 40GB/80GB, RTX 3090/4090 |
| **预装深度学习环境** | PyTorch、CUDA、常用库已配置 |
| **持久存储** | 数据不会丢失 |
| **按小时计费** | A100约$1.10/hr |
| **SSH访问** | 完整终端控制 |

### 2.2 Lambda Labs使用步骤

#### Step 1: 注册并创建实例

1. 访问 [Lambda Labs](https://lambdalabs.com/)
2. 注册账户并充值
3. 点击 `Launch Instance`
4. 选择GPU类型：
   - **A100 40GB**（推荐）：适合7B-14B模型
   - **RTX 3090**（经济）：适合1.5B-7B模型，约$0.50/hr
5. 选择镜像：`PyTorch 2.1 + Python 3.10`

#### Step 2: SSH连接实例

```bash
# Lambda Labs会提供SSH命令
ssh ubuntu@<instance-ip> -p <port>

# 或使用Lambda提供的密钥
ssh -i ~/.ssh/lambda_key ubuntu@<instance-ip>
```

#### Step 3: 验证GPU

```bash
nvidia-smi
# 应显示类似：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  A100-SXM4-40GB      On   | 00000000:00:04.0 Off |                    0 |
# |-------------------------------+----------------------+----------------------+
```

### 2.3 Lambda Labs环境设置

```bash
# === 克隆项目 ===
git clone https://github.com/your-repo/method-thinker.git
cd method-thinker

# === 安装依赖（Lambda Labs大部分已预装）===
pip install transformers>=4.40.0 accelerate>=0.25.0 peft>=0.7.0 trl>=0.7.0 datasets>=2.14.0
pip install pyyaml pydantic scipy sympy pandas tqdm rich python-dotenv

# === 验证环境 ===
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 2.4 Lambda Labs训练命令

#### A100 40GB配置

```bash
# 全量微调（显存充足）
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode full \
    --kb data/methodology_kb/v0/math_methods.yaml \
    --batch-size 16 \
    --epochs 3

# 或使用LoRA加速
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode methodology-injection \
    --use-lora --lora-r 32 \
    --batch-size 32
```

#### RTX 3090配置（24GB显存）

```bash
# 必须使用LoRA
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode methodology-injection \
    --use-lora --lora-r 16 \
    --batch-size 8 \
    --max-length 2048
```

### 2.5 Lambda Labs模型保存

#### 本地持久存储

```bash
# Lambda Labs实例数据持久化，直接保存即可
mkdir -p ~/models/method-thinker
cp -r outputs/checkpoints/final ~/models/method-thinker/
```

#### 上传到云存储

```bash
# 使用SCP下载到本地
scp -r ubuntu@<instance-ip>:~/method-thinker/outputs/checkpoints/final ./local_models/

# 或上传到AWS S3
aws s3 sync outputs/checkpoints/final s3://your-bucket/method-thinker-model/
```

---

## 3. RunPod训练指南

### 3.1 为什么选择RunPod

| 特性 | 说明 |
|-----|------|
| **灵活GPU选择** | A100, RTX 4090, RTX A6000等 |
| **价格透明** | 按秒计费，RTX 4090约$0.69/hr |
| **Community Cloud** | 更便宜的社区GPU |
| **Secure Cloud** | 企业级安全 |
| **Pod持久化** | 可配置持久存储 |

### 3.2 RunPod使用步骤

#### Step 1: 创建Pod

1. 访问 [RunPod](https://runpod.io/)
2. 注册账户并充值
3. 点击 `Create Pod`
4. 选择GPU：
   - **RTX 4090**（推荐性价比）：24GB显存，约$0.69/hr
   - **A100 80GB**（高性能）：适合大模型
5. 选择模板：`RunPod PyTorch` 或自定义
6. 配置持久存储：添加 `Network Volume`（推荐50GB+）

#### Step 2: 连接Pod

RunPod提供多种连接方式：

```bash
# 方式1: SSH连接（在Pod详情页获取命令）
ssh root@<pod-ip> -p <port> -i <key-file>

# 方式2: RunPod Terminal（网页端）
# 点击Pod的 "Connect" → "Terminal"

# 方式3: Jupyter Lab（网页端）
# 点击Pod的 "Connect" → "Jupyter Lab"
```

### 3.3 RunPod环境设置

```bash
# === 克隆项目（到持久存储目录）===
cd /workspace  # Network Volume挂载点
git clone https://github.com/your-repo/method-thinker.git
cd method-thinker

# === 安装依赖 ===
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 accelerate>=0.25.0 peft>=0.7.0 trl>=0.7.0 datasets>=2.14.0
pip install pyyaml pydantic scipy sympy pandas tqdm rich python-dotenv

# === 验证GPU ===
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"
```

### 3.4 RunPod训练命令

#### RTX 4090配置（24GB显存）

```bash
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode methodology-injection \
    --use-lora --lora-r 16 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --max-length 2048 \
    --kb data/methodology_kb/v0/math_methods.yaml
```

#### A100配置（80GB显存）

```bash
# 全量微调
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode full \
    --batch-size 16 \
    --epochs 3

# 多样性训练
python scripts/train_sft.py \
    --mode diversity \
    --methods-per-problem 4 \
    --batch-size 32
```

### 3.5 RunPod模型保存

```bash
# 保存到持久存储（Network Volume）
mkdir -p /workspace/models/method-thinker
cp -r outputs/checkpoints/final /workspace/models/method-thinker/

# 从RunPod下载到本地
# 方式1: 通过RunPod文件管理器下载
# 方式2: SCP下载
scp -r root@<pod-ip>:/workspace/models/method-thinker ./local_models/
```

---

## 4. 通用环境设置命令

### 4.1 基础依赖安装

```bash
# PyTorch（根据CUDA版本选择）
# CUDA 12.1
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 训练相关库
pip install transformers>=4.40.0 accelerate>=0.25.0 peft>=0.7.0 trl>=0.7.0 datasets>=2.14.0

# 数据处理和评估
pip install pyyaml pydantic numpy scipy sympy pandas tqdm rich python-dotenv
```

### 4.2 可选依赖

```bash
# DeepSpeed分布式训练（多GPU）
pip install deepspeed>=0.12.0

# bitsandbytes量化（4-bit训练）
pip install bitsandbytes>=0.41.0

# Weights & Biases实验追踪
pip install wandb>=0.15.0

# TensorBoard可视化
pip install tensorboard>=2.14.0
```

### 4.3 环境验证脚本

```python
# scripts/verify_environment.py
import torch
import transformers
import accelerate
import peft
import trl

print("=" * 50)
print("环境验证报告")
print("=" * 50)
print(f"PyTorch版本: {torch.__version__}")
print(f"Transformers版本: {transformers.__version__}")
print(f"Accelerate版本: {accelerate.__version__}")
print(f"PEFT版本: {peft.__version__}")
print(f"TRL版本: {trl.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"支持BF16: {torch.cuda.is_bf16_supported()}")
print("=" * 50)
```

---

## 5. 训练启动命令

### 5.1 基础训练命令

```bash
# 方法论注入训练
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode methodology-injection \
    --kb data/methodology_kb/v0/math_methods.yaml

# 多样性训练
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode diversity \
    --methods-per-problem 4

# 反思强化训练
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode reflection

# 完整训练流程
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode full \
    --kb data/methodology_kb/v0/math_methods.yaml
```

### 5.2 LoRA训练命令（节省显存）

```bash
# 启用LoRA
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode methodology-injection \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32

# 不同GPU配置的LoRA参数推荐
# T4 16GB:    lora-r=16, batch_size=4, max_length=2048
# RTX 3090:   lora-r=32, batch_size=8, max_length=4096
# A100 40GB:  lora-r=64, batch_size=16, max_length=4096
```

### 5.3 分布式训练命令（多GPU）

```bash
# 单机多GPU
torchrun --nproc_per_node=4 scripts/train_sft.py \
    --config configs/training_config.yaml \
    --mode methodology-injection

# DeepSpeed配置
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --deepspeed configs/deepspeed_config.json
```

### 5.4 恢复训练命令

```bash
# 从检查点恢复
python scripts/train_sft.py \
    --config configs/training_config.yaml \
    --resume outputs/checkpoints/checkpoint-epoch-2
```

---

## 6. 模型下载与保存

### 6.1 基座模型下载

#### 直接下载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Math-1.5B"

# 自动下载（首次需要时间）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

#### 使用镜像加速（国内）

```bash
# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或在Python中
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

#### 手动下载到本地

```bash
# 使用huggingface-cli
huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir ./models/Qwen2.5-Math-1.5B

# 训练时使用本地路径
python scripts/train_sft.py --base-model ./models/Qwen2.5-Math-1.5B
```

### 6.2 训练后模型保存

#### 本地保存

```python
# 训练完成后保存
trainer.save_checkpoint("outputs/checkpoints/final")

# 或手动保存
model.save_pretrained("outputs/checkpoints/final")
tokenizer.save_pretrained("outputs/checkpoints/final")
```

#### HuggingFace Hub上传

```bash
# 登录
huggingface-cli login

# 上传模型
huggingface-cli upload your-username/method-thinker-v1 outputs/checkpoints/final
```

#### Python上传脚本

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="your-username/method-thinker-v1", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="outputs/checkpoints/final",
    repo_id="your-username/method-thinker-v1",
    repo_type="model"
)

# 创建模型卡片
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id="your-username/method-thinker-v1"
)
```

### 6.3 Google Drive保存（Colab）

```python
# 挂载Drive
from google.colab import drive
drive.mount('/content/drive')

# 创建目录
import os
save_dir = '/content/drive/MyDrive/method-thinker-models'
os.makedirs(save_dir, exist_ok=True)

# 保存模型
import shutil
shutil.copytree('outputs/checkpoints/final', f'{save_dir}/final-model-v1')
```

---

## 7. 常见问题与解决方案

### 7.1 显存不足 (OOM)

**症状**: `CUDA out of memory` 错误

**解决方案**:

```bash
# 方案1: 减小batch_size
--batch-size 4

# 方案2: 增加梯度累积
--gradient-accumulation-steps 8

# 方案3: 减小序列长度
--max-length 2048

# 方案4: 启用gradient checkpointing
# 在配置文件中添加：
# training:
#   gradient_checkpointing: true

# 方案5: 使用LoRA
--use-lora --lora-r 16

# 方案6: 使用4-bit量化
pip install bitsandbytes
# 在配置中：
# quantization:
#   enabled: true
#   method: bitsandbytes
#   bits: 4
```

### 7.2 训练速度慢

**解决方案**:

```bash
# 方案1: 使用BF16加速（如果GPU支持）
--bf16

# 方案2: 启用Flash Attention
# 配置文件中：
# model:
#   use_flash_attention: true

# 方案3: 增加dataloader workers
--dataloader-num-workers 4

# 方案4: 使用vLLM推理加速
pip install vllm
```

### 7.3 模型下载失败

**解决方案**:

```bash
# 方案1: 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方案2: 手动下载后使用本地路径
huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir ./models/base

# 方案3: 设置代理
export HTTP_PROXY=http://proxy-server:port
export HTTPS_PROXY=http://proxy-server:port
```

### 7.4 Colab会话断开

**解决方案**:

```python
# 自动保存检查点脚本
import time
import os

def periodic_save(trainer, interval_minutes=30):
    """定期保存检查点防止丢失"""
    while True:
        time.sleep(interval_minutes * 60)
        checkpoint_path = f"outputs/checkpoints/auto_save_{int(time.time())}"
        trainer.save_checkpoint(checkpoint_path)
        print(f"自动保存: {checkpoint_path}")

# 启动后台线程
import threading
save_thread = threading.Thread(target=periodic_save, args=(trainer,), daemon=True)
save_thread.start()
```

### 7.5 训练loss不下降

**解决方案**:

1. **检查学习率**: 降低或提高 `--learning-rate`
2. **检查数据质量**: 确保训练数据格式正确
3. **检查模型加载**: 确认基座模型正确加载
4. **增加warmup**: 设置 `--warmup-ratio 0.1`
5. **调整LoRA参数**: 尝试不同的 `lora-r` 值

### 7.6 评估结果不理想

**解决方案**:

```bash
# 增加评估样本数
--eval-samples 64

# 使用Pass@K评估
python scripts/run_evaluation.py \
    --model outputs/checkpoints/final \
    --benchmark aime25 \
    --pass-k 16 64

# 检查推理参数
--temperature 0.7 --top-p 0.9
```

---

## 附录A: GPU显存估算表

| 模型大小 | LoRA r=16 | LoRA r=32 | 全量微调 |
|---------|-----------|-----------|----------|
| 1.5B | 8GB | 10GB | 16GB |
| 7B | 16GB | 20GB | 40GB+ |
| 14B | 24GB | 32GB | 80GB+ |

**推荐配置**:
- T4 16GB: 1.5B模型 + LoRA r=16
- RTX 3090 24GB: 7B模型 + LoRA r=16 或 1.5B全量微调
- A100 40GB: 7B模型 + LoRA r=32 或 全量微调
- A100 80GB: 14B模型训练

---

## 附录B: 配置文件参考

详见:
- `configs/training_config.yaml` - 标准训练配置
- `configs/training_config_colab.yaml` - Colab优化配置

---

## 附录C: 相关文档

- [用户指南](user_guide.md)
- [API参考](api_reference.md)
- [验证系统](validation_system.md)
- [部署指南](deployment.md)