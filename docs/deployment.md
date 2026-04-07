# MethodThinker 部署指南

> 版本: v1.0 | 更新日期: 2026-04-07

## 目录

1. [环境要求](#环境要求)
2. [快速部署](#快速部署)
3. [详细安装步骤](#详细安装步骤)
4. [配置说明](#配置说明)
5. [验证安装](#验证安装)
6. [生产环境部署](#生产环境部署)
7. [常见问题](#常见问题)

---

## 环境要求

### 系统要求

| 要求 | 最低配置 | 推荐配置 |
|-----|---------|---------|
| 操作系统 | Linux (Ubuntu 20.04+) | Linux (Ubuntu 22.04) |
| Python | 3.10+ | 3.12 |
| 内存 | 16 GB | 32 GB+ |
| 存储 | 50 GB | 100 GB+ SSD |

### GPU 要求（可选，用于训练）

| 用途 | GPU型号 | 显存 | 说明 |
|-----|--------|------|------|
| 验证测试 | RTX 3060+ | 8GB+ | 推理验证 |
| SFT训练 (1.5B) | RTX 4090 | 24GB | 全量微调 |
| SFT训练 (7B) | H800/A100 | 80GB | 大模型训练 |
| 推理加速 | RTX 4090 | 24GB | vLLM加速 |

### 软件依赖

| 依赖 | 版本 | 用途 |
|-----|------|------|
| CUDA | 12.x | GPU支持 |
| PyTorch | 2.1+ | 深度学习框架 |
| Transformers | 4.40+ | 模型库 |
| vLLM | 0.4+ | 推理加速（可选） |

---

## 快速部署

### 一键安装脚本

```bash
# 下载并运行安装脚本
curl -fsSL https://raw.githubusercontent.com/your-org/method-thinker/main/scripts/install.sh | bash

# 或使用 pip 直接安装
pip install method-thinker
```

### 最小化验证安装

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/method-thinker.git
cd method-thinker

# 2. 安装核心依赖
pip install -r requirements.txt

# 3. 快速验证
python -c "from src.validation.pipeline import ValidationPipeline; print('OK')"
```

---

## 详细安装步骤

### Step 1: Python 环境准备

```bash
# 推荐：使用 conda 创建独立环境
conda create -n method-thinker python=3.12
conda activate method-thinker

# 或使用 venv
python3.12 -m venv venv
source venv/bin/activate
```

### Step 2: 安装依赖

```bash
# 安装全部依赖
pip install -r requirements.txt

# 或按功能分组安装（节省空间）

# 核心依赖（必需）
pip install torch>=2.1.0 transformers>=4.40.0 pyyaml>=6.0 pydantic>=2.0.0

# 验证系统依赖
pip install scipy>=1.10.0 sympy>=1.12 pytest>=7.0.0

# 训练系统依赖（可选，用于训练）
pip install accelerate>=0.25.0 peft>=0.7.0 trl>=0.7.0 datasets>=2.14.0

# API 客户端（可选，用于 Layer 2）
pip install openai>=1.0.0 dashscope>=1.14.0

# 推理加速（可选）
pip install vllm>=0.4.0
```

### Step 3: GPU 环境验证（可选）

```bash
# 检查 CUDA 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# 检查 GPU 信息
nvidia-smi
```

### Step 4: API 密钥配置（可选）

Layer 2 多模型验证需要外部 API。配置环境变量：

```bash
# 方式 1: 直接设置环境变量
export DEEPSEEK_API_KEY="your-deepseek-key"
export QWEN_API_KEY="your-qwen-key"
export OPENAI_API_KEY="your-openai-key"

# 方式 2: 使用 .env 文件
cat > .env << EOF
DEEPSEEK_API_KEY=your-deepseek-key
QWEN_API_KEY=your-qwen-key
OPENAI_API_KEY=your-openai-key
EOF

# 加载 .env
pip install python-dotenv
python -c "from dotenv import load_dotenv; load_dotenv()"
```

### Step 5: 数据准备

```bash
# 创建数据目录
mkdir -p data/methodology_kb data/test_sets data/train_data

# 验证种子知识库存在
ls data/methodology_kb/v0/math_methods.yaml
```

### Step 6: 配置文件检查

```bash
# 检查配置文件
ls configs/*.yaml

# 查看验证配置
cat configs/validation_config.yaml
```

---

## 配置说明

### 配置文件结构

```
configs/
├── validation_config.yaml    # 验证系统配置
├── training_config.yaml      # 训练系统配置
└── evaluation_config.yaml    # 评估配置
```

### 关键配置项

#### 验证配置 (validation_config.yaml)

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `layer0.enabled` | Layer 0 快速过滤 | true |
| `layer1.enabled` | Layer 1 自我反思 | true |
| `layer2.enabled` | Layer 2 多模型验证 | true |
| `layer3.enabled` | Layer 3 测试驱动 | true |
| `budget.total` | API 预算（美元） | 500 |

#### 训练配置 (training_config.yaml)

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `model.base_model` | 基座模型 | Qwen/Qwen2.5-Math-1.5B |
| `training.num_epochs` | 训练轮数 | 3 |
| `training.batch_size` | 批大小 | 8 |
| `training.learning_rate` | 学习率 | 5e-5 |

#### 评估配置 (evaluation_config.yaml)

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `benchmarks.aime25.enabled` | AIME25 基准 | true |
| `pass_at_k.k_values` | Pass@K 值列表 | [1,8,16,64] |
| `targets.primary.target` | Pass@64 目标 | 0.75 |

### 配置修改建议

```yaml
# 低预算配置 - 只使用免费层级
layer2:
  enabled: false
layer3:
  enabled: false

# 高性能配置 - 启用所有层级
layer2:
  enabled: true
  models: [deepseek_v3, qwen_math, gpt4o_mini]
layer3:
  enabled: true
  min_test_count: 50
```

---

## 验证安装

### 功能测试

```bash
# 1. 导入测试
python -c "
from src.validation.pipeline import ValidationPipeline
from src.validation.config import ValidationConfig
from src.kb.knowledge_base import KnowledgeBase
print('导入测试通过')
"

# 2. 单元测试
python -m pytest tests/ -v --tb=short

# 3. 验证流水线测试
python scripts/run_validation.py \
  --kb data/methodology_kb/v0/math_methods.yaml \
  --layers 0 \
  --limit 5 \
  --verbose
```

### 性能基准测试

```bash
# Layer 0 性能测试（应 <50ms）
python -c "
import time
from src.validation.layer0_fast_filter import Layer0FastFilter
layer0 = Layer0FastFilter({'methods': {}})
method = {
    'method_id': 'TEST_001',
    'name': '测试方法',
    'description': '这是一个测试方法',
    'applicability': [{'condition': '测试条件'}],
    'template': {'steps': ['步骤1', '步骤2']}
}
start = time.time()
result = layer0.validate(method)
elapsed = time.time() - start
print(f'延迟: {elapsed*1000:.2f}ms (目标<50ms)')
print(f'结果: {result.passed}')
"
```

### GPU 测试（可选）

```bash
# GPU 推理测试
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen2.5-Math-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

prompt = 'What is 2+2?'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
print('GPU推理测试通过')
"
```

---

## 生产环境部署

### Docker 部署

```bash
# 构建镜像
docker build -t method-thinker:latest .

# 运行容器
docker run -d \
  --name method-thinker \
  --gpus all \
  -v ./data:/app/data \
  -v ./configs:/app/configs \
  -v ./results:/app/results \
  -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
  -e QWEN_API_KEY=$QWEN_API_KEY \
  method-thinker:latest
```

#### Dockerfile 示例

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/

# 运行验证
CMD ["python", "scripts/run_validation.py", "--layers", "0"]
```

### Kubernetes 部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: method-thinker
spec:
  replicas: 1
  selector:
    matchLabels:
      app: method-thinker
  template:
    metadata:
      labels:
        app: method-thinker
    spec:
      containers:
      - name: method-thinker
        image: method-thinker:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: DEEPSEEK_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: deepseek
        volumeMounts:
        - name: data
          mountPath: /app/data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: method-thinker-data
```

### 分布式训练部署

```bash
# 多 GPU 训练（使用 accelerate）
accelerate config  # 配置分布式环境

# 启动分布式训练
accelerate launch scripts/train_sft.py \
  --config configs/training_config.yaml \
  --distributed true
```

### API 服务部署

```bash
# 使用 vLLM 部署推理服务
python -m vllm.entrypoints.api_server \
  --model Qwen/Qwen2.5-Math-1.5B \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000

# 验证服务可用
curl http://localhost:8000/v1/models
```

---

## 常见问题

### Q1: CUDA 内存不足怎么办？

**解决方案:**

```yaml
# 1. 减小批大小
training:
  batch_size: 4
  gradient_accumulation_steps: 8

# 2. 使用量化
quantization:
  enabled: true
  bits: 4
  method: bitsandbytes

# 3. 使用 LoRA 替代全量微调
quantization:
  lora:
    enabled: true
    lora_r: 16
```

### Q2: API 调用超时怎么办？

**解决方案:**

```yaml
# 增加超时和重试配置
layer2:
  retry:
    max_attempts: 5
    backoff_factor: 3
    initial_delay: 2.0

advanced:
  timeouts:
    layer2: 120  # 增加到 2 分钟
```

### Q3: 验证结果异常（全部失败）

**诊断步骤:**

```bash
# 1. 检查知识库格式
python -c "
from src.kb.knowledge_base import KnowledgeBase
kb = KnowledgeBase.from_yaml('data/methodology_kb/v0/math_methods.yaml')
print(f'方法数: {len(kb.methods)}')
"

# 2. 详细输出模式
python scripts/run_validation.py --verbose --limit 1

# 3. 单层测试
python scripts/run_validation.py --layers 0 --limit 5 --verbose
```

### Q4: 训练不收敛怎么办？

**解决方案:**

```yaml
# 1. 降低学习率
training:
  learning_rate: 1e-5
  warmup_ratio: 0.2

# 2. 启用早停
early_stopping:
  enabled: true
  patience: 5
  threshold: 0.001

# 3. 检查数据质量
data:
  preprocessing:
    filter_unverified: true
```

### Q5: 如何监控训练进度？

**解决方案:**

```bash
# 使用 TensorBoard
tensorboard --logdir outputs/logs --port 6006

# 查看日志
tail -f logs/training.log

# GPU 监控
watch -n 1 nvidia-smi
```

### Q6: 如何恢复训练？

**解决方案:**

```bash
# 从检查点恢复
python scripts/train_sft.py \
  --config configs/training_config.yaml \
  --resume outputs/checkpoints/checkpoint-epoch-2

# 或在配置文件中设置
resume:
  enabled: true
  checkpoint_path: outputs/checkpoints/checkpoint-epoch-2
```

---

## 附录

### 相关文档

| 文档 | 说明 |
|-----|------|
| `docs/user_guide.md` | 用户指南 |
| `docs/validation_system.md` | 验证系统详解 |
| `docs/project_summary.md` | 项目总结 |
| `configs/validation_config.yaml` | 验证配置示例 |

### 环境变量清单

| 变量 | 说明 | 必需 |
|-----|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | Layer 2 可选 |
| `QWEN_API_KEY` | 阿里云 Qwen API 密钥 | Layer 2 可选 |
| `OPENAI_API_KEY` | OpenAI API 密钥 | Layer 2 可选 |
| `CUDA_VISIBLE_DEVICES` | GPU 设备选择 | GPU 训练可选 |
| `HF_TOKEN` | HuggingFace Token | 私有模型可选 |

### 检查清单

**部署前检查:**

- [ ] Python 版本 ≥ 3.10
- [ ] 依赖安装完成
- [ ] 配置文件存在
- [ ] 知识库数据存在
- [ ] API 密钥配置（如需要）
- [ ] GPU 可用（如需要）

**部署后验证:**

- [ ] 导入测试通过
- [ ] 单元测试通过
- [ ] 验证流水线运行正常
- [ ] GPU 推理正常（如需要）
- [ ] API 调用正常（如需要）

---

*文档生成时间: 2026-04-07*