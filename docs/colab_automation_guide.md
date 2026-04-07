# MethodThinker Colab自动化训练指南

> **一键云端训练** - 使用papermill参数化执行，实现自动化训练流程

**更新日期**: 2026-04-07

---

## 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [快速开始](#快速开始)
4. [配置详解](#配置详解)
5. [预设训练方案](#预设训练方案)
6. [Google Drive同步](#google-drive同步)
7. [定时训练](#定时训练)
8. [常见问题](#常见问题)

---

## 概述

### 自动化方案

MethodThinker提供三种自动化训练方式：

| 方式 | 工具 | 适用场景 | 复杂度 |
|-----|------|---------|-------|
| **Papermill本地执行** | papermill | 本地GPU/Colab手动执行 | 低 |
| **Drive同步自动化** | colab_automation.py | 自动上传数据、下载结果 | 中 |
| **定时训练** | gcloud scheduler | 定期自动训练 | 高 |

### 核心组件

```
scripts/colab_automation.py  - 自动化脚本
notebooks/train_automated.ipynb - 参数化训练notebook
configs/colab_automation.yaml - 自动化配置
```

### 工作流程

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  准备阶段       │ -> │  执行阶段       │ -> │  结果收集       │
│                 │    │                 │    │                 │
│  - 检查依赖     │    │  - 参数注入     │    │  - 下载模型     │
│  - 上传数据     │    │  - papermill执行│    │  - 提取报告     │
│  - 验证环境     │    │  - 训练执行     │    │  - 保存结果     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 环境准备

### 1. 安装依赖

```bash
# 核心依赖
pip install papermill>=2.4.0

# Google Drive API（可选，用于自动同步）
pip install google-api-python-client google-auth

# Google Cloud SDK（可选，用于定时训练）
# 参见: https://cloud.google.com/sdk/docs/install
```

### 2. Google Cloud项目设置（可选）

如果需要Drive自动同步或定时训练：

1. 创建Google Cloud项目: https://console.cloud.google.com/
2. 启用API:
   - Google Drive API
   - Google Cloud Scheduler API（定时训练需要）
3. 创建服务账号:
   - 控制台 → IAM → 服务账号 → 创建
   - 下载JSON密钥文件
4. 设置凭据:

```yaml
# configs/colab_automation.yaml
google_cloud:
  project_id: "your-project-id"
  credentials_path: "~/.config/gcloud/service-account.json"
```

### 3. 验证环境

```bash
# 运行准备命令
python scripts/colab_automation.py prepare

# 输出示例
# ✅ 本地数据检查完成
# ✅ Notebook模板存在
# ✅ 依赖验证通过
```

---

## 快速开始

### 一键训练命令

```bash
# 快速验证训练（约15分钟）
python scripts/colab_automation.py run --preset quick

# 标准训练（约1小时）
python scripts/colab_automation.py run --preset standard

# 完整训练（约3小时）
python scripts/colab_automation.py run --preset full --download-results
```

### 单步执行

```bash
# 1. 准备环境
python scripts/colab_automation.py prepare --sync-drive

# 2. 执行训练
python scripts/colab_automation.py execute --preset standard

# 3. 下载结果（需要Drive配置）
python scripts/colab_automation.py run --download-results
```

### 自定义参数

```bash
# 自定义模型和参数
python scripts/colab_automation.py execute \
    --base-model Qwen/Qwen2.5-Math-7B \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --max-length 4096

# 使用自定义notebook
python scripts/colab_automation.py execute \
    --notebook notebooks/custom_train.ipynb
```

---

## 配置详解

### 配置文件结构

```yaml
# configs/colab_automation.yaml

# Google Cloud配置
google_cloud:
  project_id: "your-project-id"
  credentials_path: "path/to/credentials.json"
  region: "us-central1"

# Google Drive配置
google_drive:
  enabled: true
  base_folder: "method-thinker-models"
  data_folder: "training-data"
  results_folder: "training-results"

# 默认训练参数
training:
  base_model: "Qwen/Qwen2.5-Math-1.5B"
  epochs: 3
  batch_size: 4
  learning_rate: 5e-5
  max_length: 2048
  lora_r: 16
  lora_alpha: 32

# 预设配置
presets:
  quick: {...}
  standard: {...}
  full: {...}

# 路径配置
paths:
  notebook_template: "notebooks/train_automated.ipynb"
  output_notebook: "outputs/executed_notebooks/"
  results_dir: "outputs/colab_results/"
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `base_model` | Qwen/Qwen2.5-Math-1.5B | 基座模型路径 |
| `epochs` | 3 | 训练轮数 |
| `batch_size` | 4 | 批大小（T4显存限制） |
| `learning_rate` | 5e-5 | 学习率 |
| `max_length` | 2048 | 最大序列长度 |
| `lora_r` | 16 | LoRA秩 |
| `lora_alpha` | 32 | LoRA缩放参数 |
| `hf_mirror` | https://hf-mirror.com | HuggingFace镜像（国内加速） |
| `max_samples` | 1000 | 最大训练样本数 |

---

## 预设训练方案

### Quick - 快速验证

```yaml
quick:
  epochs: 1
  max_samples: 100
  batch_size: 4
  description: "快速验证训练 - 约15分钟完成"
```

**适用场景**:
- 验证训练流程是否正常
- 测试新模型兼容性
- 快速原型验证

**命令**:
```bash
python scripts/colab_automation.py run --preset quick
```

### Standard - 标准训练

```yaml
standard:
  epochs: 3
  max_samples: 1000
  batch_size: 4
  description: "标准训练 - 约1小时完成"
```

**适用场景**:
- 日常训练任务
- 模型迭代更新
- 中等规模数据训练

**命令**:
```bash
python scripts/colab_automation.py run --preset standard --download-results
```

### Full - 完整训练

```yaml
full:
  epochs: 5
  max_samples: 5000
  batch_size: 4
  gradient_accumulation_steps: 8
  description: "完整训练 - 约3小时完成"
```

**适用场景**:
- 正式版本训练
- 大规模数据训练
- 最高质量模型

**命令**:
```bash
python scripts/colab_automation.py run --preset full --download-results
```

---

## Google Drive同步

### 自动上传训练数据

```bash
# 同步本地数据到Drive
python scripts/colab_automation.py prepare --sync-drive

# 上传的文件结构
# method-thinker-models/
#   └── training-data/
#       ├── train.json
#       ├── val.json
#       ├── test.json
#       └── math_methods.yaml (知识库)
```

### 自动下载训练结果

```bash
# 执行训练并下载结果
python scripts/colab_automation.py run --download-results

# 结果保存位置
# outputs/colab_results/
#   └── results_20260407_120000.zip
```

### Drive文件结构

```
method-thinker-models/            # 根文件夹
├── training-data/               # 训练数据
│   ├── train.json
│   ├── val.json
│   └── math_methods.yaml
├── training-results/            # 训练结果
│   ├── model_20260407_120000/   # 模型检查点
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── training_report.json
│   └── model_20260408_090000/
└── notebooks/                   # Notebook备份
    └── train_automated.ipynb
```

---

## 定时训练

### 本地定时（使用cron）

```bash
# 设置每周一9点自动训练
python scripts/colab_automation.py schedule --cron "0 9 * * 1"

# 输出指导
# ✅ 调度脚本已创建: scripts/scheduled/scheduled_training_xxx.sh
# 请添加到crontab:
#   crontab -e
#   添加行: 0 9 * * 1 /path/to/script.sh
```

### Cron表达式示例

| 表达式 | 说明 |
|-------|------|
| `0 9 * * *` | 每天9点 |
| `0 9 * * 1` | 每周一9点 |
| `0 9,21 * * *` | 每天9点和21点 |
| `0 */6 * * *` | 每6小时 |
| `30 8 * * 1-5` | 工作日8:30 |

### Google Cloud Scheduler

```bash
# 配置GCP凭据后使用
python scripts/colab_automation.py schedule \
    --cron "0 9 * * 1" \
    --scheduler gcloud

# 需要配置
# google_cloud:
#   project_id: "your-project-id"
#   credentials_path: "path/to/credentials.json"
```

---

## 常见问题

### Q1: papermill执行失败

**症状**: `ModuleNotFoundError: No module named 'papermill'`

**解决**:
```bash
pip install papermill>=2.4.0
```

### Q2: Drive同步权限错误

**症状**: `PermissionDenied: 403 Request had insufficient authentication scopes`

**解决**:
1. 确认服务账号有Drive访问权限
2. 在GCP控制台添加角色: `Drive File Editor`
3. 检查credentials_path配置是否正确

### Q3: 训练内存不足

**症状**: `OutOfMemoryError: CUDA out of memory`

**解决**:
```yaml
# 减小batch_size和max_length
training:
  batch_size: 2
  max_length: 1024
  gradient_accumulation_steps: 16  # 保持有效batch=32
```

### Q4: Colab断连问题

**解决**: 在notebook中添加防断连脚本

```javascript
// 在浏览器控制台运行
function KeepAlive() {
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepAlive, 300000);  // 每5分钟点击一次
```

### Q5: 模型下载慢

**解决**: 使用HuggingFace镜像

```yaml
training:
  hf_mirror: "https://hf-mirror.com"
```

或在notebook中设置:
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

## 命令参考

### colab_automation.py命令

| 命令 | 说明 | 主要参数 |
|-----|------|---------|
| `prepare` | 准备训练环境 | `--sync-drive` |
| `execute` | 执行参数化notebook | `--preset`, `--base-model`, `--epochs` |
| `run` | 完整运行流程 | `--preset`, `--download-results` |
| `schedule` | 设置定时训练 | `--cron` |

### 全局参数

```bash
--config PATH          # 配置文件路径
--preset NAME          # 预设配置 (quick/standard/full)
--base-model NAME      # 基座模型
--epochs N             # 训练轮数
--batch-size N         # 批大小
--learning-rate RATE   # 学习率
--max-length N         # 最大序列长度
```

---

## 相关文档

- [云端训练指南](cloud_training_guide.md)
- [训练配置详解](training_config.md)
- [用户指南](user_guide.md)
- [API参考](api_reference.md)

---

## 联系与反馈

如有问题或建议，请：
1. 提交Issue: GitHub Issues
2. 查看文档: docs/
3. 加入讨论: Discord/Slack社区