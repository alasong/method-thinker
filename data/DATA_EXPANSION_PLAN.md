# 训练数据扩充计划

> 创建时间: 2026-04-08

## 当前状态

| 数据集 | 样本数 | 来源 | 状态 |
|-------|--------|------|------|
| train.json | 120 | AIME样本 | 已有 |

## 目标

- 短期目标: 1000+ 样本
- 中期目标: 5000+ 样本
- 长期目标: 10000+ 样本

## 可用数据源

### 1. AIME数据集 (推荐优先)

| 数据集 | 样本数 | HuggingFace路径 | 难度 |
|-------|--------|----------------|------|
| AIME 1983-2024 | ~1000 | EleutherAI/aime-1983-2024 | 高 |
| AIME历年真题 | ~500 | yukuai/AIME | 高 |

**下载命令:**
```python
from datasets import load_dataset
dataset = load_dataset("EleutherAI/aime-1983-2024")
```

### 2. MATH数据集

| 数据集 | 样本数 | HuggingFace路径 | 难度 |
|-------|--------|----------------|------|
| Competition Math | 12500 | hendrycks/competition_math | 中高 |

**特点:**
- 包含AMC、AIME、IMO等竞赛题目
- 有完整解答过程
- 分类清晰

### 3. AMC数据集

| 数据集 | 样本数 | HuggingFace路径 | 难度 |
|-------|--------|----------------|------|
| AMC | ~10000 | mdegyes/AMC | 中 |

### 4. GSM8K (可选，较低难度)

| 数据集 | 样本数 | HuggingFace路径 | 难度 |
|-------|--------|----------------|------|
| GSM8K | 8500 | openai/gsm8k | 低 |

## 数据格式转换

当前格式:
```json
{
  "problem": "问题描述",
  "method_selection": "方法选择理由",
  "solution_steps": ["步骤1", "步骤2"],
  "final_answer": "最终答案",
  "method_id": "ALG_001",
  "method_name": "方法名称",
  "problem_type": "ALGEBRA",
  "difficulty": 2,
  "annotations": [...]
}
```

需要转换脚本将公开数据集转换为此格式。

## 执行计划

### Phase 1: AIME数据 (Week 1)

1. 下载EleutherAI/aime-1983-2024数据集
2. 编写转换脚本
3. 生成1000+样本

### Phase 2: MATH数据 (Week 2)

1. 下载hendrycks/competition_math
2. 筛选高质量样本
3. 生成5000+样本

### Phase 3: 数据增强 (Week 3)

1. 方法变体生成
2. 难度调整
3. 同义改写

## 同步Vibe Thinker

未找到vibe thinker项目目录。如需同步，请提供:
- 项目路径
- 数据格式要求
- 需要同步的数据类型

## 下载脚本

```python
# scripts/download_datasets.py
from datasets import load_dataset
import json

def download_aime():
    """下载AIME数据集"""
    dataset = load_dataset("EleutherAI/aime-1983-2024")
    # 转换格式...
    return dataset

def download_math():
    """下载MATH数据集"""
    dataset = load_dataset("hendrycks/competition_math")
    # 转换格式...
    return dataset

if __name__ == "__main__":
    aime_data = download_aime()
    math_data = download_math()
    print(f"AIME: {len(aime_data)} samples")
    print(f"MATH: {len(math_data)} samples")
```

---

*文档生成时间: 2026-04-08*
