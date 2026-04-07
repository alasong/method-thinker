# MethodThinker 用户指南

> 版本: v1.0 | 更新日期: 2026-04-07

## 目录

1. [快速开始](#快速开始)
2. [系统概览](#系统概览)
3. [方法论验证](#方法论验证)
4. [方法论提炼](#方法论提炼)
5. [模型训练](#模型训练)
6. [知识库管理](#知识库管理)
7. [常见问题](#常见问题)

---

## 快速开始

### 环境要求

| 要求 | 版本 |
|-----|------|
| Python | 3.10+ |
| CUDA | 12.x (可选，用于GPU训练) |
| 内存 | 16GB+ |
| GPU | RTX 4090/H800 (可选) |

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/method-thinker.git
cd method-thinker

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "from src.validation.pipeline import ValidationPipeline; print('安装成功')"
```

### 验证安装

```bash
# 运行快速测试
python scripts/run_validation.py --layers 0 --kb data/methodology_kb/v0/math_methods.yaml --limit 5
```

---

## 系统概览

MethodThinker是一个自动化方法论提炼与验证系统，通过迭代优化提升模型的数学推理能力。

### 核心流程

```
问题集 ──→ 方法论提炼 ──→ 验证系统 ──→ 知识库更新 ──→ 模型训练 ──→ 性能评估
    │                                                    │
    └───────────────────── 反馈循环 ←─────────────────────┘
```

### 主要模块

| 模块 | 功能 | 入口文件 |
|-----|------|---------|
| 验证系统 | 四层混合验证 | `scripts/run_validation.py` |
| 提炼系统 | 从解答提炼方法论 | `scripts/run_extraction.py` |
| 训练系统 | 方法论注入训练 | `scripts/train_sft.py` |
| 知识库 | 方法论存储与查询 | `src/kb/knowledge_base.py` |

---

## 方法论验证

验证系统是MethodThinker的核心组件，用于自动化评估方法论的有效性。

### 基本使用

```bash
# 运行完整验证（所有4层）
python scripts/run_validation.py --kb data/methodology_kb/v0/math_methods.yaml

# 只运行快速过滤（Layer 0）- 适合大量方法初步筛选
python scripts/run_validation.py --layers 0 --kb data/methodology_kb/v0/math_methods.yaml

# Layer 0 + Layer 1 - 适合验证现有KB方法
python scripts/run_validation.py --layers 0,1

# 核心验证层（Layer 2 + Layer 3）- 深度验证
python scripts/run_validation.py --layers 2,3

# 验证特定方法
python scripts/run_validation.py --method-id ALG_001 --verbose
```

### 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--config` | 配置文件路径 | `configs/validation_config.yaml` |
| `--kb` | 知识库路径 | `data/methodology_kb/v0/math_methods.yaml` |
| `--method-id` | 指定方法ID验证 | 全部方法 |
| `--layers` | 验证层级 | `all` |
| `--verbose` | 详细输出 | 关闭 |
| `--output` | 结果输出文件 | 无 |
| `--limit` | 限制验证数量 | 全部 |

### 验证层级选择指南

| 场景 | 推荐层级 | 说明 |
|-----|---------|------|
| 大量新方法初步筛选 | Layer 0 | 低成本，快速过滤 |
| 验证KB中已有方法 | Layer 0 | 检查格式完整性 |
| 新提炼方法论验证 | Layer 0, 1, 2 | 完整验证，排除低质量 |
| 核心方法深度验证 | Layer 0, 1, 2, 3 | 全流程验证 |
| 预算有限时 | Layer 0, 1 | 只用免费层级 |

### 输出解读

**标准输出示例:**

```
[1/24] 验证方法: 变量替换法 (ALG_001)
[PASS] Layer 0 (快速过滤) - 置信度: 1.00
[PASS] 集成决策 - 置信度: 0.87
  详情:
    weighted_score: 0.87
    reason: 所有层级验证通过

============================================================
验证完成 - 耗时: 0.05s
============================================================
总计: 24 个方法
通过: 24 (100.0%)
失败: 0 (0.0%)
```

**状态说明:**

| 状态 | 含义 |
|-----|------|
| `[PASS]` | 验证通过 |
| `[FAIL]` | 验证失败 |
| `置信度` | 验证可信程度（0-1） |

---

## 方法论提炼

从成功解答中提炼新的方法论模式。

### 基本使用

```bash
# 从解答文件提炼方法论
python scripts/run_extraction.py --input data/solutions.json --output data/methodology_kb/v1/

# 分析现有知识库模式
python scripts/run_extraction.py --kb data/methodology_kb/v0/math_methods.yaml

# 使用辅助模型提炼
python scripts/run_extraction.py --input data/solutions.json --model deepseek_v3
```

### 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--input` | 解答数据文件 | 无 |
| `--kb` | 现有知识库路径 | `data/methodology_kb/v0/math_methods.yaml` |
| `--output` | 输出目录 | `data/methodology_kb/v1/` |
| `--model` | 辅助提炼模型 | deepseek_v3 |
| `--min-samples` | 最小样本数 | 3 |

### 解答数据格式

输入解答文件应包含以下字段:

```json
[
  {
    "problem_id": "AIME_2024_001",
    "problem": "问题描述...",
    "solution": "解答过程...",
    "correct": true,
    "problem_type": "ALGEBRA"
  }
]
```

---

## 模型训练

使用方法论知识库训练模型。

### SFT训练

```bash
# 基础SFT训练
python scripts/train_sft.py --config configs/training_config.yaml

# 指定基座模型
python scripts/train_sft.py --base-model Qwen/Qwen2.5-Math-1.5B

# 方法论注入训练
python scripts/train_sft.py --mode methodology-injection --kb data/methodology_kb/v1/

# 多样性训练
python scripts/train_sft.py --mode diversity --methods-per-problem 4
```

### 训练模式

| 模式 | 说明 | 参数 |
|-----|------|------|
| `methodology-injection` | 方法论注入训练 | `--kb` |
| `diversity` | 多样性训练 | `--methods-per-problem` |
| `reflection` | 反思强化训练 | 无 |

### 训练参数

关键训练参数可通过配置文件或命令行指定:

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--base-model` | 基座模型 | `Qwen/Qwen2.5-Math-1.5B` |
| `--output-dir` | 输出目录 | `outputs/checkpoints` |
| `--epochs` | 训练轮数 | 3 |
| `--batch-size` | 批大小 | 8 |
| `--learning-rate` | 学习率 | 5e-5 |
| `--max-length` | 最大序列长度 | 4096 |

---

## 知识库管理

### 知识库结构

知识库采用YAML格式存储方法论:

```yaml
methods:
  - method_id: ALG_001
    name: 变量替换法
    category: ALGEBRA
    description: 通过引入新变量简化表达式结构...
    applicability:
      - condition: 表达式中存在重复结构
        keywords: [对称, 重复, 循环]
        problem_types: [代数方程, 不等式]
    template:
      steps:
        - 识别表达式中的重复模式
        - 选择合适的替换变量
        - 进行变量替换简化
        - 求解简化后的表达式
        - 回代得到原问题的解
      common_tricks: [选择能消除对称性的变量]
      pitfall_warnings: [避免引入更复杂的表达式]
    difficulty: 3
    frequency: 0.85
    related_methods: [ALG_002, ALG_003]
    examples: [AIME_2024_001, AIME_2024_002]
```

### 方法字段说明

| 字段 | 类型 | 必填 | 说明 |
|-----|------|------|------|
| `method_id` | string | 是 | 方法唯一标识（格式: `[A-Z]{3}_\d{3}`） |
| `name` | string | 是 | 方法名称（2-50字符） |
| `category` | string | 是 | 类别（ALGEBRA/GEOMETRY/NUMBER_THEORY/COMBINATORICS/GENERAL） |
| `description` | string | 是 | 详细描述（≥20字符） |
| `applicability` | list | 是 | 适用条件列表 |
| `template` | dict | 是 | 执行步骤模板（≥2步骤） |
| `difficulty` | int | 否 | 难度等级（1-5，默认3） |
| `frequency` | float | 否 | 使用频率（0-1，默认0.5） |
| `related_methods` | list | 否 | 相关方法ID列表 |
| `examples` | list | 否 | 示例题目ID列表 |

### Python API使用

```python
from src.kb.knowledge_base import KnowledgeBase, Method

# 加载知识库
kb = KnowledgeBase.from_yaml('data/methodology_kb/v0/math_methods.yaml')

# 查询方法
method = kb.get_method('ALG_001')
print(f"方法名: {method.name}")
print(f"步骤: {method.template['steps']}")

# 获取适用的方法
applicable = kb.get_applicable_methods(
    problem="求解对称方程",
    problem_type="ALGEBRA"
)

# 添加新方法
new_method = Method(
    method_id='ALG_100',
    name='新方法',
    category='ALGEBRA',
    description='这是一个新的解题方法...',
    applicability=[{'condition': '特定条件'}],
    template={'steps': ['步骤1', '步骤2']}
)
kb.add_method(new_method)

# 保存知识库
kb.save('data/methodology_kb/v1/math_methods.json')
```

---

## 常见问题

### Q1: Layer 0全部失败，为什么？

**原因:** 可能是重复检查误报。当验证KB中已有方法时，系统会检测到重复。

**解决:** 使用空KB或确保每个方法有唯一ID:
```bash
python scripts/run_validation.py --layers 0 --kb empty.yaml
```

### Q2: Layer 2需要哪些API配置？

**所需API:**
- DeepSeek API (环境变量: `DEEPSEEK_API_KEY`)
- Qwen API (环境变量: `QWEN_API_KEY`)
- OpenAI API (环境变量: `OPENAI_API_KEY`)

**配置方法:**
```bash
export DEEPSEEK_API_KEY="your-key"
export QWEN_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

### Q3: 如何跳过某层验证？

使用`--layers`参数指定要运行的层级:
```bash
# 只运行Layer 0和Layer 1
python scripts/run_validation.py --layers 0,1

# 不运行Layer 3
python scripts/run_validation.py --layers 0,1,2
```

### Q4: 验证通过但置信度低怎么办？

**诊断步骤:**
1. 使用`--verbose`查看详细输出
2. 检查各层输出，找出置信度低的层级
3. 针对性地改进方法论

```bash
python scripts/run_validation.py --method-id ALG_001 --verbose
```

### Q5: 如何控制API预算？

**方法:**
1. 修改配置文件中的预算设置:
```yaml
budget:
  total: 500
  alert_threshold: 0.8
  hard_limit: 0.95
```

2. 优先使用免费层级（Layer 0, Layer 1）
3. 减少Layer 2模型数量

### Q6: 训练需要多少GPU资源？

| 训练模式 | GPU型号 | 估算时间 |
|---------|--------|---------|
| SFT (1.5B) | RTX 4090 | 10-20小时/轮 |
| SFT (7B) | H800 | 20-40小时/轮 |
| 多样性训练 | RTX 4090 | 5-10小时 |

### Q7: 如何查看验证结果统计？

```bash
# 输出到JSON文件
python scripts/run_validation.py --output results/validation_report.json

# 查看结果
cat results/validation_report.json
```

---

## 附录

### 相关文档

| 文档 | 说明 |
|-----|------|
| `docs/api_reference.md` | API参考文档 |
| `docs/validation_system.md` | 验证系统详解 |
| `configs/validation_config.yaml` | 验证配置示例 |
| `configs/training_config.yaml` | 训练配置示例 |

### 目录结构

```
method-thinker/
├── src/                     # 源代码
│   ├── validation/          # 验证系统
│   ├── extraction/          # 提炼系统
│   ├── kb/                  # 知识库管理
│   ├── training/            # 训练模块
│   ├── data/                # 数据处理
│   └── iteration/           # 迭代控制
├── scripts/                 # 运行脚本
│   ├── run_validation.py
│   ├── run_extraction.py
│   └── train_sft.py
├── configs/                 # 配置文件
├── data/                    # 数据文件
│   ├── methodology_kb/      # 方法论知识库
│   ├── test_sets/           # 测试集
│   └── train_data/          # 训练数据
├── tests/                   # 测试文件
├── docs/                    # 文档
└── outputs/                 # 输出目录
```

---

*文档生成时间: 2026-04-07*