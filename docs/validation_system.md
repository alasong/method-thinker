# MethodThinker 验证系统

## 目录

1. [系统概述](#系统概述)
2. [验证层级详解](#验证层级详解)
3. [CLI使用指南](#cli使用指南)
4. [配置说明](#配置说明)
5. [集成指南](#集成指南)
6. [最佳实践](#最佳实践)

---

## 系统概述

MethodThinker验证系统是一个多层混合验证框架，用于自动化评估方法论的有效性。系统通过四个验证层级逐步递进，从低成本快速过滤到高成本深度验证，实现高效的质量控制。

### 设计目标

| 目标 | 指标 |
|-----|-----|
| 验证准确率 | 80-85% |
| 覆盖率 | 95%+ |
| 总预算 | $500 |
| 自动化程度 | 100% |

### 系统架构

```
输入方法论
    │
    ▼
┌─────────────────────────────────────┐
│          验证流水线                  │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Layer 0: 快速过滤             │ │ ← 免费，~10ms
│  │ - 语法检查                    │ │
│  │ - 字段完整性                  │ │
│  │ - 值域约束                    │ │
│  │ - 去重检查                    │ │
│  └───────────────────────────────┘ │
│              │                      │
│              ▼                      │
│  ┌───────────────────────────────┐ │
│  │ Layer 1: 自我反思             │ │ ← 免费，~500ms
│  │ - 内部一致性检查              │ │
│  │ - 多角度反思                  │ │
│  │ - 迭代改进                    │ │
│  └───────────────────────────────┘ │
│              │                      │
│              ▼                      │
│  ┌───────────────────────────────┐ │
│  │ Layer 2: 多模型验证           │ │ ← $0.05/方法，~5s
│  │ - DeepSeek-V3评估             │ │
│  │ - Qwen-Math-72B评估           │ │
│  │ - GPT-4o-mini评估             │ │
│  │ - 多数投票+否决机制           │ │
│  └───────────────────────────────┘ │
│              │                      │
│              ▼                      │
│  ┌───────────────────────────────┐ │
│  │ Layer 3: 测试驱动             │ │ ← GPU时，~30s
│  │ - 相关测试选择                │ │
│  │ - 实际解答验证                │ │
│  │ - 统计分析                    │ │
│  └───────────────────────────────┘ │
│              │                      │
│              ▼                      │
│  ┌───────────────────────────────┐ │
│  │ 集成决策引擎                  │ │ ← 免费，~10ms
│  │ - 加权投票                    │ │
│  │ - 否决检查                    │ │
│  │ - 最终判定                    │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
              │
              ▼
        通过/失败判定
```

---

## 验证层级详解

### Layer 0: 快速过滤层

**职责**: 用最小成本快速识别明显问题，避免后续资源浪费。

**检查内容**:

| 检查项 | 说明 | 延迟 |
|-------|------|------|
| 语法检查 | JSON/YAML格式是否正确 | ~1ms |
| 字段完整性 | 必要字段是否齐全 | ~2ms |
| 值域检查 | difficulty/frequency等是否在合理范围 | ~2ms |
| 去重检查 | method_id是否重复，名称是否相似 | ~3ms |
| 描述质量 | 是否过于泛化 | ~2ms |

**必要字段**:
- `method_id`: 方法唯一标识（格式: `[A-Z]{3}_\d{3}`，如 `ALG_001`）
- `name`: 方法名称（2-50字符）
- `description`: 方法描述（≥20字符）
- `applicability`: 适用条件列表
- `template`: 执行步骤模板（≥2步骤）

**值域约束**:
- `difficulty`: 1-5（难度等级）
- `frequency`: 0-1（使用频率）
- `success_rate`: 0-1（成功率）

**通过条件**: 所有检查项通过，置信度=1.0

**示例输出**:
```
[PASS] Layer 0 (快速过滤) - 置信度: 1.00
  详情:
    fast_filter: passed
```

---

### Layer 1: 自我反思层

**职责**: 让模型自己检查自己，模拟人类自我纠错。

**验证流程**:

1. **自我批判**: 模型扮演批评者角色审视方法论
2. **多角度检查**:
   - 逻辑一致性：步骤是否连贯
   - 适用范围：条件是否明确
   - 可执行性：步骤是否具体
   - 完整性：是否有遗漏
3. **迭代改进**: 发现问题后尝试修正（最多3次）
4. **收敛判断**: 改进是否稳定

**通过条件**: 自我反思通过，或迭代改进后通过

**配置参数**:
- `max_iterations`: 最大迭代次数（默认3）

**示例输出**:
```
[PASS] Layer 1 (自我反思) - 置信度: 0.85
  详情:
    reflections: [
      {iteration: 0, passed: false, issues: ["适用条件过于模糊"]},
      {iteration: 1, passed: true, suggestions: ["已明确适用范围"]}
    ]
```

---

### Layer 2: 多模型验证层（核心）

**职责**: 用多个模型相互验证，打破单一模型偏见。

**模型配置**:

| 模型 | 提供商 | 擅长领域 | 成本 | 延迟 |
|-----|-------|---------|------|------|
| DeepSeek-V3 | DeepSeek | 推理、数学、编程 | $0.02 | 2s |
| Qwen-Math-72B | Alibaba | 数学、推理 | $0.015 | 3s |
| GPT-4o-mini | OpenAI | 通用、推理 | $0.01 | 1.5s |

**评估维度**:

| 维度 | 说明 | 评分范围 |
|-----|------|---------|
| 正确性 | 方法原理是否正确 | 0-10 |
| 完整性 | 步骤是否完整 | 0-10 |
| 适用性 | 适用条件是否合理 | 0-10 |
| 清晰度 | 描述是否清晰 | 0-10 |
| 实用性 | 实际价值 | 0-10 |

**决策机制**:

1. **加权平均分**: `sum(score * confidence) / sum(confidence)`
2. **多数投票**: 认可比例 ≥ 60% → 通过
3. **否决机制**: 2个以上模型评分 < 5 → 强制失败

**通过条件**:
- 无否决 + 认可比例 ≥ `approval_threshold`（默认0.6）

**配置参数**:
- `models`: 使用的模型列表
- `approval_threshold`: 通过阈值（默认0.6）
- `veto_threshold`: 否决阈值（默认0.3）

**示例输出**:
```
[PASS] Layer 2 (多模型验证) - 置信度: 0.67
  详情:
    assessments:
      - model: deepseek_v3, score: 8.5, confidence: 0.9
      - model: qwen_math, score: 7.0, confidence: 0.8
      - model: gpt4o_mini, score: 7.5, confidence: 0.7
    approve_rate: 1.0 (3/3 models approve)
```

---

### Layer 3: 测试驱动验证层

**职责**: 用实际测试验证方法有效性。

**验证流程**:

1. **测试选择**: 根据方法适用条件选择相关测试（最多50个）
2. **解答生成**: 模型使用方法解答问题
3. **答案验证**: 检查答案正确性
4. **统计分析**: 计算成功率、置信度等

**统计指标**:

| 指标 | 说明 |
|-----|------|
| success_rate | 正确率 |
| difficulty_stats | 各难度正确率 |
| avg_execution_time | 平均解答时间 |
| avg_steps | 平均步骤数 |

**通过条件**: success_rate ≥ `pass_threshold`（默认0.6）

**置信度计算**: 基于95%置信区间，`confidence = 1 - 2 * SE`

**配置参数**:
- `min_test_count`: 最少测试数量（默认20）
- `pass_threshold`: 通过阈值（默认0.6）

**示例输出**:
```
[PASS] Layer 3 (测试驱动) - 置信度: 0.82
  详情:
    statistics:
      success_rate: 0.72
      total_tests: 25
      difficulty_stats: {1: {correct: 5/5}, 2: {correct: 8/10}, ...}
```

---

### 集成决策引擎

**职责**: 综合各层验证结果做出最终决策。

**权重配置**:

| 层级 | 权重 | 说明 |
|-----|------|------|
| Layer 0 | 0.05 | 快速过滤权重低 |
| Layer 1 | 0.15 | 自我反思权重中低 |
| Layer 2 | 0.40 | 多模型验证权重高（核心） |
| Layer 3 | 0.40 | 测试验证权重高（核心） |

**决策流程**:

1. **加权得分**: `sum((confidence if passed else 1-confidence) * weight)`
2. **否决检查**: Layer 2/3 高置信度失败 → 强制否决
3. **一致通过**: 所有层级通过 → 通过
4. **核心失败**: Layer 2/3 失败 → 失败
5. **加权判定**: 加权得分 > 0.5 → 通过/失败

**示例输出**:
```
[PASS] 集成决策 - 置信度: 0.87
  详情:
    weighted_score: 0.87
    reason: 所有层级验证通过
```

---

## CLI使用指南

### 基本用法

```bash
# 运行完整验证流水线（所有层级）
python scripts/run_validation.py --kb data/methodology_kb/v0/math_methods.yaml

# 只运行指定层级
python scripts/run_validation.py --layers 0,1 --kb data/methodology_kb/v0/math_methods.yaml

# 验证特定方法
python scripts/run_validation.py --method-id ALG_001 --layers 0,2,3

# 详细输出模式
python scripts/run_validation.py --verbose --layers 0

# 输出结果到文件
python scripts/run_validation.py --output results/validation_report.json
```

### 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--config` | 配置文件路径 | `configs/validation_config.yaml` |
| `--kb` | 知识库路径 | `data/methodology_kb/v0/math_methods.yaml` |
| `--method-id` | 要验证的方法ID | 全部方法 |
| `--layers` | 要运行的层级 | `all`（即0,1,2,3） |
| `--verbose` | 详细输出 | 关闭 |
| `--output` | 结果输出文件 | 无 |
| `--limit` | 限制验证数量 | 全部 |

### 层级选择示例

```bash
# 只做快速过滤（适合大量方法的初步筛选）
python scripts/run_validation.py --layers 0

# Layer 0 + Layer 1（适合验证现有KB方法）
python scripts/run_validation.py --layers 0,1

# 核心验证层（Layer 2 + Layer 3）
python scripts/run_validation.py --layers 2,3

# 完整验证流程
python scripts/run_validation.py --layers all
```

### 输出格式

**标准输出**:
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

各层验证统计:
  Layer 0: 24/24 (100.0%)
```

**JSON输出**（使用`--output`）:
```json
{
  "passed": 24,
  "failed": 0,
  "layer_stats": {
    "0": {"passed": 24, "total": 24}
  },
  "failed_methods": [],
  "elapsed_time": 0.05,
  "timestamp": "2026-04-07T...",
  "config": "configs/validation_config.yaml",
  "layers": [0]
}
```

---

## 配置说明

### 配置文件结构

```yaml
# configs/validation_config.yaml

# Layer 0: 快速过滤
layer0:
  enabled: true
  strict_mode: true  # 严格模式

# Layer 1: 自我反思
layer1:
  enabled: true
  max_iterations: 3  # 最大迭代次数

# Layer 2: 多模型验证
layer2:
  enabled: true
  models:
    - deepseek_v3
    - qwen_math
    - gpt4o_mini
  approval_threshold: 0.6  # 通过阈值
  veto_threshold: 0.3      # 否决阈值

# Layer 3: 测试驱动验证
layer3:
  enabled: true
  min_test_count: 20  # 最少测试数量
  pass_threshold: 0.6 # 通过阈值

# 集成决策权重
ensemble:
  weights:
    0: 0.05  # Layer 0 权重
    1: 0.15  # Layer 1 权重
    2: 0.40  # Layer 2 权重
    3: 0.40  # Layer 3 权重

# 预算配置
budget:
  total: 500           # 总预算（美元）
  alert_threshold: 0.8 # 告警阈值（80%预算时告警）
```

### 参数详解

#### Layer 0 配置

| 参数 | 类型 | 说明 | 默认值 |
|-----|------|------|-------|
| `enabled` | bool | 是否启用 | true |
| `strict_mode` | bool | 严格模式（失败即终止） | true |

#### Layer 1 配置

| 参数 | 类型 | 说明 | 默认值 |
|-----|------|------|-------|
| `enabled` | bool | 是否启用 | true |
| `max_iterations` | int | 最大迭代改进次数 | 3 |

#### Layer 2 配置

| 参数 | 类型 | 说明 | 默认值 |
|-----|------|------|-------|
| `enabled` | bool | 是否启用 | true |
| `models` | list[str] | 验证模型列表 | ['deepseek_v3', 'qwen_math', 'gpt4o_mini'] |
| `approval_threshold` | float | 认可比例阈值 | 0.6 |
| `veto_threshold` | float | 否决阈值 | 0.3 |

#### Layer 3 配置

| 参数 | 类型 | 说明 | 默认值 |
|-----|------|------|-------|
| `enabled` | bool | 是否启用 | true |
| `min_test_count` | int | 最少测试数量 | 20 |
| `pass_threshold` | float | 成功率阈值 | 0.6 |

#### 集成决策配置

| 参数 | 类型 | 说明 | 默认值 |
|-----|------|------|-------|
| `weights` | dict[int, float] | 各层权重 | {0:0.05, 1:0.15, 2:0.40, 3:0.40} |

#### 预算配置

| 参数 | 类型 | 说明 | 默认值 |
|-----|------|------|-------|
| `total` | float | 总预算（美元） | 500 |
| `alert_threshold` | float | 告警阈值 | 0.8 |

---

## 集成指南

### Python API使用

```python
from src.validation.pipeline import ValidationPipeline
from src.validation.config import ValidationConfig
from src.validation.layer0_fast_filter import Layer0FastFilter

# 加载配置
config = ValidationConfig.from_yaml('configs/validation_config.yaml')

# 创建验证流水线
pipeline = ValidationPipeline(config=config)

# 验证单个方法
method = {
    'method_id': 'ALG_001',
    'name': '变量替换法',
    'description': '通过引入新变量简化表达式...',
    'applicability': [{'condition': '存在重复结构'}],
    'template': {'steps': ['识别模式', '选择变量', '变换表达式']}
}

result = pipeline.run(method, skip_layers=[2, 3])  # 只运行Layer 0, 1

# 检查结果
if result.passed:
    print(f"验证通过，置信度: {result.confidence}")
else:
    print(f"验证失败，问题: {result.issues}")
```

### 单独使用各层验证器

```python
# Layer 0: 快速过滤
from src.validation.layer0_fast_filter import Layer0FastFilter

layer0 = Layer0FastFilter(existing_kb={'methods': {}})
result = layer0.validate(method)

# Layer 1: 自我反思（需要模型）
from src.validation.layer1_self_reflection import Layer1SelfReflection

layer1 = Layer1SelfReflection(model=my_model, max_iterations=3)
result = layer1.validate(method)

# Layer 2: 多模型验证（需要模型客户端）
from src.validation.layer2_multi_model import Layer2MultiModelValidation

layer2 = Layer2MultiModelValidation(
    model_clients={'deepseek_v3': deepseek_client, ...},
    budget=500.0,
    approval_threshold=0.6
)
result = layer2.validate(method)

# Layer 3: 测试驱动（需要测试数据）
from src.validation.layer3_test_driven import Layer3TestDrivenValidation, TestCase

test_dataset = [TestCase(problem='...', answer='...', difficulty=3)]
layer3 = Layer3TestDrivenValidation(model=my_model, test_dataset=test_dataset)
result = layer3.validate(method)

# 集成决策
from src.validation.ensemble_decision import EnsembleDecisionEngine, LayerResult

ensemble = EnsembleDecisionEngine(layer_weights={0: 0.05, 1: 0.15, ...})
layer_results = [
    LayerResult(layer=0, passed=True, confidence=1.0, issues=[]),
    LayerResult(layer=1, passed=True, confidence=0.8, issues=[])
]
result = ensemble.decide(layer_results)
```

### 自定义验证流程

```python
def custom_validation_pipeline(method, config):
    """自定义验证流程示例"""
    
    # 1. 快速过滤
    layer0 = Layer0FastFilter({'methods': {}})
    result0 = layer0.validate(method)
    
    if not result0.passed:
        return result0  # 快速失败
    
    # 2. 自我反思（可选）
    if config.layer1.enabled:
        layer1 = Layer1SelfReflection(model, config.layer1.max_iterations)
        result1 = layer1.validate(method)
        
        if not result1.passed:
            # 尝试改进
            improved = layer1._improve_method(method, result1.details['critique'])
            # 重新验证
            result1_retry = layer1.validate(improved)
    
    # 3. 多模型验证（核心方法）
    if config.layer2.enabled:
        layer2 = Layer2MultiModelValidation(clients, budget)
        result2 = layer2.validate(method)
    
    # 4. 集成决策
    ensemble = EnsembleDecisionEngine(config.ensemble.weights)
    return ensemble.decide([
        LayerResult(layer=0, passed=result0.passed, confidence=result0.confidence),
        LayerResult(layer=1, passed=result1.passed, confidence=result1.confidence),
        LayerResult(layer=2, passed=result2.passed, confidence=result2.confidence)
    ])
```

---

## 最佳实践

### 验证流程选择

| 场景 | 推荐层级 | 说明 |
|-----|---------|------|
| 大量新方法初步筛选 | Layer 0 | 低成本，快速过滤 |
| 验证KB中已有方法 | Layer 0 | 检查格式完整性 |
| 新提炼方法论验证 | Layer 0, 1, 2 | 完整验证，排除低质量 |
| 核心方法深度验证 | Layer 0, 1, 2, 3 | 全流程验证 |
| 预算有限时 | Layer 0, 1 | 只用免费层级 |

### 成本控制

```bash
# 优先使用Layer 0过滤
python scripts/run_validation.py --layers 0 --limit 100

# 只有通过Layer 0的方法才进入后续验证
# 在代码中实现分层过滤
```

**预算估算**:
- Layer 0: 免费
- Layer 1: 免费（本地推理）
- Layer 2: ~$0.05/方法（3个模型）
- Layer 3: GPU时成本

### 调试建议

1. **启用verbose模式**: 查看详细验证过程
   ```bash
   python scripts/run_validation.py --verbose --layers 0
   ```

2. **限制验证数量**: 测试时只验证少量方法
   ```bash
   python scripts/run_validation.py --limit 5 --verbose
   ```

3. **输出JSON结果**: 方便后续分析
   ```bash
   python scripts/run_validation.py --output results.json
   ```

4. **检查配置文件**: 确保参数正确
   ```bash
   cat configs/validation_config.yaml
   ```

### 常见问题

**Q: Layer 0全部失败，为什么？**
A: 可能是重复检查误报。验证KB中已有方法时，使用空KB或`--new-method`模式。

**Q: Layer 2需要哪些模型API？**
A: DeepSeek API、Qwen API或OpenAI API。配置在`configs/validation_config.yaml`中。

**Q: 如何跳过某层验证？**
A: 使用`--layers`参数指定要运行的层级，如`--layers 0,1`。

**Q: 验证通过但置信度低怎么办？**
A: 检查各层输出，找出置信度低的层级，针对性地改进方法。

---

## 附录

### 方法格式规范

```yaml
method_id: ALG_001  # 格式: [A-Z]{3}_\d{3}
name: 变量替换法    # 2-50字符
category: ALGEBRA   # 分类
description: 通过引入新变量简化表达式结构，适用于存在重复或对称模式的问题  # ≥20字符
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
difficulty: 3       # 1-5
frequency: 0.85     # 0-1
```

### 相关文件

| 文件 | 说明 |
|-----|------|
| `src/validation/layer0_fast_filter.py` | Layer 0实现 |
| `src/validation/layer1_self_reflection.py` | Layer 1实现 |
| `src/validation/layer2_multi_model.py` | Layer 2实现 |
| `src/validation/layer3_test_driven.py` | Layer 3实现 |
| `src/validation/ensemble_decision.py` | 集成决策引擎 |
| `src/validation/pipeline.py` | 验证流水线 |
| `src/validation/config.py` | 配置管理 |
| `configs/validation_config.yaml` | 配置文件 |
| `scripts/run_validation.py` | CLI脚本 |

---

*文档生成时间: 2026-04-07*
*版本: v1.0*