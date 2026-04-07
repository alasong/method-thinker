# MethodThinker 训练数据说明

> 本文档介绍MethodThinker项目的训练数据格式、结构和使用方法

## 目录结构

```
data/train_data/
├── train.json          # 训练集数据
├── val.json            # 验证集数据
├── test.json           # 测试集数据（内部评估用）
├── README.md           # 本说明文档
└── backups/            # 数据备份目录
```

## 数据格式

### 标准数据格式 (JSON)

每个样本包含以下字段:

```json
{
  "problem_id": "AIME2024001",
  "problem": "问题描述...",
  "problem_type": "ALGEBRA",
  "difficulty": 3,
  
  "candidate_methods": [
    {"name": "substitution", "description": "代入法"},
    {"name": "induction", "description": "归纳法"}
  ],
  "selected_method": "substitution",
  "selection_reasoning": "选择代入法的理由...",
  
  "solution_steps": [
    "步骤1: 设变量...",
    "步骤2: 建立方程...",
    "步骤3: 求解..."
  ],
  "solution_annotations": [
    "annotation1: 关键点说明",
    "annotation2: 方法应用点"
  ],
  
  "reflection": "解题后的反思和总结...",
  
  "source": "aime",
  "verified": true
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|-----|------|-----|------|
| `problem_id` | string | 是 | 问题唯一标识符，格式: `[来源][年份][编号]` |
| `problem` | string | 是 | 问题完整描述 |
| `problem_type` | string | 是 | 问题类型，见下文类型列表 |
| `difficulty` | int | 是 | 难度级别 (1-5) |
| `candidate_methods` | list | 可选 | 候选方法列表 |
| `selected_method` | string | 可选 | 最终选择的方法 |
| `selection_reasoning` | string | 可选 | 方法选择的推理过程 |
| `solution_steps` | list | 是 | 解答步骤列表 |
| `solution_annotations` | list | 可选 | 步骤注解 |
| `reflection` | string | 可选 | 解题反思 |
| `source` | string | 可选 | 数据来源标识 |
| `verified` | bool | 可选 | 是否已验证正确性 |

### 问题类型列表

| 类型代码 | 中文名称 | 英文名称 | 示例题型 |
|---------|---------|---------|---------|
| `ALGEBRA` | 代数 | Algebra | 方程、多项式、不等式 |
| `GEOMETRY` | 几何 | Geometry | 平面几何、立体几何 |
| `NUMBER_THEORY` | 数论 | Number Theory | 整除、质数、模运算 |
| `COMBINATORICS` | 组合 | Combinatorics | 计数、排列组合 |
| `CALCULUS` | 微积分 | Calculus | 极限、导数、积分 |
| `PROBABILITY` | 概率 | Probability | 概率计算、期望 |
| `LOGIC` | 逻辑推理 | Logic | 逻辑证明、反证 |
| `GENERAL` | 通用 | General | 跨领域综合题 |

### 难度级别定义

| 难度 | 名称 | 描述 | 目标比例 |
|-----|------|------|---------|
| 1 | 基础 | 基础概念和简单计算 | 15% |
| 2 | 入门 | 入门级竞赛题目 | 25% |
| 3 | 中等 | 中等难度竞赛题目 | 30% |
| 4 | 进阶 | 进阶竞赛题目 | 20% |
| 5 | 挑战 | 高难度竞赛题目 | 10% |

## 数据来源

### 支持的数据来源

| 来源标识 | 名称 | 说明 |
|---------|------|------|
| `aime` | AIME | American Invitational Mathematics Examination |
| `amc` | AMC | American Mathematics Competition |
| `hmmt` | HMMT | Harvard-MIT Mathematics Tournament |
| `olympiad` | 数学奥林匹克 | 各类国家级/国际级数学竞赛 |
| `textbook` | 教材习题 | 数学教材中的典型习题 |
| `generated` | AI生成 | 通过数据生成脚本创建的数据 |

### 数据来源权重

合并数据集时，不同来源的数据会按照权重进行优先级排序:

- `aime`, `hmmt`, `amc`: 权重 1.0 (标准竞赛数据)
- `olympiad`: 权重 1.2 (高难度竞赛数据优先)
- `textbook`: 权重 0.8 (基础数据)
- `generated`: 权重 0.7 (AI生成数据需验证)

## 数据集管理

### CLI工具使用

使用 `scripts/manage_datasets.py` 进行数据集管理:

```bash
# 加载并验证数据集
python scripts/manage_datasets.py load data/train_data/train.json --validate

# 显示统计信息
python scripts/manage_datasets.py stats data/train_data/train.json --verbose

# 合并数据集
python scripts/manage_datasets.py merge data/*.json -o data/merged.json

# 分割数据集 (训练/验证/测试)
python scripts/manage_datasets.py split data/raw_data.json \
    --ratio 0.8,0.1,0.1 \
    -o data/train_data

# 格式转换
python scripts/manage_datasets.py convert data/train.json --to jsonl

# 按难度筛选
python scripts/manage_datasets.py filter data/train.json \
    --difficulty 3-5 \
    -o data/filtered.json

# 平衡类别分布
python scripts/manage_datasets.py balance data/train.json \
    --strategy hybrid \
    -o data/balanced.json

# 检查类别平衡状态
python scripts/manage_datasets.py balance data/train.json --check

# 导出数据集摘要
python scripts/manage_datasets.py summary data/train.json -o SUMMARY.md
```

### 配置文件

数据集配置位于 `configs/dataset_config.yaml`:

- 分割比例设置
- 难度分布目标
- 类别平衡策略
- 数据验证规则
- 输出格式配置

## 数据处理流程

### 标准处理流程

```
原始数据 → 验证 → 去重 → 分割 → 平衡 → 增强 → 最终数据集
```

1. **原始数据收集**: 从竞赛、教材等来源收集原始数据
2. **数据验证**: 检查字段完整性、格式正确性
3. **去重处理**: 基于 `problem_id` 或语义相似度去重
4. **数据分割**: 按比例分割为训练/验证/测试集
5. **类别平衡**: 确保问题类型和难度分布均衡
6. **数据增强**: 通过改写、方法变体等方式增强数据
7. **最终输出**: 生成可用于训练的数据集

### 数据分割策略

默认分割比例:
- 训练集: 80%
- 验证集: 10%
- 测试集: 10%

分割采用**分层抽样**策略，确保:
- 各问题类型在各数据集中比例一致
- 各难度级别在各数据集中比例一致

### 类别平衡策略

三种平衡策略可选:

| 策略 | 方法 | 适用场景 |
|-----|------|---------|
| `oversample` | 过采样少数类 | 数据量较小，不丢失信息 |
| `undersample` | 欠采样多数类 | 数据量较大，减少冗余 |
| `hybrid` | 混合策略 | 中等数据量，平衡效率与完整性 |

## 数据验证

### 验证规则

自动验证检查以下内容:

1. **字段完整性**: 必填字段是否存在
2. **值域约束**: 难度值是否在 [1,5] 范围
3. **类型有效性**: 问题类型是否在有效列表中
4. **唯一性**: `problem_id` 是否唯一
5. **长度约束**: 解答步骤至少有1个

### 验证示例

```bash
# 加载并验证数据集
python scripts/manage_datasets.py load data/train.json --validate

# 输出示例:
# 加载: data/train.json (1000 样本)
# 验证发现 5 个问题:
#   - 样本 42: 难度值 6 超出范围 [1,5]
#   - 样本 78: 缺少必要字段 'problem_type'
#   - ...
```

## 数据增强

### 支持的增强方法

| 方法 | 说明 | 默认比例 |
|-----|------|---------|
| `paraphrase` | 同义改写问题描述 | 20% |
| `difficulty_adjust` | 难度级别调整 | 10% |
| `method_variation` | 同一问题的不同解法 | 15% |

### 增强约束

- 保持答案一致性
- 保持问题类型不变
- 最大增强比例不超过50%

## 统计分析

### 统计维度

生成的统计报告包含:

- 总样本数
- 问题类型分布
- 难度分布
- 验证比例
- 平均解答步骤数
- 数据来源分布
- 常用方法统计

### 统计示例输出

```
数据集统计信息
==================================================
总样本数: 1000
验证比例: 85.00%
平均解答步骤: 5.2

问题类型分布:
  ALGEBRA: 250 (25.0%)
  GEOMETRY: 150 (15.0%)
  NUMBER_THEORY: 150 (15.0%)
  COMBINATORICS: 200 (20.0%)
  ...

难度分布:
  难度 1: 150 (15.0%)
  难度 2: 250 (25.0%)
  难度 3: 300 (30.0%)
  难度 4: 200 (20.0%)
  难度 5: 100 (10.0%)
```

## 与训练系统集成

### 训练脚本使用

训练数据与 `scripts/train_sft.py` 配合使用:

```bash
# 使用默认数据路径训练
python scripts/train_sft.py --config configs/training_config.yaml

# 数据路径在 configs/training_config.yaml 中配置:
data:
  train_path: "data/train_data/train.json"
  val_path: "data/train_data/val.json"
  test_path: "data/train_data/test.json"
```

### 数据加载流程

训练时数据通过 `src/data/dataset.py` 加载:

```python
from src.data.dataset import MethodologyDataset

# 加载训练数据
train_data = MethodologyDataset("data/train_data/train.json")

# 查看数据
print(f"训练样本数: {len(train_data)}")
for sample in train_data.samples[:5]:
    print(f"问题: {sample.problem[:50]}...")
    print(f"类型: {sample.problem_type}, 难度: {sample.difficulty}")
```

## 最佳实践

### 数据收集建议

1. **多样化来源**: 从多个竞赛/教材收集数据
2. **难度覆盖**: 确保1-5难度级别都有足够数据
3. **类型平衡**: 各问题类型比例尽量均衡
4. **验证优先**: 使用已验证的数据源

### 数据管理建议

1. **定期备份**: 修改前自动备份原始数据
2. **版本管理**: 使用日期/版本号标识数据集版本
3. **日志记录**: 记录每次数据处理操作
4. **配置分离**: 数据处理参数通过配置文件管理

### 常见问题解决

| 问题 | 解决方案 |
|-----|---------|
| 数据不平衡 | 使用 `balance` 命令进行平衡 |
| 缺少验证数据 | 使用 `filter --verified true` 筛选 |
| 格式不兼容 | 使用 `convert` 命令转换格式 |
| 数据量不足 | 启用数据增强功能 |

## 相关文档

- [训练配置说明](../configs/training_config.yaml)
- [数据集配置](../configs/dataset_config.yaml)
- [验证系统文档](../docs/validation_system.md)
- [项目README](../README.md)

## 更新日志

| 日期 | 版本 | 更新内容 |
|-----|------|---------|
| 2026-04-07 | v1.0 | 初始版本，定义数据格式和管理流程 |

---

如有问题或建议，请参考项目文档或联系项目维护者。