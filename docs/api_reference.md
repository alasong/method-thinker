# MethodThinker API参考文档

> 版本: v1.0 | 更新日期: 2026-04-07

## 目录

1. [验证系统API](#验证系统api)
2. [提炼系统API](#提炼系统api)
3. [知识库API](#知识库api)
4. [训练系统API](#训练系统api)
5. [数据集API](#数据集api)
6. [迭代控制API](#迭代控制api)

---

## 验证系统API

### ValidationPipeline

验证流水线，整合四层验证系统。

**位置:** `src/validation/pipeline.py`

```python
from src.validation.pipeline import ValidationPipeline, ValidationConfig

# 创建流水线
config = ValidationConfig.from_yaml('configs/validation_config.yaml')
pipeline = ValidationPipeline(config=config)

# 验证单个方法
result = pipeline.run(method_dict)

# 批量验证
results = pipeline.run_batch(methods_list)
```

#### ValidationConfig

```python
@dataclass
class ValidationConfig:
    """验证配置"""
    layer0_enabled: bool = True
    layer1_enabled: bool = True
    layer2_enabled: bool = True
    layer3_enabled: bool = True
    
    ensemble_weights: Dict[int, float] = {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.40}
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ValidationConfig':
        """从YAML文件加载配置"""
        ...
```

#### run()

```python
def run(
    self,
    method: Dict,
    skip_layers: Optional[List[int]] = None
) -> ValidationResult:
    """运行验证流水线
    
    Args:
        method: 方法论字典
        skip_layers: 要跳过的层级列表
        
    Returns:
        ValidationResult: 验证结果
    """
```

---

### Layer0FastFilter

快速过滤器，执行语法、格式和完整性检查。

**位置:** `src/validation/layer0_fast_filter.py`

```python
from src.validation.layer0_fast_filter import Layer0FastFilter, ValidationResult

layer0 = Layer0FastFilter(existing_kb={'methods': {}})
result = layer0.validate(method_dict)

# 检查结果
if result.passed:
    print(f"通过，置信度: {result.confidence}")
else:
    print(f"失败，问题: {result.issues}")
```

#### ValidationResult

```python
@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    layer: int
    confidence: float
    issues: List[str]
    details: Dict = field(default_factory=dict)
```

#### validate()

```python
def validate(self, method: Dict) -> ValidationResult:
    """验证方法论
    
    Args:
        method: 方法论字典
        
    Returns:
        ValidationResult: 验证结果
    """
```

#### 检查内容

| 检查项 | 方法 |
|-------|------|
| 字段完整性 | `_check_required_fields()` |
| 格式验证 | `_check_field_formats()` |
| 值域约束 | `_check_value_constraints()` |
| 去重检查 | `_check_duplicates()` |
| 描述质量 | `_check_description_quality()` |

---

### Layer1SelfReflection

自我反思验证层，让模型自我检查。

**位置:** `src/validation/layer1_self_reflection.py`

```python
from src.validation.layer1_self_reflection import Layer1SelfReflection

layer1 = Layer1SelfReflection(
    model=assistant_model,
    max_iterations=3
)
result = layer1.validate(method_dict)
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `model` | Any | None | 辅助模型 |
| `max_iterations` | int | 3 | 最大迭代改进次数 |

#### 方法

```python
def validate(self, method: Dict) -> ValidationResult:
    """验证方法论
    
    流程:
    1. 自我批判
    2. 多角度反思
    3. 迭代改进
    4. 收敛判断
    """
```

---

### Layer2MultiModelValidation

多模型验证层，使用多个外部模型交叉验证。

**位置:** `src/validation/layer2_multi_model.py`

```python
from src.validation.layer2_multi_model import Layer2MultiModelValidation

layer2 = Layer2MultiModelValidation(
    model_clients={
        'deepseek_v3': deepseek_client,
        'qwen_math': qwen_client,
        'gpt4o_mini': openai_client
    },
    budget=500.0,
    approval_threshold=0.6,
    veto_threshold=0.3
)
result = layer2.validate(method_dict)
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `model_clients` | Dict[str, Client] | {} | 模型客户端字典 |
| `budget` | float | 500.0 | 总预算 |
| `approval_threshold` | float | 0.6 | 认可比例阈值 |
| `veto_threshold` | float | 0.3 | 否决阈值 |

#### 方法

```python
def validate(self, method: Dict) -> ValidationResult:
    """验证方法论
    
    流程:
    1. 并行调用多个模型评估
    2. 收集评估结果
    3. 多数投票决策
    4. 否决机制检查
    """

def assess_method(self, method: Dict, model_name: str) -> Dict:
    """单个模型评估
    
    Returns:
        Dict: 包含score, confidence, assessment
    """
```

---

### Layer3TestDrivenValidation

测试驱动验证层，用实际测试验证方法有效性。

**位置:** `src/validation/layer3_test_driven.py`

```python
from src.validation.layer3_test_driven import Layer3TestDrivenValidation, TestCase

test_dataset = [
    TestCase(
        problem="问题描述",
        answer="正确答案",
        difficulty=3,
        problem_type="ALGEBRA"
    )
]

layer3 = Layer3TestDrivenValidation(
    model=test_model,
    test_dataset=test_dataset,
    min_test_count=20,
    pass_threshold=0.6
)
result = layer3.validate(method_dict)
```

#### TestCase

```python
@dataclass
class TestCase:
    """测试用例"""
    problem: str
    answer: str
    difficulty: int = 3
    problem_type: str = ""
    problem_id: str = ""
```

#### 方法

```python
def validate(self, method: Dict) -> ValidationResult:
    """验证方法论
    
    流程:
    1. 选择相关测试
    2. 生成解答
    3. 验证答案
    4. 统计分析
    """

def select_tests(self, method: Method, max_count: int = 50) -> List[TestCase]:
    """选择相关测试用例"""

def verify_answer(self, predicted: str, expected: str) -> bool:
    """验证答案正确性"""
```

---

### EnsembleDecisionEngine

集成决策引擎，综合各层验证结果。

**位置:** `src/validation/ensemble_decision.py`

```python
from src.validation.ensemble_decision import EnsembleDecisionEngine, LayerResult

ensemble = EnsembleDecisionEngine(
    layer_weights={0: 0.05, 1: 0.15, 2: 0.40, 3: 0.40}
)

layer_results = [
    LayerResult(layer=0, passed=True, confidence=1.0, issues=[]),
    LayerResult(layer=1, passed=True, confidence=0.8, issues=[]),
    LayerResult(layer=2, passed=True, confidence=0.67, issues=[])
]

result = ensemble.decide(layer_results)
```

#### LayerResult

```python
@dataclass
class LayerResult:
    """层级验证结果"""
    layer: int
    passed: bool
    confidence: float
    issues: List[str]
    weight: float = 0.0
    details: Dict = field(default_factory=dict)
```

#### 方法

```python
def decide(self, layer_results: List[LayerResult]) -> ValidationResult:
    """综合决策
    
    流程:
    1. 加权得分计算
    2. 否决检查
    3. 一致通过检查
    4. 最终判定
    """
```

---

## 提炼系统API

### MethodologyExtractor

方法论提取器，从成功解答中提炼方法论。

**位置:** `src/extraction/methodology_extractor.py`

```python
from src.extraction.methodology_extractor import MethodologyExtractor, Method

extractor = MethodologyExtractor(
    assistant_model=model,
    min_samples=3
)

# 从解答提炼
solutions = [
    {'problem': '...', 'solution': '...', 'correct': True, 'problem_type': 'ALGEBRA'}
]
methods = extractor.extract_from_solutions(solutions)
```

#### Method

```python
@dataclass
class Method:
    """方法论定义"""
    method_id: str
    name: str
    category: str
    description: str
    applicability: List[Dict] = field(default_factory=list)
    template: Dict = field(default_factory=dict)
    difficulty: int = 3
    frequency: float = 0.5
    related_methods: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
```

#### 方法

```python
def extract_from_solutions(self, solutions: List[Dict]) -> List[Method]:
    """从解答集提炼方法论
    
    Args:
        solutions: 解答列表
        
    Returns:
        List[Method]: 提炼出的方法论列表
    """

def _cluster_solutions(self, solutions: List[Dict]) -> List[List[Dict]]:
    """聚类相似解答"""

def _extract_method_from_cluster(self, cluster: List[Dict]) -> Optional[Method]:
    """从解答簇提炼方法论"""

def validate_extracted_method(self, method: Method) -> bool:
    """验证提取的方法论是否有效"""
```

---

### PatternMiner

模式挖掘器，分析方法论模式。

**位置:** `src/extraction/pattern_miner.py`

```python
from src.extraction.pattern_miner import PatternMiner

miner = PatternMiner()

# 分析关键词模式
patterns = miner.analyze_keywords(methods_list)

# 分析步骤模式
step_patterns = miner.analyze_steps(methods_list)
```

---

## 知识库API

### KnowledgeBase

方法论知识库管理。

**位置:** `src/kb/knowledge_base.py`

```python
from src.kb.knowledge_base import KnowledgeBase, Method

# 加载知识库
kb = KnowledgeBase.from_yaml('data/methodology_kb/v0/math_methods.yaml')

# 或从JSON加载
kb = KnowledgeBase.load('data/methodology_kb/v0/math_methods.json')
```

#### 方法

```python
def add_method(self, method: Method) -> None:
    """添加方法到知识库"""

def get_method(self, method_id: str) -> Optional[Method]:
    """获取方法"""

def get_applicable_methods(
    self, 
    problem: str, 
    problem_type: str
) -> List[tuple]:
    """获取适用于给定问题的方法
    
    Returns:
        List[tuple]: (Method, score) 列表，按分数降序
    """

def get_methods_by_category(self, category: str) -> List[Method]:
    """获取某类别的所有方法"""

def find_similar_methods(
    self, 
    method: Method, 
    threshold: float = 0.8
) -> List[Method]:
    """查找相似方法"""

def save(self, path: str) -> None:
    """保存知识库"""

@classmethod
def from_yaml(cls, path: str) -> 'KnowledgeBase':
    """从YAML文件加载"""

@classmethod
def load(cls, path: str) -> 'KnowledgeBase':
    """从JSON文件加载"""
```

---

### IncrementalUpdater

增量更新器，管理知识库版本。

**位置:** `src/kb/incremental_updater.py`

```python
from src.kb.incremental_updater import IncrementalUpdater

updater = IncrementalUpdater(kb)

# 添加新方法
updater.add_or_merge(new_method)

# 检查冲突
conflicts = updater.detect_conflicts(new_method)

# 生成更新报告
report = updater.generate_update_report()
```

---

## 训练系统API

### MethodThinkerTrainer

模型训练器。

**位置:** `src/training/trainer.py`

```python
from src.training.trainer import MethodThinkerTrainer, TrainingConfig

config = TrainingConfig(
    base_model="Qwen/Qwen2.5-Math-1.5B",
    output_dir="outputs/checkpoints",
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5
)

trainer = MethodThinkerTrainer(config)
trainer.setup()

# 方法论注入训练
results = trainer.train_methodology_injection(train_data, val_data)

# 多样性训练
results = trainer.train_diversity(train_data, methods_per_problem=4)

# 反思强化训练
results = trainer.train_reflection(train_data)

# 保存检查点
trainer.save_checkpoint("outputs/checkpoints/v1")
```

#### TrainingConfig

```python
@dataclass
class TrainingConfig:
    """训练配置"""
    base_model: str = "Qwen/Qwen2.5-Math-1.5B"
    output_dir: str = "outputs/checkpoints"
    
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 4096
    
    method_selection_weight: float = 0.3
    solution_generation_weight: float = 0.4
    reflection_weight: float = 0.3
```

---

## 数据集API

### MethodologyDataset

方法论训练数据集。

**位置:** `src/data/dataset.py`

```python
from src.data.dataset import MethodologyDataset, MethodologySample

# 加载数据集
dataset = MethodologyDataset('data/train_data/train.json')

# 过滤
algebra_set = dataset.filter_by_type('ALGEBRA')
easy_set = dataset.filter_by_difficulty(1, 2)

# 划分
train, val, test = dataset.split([0.8, 0.1, 0.1])

# 保存
dataset.save('data/train_data/filtered.json')
```

#### MethodologySample

```python
@dataclass
class MethodologySample:
    """方法论数据样本"""
    problem_id: str
    problem: str
    problem_type: str
    difficulty: int
    
    candidate_methods: List[Dict]
    selected_method: str
    selection_reasoning: str
    
    solution_steps: List[str]
    solution_annotations: List[str]
    
    reflection: str
    
    source: str
    verified: bool = False
```

---

### DataCollator

数据整理器。

**位置:** `src/data/collator.py`

```python
from src.data.collator import MethodologyCollator

collator = MethodologyCollator(
    tokenizer=tokenizer,
    max_length=4096,
    pad_to_multiple_of=8=True
)

# 整理批次
batch = collator([sample1, sample2, sample3])
```

---

## 迭代控制API

### IterationController

迭代控制器，管理迭代提炼流程。

**位置:** `src/iteration/iteration_controller.py`

```python
from src.iteration.iteration_controller import IterationController

controller = IterationController(
    max_iterations=5,
    convergence_threshold=0.02
)

# 开始迭代
controller.start_iteration()

# 检查状态
status = controller.get_status()

# 检查收敛
if controller.check_convergence(previous_score, current_score):
    controller.stop_iteration()
```

---

### ConvergenceDetector

收敛检测器。

**位置:** `src/iteration/convergence_detector.py`

```python
from src.iteration.convergence_detector import ConvergenceDetector

detector = ConvergenceDetector(
    threshold=0.02,
    min_iterations=3,
    window_size=5
)

# 检测收敛
is_converged = detector.detect(score_history)

# 检测退化
is_degraded = detector.detect_degradation(previous_score, current_score)
```

---

## 附录

### 类型定义汇总

```python
# 验证结果
@dataclass
class ValidationResult:
    passed: bool
    layer: int
    confidence: float
    issues: List[str]
    details: Dict

# 层级结果
@dataclass
class LayerResult:
    layer: int
    passed: bool
    confidence: float
    issues: List[str]
    weight: float
    details: Dict

# 方法论
@dataclass
class Method:
    method_id: str
    name: str
    category: str
    description: str
    applicability: List[Dict]
    template: Dict
    difficulty: int
    frequency: float
    related_methods: List[str]
    examples: List[str]

# 测试用例
@dataclass
class TestCase:
    problem: str
    answer: str
    difficulty: int
    problem_type: str
    problem_id: str

# 数据样本
@dataclass
class MethodologySample:
    problem_id: str
    problem: str
    problem_type: str
    difficulty: int
    candidate_methods: List[Dict]
    selected_method: str
    selection_reasoning: str
    solution_steps: List[str]
    solution_annotations: List[str]
    reflection: str
    source: str
    verified: bool
```

### 常用导入

```python
# 验证系统
from src.validation.pipeline import ValidationPipeline, ValidationConfig
from src.validation.layer0_fast_filter import Layer0FastFilter, ValidationResult
from src.validation.layer1_self_reflection import Layer1SelfReflection
from src.validation.layer2_multi_model import Layer2MultiModelValidation
from src.validation.layer3_test_driven import Layer3TestDrivenValidation, TestCase
from src.validation.ensemble_decision import EnsembleDecisionEngine, LayerResult

# 提炼系统
from src.extraction.methodology_extractor import MethodologyExtractor, Method
from src.extraction.pattern_miner import PatternMiner

# 知识库
from src.kb.knowledge_base import KnowledgeBase, Method
from src.kb.incremental_updater import IncrementalUpdater

# 训练系统
from src.training.trainer import MethodThinkerTrainer, TrainingConfig

# 数据集
from src.data.dataset import MethodologyDataset, MethodologySample
from src.data.collator import MethodologyCollator

# 迭代控制
from src.iteration.iteration_controller import IterationController
from src.iteration.convergence_detector import ConvergenceDetector
```

---

*文档生成时间: 2026-04-07*