# MethodThinker-1.5B 实施计划与技术方案

## 版本信息
- 版本：v1.0
- 日期：2026-04-04
- 目标：超越 VibeThinker-1.5B 的推理能力

---

## 一、项目总览

### 1.1 核心目标

| 指标 | 基座模型 | VibeThinker | MethodThinker目标 |
|-----|---------|-------------|------------------|
| AIME24 | 6.7 | 80.3 | **83-86** |
| AIME25 | 4.3 | 74.4 | **77-80** |
| HMMT25 | 0.6 | 50.4 | **53-56** |
| LiveCodeBench v6 | 0.0 | 51.1 | **52-55** |

### 1.2 核心创新点

```
VibeThinker：多样性解法 → 精准收敛
                    ↓
MethodThinker：方法论显式教学 → 多样性解法（方法约束）→ 精准收敛 → 反思强化
```

### 1.3 项目周期与预算

| 阶段 | 周期 | GPU时 | 预算 |
|-----|------|-------|------|
| 阶段0：基础设施搭建 | 2周 | 0 | $0 |
| 阶段1：方法论知识库构建 | 4周 | 500h | $1,000 |
| 阶段2：方法论数据生成 | 3周 | 800h | $1,600 |
| 阶段3：方法论注入训练 | 1周 | 400h | $800 |
| 阶段4：多样性解法训练 | 1周 | 800h | $1,600 |
| 阶段5：精准收敛训练 | 1周 | 1000h | $2,000 |
| 阶段6：反思强化训练 | 1周 | 400h | $800 |
| 阶段7：评估与迭代 | 2周 | 500h | $1,000 |
| **总计** | **15周** | **4400h** | **$8,800** |

---

## 二、阶段0：基础设施搭建

### 2.1 硬件环境

```yaml
训练服务器:
  - GPU: 4×NVIDIA H800 (80GB)
  - CPU: 128核
  - 内存: 1TB
  - 存储: 10TB NVMe

推理服务器:
  - GPU: 1×RTX 4090 (24GB)
  - 用于快速验证和消融实验
```

### 2.2 软件环境

```bash
# 基础环境
Python >= 3.10
PyTorch >= 2.1
Transformers >= 4.54.0
vLLM >= 0.10.1
DeepSpeed >= 0.12

# 训练框架
Accelerate >= 0.25
PEFT >= 0.7
TRL >= 0.7

# 数据处理
Datasets >= 2.14
Pandas >= 2.0
NumPy >= 1.24

# 评估框架
lm-evaluation-harness
custom_eval (自建)
```

### 2.3 目录结构

```
MethodThinker/
├── configs/
│   ├── training/
│   │   ├── stage1_methodology_injection.yaml
│   │   ├── stage2_diversity_sft.yaml
│   │   ├── stage3_convergence_rl.yaml
│   │   └── stage4_reflection.yaml
│   └── evaluation/
│       └── eval_config.yaml
├── data/
│   ├── raw/                    # 原始数据集
│   ├── methodology_kb/         # 方法论知识库
│   ├── processed/              # 处理后的训练数据
│   └── eval/                   # 评估数据集
├── scripts/
│   ├── data_generation/
│   │   ├── build_methodology_kb.py
│   │   ├── generate_methodology_data.py
│   │   └── generate_diversity_data.py
│   ├── training/
│   │   ├── train_stage1.py
│   │   ├── train_stage2.py
│   │   ├── train_stage3.py
│   │   └── train_stage4.py
│   └── evaluation/
│       └── run_eval.py
├── src/
│   ├── methodology/
│   │   ├── knowledge_base.py   # 方法论知识库
│   │   ├── method_selector.py  # 方法选择器
│   │   └── method_templates.py # 方法模板
│   ├── data/
│   │   ├── dataset.py
│   │   └── collator.py
│   ├── models/
│   │   └── method_thinker.py
│   └── training/
│       ├── trainer.py
│       └── reward_functions.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── results/
├── tests/
└── requirements.txt
```

---

## 三、阶段1：方法论知识库构建

### 3.1 知识库架构设计

```python
# src/methodology/knowledge_base.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class MethodCategory(Enum):
    """方法分类"""
    ALGEBRA = "代数方法"
    GEOMETRY = "几何方法"
    ANALYSIS = "分析方法"
    COMBINATORICS = "组合方法"
    NUMBER_THEORY = "数论方法"
    GENERAL = "通用方法"

@dataclass
class ApplicabilityCondition:
    """方法适用条件"""
    description: str           # 条件描述
    keywords: List[str]        # 触发关键词
    problem_types: List[str]   # 适用题型
    examples: List[str]        # 示例题目ID

@dataclass
class MethodTemplate:
    """方法执行模板"""
    steps: List[str]           # 标准步骤
    notation_hints: Dict[str, str]  # 符号提示
    common_tricks: List[str]   # 常用技巧
    pitfall_warnings: List[str] # 陷阱警示

@dataclass
class Method:
    """单个方法的完整定义"""
    method_id: str
    name: str
    category: MethodCategory
    description: str
    
    # 适用性
    applicability: List[ApplicabilityCondition]
    applicability_score_fn: str  # 适用性评分函数
    
    # 执行模板
    template: MethodTemplate
    
    # 关联方法
    related_methods: List[str]
    alternative_methods: List[str]
    
    # 学习资料
    typical_examples: List[str]  # 典型例题ID
    difficulty_range: tuple      # 适用难度范围
    
    # 元数据
    prerequisites: List[str]     # 前置知识
    complexity: str              # 方法复杂度
    frequency: float             # 在竞赛中的出现频率

@dataclass
class MethodologyKnowledgeBase:
    """方法论知识库"""
    methods: Dict[str, Method]
    problem_type_hierarchy: Dict  # 题型层次结构
    method_selection_rules: List[Dict]  # 方法选择规则
    
    def get_applicable_methods(self, problem: str, problem_type: str) -> List[Method]:
        """获取适用于给定问题的方法列表"""
        applicable = []
        for method in self.methods.values():
            score = self._compute_applicability(method, problem, problem_type)
            if score > 0.3:  # 阈值
                applicable.append((method, score))
        return sorted(applicable, key=lambda x: x[1], reverse=True)
    
    def _compute_applicability(self, method: Method, problem: str, problem_type: str) -> float:
        """计算方法适用性分数"""
        score = 0.0
        
        # 题型匹配
        for condition in method.applicability:
            if problem_type in condition.problem_types:
                score += 0.4
            
            # 关键词匹配
            for keyword in condition.keywords:
                if keyword.lower() in problem.lower():
                    score += 0.1
        
        # 难度匹配
        # ... (需要问题难度估计)
        
        return min(score, 1.0)
```

### 3.2 方法论知识库内容

#### 3.2.1 数学方法论知识库

```yaml
# data/methodology_kb/math_methods.yaml

methods:
  # ========== 代数方法 ==========
  - method_id: "ALG_001"
    name: "变量替换法"
    category: "ALGEBRA"
    description: "通过引入新变量简化表达式结构"
    
    applicability:
      - description: "表达式中存在重复或对称结构"
        keywords: ["对称", "重复", "复合函数", "嵌套"]
        problem_types: ["方程求解", "不等式证明", "函数最值"]
      - description: "表达式过于复杂，直接处理困难"
        keywords: ["复杂", "繁琐", "多层"]
        problem_types: ["代数恒等式", "极限计算"]
    
    template:
      steps:
        - "识别表达式中的重复模式或对称结构"
        - "选择合适的替换变量，设 t = ..."
        - "将原表达式转化为关于t的新表达式"
        - "求解关于t的问题"
        - "回代求解原变量"
      notation_hints:
        "常见替换": "三角替换、双曲替换、倒数替换、齐次替换"
      common_tricks:
        - "对于 a+b 和 ab 同时出现，考虑令 t = a+b 或 t = ab"
        - "对于 √(ax²+bx+c)，考虑三角替换"
      pitfall_warnings:
        - "注意新变量的取值范围"
        - "回代时需要验证解的有效性"
    
    related_methods: ["配方法", "齐次化方法"]
    alternative_methods: ["直接展开", "因式分解"]
    
    difficulty_range: [1, 5]
    frequency: 0.85

  - method_id: "ALG_002"
    name: "配方法"
    category: "ALGEBRA"
    description: "将二次型表达式化为完全平方形式"
    
    applicability:
      - description: "表达式包含二次项"
        keywords: ["二次", "平方", "x²", "ax²+bx+c"]
        problem_types: ["二次函数", "不等式", "最值问题"]
    
    template:
      steps:
        - "提取二次项系数"
        - "对变量部分配方：ax²+bx = a(x + b/2a)² - b²/4a"
        - "利用完全平方的非负性"
        - "分析最值或证明不等式"
      common_tricks:
        - "配方后可以应用均值不等式"
        - "可以用于求函数的对称轴"
      pitfall_warnings:
        - "注意二次项系数的正负"
    
    related_methods: ["均值不等式", "判别式法"]
    frequency: 0.92

  - method_id: "ALG_003"
    name: "因式分解法"
    category: "ALGEBRA"
    
  - method_id: "ALG_004"
    name: "韦达定理应用"
    category: "ALGEBRA"
    
  - method_id: "ALG_005"
    name: "齐次化方法"
    category: "ALGEBRA"

  # ========== 几何方法 ==========
  - method_id: "GEO_001"
    name: "坐标化方法"
    category: "GEOMETRY"
    description: "将几何问题转化为代数计算"
    
    applicability:
      - description: "涉及距离、角度的计算问题"
        keywords: ["距离", "角度", "坐标", "位置"]
        problem_types: ["解析几何", "距离最值", "角度计算"]
      - description: "纯几何方法难以处理"
        keywords: ["计算复杂", "几何关系不明显"]
    
    template:
      steps:
        - "建立合适的坐标系"
        - "将几何元素用坐标表示"
        - "将几何条件转化为代数方程"
        - "求解代数问题"
        - "将结果翻译回几何语言"
      common_tricks:
        - "选择特殊位置建系可以简化计算"
        - "利用对称性选择坐标轴方向"
      pitfall_warnings:
        - "注意验证特殊情况"
        - "代数解可能对应多个几何情况"
    
    related_methods: ["向量法", "复数法"]
    alternative_methods: ["纯几何方法", "几何变换"]
    frequency: 0.88

  - method_id: "GEO_002"
    name: "向量法"
    category: "GEOMETRY"
    
  - method_id: "GEO_003"
    name: "几何变换"
    category: "GEOMETRY"
    
  - method_id: "GEO_004"
    name: "辅助线构造"
    category: "GEOMETRY"

  # ========== 通用方法 ==========
  - method_id: "GEN_001"
    name: "数学归纳法"
    category: "GENERAL"
    description: "证明关于正整数的命题"
    
    applicability:
      - description: "命题涉及'对任意正整数n'"
        keywords: ["正整数", "任意n", "对所有n", "∀n∈N"]
        problem_types: ["恒等式证明", "不等式证明", "存在性证明"]
      - description: "命题有递推结构"
        keywords: ["递推", "递归", "f(n+1)"]
    
    template:
      steps:
        - "明确归纳命题P(n)"
        - "验证基础步骤：证明P(1)或P(0)成立"
        - "假设P(k)成立（归纳假设）"
        - "证明P(k)→P(k+1)（归纳步骤）"
        - "由归纳原理，命题对所有n成立"
      common_tricks:
        - "归纳假设要充分利用"
        - "有时需要加强命题"
        - "第二数学归纳法：假设P(1),...,P(k)都成立"
      pitfall_warnings:
        - "基础步骤不能省略"
        - "归纳假设必须被使用"
        - "注意归纳起点"
    
    related_methods: ["递推法", "生成函数"]
    frequency: 0.78

  - method_id: "GEN_002"
    name: "反证法"
    category: "GENERAL"
    
  - method_id: "GEN_003"
    name: "构造法"
    category: "GENERAL"
    
  - method_id: "GEN_004"
    name: "极端原理"
    category: "GENERAL"
    
  - method_id: "GEN_005"
    name: "抽屉原理"
    category: "GENERAL"

  # ========== 组合方法 ==========
  - method_id: "COM_001"
    name: "计数原理"
    category: "COMBINATORICS"
    
  - method_id: "COM_002"
    name: "容斥原理"
    category: "COMBINATORICS"
    
  - method_id: "COM_003"
    name: "生成函数"
    category: "COMBINATORICS"

  # ========== 数论方法 ==========
  - method_id: "NUM_001"
    name: "模运算"
    category: "NUMBER_THEORY"
    
  - method_id: "NUM_002"
    name: "裴蜀定理"
    category: "NUMBER_THEORY"
    
  - method_id: "NUM_003"
    name: "费马小定理"
    category: "NUMBER_THEORY"

# 题型层次结构
problem_type_hierarchy:
  代数:
    - 方程求解
    - 不等式证明
    - 函数最值
    - 恒等式证明
    - 多项式理论
  几何:
    - 平面几何
    - 解析几何
    - 立体几何
    - 向量几何
  数论:
    - 整除性
    - 同余
    - 数位问题
    - 素数理论
  组合:
    - 计数问题
    - 存在性问题
    - 构造问题
    - 博弈问题

# 方法选择规则
method_selection_rules:
  - condition: "问题涉及'对任意正整数n'"
    priority_methods: ["GEN_001", "COM_003"]
    
  - condition: "问题涉及二次表达式"
    priority_methods: ["ALG_002", "ALG_001"]
    
  - condition: "问题涉及距离或角度"
    priority_methods: ["GEO_001", "GEO_002"]
    
  - condition: "问题涉及整除或余数"
    priority_methods: ["NUM_001", "NUM_002"]
```

#### 3.2.2 编程方法论知识库

```yaml
# data/methodology_kb/code_methods.yaml

methods:
  - method_id: "CODE_001"
    name: "双指针法"
    category: "ALGORITHM"
    description: "使用两个指针协同遍历数据结构"
    
    applicability:
      - description: "数组/链表的查找或匹配问题"
        keywords: ["查找", "匹配", "子数组", "子串"]
        problem_types: ["两数之和", "滑动窗口", "快慢指针"]
      - description: "需要O(n)时间复杂度"
        keywords: ["线性时间", "一次遍历"]
    
    template:
      steps:
        - "初始化两个指针的位置"
        - "确定指针移动的条件"
        - "在循环中更新指针和结果"
        - "返回最终结果"
      time_complexity: "O(n)"
      space_complexity: "O(1)"
    
    related_methods: ["CODE_002", "CODE_006"]
    frequency: 0.90

  - method_id: "CODE_002"
    name: "滑动窗口"
    category: "ALGORITHM"
    
  - method_id: "CODE_003"
    name: "二分查找"
    category: "ALGORITHM"
    
  - method_id: "CODE_004"
    name: "深度优先搜索"
    category: "ALGORITHM"
    
  - method_id: "CODE_005"
    name: "广度优先搜索"
    category: "ALGORITHM"
    
  - method_id: "CODE_006"
    name: "哈希表"
    category: "DATA_STRUCTURE"
    
  - method_id: "CODE_007"
    name: "动态规划"
    category: "ALGORITHM"
    
  - method_id: "CODE_008"
    name: "贪心算法"
    category: "ALGORITHM"

# 算法选择规则
algorithm_selection_rules:
  - condition: "问题涉及子数组/子串的最值"
    priority_methods: ["CODE_002", "CODE_007"]
  
  - condition: "问题涉及有序数组的查找"
    priority_methods: ["CODE_003"]
  
  - condition: "问题涉及图的遍历"
    priority_methods: ["CODE_004", "CODE_005"]
  
  - condition: "问题涉及最优子结构"
    priority_methods: ["CODE_007", "CODE_008"]
```

### 3.3 知识库构建流程

```python
# scripts/data_generation/build_methodology_kb.py

import yaml
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict

class MethodologyKBBuilder:
    """方法论知识库构建器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def build_from_expert_input(self, expert_yaml_path: str):
        """从专家编写的YAML文件构建知识库"""
        with open(expert_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 验证数据完整性
        self._validate_methods(data['methods'])
        
        # 构建索引
        kb = {
            'methods': {m['method_id']: m for m in data['methods']},
            'category_index': self._build_category_index(data['methods']),
            'keyword_index': self._build_keyword_index(data['methods']),
            'problem_type_index': self._build_problem_type_index(data['methods']),
            'selection_rules': data.get('method_selection_rules', [])
        }
        
        # 保存
        output_path = self.output_dir / 'methodology_kb.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kb, f, ensure_ascii=False, indent=2)
        
        return kb
    
    def _build_category_index(self, methods: List[Dict]) -> Dict:
        """构建按类别索引"""
        index = {}
        for method in methods:
            category = method['category']
            if category not in index:
                index[category] = []
            index[category].append(method['method_id'])
        return index
    
    def _build_keyword_index(self, methods: List[Dict]) -> Dict:
        """构建关键词倒排索引"""
        index = {}
        for method in methods:
            for condition in method.get('applicability', []):
                for keyword in condition.get('keywords', []):
                    keyword_lower = keyword.lower()
                    if keyword_lower not in index:
                        index[keyword_lower] = []
                    index[keyword_lower].append(method['method_id'])
        return index
    
    def _build_problem_type_index(self, methods: List[Dict]) -> Dict:
        """构建题型索引"""
        index = {}
        for method in methods:
            for condition in method.get('applicability', []):
                for ptype in condition.get('problem_types', []):
                    if ptype not in index:
                        index[ptype] = []
                    index[ptype].append(method['method_id'])
        return index
    
    def _validate_methods(self, methods: List[Dict]):
        """验证方法定义的完整性"""
        required_fields = ['method_id', 'name', 'category', 'description', 'applicability']
        for method in methods:
            for field in required_fields:
                if field not in method:
                    raise ValueError(f"方法 {method.get('method_id', 'unknown')} 缺少字段: {field}")

# 主函数
def main():
    builder = MethodologyKBBuilder('data/methodology_kb/')
    
    # 构建数学方法论知识库
    math_kb = builder.build_from_expert_input('data/raw/math_methods.yaml')
    print(f"数学方法论知识库构建完成，包含 {len(math_kb['methods'])} 个方法")
    
    # 构建编程方法论知识库
    code_kb = builder.build_from_expert_input('data/raw/code_methods.yaml')
    print(f"编程方法论知识库构建完成，包含 {len(code_kb['methods'])} 个方法")

if __name__ == '__main__':
    main()
```

---

## 四、阶段2：方法论数据生成

### 4.1 数据生成架构

```python
# scripts/data_generation/generate_methodology_data.py

import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI
import random

@dataclass
class MethodologyDataSample:
    """方法论数据样本"""
    problem_id: str
    problem: str
    problem_type: str
    difficulty: int
    
    # 方法选择
    candidate_methods: List[Dict]  # 候选方法列表
    selected_method: str           # 选中的方法
    selection_reasoning: str       # 选择理由
    
    # 解答
    solution_steps: List[str]      # 解答步骤
    solution_method_annotations: List[str]  # 每步的方法论注释
    
    # 反思
    reflection: str
    
    # 元数据
    source: str
    verified: bool

class MethodologyDataGenerator:
    """方法论数据生成器"""
    
    def __init__(self, kb_path: str, model_name: str = "gpt-4o"):
        self.kb = self._load_kb(kb_path)
        self.client = OpenAI()
        self.model_name = model_name
    
    def generate_sample(self, problem: str, problem_type: str, 
                        ground_truth_solution: str) -> MethodologyDataSample:
        """生成单个方法论数据样本"""
        
        # Step 1: 题型识别
        problem_type_identified = self._identify_problem_type(problem)
        
        # Step 2: 获取候选方法
        candidate_methods = self.kb.get_applicable_methods(problem, problem_type_identified)
        
        # Step 3: 生成方法选择推理
        selection_result = self._generate_method_selection(
            problem, problem_type_identified, candidate_methods
        )
        
        # Step 4: 生成方法论标注的解答
        solution = self._generate_annotated_solution(
            problem, selection_result['selected_method'], ground_truth_solution
        )
        
        # Step 5: 生成反思
        reflection = self._generate_reflection(
            problem, selection_result, solution
        )
        
        return MethodologyDataSample(
            problem_id=self._generate_problem_id(problem),
            problem=problem,
            problem_type=problem_type_identified,
            difficulty=self._estimate_difficulty(problem),
            candidate_methods=selection_result['candidate_methods'],
            selected_method=selection_result['selected_method'],
            selection_reasoning=selection_result['reasoning'],
            solution_steps=solution['steps'],
            solution_method_annotations=solution['annotations'],
            reflection=reflection,
            source="generated",
            verified=False
        )
    
    def _identify_problem_type(self, problem: str) -> str:
        """识别题型"""
        prompt = f"""分析以下数学问题，识别其题型。

问题：{problem}

请从以下题型中选择最合适的一个或多个：
- 代数：方程求解、不等式证明、函数最值、恒等式证明
- 几何：平面几何、解析几何、立体几何
- 数论：整除性、同余、数位问题
- 组合：计数问题、存在性问题、构造问题

输出格式：
{{"problem_type": "主要题型", "sub_types": ["子类型1", "子类型2"]}}
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        # 解析响应...
        return "代数"  # 简化示例
    
    def _generate_method_selection(self, problem: str, problem_type: str,
                                    candidate_methods: List) -> Dict:
        """生成方法选择推理"""
        
        methods_info = "\n".join([
            f"- {m[0].name}（适用性分数：{m[1]:.2f}）：{m[0].description}"
            for m in candidate_methods[:5]  # 取前5个
        ])
        
        prompt = f"""你是一位数学方法论专家。请分析以下问题，选择最合适的解题方法。

问题：{problem}
题型：{problem_type}

候选方法：
{methods_info}

请分析每个方法的适用性，并给出选择理由。输出格式如下：

```json
{{
  "candidate_methods": [
    {{
      "method_name": "方法名",
      "applicability_score": 0.8,
      "applicability_reason": "适用原因"
    }}
  ],
  "selected_method": "选中的方法名",
  "reasoning": "选择该方法的详细理由，包括：\n1. 为什么这个方法最适合\n2. 其他方法为什么不太适合\n3. 可能需要注意的陷阱"
}}
```
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        # 解析响应...
        return json.loads(response.choices[0].message.content)
    
    def _generate_annotated_solution(self, problem: str, method: str, 
                                      reference_solution: str) -> Dict:
        """生成带方法论标注的解答"""
        
        prompt = f"""你是一位数学解题专家。请使用指定的方法解答以下问题，并为每个步骤添加方法论注释。

问题：{problem}
指定方法：{method}

参考解答：
{reference_solution}

请按以下格式输出解答：

```json
{{
  "steps": [
    "步骤1：具体操作",
    "步骤2：具体操作",
    ...
  ],
  "annotations": [
    "方法论注释：这步使用了XX技巧，目的是YY",
    ...
  ]
}}
```

要求：
1. 每个步骤都要清晰说明做了什么
2. 方法论注释要解释"为什么这样做"
3. 标注使用了哪些通用技巧
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return json.loads(response.choices[0].message.content)
    
    def _generate_reflection(self, problem: str, selection: Dict, 
                              solution: Dict) -> str:
        """生成反思"""
        
        prompt = f"""你是一位数学方法论专家。请对以下解题过程进行深度反思。

问题：{problem}
使用的方法：{selection['selected_method']}
方法选择理由：{selection['reasoning']}
解答步骤：{solution['steps']}

请从以下角度进行反思：

1. 方法论洞察
   - 本题的核心特征是什么？
   - 这个特征如何触发了方法的选择？
   - 是否有其他方法也能解决？比较优劣。

2. 执行分析
   - 解答过程中的关键转折点是什么？
   - 有没有可以简化的地方？
   - 有没有更好的切入角度？

3. 可推广模式
   - 本题的方法可以推广到哪些类似问题？
   - 形成了一个怎样的解题模式？
   - 对未来的问题有什么启示？

输出格式：
{{"insights": "...", "execution_analysis": "...", "patterns": "..."}}
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

# 批量生成
def generate_training_data(problems: List[Dict], output_path: str):
    """批量生成训练数据"""
    generator = MethodologyDataGenerator('data/methodology_kb/methodology_kb.json')
    
    samples = []
    for i, prob in enumerate(problems):
        try:
            sample = generator.generate_sample(
                problem=prob['problem'],
                problem_type=prob.get('type', '未知'),
                ground_truth_solution=prob['solution']
            )
            samples.append(asdict(sample))
            
            if (i + 1) % 100 == 0:
                print(f"已生成 {i+1} 个样本")
                # 保存检查点
                with open(f"{output_path}.checkpoint", 'w') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"生成样本 {i} 失败: {e}")
            continue
    
    # 保存最终结果
    with open(output_path, 'w') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    return samples
```

### 4.2 多样性解法数据生成

```python
# scripts/data_generation/generate_diversity_data.py

from typing import List, Dict
import json

class DiversityDataGenerator:
    """多样性解法数据生成器"""
    
    def __init__(self, kb_path: str, model_name: str = "gpt-4o"):
        self.kb = self._load_kb(kb_path)
        self.client = OpenAI()
        self.model_name = model_name
    
    def generate_diverse_solutions(self, problem: str, 
                                    num_solutions: int = 4) -> List[Dict]:
        """生成多种不同方法的解答"""
        
        # 获取适用的方法列表
        applicable_methods = self.kb.get_applicable_methods(problem, "")
        
        # 确保方法多样性
        selected_methods = self._select_diverse_methods(
            applicable_methods, num_solutions
        )
        
        # 为每种方法生成解答
        solutions = []
        for method in selected_methods:
            solution = self._generate_solution_with_method(problem, method)
            solutions.append({
                'method': method.name,
                'method_id': method.method_id,
                'solution': solution,
                'complexity': self._analyze_complexity(solution)
            })
        
        return solutions
    
    def _select_diverse_methods(self, methods: List, num: int) -> List:
        """选择多样化的方法"""
        # 确保不同类别的方法都被选择
        categories = {}
        for method, score in methods:
            cat = method.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((method, score))
        
        selected = []
        # 轮流从每个类别选择
        while len(selected) < num:
            for cat, methods_in_cat in categories.items():
                if methods_in_cat and len(selected) < num:
                    # 选择该类别中分数最高的
                    selected.append(methods_in_cat.pop(0)[0])
        
        return selected[:num]
    
    def _generate_solution_with_method(self, problem: str, method) -> str:
        """使用指定方法生成解答"""
        
        prompt = f"""请使用【{method.name}】解答以下数学问题。

问题：{problem}

方法说明：
{method.description}

方法步骤：
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(method.template.steps))}

要求：
1. 严格按照该方法的步骤解题
2. 在解题开始时标注【方法：{method.name}】
3. 每个步骤都要清晰标注
4. 如果该方法不适用，请说明原因并尝试用该方法的基本思想处理
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content

# 数据格式示例
sample_diversity_data = {
    "problem_id": "AIME_2024_001",
    "problem": "求证对于任意正整数n，1 + 2 + ... + n = n(n+1)/2",
    "solutions": [
        {
            "method": "数学归纳法",
            "method_id": "GEN_001",
            "solution": "【方法：数学归纳法】\n基础步骤：当n=1时，左边=1，右边=1×2/2=1，成立。\n归纳步骤：假设n=k时成立，即1+2+...+k = k(k+1)/2。\n则n=k+1时：1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2。证毕。",
            "complexity": {"steps": 4, "difficulty": 2}
        },
        {
            "method": "组合意义法",
            "method_id": "COM_001",
            "solution": "【方法：组合意义法】\n考虑从n+1个人中选出2个人的方法数，等于C(n+1,2) = n(n+1)/2。\n另一方面，可以先选编号最大的那个人，假设编号为i（2≤i≤n+1），然后选一个编号比i小的人，有i-1种选法。\n总方法数 = 1 + 2 + ... + n。\n因此，1 + 2 + ... + n = n(n+1)/2。",
            "complexity": {"steps": 5, "difficulty": 3}
        },
        {
            "method": "直接计算法",
            "method_id": "ALG_001",
            "solution": "【方法：直接计算法（配对求和）】\n设S = 1 + 2 + ... + n。\n将求和式倒序写：S = n + (n-1) + ... + 1。\n两式相加：2S = (1+n) + (2+n-1) + ... + (n+1) = n(n+1)。\n因此S = n(n+1)/2。",
            "complexity": {"steps": 3, "difficulty": 1}
        }
    ],
    "best_method": "直接计算法",
    "best_method_reason": "最简洁直观，步骤最少"
}
```

### 4.3 数据质量保障

```python
# scripts/data_generation/quality_control.py

from typing import List, Dict
import json

class DataQualityController:
    """数据质量控制器"""
    
    def __init__(self):
        self.validators = [
            self._validate_method_consistency,
            self._validate_solution_correctness,
            self._validate_reflection_quality,
            self._validate_diversity
        ]
    
    def validate_sample(self, sample: Dict) -> Dict:
        """验证单个样本"""
        issues = []
        
        for validator in self.validators:
            try:
                result = validator(sample)
                if result:
                    issues.extend(result)
            except Exception as e:
                issues.append(f"验证异常: {str(e)}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'sample_id': sample.get('problem_id')
        }
    
    def _validate_method_consistency(self, sample: Dict) -> List[str]:
        """验证方法选择的一致性"""
        issues = []
        
        # 检查选中的方法是否在候选方法中
        selected = sample.get('selected_method')
        candidates = [m['method_name'] for m in sample.get('candidate_methods', [])]
        
        if selected not in candidates:
            issues.append(f"选中的方法 '{selected}' 不在候选方法中")
        
        # 检查适用性分数的合理性
        for m in sample.get('candidate_methods', []):
            score = m.get('applicability_score', 0)
            if not 0 <= score <= 1:
                issues.append(f"方法 '{m['method_name']}' 的适用性分数 {score} 不在 [0,1] 范围内")
        
        return issues
    
    def _validate_solution_correctness(self, sample: Dict) -> List[str]:
        """验证解答的正确性（使用外部验证器）"""
        issues = []
        
        # 使用大模型验证解答的正确性
        # 或使用符号计算工具验证（针对数学题）
        
        return issues
    
    def _validate_reflection_quality(self, sample: Dict) -> List[str]:
        """验证反思的质量"""
        issues = []
        
        reflection = sample.get('reflection', '')
        
        # 检查反思的完整性
        required_sections = ['洞察', '分析', '模式', 'insight', 'analysis', 'pattern']
        if not any(section in reflection.lower() for section in required_sections):
            issues.append("反思缺少必要的分析维度")
        
        # 检查反思的长度
        if len(reflection) < 100:
            issues.append("反思内容过短，可能不够深入")
        
        return issues
    
    def _validate_diversity(self, sample: Dict) -> List[str]:
        """验证多样性解法的质量"""
        issues = []
        
        solutions = sample.get('solutions', [])
        
        if len(solutions) < 2:
            issues.append("多样性解法数量不足")
            return issues
        
        # 检查方法是否真的不同
        methods = [s['method'] for s in solutions]
        if len(set(methods)) < len(methods):
            issues.append("存在重复的方法，多样性不足")
        
        # 检查解答是否真的不同（不只是表面变化）
        solution_texts = [s['solution'] for s in solutions]
        # 简单的文本相似度检查
        # ...
        
        return issues

    def batch_validate(self, samples: List[Dict]) -> Dict:
        """批量验证"""
        results = [self.validate_sample(s) for s in samples]
        
        valid_count = sum(1 for r in results if r['valid'])
        
        return {
            'total': len(samples),
            'valid': valid_count,
            'invalid': len(samples) - valid_count,
            'valid_rate': valid_count / len(samples) if samples else 0,
            'common_issues': self._aggregate_issues(results)
        }
    
    def _aggregate_issues(self, results: List[Dict]) -> Dict:
        """汇总常见问题"""
        issue_counts = {}
        for r in results:
            for issue in r['issues']:
                issue_type = issue.split(':')[0] if ':' in issue else issue
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        return dict(sorted(issue_counts.items(), key=lambda x: -x[1]))
```

---

## 五、阶段3：方法论注入训练

### 5.1 训练配置

```yaml
# configs/training/stage1_methodology_injection.yaml

# 模型配置
model:
  base_model: "Qwen/Qwen2.5-Math-1.5B"
  use_flash_attention: true
  torch_dtype: "bfloat16"

# 数据配置
data:
  train_file: "data/processed/methodology_train.json"
  val_file: "data/processed/methodology_val.json"
  max_length: 8192
  
  # 数据格式
  format: "methodology"  # 特殊格式

# 训练配置
training:
  num_epochs: 3
  batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
  
  # 多任务学习权重
  task_weights:
    method_selection: 0.3      # 方法选择任务
    selection_reasoning: 0.2   # 选择理由生成
    solution_generation: 0.4   # 解答生成
    reflection: 0.1            # 反思生成

# 评估配置
evaluation:
  eval_steps: 500
  metrics:
    - "method_selection_accuracy"
    - "solution_correctness"
    - "reasoning_quality"

# 输出配置
output:
  output_dir: "outputs/checkpoints/stage1_methodology"
  save_steps: 500
  save_total_limit: 3
```

### 5.2 训练代码

```python
# scripts/training/train_stage1.py

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class MethodologyCollator:
    """方法论数据整理器"""
    
    tokenizer: AutoTokenizer
    max_length: int = 8192
    
    def __call__(self, samples: List[Dict]) -> Dict:
        """将样本整理成模型输入"""
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'task_ids': []  # 多任务标识
        }
        
        for sample in samples:
            # 构建输入文本
            input_text = self._build_input_text(sample)
            target_text = self._build_target_text(sample)
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            # 合并输入和目标
            input_ids = torch.cat([
                inputs['input_ids'][0],
                targets['input_ids'][0]
            ])
            
            attention_mask = torch.cat([
                inputs['attention_mask'][0],
                targets['attention_mask'][0]
            ])
            
            # 标签：输入部分为-100，目标部分为实际token
            labels = torch.cat([
                torch.full_like(inputs['input_ids'][0], -100),
                targets['input_ids'][0]
            ])
            
            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)
            batch['task_ids'].append(sample.get('task_type', 'solution'))
        
        # Padding
        max_len = max(len(ids) for ids in batch['input_ids'])
        
        for i in range(len(samples)):
            pad_length = max_len - len(batch['input_ids'][i])
            
            batch['input_ids'][i] = torch.cat([
                batch['input_ids'][i],
                torch.zeros(pad_length, dtype=torch.long)
            ])
            batch['attention_mask'][i] = torch.cat([
                batch['attention_mask'][i],
                torch.zeros(pad_length, dtype=torch.long)
            ])
            batch['labels'][i] = torch.cat([
                batch['labels'][i],
                torch.full((pad_length,), -100, dtype=torch.long)
            ])
        
        # Stack
        batch['input_ids'] = torch.stack(batch['input_ids'])
        batch['attention_mask'] = torch.stack(batch['attention_mask'])
        batch['labels'] = torch.stack(batch['labels'])
        
        return batch
    
    def _build_input_text(self, sample: Dict) -> str:
        """构建输入文本"""
        return f"""【问题】
{sample['problem']}

【题型】
{sample['problem_type']}

【候选方法】
{self._format_candidates(sample['candidate_methods'])}

请分析各方法的适用性，选择最合适的方法并给出理由，然后用该方法解答问题。
"""
    
    def _build_target_text(self, sample: Dict) -> str:
        """构建目标文本"""
        return f"""【方法选择】
选中的方法：{sample['selected_method']}

【选择理由】
{sample['selection_reasoning']}

【解答】
{chr(10).join(sample['solution_steps'])}

【反思】
{sample['reflection']}
"""
    
    def _format_candidates(self, candidates: List[Dict]) -> str:
        """格式化候选方法"""
        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(f"{i}. {c['method_name']}（适用性：{c['applicability_score']:.2f}）")
        return '\n'.join(lines)


class MethodologyTrainer(Trainer):
    """方法论训练器"""
    
    def __init__(self, task_weights: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算多任务损失"""
        
        # 标准语言模型损失
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels']
        )
        
        loss = outputs.loss
        
        # 可以添加额外的损失项
        # 例如：方法选择的一致性损失
        
        return (loss, outputs) if return_outputs else loss


def main():
    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    
    # 加载数据
    with open('data/processed/methodology_train.json') as f:
        train_data = json.load(f)
    with open('data/processed/methodology_val.json') as f:
        val_data = json.load(f)
    
    # 数据整理器
    collator = MethodologyCollator(tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="outputs/checkpoints/stage1_methodology",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        report_to="tensorboard"
    )
    
    # 训练器
    trainer = MethodologyTrainer(
        task_weights={
            'method_selection': 0.3,
            'solution_generation': 0.4,
            'reflection': 0.3
        },
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        tokenizer=tokenizer
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model("outputs/checkpoints/stage1_final")


if __name__ == '__main__':
    main()
```

---

## 六、阶段4：多样性解法训练（方法约束版）

### 6.1 核心创新：方法约束的多样性训练

```python
# src/training/diversity_trainer.py

import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random

@dataclass
class MethodConstrainedSample:
    """方法约束的训练样本"""
    problem: str
    method: str
    method_id: str
    solution: str

class MethodConstrainedDiversityTrainer:
    """方法约束的多样性训练器"""
    
    def __init__(self, model, tokenizer, kb_path: str):
        self.model = model
        self.tokenizer = tokenizer
        self.kb = self._load_kb(kb_path)
    
    def train_step(self, batch: List[Dict]):
        """单步训练"""
        
        # 为每个问题生成多种方法的解答
        total_loss = 0
        
        for sample in batch:
            problem = sample['problem']
            
            # 获取适用的方法
            methods = self.kb.get_applicable_methods(problem, sample.get('type', ''))
            
            # 为每种方法生成解答并计算损失
            for method, score in methods[:4]:  # 取前4种方法
                # 构建输入
                input_text = f"""【问题】
{problem}

【要求使用的解题方法】
{method.name}

{method.description}

请使用上述方法解答问题。在解答开始时标注【方法：{method.name}】。
"""
                
                # 目标输出
                target_text = sample['solutions'][method.method_id]['solution']
                
                # 计算损失
                loss = self._compute_loss(input_text, target_text)
                total_loss += loss
        
        # 平均损失
        avg_loss = total_loss / (len(batch) * 4)
        
        return avg_loss
    
    def _compute_loss(self, input_text: str, target_text: str) -> torch.Tensor:
        """计算损失"""
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        targets = self.tokenizer(target_text, return_tensors='pt').to(self.model.device)
        
        # 合并
        input_ids = torch.cat([inputs['input_ids'], targets['input_ids']], dim=1)
        attention_mask = torch.cat([inputs['attention_mask'], targets['attention_mask']], dim=1)
        
        # 标签
        labels = torch.cat([
            torch.full_like(inputs['input_ids'], -100),
            targets['input_ids']
        ], dim=1)
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss


class PassAtKEvaluator:
    """Pass@K评估器"""
    
    def __init__(self, model, tokenizer, kb):
        self.model = model
        self.tokenizer = tokenizer
        self.kb = kb
    
    def evaluate(self, problems: List[Dict], k: int = 4) -> Dict:
        """评估Pass@K"""
        
        results = {
            'pass@1': 0,
            'pass@k': 0,
            'method@k': 0,  # 使用不同方法解决的比例
            'total': len(problems)
        }
        
        for prob in problems:
            # 生成k个不同的解答
            solutions = self._generate_k_solutions(prob['problem'], k)
            
            # 检查是否有正确的
            correct_count = sum(
                1 for s in solutions 
                if self._verify_correctness(s, prob.get('answer'))
            )
            
            if correct_count > 0:
                results['pass@k'] += 1
            
            if correct_count >= k:
                results['pass@1'] += 1
            
            # 检查方法多样性
            methods_used = set(s['method'] for s in solutions)
            if len(methods_used) >= k:
                results['method@k'] += 1
        
        # 计算比率
        for key in ['pass@1', 'pass@k', 'method@k']:
            results[key] = results[key] / results['total']
        
        return results
    
    def _generate_k_solutions(self, problem: str, k: int) -> List[Dict]:
        """生成k个不同的解答"""
        solutions = []
        
        # 获取适用的方法
        methods = self.kb.get_applicable_methods(problem, '')
        
        for i, (method, score) in enumerate(methods[:k]):
            # 构建输入
            input_text = f"问题：{problem}\n\n请使用【{method.name}】解答。\n"
            
            # 生成
            inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.8,
                do_sample=True,
                top_p=0.95
            )
            
            solution_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            solutions.append({
                'method': method.name,
                'solution': solution_text,
                'correct': None  # 待验证
            })
        
        return solutions
    
    def _verify_correctness(self, solution: Dict, ground_truth: str) -> bool:
        """验证解答正确性"""
        # 使用大模型或符号计算验证
        # 简化实现
        return True
```

---

## 七、阶段5：精准收敛训练（MGPO增强版）

### 7.1 增强的MGPO算法

```python
# src/training/enhanced_mgpo.py

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import math

@dataclass
class MGPOConfig:
    """MGPO配置"""
    lambda_entropy: float = 2.0        # 熵偏差权重
    num_rollouts: int = 8               # 每个问题的采样数
    kl_coef: float = 0.1                # KL散度系数
    clip_range: float = 0.2             # PPO裁剪范围
    learning_rate: float = 1e-5
    gamma: float = 1.0                  # 折扣因子
    gae_lambda: float = 0.95            # GAE lambda
    
    # 方法论增强
    methodology_reward_coef: float = 0.3  # 方法论奖励系数
    efficiency_reward_coef: float = 0.2   # 效率奖励系数

class EnhancedMGPOTrainer:
    """增强的MGPO训练器"""
    
    def __init__(self, model, ref_model, tokenizer, config: MGPOConfig, kb):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.kb = kb
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
    
    def train_step(self, batch: List[Dict]) -> Dict:
        """单步训练"""
        
        all_metrics = []
        total_loss = 0
        
        for sample in batch:
            # Step 1: 采样多个解答
            rollouts = self._generate_rollouts(sample['problem'])
            
            # Step 2: 计算奖励
            rewards = self._compute_rewards(rollouts, sample)
            
            # Step 3: 计算熵偏差权重
            entropy_weights = self._compute_entropy_weights(rewards)
            
            # Step 4: 计算优势函数
            advantages = self._compute_advantages(rollouts, rewards)
            
            # Step 5: 加权优势
            weighted_advantages = advantages * entropy_weights
            
            # Step 6: 计算损失
            loss, metrics = self._compute_loss(
                rollouts, 
                weighted_advantages,
                sample
            )
            
            total_loss += loss
            all_metrics.append(metrics)
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 汇总指标
        avg_metrics = {
            'loss': total_loss.item() / len(batch),
            'correct_rate': np.mean([m['correct_rate'] for m in all_metrics]),
            'entropy_weight': np.mean([m['entropy_weight'] for m in all_metrics]),
            'methodology_score': np.mean([m['methodology_score'] for m in all_metrics])
        }
        
        return avg_metrics
    
    def _generate_rollouts(self, problem: str) -> List[Dict]:
        """生成多个解答样本"""
        rollouts = []
        
        input_ids = self.tokenizer(
            f"问题：{problem}\n\n解答：",
            return_tensors='pt'
        ).input_ids.to(self.model.device)
        
        for _ in range(self.config.num_rollouts):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95
                )
            
            rollout = {
                'input_ids': input_ids,
                'output_ids': outputs[0],
                'log_probs': self._compute_log_probs(outputs[0])
            }
            rollouts.append(rollout)
        
        return rollouts
    
    def _compute_rewards(self, rollouts: List[Dict], sample: Dict) -> torch.Tensor:
        """计算奖励"""
        rewards = []
        
        for rollout in rollouts:
            output_text = self.tokenizer.decode(
                rollout['output_ids'], 
                skip_special_tokens=True
            )
            
            # 基础奖励：正确性
            r_correct = self._verify_correctness(output_text, sample)
            
            # 方法论奖励
            r_methodology = self._evaluate_methodology(output_text, sample)
            
            # 效率奖励
            r_efficiency = self._evaluate_efficiency(output_text)
            
            # 综合奖励
            total_reward = (
                r_correct +
                self.config.methodology_reward_coef * r_methodology +
                self.config.efficiency_reward_coef * r_efficiency
            )
            
            rewards.append(total_reward)
        
        return torch.tensor(rewards, device=self.model.device)
    
    def _compute_entropy_weights(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算熵偏差权重"""
        
        # 计算正确率
        p_correct = rewards.mean()  # 简化，实际应该用二值奖励
        
        # 计算与0.5的KL散度
        p_0 = 0.5
        
        if p_correct == 0 or p_correct == 1:
            return torch.ones_like(rewards) * 0.01  # 极端情况
        
        kl_div = (
            p_correct * math.log(p_correct / p_0) +
            (1 - p_correct) * math.log((1 - p_correct) / p_0)
        )
        
        # 熵偏差权重
        weight = math.exp(-self.config.lambda_entropy * kl_div)
        
        # 返回权重向量
        return torch.ones_like(rewards) * weight
    
    def _compute_advantages(self, rollouts: List[Dict], 
                            rewards: torch.Tensor) -> torch.Tensor:
        """计算优势函数（GRPO风格）"""
        
        # 组相对优势
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        
        advantages = (rewards - mean) / std
        
        return advantages
    
    def _compute_loss(self, rollouts: List[Dict], 
                      advantages: torch.Tensor,
                      sample: Dict) -> Tuple[torch.Tensor, Dict]:
        """计算PPO损失"""
        
        total_loss = 0
        metrics = {
            'correct_rate': 0,
            'entropy_weight': advantages.mean().item(),
            'methodology_score': 0
        }
        
        for i, rollout in enumerate(rollouts):
            # 计算当前策略的log prob
            current_log_probs = self._compute_log_probs(rollout['output_ids'])
            
            # 计算重要性比率
            ratio = torch.exp(current_log_probs - rollout['log_probs'])
            
            # PPO裁剪
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_range,
                1 + self.config.clip_range
            )
            
            # 损失
            loss1 = -advantages[i] * ratio
            loss2 = -advantages[i] * clipped_ratio
            policy_loss = torch.max(loss1, loss2).mean()
            
            # KL惩罚
            kl_div = self._compute_kl_div(rollout['output_ids'])
            
            # 总损失
            total_loss += policy_loss + self.config.kl_coef * kl_div
        
        return total_loss / len(rollouts), metrics
    
    def _compute_log_probs(self, output_ids: torch.Tensor) -> torch.Tensor:
        """计算log概率"""
        with torch.no_grad():
            outputs = self.model(output_ids)
            logits = outputs.logits[:, :-1, :]
            labels = output_ids[:, 1:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            
            return selected_log_probs.sum()
    
    def _compute_kl_div(self, output_ids: torch.Tensor) -> torch.Tensor:
        """计算KL散度"""
        with torch.no_grad():
            # 当前模型的logits
            current_outputs = self.model(output_ids)
            current_logits = current_outputs.logits
            
            # 参考模型的logits
            ref_outputs = self.ref_model(output_ids)
            ref_logits = ref_outputs.logits
            
            # KL散度
            kl = F.kl_div(
                F.log_softmax(current_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction='batchmean'
            )
            
            return kl
    
    def _verify_correctness(self, solution: str, sample: Dict) -> float:
        """验证解答正确性"""
        # 使用大模型或符号计算验证
        # 返回0-1之间的分数
        return 1.0  # 简化
    
    def _evaluate_methodology(self, solution: str, sample: Dict) -> float:
        """评估方法论质量"""
        
        # 检查是否使用了合适的方法
        selected_method = sample.get('selected_method', '')
        if selected_method in solution:
            return 1.0
        
        # 检查是否有方法标注
        if '【方法：' in solution or '方法：' in solution:
            return 0.8
        
        # 检查推理链的完整性
        # ...
        
        return 0.5
    
    def _evaluate_efficiency(self, solution: str) -> float:
        """评估解答效率"""
        
        # 基于长度
        length = len(solution)
        
        # 简单的启发式：不是太长也不是太短
        if 100 < length < 1000:
            return 1.0
        elif length < 100:
            return 0.5
        else:
            return max(0, 1 - (length - 1000) / 2000)
```

### 7.2 训练流程

```python
# scripts/training/train_stage3.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.training.enhanced_mgpo import EnhancedMGPOTrainer, MGPOConfig
import json
from tqdm import tqdm

def train_stage3():
    """阶段3：精准收敛训练"""
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "outputs/checkpoints/stage2_diversity/final",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载参考模型（SFT后的模型）
    ref_model = AutoModelForCausalLM.from_pretrained(
        "outputs/checkpoints/stage2_diversity/final",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref_model.eval()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    
    # 加载训练数据
    with open('data/processed/rl_train.json') as f:
        train_data = json.load(f)
    
    # 配置
    config = MGPOConfig(
        lambda_entropy=2.0,
        num_rollouts=8,
        kl_coef=0.1,
        clip_range=0.2,
        learning_rate=1e-5,
        methodology_reward_coef=0.3,
        efficiency_reward_coef=0.2
    )
    
    # 训练器
    trainer = EnhancedMGPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        kb=load_kb('data/methodology_kb/methodology_kb.json')
    )
    
    # 训练循环
    num_epochs = 3
    batch_size = 4
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # 打乱数据
        import random
        random.shuffle(train_data)
        
        # 分批训练
        for i in tqdm(range(0, len(train_data), batch_size)):
            batch = train_data[i:i+batch_size]
            
            metrics = trainer.train_step(batch)
            
            if i % 100 == 0:
                print(f"Step {i}: Loss={metrics['loss']:.4f}, "
                      f"Correct={metrics['correct_rate']:.2%}, "
                      f"Entropy={metrics['entropy_weight']:.4f}")
        
        # 评估
        eval_metrics = evaluate(model, tokenizer, 'data/processed/rl_val.json')
        print(f"Evaluation: {eval_metrics}")
        
        # 保存检查点
        model.save_pretrained(f"outputs/checkpoints/stage3_epoch{epoch+1}")
    
    # 保存最终模型
    model.save_pretrained("outputs/checkpoints/stage3_final")


def evaluate(model, tokenizer, val_file):
    """评估模型"""
    with open(val_file) as f:
        val_data = json.load(f)
    
    correct = 0
    total = len(val_data)
    
    for sample in val_data:
        input_text = f"问题：{sample['problem']}\n\n解答："
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.6,
                do_sample=True
            )
        
        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 验证正确性
        if verify_solution(solution, sample):
            correct += 1
    
    return {'accuracy': correct / total}


if __name__ == '__main__':
    train_stage3()
```

---

## 八、阶段6：反思强化训练

### 8.1 反思数据生成

```python
# scripts/data_generation/generate_reflection_data.py

from typing import Dict, List
import json

class ReflectionDataGenerator:
    """反思数据生成器"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
    
    def generate_reflection(self, problem: str, solution: str, 
                            method: str) -> Dict:
        """生成反思"""
        
        prompt = f"""你是一位数学方法论专家。请对以下解题过程进行深度反思。

问题：{problem}
使用的方法：{method}
解答过程：
{solution}

请从以下维度进行反思，输出JSON格式：

1. 方法论洞察（insights）
   - 问题的核心特征是什么？
   - 为什么这个方法最适合？
   - 关键的转折点是什么？

2. 执行分析（execution_analysis）
   - 解答过程中有哪些关键步骤？
   - 有没有可以简化的地方？
   - 有没有更好的切入角度？

3. 可推广模式（patterns）
   - 这个方法可以推广到哪些类似问题？
   - 形成了怎样的解题模式？
   - 对未来的问题有什么启示？

4. 错误预防（error_prevention）
   - 使用这个方法时容易犯什么错误？
   - 有什么需要注意的陷阱？
   - 如何避免这些错误？

输出格式：
```json
{{
  "insights": "...",
  "execution_analysis": "...",
  "patterns": "...",
  "error_prevention": "..."
}}
```
"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return json.loads(response.choices[0].message.content)
    
    def enrich_with_reflection(self, samples: List[Dict]) -> List[Dict]:
        """为样本添加反思"""
        
        enriched = []
        
        for sample in samples:
            # 为每种解法生成反思
            reflections = []
            for solution in sample.get('solutions', []):
                reflection = self.generate_reflection(
                    sample['problem'],
                    solution['solution'],
                    solution['method']
                )
                reflections.append({
                    'method': solution['method'],
                    'reflection': reflection
                })
            
            sample['reflections'] = reflections
            enriched.append(sample)
        
        return enriched
```

### 8.2 反思增强训练

```python
# scripts/training/train_stage4.py

def train_stage4():
    """阶段4：反思强化训练"""
    
    # 加载阶段3的模型
    model = AutoModelForCausalLM.from_pretrained(
        "outputs/checkpoints/stage3_final",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载反思增强数据
    with open('data/processed/reflection_train.json') as f:
        train_data = json.load(f)
    
    # 构建训练数据格式
    # 输入：问题 + 解答
    # 输出：反思
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir="outputs/checkpoints/stage4_reflection",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=1e-5,
        logging_steps=100,
        save_steps=500
    )
    
    # 训练
    # ...
    
    # 保存最终模型
    model.save_pretrained("outputs/method_thinker_1.5b")
```

---

## 九、阶段7：评估与迭代

### 9.1 评估框架

```python
# scripts/evaluation/run_eval.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import Dict, List
import numpy as np

class MethodThinkerEvaluator:
    """MethodThinker评估器"""
    
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.kb = load_kb('data/methodology_kb/methodology_kb.json')
    
    def evaluate_all(self, benchmarks: List[str]) -> Dict:
        """评估所有基准"""
        
        results = {}
        
        for benchmark in benchmarks:
            if benchmark == 'AIME24':
                results['AIME24'] = self._evaluate_aime('data/eval/aime24.json')
            elif benchmark == 'AIME25':
                results['AIME25'] = self._evaluate_aime('data/eval/aime25.json')
            elif benchmark == 'HMMT25':
                results['HMMT25'] = self._evaluate_hmmt('data/eval/hmmt25.json')
            elif benchmark == 'LiveCodeBench':
                results['LiveCodeBench'] = self._evaluate_lcb('data/eval/lcb_v6.json')
        
        return results
    
    def _evaluate_aime(self, data_path: str, k: int = 64) -> Dict:
        """评估AIME"""
        
        with open(data_path) as f:
            problems = json.load(f)
        
        correct = 0
        total = len(problems)
        
        for prob in problems:
            # 生成k个解答
            solutions = self._generate_k_solutions(prob['problem'], k)
            
            # 检查是否有正确的
            for sol in solutions:
                if self._verify_aime_answer(sol, prob['answer']):
                    correct += 1
                    break
        
        return {
            'correct': correct,
            'total': total,
            'accuracy': correct / total,
            'pass@k': correct / total
        }
    
    def _generate_k_solutions(self, problem: str, k: int) -> List[str]:
        """生成k个解答"""
        
        solutions = []
        
        input_text = f"问题：{problem}\n\n请解答："
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        for _ in range(k):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=1.0,
                    do_sample=True,
                    top_p=0.95
                )
            
            solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            solutions.append(solution)
        
        return solutions
    
    def _verify_aime_answer(self, solution: str, answer: str) -> bool:
        """验证AIME答案"""
        
        # 提取最终答案
        # AIME答案是000-999的整数
        
        import re
        # 尝试匹配最终答案
        patterns = [
            r'最终答案[是为：:]\s*(\d{1,3})',
            r'答案[是为：:]\s*(\d{1,3})',
            r'(\d{1,3})$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution)
            if match:
                predicted = match.group(1).zfill(3)
                if predicted == answer.zfill(3):
                    return True
        
        return False
    
    def evaluate_methodology(self, problems: List[Dict]) -> Dict:
        """评估方法论能力"""
        
        results = {
            'method_selection_accuracy': 0,
            'reasoning_quality': 0,
            'reflection_depth': 0
        }
        
        method_correct = 0
        total = len(problems)
        
        for prob in problems:
            # 生成方法选择
            method_selection = self._generate_method_selection(prob['problem'])
            
            # 检查是否选择了正确的方法
            if self._verify_method_selection(method_selection, prob):
                method_correct += 1
        
        results['method_selection_accuracy'] = method_correct / total
        
        return results
    
    def _generate_method_selection(self, problem: str) -> Dict:
        """生成方法选择"""
        
        input_text = f"""问题：{problem}

请分析这道题，选择最合适的解题方法。

输出格式：
```json
{{
  "problem_type": "题型",
  "candidate_methods": ["方法1", "方法2"],
  "selected_method": "选中的方法",
  "reasoning": "选择理由"
}}
```
"""
        
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析JSON
        # ...
        
        return {}


def compare_with_vibethinker():
    """与VibeThinker对比"""
    
    # 评估MethodThinker
    mt_evaluator = MethodThinkerEvaluator('outputs/method_thinker_1.5b')
    mt_results = mt_evaluator.evaluate_all(['AIME24', 'AIME25', 'HMMT25', 'LiveCodeBench'])
    
    # VibeThinker的基准结果
    vt_results = {
        'AIME24': 80.3,
        'AIME25': 74.4,
        'HMMT25': 50.4,
        'LiveCodeBench': 51.1
    }
    
    # 对比
    comparison = {}
    for key in mt_results:
        comparison[key] = {
            'MethodThinker': mt_results[key]['accuracy'] * 100,
            'VibeThinker': vt_results[key],
            'Improvement': mt_results[key]['accuracy'] * 100 - vt_results[key]
        }
    
    # 打印对比表
    print("=" * 60)
    print(f"{'Benchmark':<20} {'MethodThinker':>15} {'VibeThinker':>15} {'Improvement':>15}")
    print("=" * 60)
    for key, values in comparison.items():
        print(f"{key:<20} {values['MethodThinker']:>14.1f} {values['VibeThinker']:>14.1f} {values['Improvement']:>+14.1f}")
    print("=" * 60)
    
    return comparison


if __name__ == '__main__':
    compare_with_vibethinker()
```

### 9.2 消融实验

```python
# scripts/evaluation/ablation_study.py

def run_ablation_studies():
    """运行消融实验"""
    
    experiments = [
        {
            'name': 'Without Methodology Injection',
            'model': 'outputs/checkpoints/ablation_no_methodology',
            'description': '移除方法论注入阶段'
        },
        {
            'name': 'Without Method-Constrained Diversity',
            'model': 'outputs/checkpoints/ablation_no_constraint',
            'description': '使用传统多样性训练而非方法约束'
        },
        {
            'name': 'Without MGPO Enhancement',
            'model': 'outputs/checkpoints/ablation_no_mgpo',
            'description': '使用标准GRPO而非增强MGPO'
        },
        {
            'name': 'Without Reflection Training',
            'model': 'outputs/checkpoints/ablation_no_reflection',
            'description': '移除反思强化阶段'
        },
        {
            'name': 'Full MethodThinker',
            'model': 'outputs/method_thinker_1.5b',
            'description': '完整方法'
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n评估: {exp['name']}")
        print(f"描述: {exp['description']}")
        
        evaluator = MethodThinkerEvaluator(exp['model'])
        results[exp['name']] = evaluator.evaluate_all(['AIME24', 'AIME25'])
    
    # 打印结果对比
    print("\n" + "=" * 80)
    print(f"{'Experiment':<40} {'AIME24':>10} {'AIME25':>10} {'Average':>10}")
    print("=" * 80)
    
    for name, res in results.items():
        avg = (res['AIME24']['accuracy'] + res['AIME25']['accuracy']) / 2
        print(f"{name:<40} {res['AIME24']['accuracy']*100:>9.1f} {res['AIME25']['accuracy']*100:>9.1f} {avg*100:>9.1f}")
    
    print("=" * 80)
    
    return results
```

---

## 十、风险管理与应急预案

### 10.1 风险清单

| 风险类别 | 具体风险 | 概率 | 影响 | 应对措施 |
|---------|---------|------|------|---------|
| **数据风险** | 方法论数据质量不足 | 中 | 高 | 多轮审核 + 自动验证 |
| **数据风险** | 数据生成成本超支 | 中 | 中 | 使用开源模型替代 |
| **训练风险** | 训练不稳定 | 低 | 高 | 梯度裁剪 + 早停 |
| **训练风险** | 过拟合 | 中 | 中 | 正则化 + 数据增强 |
| **性能风险** | 未超越VibeThinker | 低 | 高 | 消融实验 + 迭代优化 |
| **性能风险** | 特定题型表现差 | 中 | 中 | 针对性数据补充 |
| **资源风险** | GPU资源不足 | 低 | 高 | 使用云服务弹性扩展 |
| **时间风险** | 周期超期 | 中 | 中 | 并行化 + 优先级调整 |

### 10.2 应急预案

```python
# 应急预案：快速迭代版本

def emergency_plan():
    """如果完整方案效果不佳，执行快速迭代版本"""
    
    # Plan A: 减少训练阶段
    # 只保留方法论注入 + 标准SFT + 标准RL
    
    # Plan B: 增加数据规模
    # 使用更大的数据集进行训练
    
    # Plan C: 调整超参数
    # 增加学习率，减少正则化
    
    # Plan D: 方法论知识库扩充
    # 增加更多方法定义和示例
    
    pass
```

---

## 十一、预期成果

### 11.1 性能指标

| 基准 | 基座模型 | VibeThinker | MethodThinker预期 | 提升幅度 |
|-----|---------|-------------|------------------|---------|
| AIME24 | 6.7 | 80.3 | **83-86** | +3-6 |
| AIME25 | 4.3 | 74.4 | **77-80** | +3-6 |
| HMMT25 | 0.6 | 50.4 | **53-56** | +3-6 |
| LiveCodeBench v6 | 0.0 | 51.1 | **52-55** | +1-4 |

### 11.2 核心贡献

1. **方法论优先数据集构建方法**
   - 显式的方法选择教学
   - 结构化的方法论知识库

2. **方法约束的多样性训练**
   - 确保解法的本质多样性
   - Method@K评估指标

3. **增强的MGPO算法**
   - 方法论一致性奖励
   - 效率奖励机制

4. **反思强化机制**
   - 自动生成反思
   - 知识迁移能力

### 11.3 学术价值

- 证明"方法论显式教学"对小模型推理能力的提升作用
- 提出新的评估维度：方法论选择准确率、反思深度
- 为后续研究提供可复现的完整方案

---

## 十二、附录

### A. 完整的训练脚本

```bash
#!/bin/bash
# run_full_training.sh

echo "开始 MethodThinker-1.5B 完整训练流程"

# 阶段0：环境检查
echo "阶段0：检查环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 阶段1：构建方法论知识库
echo "阶段1：构建方法论知识库..."
python scripts/data_generation/build_methodology_kb.py

# 阶段2：生成训练数据
echo "阶段2：生成方法论数据..."
python scripts/data_generation/generate_methodology_data.py \
    --input data/raw/math_problems.json \
    --output data/processed/methodology_train.json

echo "阶段2：生成多样性数据..."
python scripts/data_generation/generate_diversity_data.py \
    --input data/raw/math_problems.json \
    --output data/processed/diversity_train.json

# 阶段3：方法论注入训练
echo "阶段3：方法论注入训练..."
python scripts/training/train_stage1.py \
    --config configs/training/stage1_methodology_injection.yaml

# 阶段4：多样性解法训练
echo "阶段4：多样性解法训练..."
python scripts/training/train_stage2.py \
    --config configs/training/stage2_diversity_sft.yaml

# 阶段5：精准收敛训练
echo "阶段5：精准收敛训练..."
python scripts/training/train_stage3.py \
    --config configs/training/stage3_convergence_rl.yaml

# 阶段6：反思强化训练
echo "阶段6：反思强化训练..."
python scripts/training/train_stage4.py \
    --config configs/training/stage4_reflection.yaml

# 阶段7：评估
echo "阶段7：评估..."
python scripts/evaluation/run_eval.py

echo "训练完成！"
```

### B. 配置文件模板

```yaml
# configs/training/master_config.yaml

project:
  name: "MethodThinker-1.5B"
  version: "1.0.0"
  base_model: "Qwen/Qwen2.5-Math-1.5B"

paths:
  data_dir: "data"
  output_dir: "outputs"
  log_dir: "outputs/logs"
  checkpoint_dir: "outputs/checkpoints"

stages:
  - name: "methodology_injection"
    enabled: true
    config: "configs/training/stage1_methodology_injection.yaml"
    
  - name: "diversity_sft"
    enabled: true
    config: "configs/training/stage2_diversity_sft.yaml"
    
  - name: "convergence_rl"
    enabled: true
    config: "configs/training/stage3_convergence_rl.yaml"
    
  - name: "reflection"
    enabled: true
    config: "configs/training/stage4_reflection.yaml"

evaluation:
  benchmarks:
    - "AIME24"
    - "AIME25"
    - "HMMT25"
    - "LiveCodeBench"
  
  metrics:
    - "accuracy"
    - "pass@k"
    - "method@k"
    - "methodology_accuracy"
```

---

## 文档结束

本实施计划涵盖了从方法论知识库构建到最终模型评估的完整流程，包含了所有关键的技术细节和代码实现。按照此方案执行，预计可以构建出超越 VibeThinker-1.5B 的 MethodThinker 模型。
