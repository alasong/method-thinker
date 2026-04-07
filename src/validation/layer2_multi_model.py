"""Layer 2: 多模型交叉验证

用多个模型相互验证，打破单一模型偏见。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import asyncio
import json


@dataclass
class ModelAssessment:
    """单个模型的评估"""
    model_name: str
    score: float  # 0-10
    confidence: float  # 0-1
    issues: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    layer: int
    confidence: float
    issues: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: str
    strength: List[str]
    cost_per_call: float
    latency: float


class Layer2MultiModelValidation:
    """Layer 2: 多模型交叉验证

    使用多个不同的模型独立评估方法论，通过多数投票和否决机制
    判断方法论的有效性。

    Attributes:
        model_clients: 模型客户端字典
        budget: 总预算
        approval_threshold: 通过阈值
        veto_threshold: 否决阈值
    """

    DEFAULT_MODELS = {
        'deepseek_v3': ModelConfig(
            name='DeepSeek-V3',
            provider='DeepSeek',
            strength=['推理', '数学', '编程'],
            cost_per_call=0.02,
            latency=2.0
        ),
        'qwen_math': ModelConfig(
            name='Qwen-Math-72B',
            provider='Alibaba',
            strength=['数学', '推理'],
            cost_per_call=0.015,
            latency=3.0
        ),
        'gpt4o_mini': ModelConfig(
            name='GPT-4o-mini',
            provider='OpenAI',
            strength=['通用', '推理'],
            cost_per_call=0.01,
            latency=1.5
        )
    }

    def __init__(
        self,
        model_clients: Dict,
        budget: float = 500.0,
        approval_threshold: float = 0.6,
        veto_threshold: float = 0.3
    ):
        """初始化多模型验证器

        Args:
            model_clients: 模型客户端字典
            budget: 总预算
            approval_threshold: 通过阈值（认可比例）
            veto_threshold: 否决阈值
        """
        self.model_clients = model_clients
        self.budget = budget
        self.spent = 0.0
        self.approval_threshold = approval_threshold
        self.veto_threshold = veto_threshold
        self.model_registry = self.DEFAULT_MODELS

    def validate(self, method: Dict) -> ValidationResult:
        """多模型验证

        Args:
            method: 待验证的方法

        Returns:
            ValidationResult: 验证结果
        """
        selected_models = self._select_models(method)
        assessments = self._parallel_assess(method, selected_models)
        decision = self._ensemble_decision(assessments)

        return ValidationResult(
            passed=decision['passed'],
            layer=2,
            confidence=decision['confidence'],
            issues=decision['issues'],
            details={
                'assessments': [a.__dict__ for a in assessments],
                'decision': decision
            }
        )

    def _select_models(self, method: Dict) -> List[str]:
        """选择验证模型"""
        category = method.get('category', 'GENERAL')

        selection_rules = {
            'ALGEBRA': ['deepseek_v3', 'qwen_math', 'gpt4o_mini'],
            'GEOMETRY': ['qwen_math', 'deepseek_v3', 'gpt4o_mini'],
            'NUMBER_THEORY': ['qwen_math', 'deepseek_v3', 'gpt4o_mini'],
            'COMBINATORICS': ['deepseek_v3', 'gpt4o_mini'],
            'GENERAL': ['deepseek_v3', 'gpt4o_mini']
        }

        return selection_rules.get(category, selection_rules['GENERAL'])

    def _parallel_assess(self, method: Dict, model_keys: List[str]) -> List[ModelAssessment]:
        """并行评估"""
        prompt = self._build_assessment_prompt(method)
        assessments = []

        for model_key in model_keys:
            if model_key not in self.model_clients:
                continue

            try:
                client = self.model_clients[model_key]
                response = client.generate(prompt, temperature=0.3)
                assessment = self._parse_assessment(model_key, response)

                # 记录成本
                cost = self.model_registry[model_key].cost_per_call
                self.spent += cost

            except Exception as e:
                assessment = ModelAssessment(
                    model_name=model_key,
                    score=5.0,
                    confidence=0.3,
                    issues=[f'模型调用失败: {e}'],
                    reasoning='调用失败'
                )

            assessments.append(assessment)

        return assessments

    def _build_assessment_prompt(self, method: Dict) -> str:
        """构建评估提示"""
        return f"""你是一位数学方法论评审专家，请评估以下方法论的质量。

方法名称：{method.get('name', '')}
方法描述：{method.get('description', '')}
适用条件：{method.get('applicability', [])}
执行步骤：{method.get('template', {}).get('steps', [])}

请从以下维度评分（每项0-10分）：
1. 正确性：方法原理是否正确？
2. 完整性：步骤是否完整？
3. 适用性：适用条件是否合理？
4. 清晰度：描述是否清晰？
5. 实用性：这个方法有实际价值吗？

输出JSON格式：
{{
    "scores": {{"correctness": X, "completeness": X, "applicability": X, "clarity": X, "practicality": X}},
    "overall_score": X,
    "confidence": 0.X,
    "issues": ["问题1", "问题2"],
    "reasoning": "评估理由"
}}
"""

    def _parse_assessment(self, model_name: str, response: str) -> ModelAssessment:
        """解析评估结果"""
        try:
            data = json.loads(response)
            return ModelAssessment(
                model_name=model_name,
                score=float(data.get('overall_score', 5)),
                confidence=float(data.get('confidence', 0.5)),
                issues=data.get('issues', []),
                reasoning=data.get('reasoning', '')
            )
        except:
            return ModelAssessment(
                model_name=model_name,
                score=5.0,
                confidence=0.3,
                issues=['解析失败'],
                reasoning='解析失败'
            )

    def _ensemble_decision(self, assessments: List[ModelAssessment]) -> Dict:
        """集成决策"""
        if not assessments:
            return {'passed': False, 'confidence': 0.0, 'issues': ['无评估结果']}

        # 计算加权平均分
        weighted_score = sum(
            a.score * a.confidence for a in assessments
        ) / sum(a.confidence for a in assessments)

        # 多数投票
        approve_count = sum(1 for a in assessments if a.score >= 7)
        approve_rate = approve_count / len(assessments)

        # 否决机制
        veto_count = sum(1 for a in assessments if a.score < 5)

        # 综合判断
        if veto_count >= 2:
            passed = False
            confidence = 0.9
        elif approve_rate >= self.approval_threshold:
            passed = True
            confidence = approve_rate
        else:
            passed = False
            confidence = 0.5

        # 收集问题
        all_issues = []
        for a in assessments:
            all_issues.extend([f"[{a.model_name}] {issue}" for issue in a.issues])

        return {
            'passed': passed,
            'confidence': confidence,
            'weighted_score': weighted_score,
            'approve_rate': approve_rate,
            'issues': list(set(all_issues))
        }

    def get_remaining_budget(self) -> float:
        """获取剩余预算"""
        return self.budget - self.spent