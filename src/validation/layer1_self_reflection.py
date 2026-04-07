"""Layer 1: 自我反思验证

让模型自己检查自己，模拟人类自我纠错。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    layer: int
    confidence: float
    issues: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


class Layer1SelfReflection:
    """Layer 1: 自我反思验证

    让模型扮演批评者的角色，对自己的方法论进行反思和改进。

    Attributes:
        model: 用于生成反思的模型
        max_iterations: 最大迭代次数
    """

    def __init__(self, model, max_iterations: int = 3):
        """初始化自我反思验证器

        Args:
            model: 用于生成反思的模型
            max_iterations: 最大迭代次数
        """
        self.model = model
        self.max_iterations = max_iterations

    def validate(self, method: Dict) -> ValidationResult:
        """自我反思验证

        Args:
            method: 待验证的方法

        Returns:
            ValidationResult: 验证结果
        """
        reflection_history = []
        current_method = method

        for iteration in range(self.max_iterations):
            critique = self._self_critique(current_method)

            reflection_history.append({
                'iteration': iteration,
                'method': current_method,
                'critique': critique
            })

            if critique.get('passed', False):
                return ValidationResult(
                    passed=True,
                    layer=1,
                    confidence=critique.get('confidence', 0.8),
                    issues=[],
                    details={'reflections': reflection_history}
                )

            if iteration < self.max_iterations - 1:
                improved = self._improve_method(current_method, critique)
                current_method = improved

        final_critique = reflection_history[-1]['critique']
        return ValidationResult(
            passed=False,
            layer=1,
            confidence=final_critique.get('confidence', 0.3),
            issues=final_critique.get('issues', ['自我反思未通过']),
            details={'reflections': reflection_history}
        )

    def _self_critique(self, method: Dict) -> Dict:
        """自我批判

        Args:
            method: 待批判的方法

        Returns:
            Dict: 批判结果
        """
        prompt = self._build_critique_prompt(method)

        try:
            response = self.model.generate(prompt, temperature=0.3)
            critique = json.loads(response)
        except Exception as e:
            critique = {
                'passed': False,
                'confidence': 0.3,
                'issues': [{'aspect': '格式', 'problem': f'解析失败: {e}'}],
                'suggestions': ['请检查输出格式']
            }

        return critique

    def _improve_method(self, method: Dict, critique: Dict) -> Dict:
        """根据批判改进方法

        Args:
            method: 原方法
            critique: 批判意见

        Returns:
            Dict: 改进后的方法
        """
        prompt = f"""请根据以下批评意见改进方法论：

原方法：
{json.dumps(method, ensure_ascii=False, indent=2)}

批评意见：
{json.dumps(critique, ensure_ascii=False, indent=2)}

请输出改进后的完整方法论JSON。保持原有字段，只改进有问题的地方。
"""

        try:
            response = self.model.generate(prompt, temperature=0.5)
            improved = json.loads(response)
            improved['method_id'] = method.get('method_id')
            return improved
        except:
            return method

    def _build_critique_prompt(self, method: Dict) -> str:
        """构建批判提示"""
        return f"""你是一位数学方法论专家，现在需要从批评的角度审视以下方法论：

方法名称：{method.get('name', '')}
方法描述：{method.get('description', '')}
适用条件：{method.get('applicability', [])}
执行步骤：{method.get('template', {}).get('steps', [])}

请从以下角度批判：

1. 【逻辑一致性】各步骤之间是否逻辑连贯？是否有跳跃或遗漏？
2. 【适用范围】适用条件是否明确？是否遗漏重要特殊情况？
3. 【可执行性】步骤是否足够具体？能直接执行吗？
4. 【完整性】是否有常见陷阱未提醒？是否有重要技巧未说明？

输出JSON格式：
{{
    "passed": true/false,
    "confidence": 0.0-1.0,
    "issues": [
        {{"aspect": "逻辑一致性", "problem": "具体问题", "severity": "高/中/低"}}
    ],
    "suggestions": ["改进建议1", "改进建议2"]
}}
"""

    def multi_perspective_reflection(self, method: Dict) -> Dict:
        """多角度反思"""
        perspectives = [
            {'role': '初学者', 'focus': '这个描述我能理解吗？'},
            {'role': '专家', 'focus': '这个方法在数学上严谨吗？'},
            {'role': '教师', 'focus': '这个方法适合教学吗？'}
        ]

        reflections = []
        for p in perspectives:
            prompt = f"""你是一位{p['role']}，请评价以下方法论：
方法：{method.get('name', '')}
关注点：{p['focus']}
请简短评价（100字以内）。
"""
            response = self.model.generate(prompt, temperature=0.5)
            reflections.append({'perspective': p['role'], 'feedback': response})

        return {'reflections': reflections}