"""Layer 3: 测试驱动验证

用实际测试验证方法有效性。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re
import time


@dataclass
class MethodTestCase:
    """方法测试用例"""
    problem: str
    answer: str
    difficulty: int = 3
    problem_type: str = "unknown"


@dataclass
class MethodTestResult:
    """方法测试结果"""
    test_case: MethodTestCase
    predicted_answer: str
    is_correct: bool
    execution_time: float
    steps_count: int


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    layer: int
    confidence: float
    issues: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


class Layer3TestDrivenValidation:
    """Layer 3: 测试驱动验证

    通过在实际测试题上应用方法来验证其有效性。

    Attributes:
        model: 用于生成解答的模型
        test_dataset: 测试数据集
        pass_threshold: 通过阈值
    """

    def __init__(
        self,
        model,
        test_dataset: List[MethodTestCase],
        pass_threshold: float = 0.6
    ):
        """初始化测试驱动验证器

        Args:
            model: 用于生成解答的模型
            test_dataset: 测试数据集
            pass_threshold: 成功率通过阈值
        """
        self.model = model
        self.test_dataset = test_dataset
        self.pass_threshold = pass_threshold

    def validate(self, method: Dict) -> ValidationResult:
        """测试驱动验证

        Args:
            method: 待验证的方法

        Returns:
            ValidationResult: 验证结果
        """
        relevant_tests = self._select_relevant_tests(method)

        if not relevant_tests:
            return ValidationResult(
                passed=False,
                layer=3,
                confidence=0.0,
                issues=['无相关测试用例'],
                details={}
            )

        test_results = self._execute_tests(method, relevant_tests)
        stats = self._compute_statistics(test_results)

        passed = stats['success_rate'] >= self.pass_threshold

        return ValidationResult(
            passed=passed,
            layer=3,
            confidence=stats['confidence'],
            issues=stats['issues'],
            details={
                'statistics': stats,
                'test_results': [
                    {
                        'problem': r.test_case.problem[:50],
                        'correct': r.is_correct
                    } for r in test_results[:10]
                ]
            }
        )

    def _select_relevant_tests(self, method: Dict) -> List[MethodTestCase]:
        """选择相关测试"""
        applicability = method.get('applicability', [])
        problem_types = []

        for app in applicability:
            problem_types.extend(app.get('problem_types', []))

        relevant = []
        for test in self.test_dataset:
            if not problem_types or test.problem_type in problem_types:
                relevant.append(test)

        return relevant[:50]  # 限制数量

    def _execute_tests(self, method: Dict, tests: List[MethodTestCase]) -> List[MethodTestResult]:
        """执行测试"""
        results = []

        for test in tests:
            input_text = self._build_test_input(test, method)

            start_time = time.time()
            output = self.model.generate(input_text, temperature=0.6)
            execution_time = time.time() - start_time

            predicted = self._extract_answer(output)
            is_correct = self._verify_answer(predicted, test.answer)
            steps_count = self._count_steps(output)

            results.append(MethodTestResult(
                test_case=test,
                predicted_answer=predicted,
                is_correct=is_correct,
                execution_time=execution_time,
                steps_count=steps_count
            ))

        return results

    def _compute_statistics(self, results: List[MethodTestResult]) -> Dict:
        """计算统计指标"""
        if not results:
            return {
                'success_rate': 0,
                'confidence': 0,
                'issues': ['无测试结果']
            }

        success_rate = sum(r.is_correct for r in results) / len(results)

        # 按难度分组
        difficulty_stats = {}
        for r in results:
            d = r.test_case.difficulty
            if d not in difficulty_stats:
                difficulty_stats[d] = {'correct': 0, 'total': 0}
            difficulty_stats[d]['total'] += 1
            if r.is_correct:
                difficulty_stats[d]['correct'] += 1

        issues = []
        if success_rate < 0.4:
            issues.append(f"成功率过低({success_rate:.1%})，方法可能无效")

        # 计算置信度
        import math
        n = len(results)
        se = math.sqrt(success_rate * (1 - success_rate) / n) if n > 0 else 0
        confidence = max(0, 1 - 2 * se)

        return {
            'success_rate': success_rate,
            'confidence': confidence,
            'total_tests': n,
            'difficulty_stats': difficulty_stats,
            'avg_execution_time': sum(r.execution_time for r in results) / n,
            'issues': issues
        }

    def _build_test_input(self, test: MethodTestCase, method: Dict) -> str:
        """构建测试输入"""
        steps = method.get('template', {}).get('steps', [])
        steps_text = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(steps))

        return f"""问题：{test.problem}

请使用【{method.get('name', '')}】解答。

方法说明：
{method.get('description', '')}

执行步骤：
{steps_text}

请按此方法解答问题。
"""

    def _extract_answer(self, output: str) -> str:
        """提取答案"""
        patterns = [
            r'最终答案[是为：:]\s*(.+)',
            r'答案[是为：:]\s*(.+)',
            r'因此[，,]?\s*(.+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1).strip()

        return output.strip().split('\n')[-1]

    def _verify_answer(self, predicted: str, expected: str) -> bool:
        """验证答案"""
        predicted = predicted.strip().lower()
        expected = expected.strip().lower()

        if predicted == expected:
            return True

        # 尝试数值匹配
        try:
            pred_num = float(re.search(r'[-+]?\d*\.?\d+', predicted).group())
            exp_num = float(re.search(r'[-+]?\d*\.?\d+', expected).group())
            return abs(pred_num - exp_num) < 1e-6
        except:
            pass

        return False

    def _count_steps(self, output: str) -> int:
        """统计步骤数"""
        return len(re.findall(r'^\d+[\.、）]', output, re.MULTILINE))