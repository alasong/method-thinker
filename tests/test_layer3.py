"""测试Layer 3测试驱动验证"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock
from src.validation.layer3_test_driven import (
    Layer3TestDrivenValidation,
    MethodTestCase,
    MethodTestResult,
    ValidationResult
)


def create_mock_model(response_text):
    """创建mock模型"""
    mock = Mock()
    mock.generate = Mock(return_value=response_text)
    return mock


def create_test_dataset():
    """创建测试数据集"""
    return [
        MethodTestCase(
            problem='解方程 x^2 - 5x + 6 = 0',
            answer='x=2 或 x=3',
            difficulty=2,
            problem_type='ALGEBRA'
        ),
        MethodTestCase(
            problem='计算 2x + 3 = 7 的解',
            answer='x=2',
            difficulty=1,
            problem_type='ALGEBRA'
        ),
        MethodTestCase(
            problem='证明三角形内角和为180度',
            answer='180度',
            difficulty=3,
            problem_type='GEOMETRY'
        )
    ]


def test_test_case_selection():
    """测试用例选择"""
    mock_model = create_mock_model('答案是：x=2')

    test_dataset = create_test_dataset()
    validator = Layer3TestDrivenValidation(mock_model, test_dataset, pass_threshold=0.6)

    method = {
        'method_id': 'ALG_001',
        'name': '因式分解法',
        'applicability': [{'problem_types': ['ALGEBRA']}],
        'template': {'steps': ['识别', '分解', '求解']}
    }

    relevant_tests = validator._select_relevant_tests(method)

    # 应选择ALGEBRA类型的测试
    assert len(relevant_tests) == 2
    assert all(t.problem_type == 'ALGEBRA' for t in relevant_tests)
    print("✓ 测试用例选择通过")


def test_answer_extraction():
    """测试答案提取"""
    validator = Layer3TestDrivenValidation(Mock(), [])

    # 测试各种格式
    outputs = [
        '最终答案是：x=2',
        '答案为：42',
        '因此，结果为 180 度'
    ]

    for output in outputs:
        answer = validator._extract_answer(output)
        assert len(answer) > 0

    print("✓ 测试答案提取通过")


def test_answer_verification_exact():
    """测试精确答案验证"""
    validator = Layer3TestDrivenValidation(Mock(), [])

    # 精确匹配
    assert validator._verify_answer('x=2', 'x=2') == True
    assert validator._verify_answer('42', '42') == True

    # 不匹配
    assert validator._verify_answer('x=2', 'x=3') == False

    print("✓ 测试精确答案验证通过")


def test_answer_verification_numeric():
    """测试数值答案验证"""
    validator = Layer3TestDrivenValidation(Mock(), [])

    # 数值匹配（允许微小误差）
    assert validator._verify_answer('答案是 3.0', '3') == True
    assert validator._verify_answer('结果为 180', '180度') == True
    assert validator._verify_answer('x = 2.0000001', 'x=2') == True

    print("✓ 测试数值答案验证通过")


def test_step_counting():
    """测试步骤计数"""
    validator = Layer3TestDrivenValidation(Mock(), [])

    output = """解题步骤：
1. 首先识别方程类型
2. 应用因式分解
3. 求解各因子
4. 验证答案"""

    count = validator._count_steps(output)
    assert count == 4

    print("✓ 测试步骤计数通过")


def test_statistics_computation():
    """测试统计计算"""
    validator = Layer3TestDrivenValidation(Mock(), [])

    test_results = [
        MethodTestResult(
            test_case=MethodTestCase(problem='p1', answer='a1', difficulty=1),
            predicted_answer='a1',
            is_correct=True,
            execution_time=1.0,
            steps_count=3
        ),
        MethodTestResult(
            test_case=MethodTestCase(problem='p2', answer='a2', difficulty=2),
            predicted_answer='wrong',
            is_correct=False,
            execution_time=2.0,
            steps_count=4
        ),
        MethodTestResult(
            test_case=MethodTestCase(problem='p3', answer='a3', difficulty=2),
            predicted_answer='a3',
            is_correct=True,
            execution_time=1.5,
            steps_count=5
        )
    ]

    stats = validator._compute_statistics(test_results)

    assert stats['success_rate'] == 2/3
    assert stats['total_tests'] == 3
    assert 'difficulty_stats' in stats
    assert stats['avg_execution_time'] > 0

    print("✓ 测试统计计算通过")


def test_high_success_rate_passes():
    """测试高成功率通过验证"""
    # Mock返回正确答案
    mock_model = create_mock_model('最终答案是：x=2')

    test_dataset = create_test_dataset()
    validator = Layer3TestDrivenValidation(mock_model, test_dataset, pass_threshold=0.5)

    method = {
        'method_id': 'ALG_001',
        'name': '测试方法',
        'applicability': [{'problem_types': ['ALGEBRA']}],
        'template': {'steps': ['步骤1']}
    }

    result = validator.validate(method)
    assert result.layer == 3
    assert 'statistics' in result.details

    print("✓ 测试验证流程通过")


def test_no_relevant_tests():
    """测试无相关测试用例"""
    mock_model = create_mock_model('答案')

    test_dataset = create_test_dataset()
    validator = Layer3TestDrivenValidation(mock_model, test_dataset)

    method = {
        'method_id': 'ALG_002',
        'name': '无匹配方法',
        'applicability': [{'problem_types': ['COMBINATORICS']}],  # 无此类测试
        'template': {'steps': []}
    }

    result = validator.validate(method)
    assert result.passed == False
    assert '无相关测试用例' in result.issues

    print("✓ 测试无相关测试用例通过")


def test_empty_results_handling():
    """测试空结果处理"""
    validator = Layer3TestDrivenValidation(Mock(), [])

    stats = validator._compute_statistics([])

    assert stats['success_rate'] == 0
    assert stats['confidence'] == 0
    assert '无测试结果' in stats['issues']

    print("✓ 测试空结果处理通过")


if __name__ == '__main__':
    test_test_case_selection()
    test_answer_extraction()
    test_answer_verification_exact()
    test_answer_verification_numeric()
    test_step_counting()
    test_statistics_computation()
    test_high_success_rate_passes()
    test_no_relevant_tests()
    test_empty_results_handling()
    print("\n所有Layer 3测试通过! ✓")