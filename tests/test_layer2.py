"""测试Layer 2多模型验证"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock
from src.validation.layer2_multi_model import (
    Layer2MultiModelValidation,
    ModelAssessment,
    ValidationResult
)


def create_mock_model_client(response_json):
    """创建mock模型客户端"""
    mock = Mock()
    mock.generate = Mock(return_value=response_json)
    return mock


def test_single_model_validation():
    """测试单个模型验证"""
    response = '{"overall_score": 8, "confidence": 0.85, "issues": [], "reasoning": "方法质量良好"}'

    model_clients = {
        'deepseek_v3': create_mock_model_client(response)
    }

    validator = Layer2MultiModelValidation(
        model_clients,
        budget=100.0,
        approval_threshold=0.6
    )

    method = {
        'method_id': 'ALG_001',
        'name': '变量替换法',
        'description': '通过引入新变量简化表达式',
        'category': 'ALGEBRA',
        'applicability': [{'condition': '测试'}],
        'template': {'steps': ['步骤1', '步骤2']}
    }

    result = validator.validate(method)
    assert result.layer == 2
    assert 'assessments' in result.details
    print("✓ 测试单模型验证通过")


def test_multi_model_consensus():
    """测试多模型一致通过"""
    # 所有模型都返回高分
    high_score_response = '{"overall_score": 8, "confidence": 0.9, "issues": [], "reasoning": "好方法"}'

    model_clients = {
        'deepseek_v3': create_mock_model_client(high_score_response),
        'qwen_math': create_mock_model_client(high_score_response),
        'gpt4o_mini': create_mock_model_client(high_score_response)
    }

    validator = Layer2MultiModelValidation(
        model_clients,
        budget=100.0,
        approval_threshold=0.6
    )

    method = {
        'method_id': 'ALG_002',
        'name': '优质方法',
        'category': 'ALGEBRA',
        'description': '完善的数学方法',
        'applicability': [],
        'template': {'steps': []}
    }

    result = validator.validate(method)
    # 三个模型高分应该通过
    assert result.passed == True, f"多模型一致高分应通过: {result.issues}"
    print("✓ 测试多模型一致通过")


def test_veto_mechanism():
    """测试否决机制"""
    # 两个模型返回低分触发否决
    low_score_response = '{"overall_score": 4, "confidence": 0.85, "issues": ["逻辑错误"], "reasoning": "方法有问题"}'

    model_clients = {
        'deepseek_v3': create_mock_model_client(low_score_response),
        'qwen_math': create_mock_model_client(low_score_response)
    }

    validator = Layer2MultiModelValidation(
        model_clients,
        budget=100.0
    )

    method = {
        'method_id': 'ALG_003',
        'name': '有问题的方法',
        'category': 'ALGEBRA',
        'description': '测试',
        'applicability': [],
        'template': {'steps': []}
    }

    result = validator.validate(method)
    # 两个低分触发否决
    assert result.passed == False, "否决机制应生效"
    print("✓ 测试否决机制通过")


def test_model_selection_by_category():
    """测试按类别选择模型"""
    model_clients = {
        'deepseek_v3': create_mock_model_client('{}'),
        'qwen_math': create_mock_model_client('{}'),
        'gpt4o_mini': create_mock_model_client('{}')
    }

    validator = Layer2MultiModelValidation(model_clients)

    # ALGEBRA类别
    algebra_models = validator._select_models({'category': 'ALGEBRA'})
    assert 'deepseek_v3' in algebra_models
    assert 'qwen_math' in algebra_models

    # GEOMETRY类别
    geometry_models = validator._select_models({'category': 'GEOMETRY'})
    assert 'qwen_math' in geometry_models

    # GENERAL类别
    general_models = validator._select_models({'category': 'GENERAL'})
    assert 'deepseek_v3' in general_models

    print("✓ 测试模型选择通过")


def test_budget_tracking():
    """测试预算追踪"""
    response = '{"overall_score": 7, "confidence": 0.8, "issues": []}'

    model_clients = {
        'deepseek_v3': create_mock_model_client(response),
        'qwen_math': create_mock_model_client(response)
    }

    validator = Layer2MultiModelValidation(
        model_clients,
        budget=10.0
    )

    method = {
        'method_id': 'ALG_004',
        'category': 'ALGEBRA',
        'name': '测试',
        'description': '测试',
        'applicability': [],
        'template': {'steps': []}
    }

    initial_budget = validator.get_remaining_budget()
    validator.validate(method)
    final_budget = validator.get_remaining_budget()

    assert final_budget < initial_budget, "预算应该减少"
    print("✓ 测试预算追踪通过")


def test_parse_assessment():
    """测试解析评估结果"""
    validator = Layer2MultiModelValidation({})

    valid_response = '{"overall_score": 7.5, "confidence": 0.85, "issues": ["小问题"], "reasoning": "总体不错"}'
    assessment = validator._parse_assessment('test_model', valid_response)

    assert assessment.model_name == 'test_model'
    assert assessment.score == 7.5
    assert assessment.confidence == 0.85
    assert len(assessment.issues) == 1

    print("✓ 测试解析评估通过")


def test_parse_assessment_failure():
    """测试解析失败处理"""
    validator = Layer2MultiModelValidation({})

    invalid_response = 'invalid json'
    assessment = validator._parse_assessment('test_model', invalid_response)

    assert assessment.score == 5.0  # 默认分数
    assert assessment.confidence == 0.3  # 低置信度
    assert '解析失败' in assessment.issues

    print("✓ 测试解析失败处理通过")


def test_ensemble_decision():
    """测试集成决策"""
    validator = Layer2MultiModelValidation({})

    assessments = [
        ModelAssessment(model_name='A', score=8, confidence=0.9, issues=[]),
        ModelAssessment(model_name='B', score=7, confidence=0.8, issues=[]),
        ModelAssessment(model_name='C', score=6, confidence=0.7, issues=['小问题'])
    ]

    decision = validator._ensemble_decision(assessments)

    assert 'passed' in decision
    assert 'confidence' in decision
    assert 'weighted_score' in decision
    assert 'approve_rate' in decision

    print("✓ 测试集成决策通过")


if __name__ == '__main__':
    test_single_model_validation()
    test_multi_model_consensus()
    test_veto_mechanism()
    test_model_selection_by_category()
    test_budget_tracking()
    test_parse_assessment()
    test_parse_assessment_failure()
    test_ensemble_decision()
    print("\n所有Layer 2测试通过! ✓")