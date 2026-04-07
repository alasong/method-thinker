"""测试Layer 1自我反思验证"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock
from src.validation.layer1_self_reflection import Layer1SelfReflection, ValidationResult


def create_mock_model(response_json):
    """创建mock模型"""
    mock = Mock()
    mock.generate = Mock(return_value=response_json)
    return mock


def test_valid_method_passes():
    """测试有效方法通过验证"""
    # Mock返回通过的批判结果
    pass_response = '{"passed": true, "confidence": 0.85, "issues": [], "suggestions": []}'
    mock_model = create_mock_model(pass_response)

    validator = Layer1SelfReflection(mock_model, max_iterations=2)

    valid_method = {
        'method_id': 'ALG_001',
        'name': '变量替换法',
        'description': '通过引入新变量简化表达式结构',
        'applicability': [{'condition': '表达式存在重复结构'}],
        'template': {'steps': ['识别模式', '选择变量', '变换', '求解', '回代']}
    }

    result = validator.validate(valid_method)
    assert result.passed == True, f"有效方法应通过: {result.issues}"
    assert result.layer == 1
    assert result.confidence >= 0.8
    print("✓ 测试有效方法通过")


def test_invalid_method_fails():
    """测试无效方法验证失败"""
    # Mock返回失败的批判结果
    fail_response = '{"passed": false, "confidence": 0.6, "issues": [{"aspect": "完整性", "problem": "缺少回代步骤"}], "suggestions": ["添加回代步骤"]}'
    mock_model = create_mock_model(fail_response)

    validator = Layer1SelfReflection(mock_model, max_iterations=2)

    invalid_method = {
        'method_id': 'ALG_002',
        'name': '不完整方法',
        'description': '描述过短',
        'applicability': [],
        'template': {'steps': ['步骤1']}
    }

    result = validator.validate(invalid_method)
    assert result.passed == False, "无效方法应失败"
    assert len(result.issues) > 0
    print("✓ 测试无效方法失败")


def test_iteration_improves_method():
    """测试迭代改进机制"""
    # 第一次批判失败，改进方法调用，第二次批判通过
    responses = [
        '{"passed": false, "confidence": 0.5, "issues": [{"aspect": "逻辑", "problem": "步骤跳跃"}], "suggestions": ["补充中间步骤"]}',
        '{"method_id": "ALG_003", "name": "改进后方法", "description": "改进后的描述", "applicability": [], "template": {"steps": ["步骤A", "中间步骤", "步骤B"]}}',
        '{"passed": true, "confidence": 0.8, "issues": [], "suggestions": []}'
    ]

    mock_model = Mock()
    mock_model.generate = Mock(side_effect=responses)

    validator = Layer1SelfReflection(mock_model, max_iterations=3)

    method = {
        'method_id': 'ALG_003',
        'name': '测试方法',
        'description': '一个需要改进的方法',
        'applicability': [{'condition': '测试条件'}],
        'template': {'steps': ['步骤A', '步骤B']}
    }

    result = validator.validate(method)
    # 第二次迭代应该返回通过
    assert result.passed == True, f"迭代改进后应通过: {result}"
    assert mock_model.generate.call_count >= 2
    print("✓ 测试迭代改进机制通过")


def test_self_critique_format():
    """测试自我批判返回格式"""
    mock_model = Mock()
    mock_model.generate = Mock(return_value='{"passed": true, "confidence": 0.9, "issues": []}')

    validator = Layer1SelfReflection(mock_model)

    method = {'name': '测试', 'description': '描述', 'applicability': [], 'template': {'steps': []}}
    critique = validator._self_critique(method)

    assert 'passed' in critique
    assert 'confidence' in critique
    assert isinstance(critique.get('passed'), bool)
    print("✓ 测试自我批判格式通过")


def test_parse_failure_handling():
    """测试解析失败处理"""
    # Mock返回无效JSON
    mock_model = Mock()
    mock_model.generate = Mock(return_value='not valid json')

    validator = Layer1SelfReflection(mock_model)

    method = {'name': '测试', 'description': '描述'}
    critique = validator._self_critique(method)

    assert critique['passed'] == False, "解析失败应返回失败"
    assert critique['confidence'] < 0.5
    assert '格式' in str(critique.get('issues', []))
    print("✓ 测试解析失败处理通过")


def test_multi_perspective_reflection():
    """测试多角度反思"""
    mock_model = Mock()
    mock_model.generate = Mock(return_value='这是测试反馈')

    validator = Layer1SelfReflection(mock_model)

    method = {'name': '测试方法'}
    result = validator.multi_perspective_reflection(method)

    assert 'reflections' in result
    assert len(result['reflections']) == 3  # 三种视角
    assert mock_model.generate.call_count == 3
    print("✓ 测试多角度反思通过")


if __name__ == '__main__':
    test_valid_method_passes()
    test_invalid_method_fails()
    test_iteration_improves_method()
    test_self_critique_format()
    test_parse_failure_handling()
    test_multi_perspective_reflection()
    print("\n所有Layer 1测试通过! ✓")