"""测试验证流水线"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, MagicMock
from src.validation.pipeline import ValidationPipeline
from src.validation.config import (
    ValidationConfig,
    Layer0Config,
    Layer1Config,
    Layer2Config,
    Layer3Config,
    EnsembleConfig,
    BudgetConfig
)
from src.validation.layer3_test_driven import TestCase
from src.validation.ensemble_decision import LayerResult, ValidationResult


def create_mock_model(response='{"passed": true, "confidence": 0.8}'):
    """创建mock模型"""
    mock = Mock()
    mock.generate = Mock(return_value=response)
    return mock


def create_mock_model_clients():
    """创建mock模型客户端字典"""
    return {
        'deepseek_v3': create_mock_model('{"overall_score": 8, "confidence": 0.9, "issues": []}'),
        'qwen_math': create_mock_model('{"overall_score": 7, "confidence": 0.8, "issues": []}')
    }


def create_test_dataset():
    """创建测试数据集"""
    return [
        TestCase(
            problem='解方程 x^2 - 4 = 0',
            answer='x=2 或 x=-2',
            difficulty=2,
            problem_type='ALGEBRA'
        )
    ]


def create_sample_method():
    """创建示例方法"""
    return {
        'method_id': 'ALG_001',
        'name': '因式分解法',
        'description': '通过因式分解求解二次方程的方法',
        'applicability': [{'condition': '二次方程', 'problem_types': ['ALGEBRA']}],
        'template': {'steps': ['识别方程', '因式分解', '求解']},
        'difficulty': 3,
        'frequency': 0.7
    }


def test_config_dataclasses():
    """测试配置数据类"""
    config = ValidationConfig()

    assert config.layer0.enabled == True
    assert config.layer1.max_iterations == 3
    assert config.layer2.models == ['deepseek_v3', 'qwen_math', 'gpt4o_mini']
    assert config.ensemble.weights[2] == 0.40

    print("✓ 测试配置数据类")


def test_custom_config():
    """测试自定义配置"""
    config = ValidationConfig(
        layer0=Layer0Config(enabled=False),
        layer1=Layer1Config(max_iterations=5),
        layer2=Layer2Config(models=['model_a', 'model_b']),
        ensemble=EnsembleConfig(weights={0: 0.1, 1: 0.1, 2: 0.4, 3: 0.4})
    )

    assert config.layer0.enabled == False
    assert config.layer1.max_iterations == 5
    assert config.layer2.models == ['model_a', 'model_b']

    print("✓ 测试自定义配置")


def test_pipeline_initialization_with_config():
    """测试流水线初始化（使用config.py的ValidationConfig）"""
    config = ValidationConfig()
    mock_model = create_mock_model()

    # 使用config初始化
    pipeline = ValidationPipeline(
        config=config,
        model=mock_model
    )

    # 检查config被正确传递
    assert pipeline.config is not None
    print("✓ 测试流水线初始化")


def test_pipeline_with_all_layers():
    """测试完整流水线初始化"""
    config = ValidationConfig()
    mock_model = create_mock_model('{"passed": true, "confidence": 0.85, "issues": []}')
    model_clients = create_mock_model_clients()
    test_dataset = create_test_dataset()

    pipeline = ValidationPipeline(
        config=config,
        model=mock_model,
        model_clients=model_clients,
        test_dataset=test_dataset
    )

    assert pipeline.layer0 is not None
    assert pipeline.layer1 is not None
    assert pipeline.layer2 is not None
    assert pipeline.layer3 is not None

    print("✓ 测试完整流水线初始化")


def test_pipeline_run():
    """测试流水线运行"""
    config = ValidationConfig()
    mock_model = create_mock_model('{"passed": true, "confidence": 0.85, "issues": []}')
    model_clients = create_mock_model_clients()
    test_dataset = create_test_dataset()

    pipeline = ValidationPipeline(
        config=config,
        model=mock_model,
        model_clients=model_clients,
        test_dataset=test_dataset
    )

    method = create_sample_method()
    result = pipeline.run(method)

    assert result is not None
    assert hasattr(result, 'passed')
    assert hasattr(result, 'layer')
    assert hasattr(result, 'confidence')

    print("✓ 测试流水线运行")


def test_skip_layers():
    """测试跳过层级"""
    config = ValidationConfig()
    mock_model = create_mock_model()
    model_clients = create_mock_model_clients()

    pipeline = ValidationPipeline(
        config=config,
        model=mock_model,
        model_clients=model_clients
    )

    method = create_sample_method()

    # 跳过Layer 1和2
    result = pipeline.run(method, skip_layers=[1, 2])

    assert result is not None

    print("✓ 测试跳过层级")


def test_layer0_fast_fail():
    """测试Layer 0快速失败"""
    config = ValidationConfig()

    # 使用会失败的Layer 0配置
    existing_kb = {
        'methods': {
            'ALG_001': {'name': '已存在方法'}
        }
    }

    pipeline = ValidationPipeline(
        config=config,
        existing_kb=existing_kb
    )

    # 方法ID重复
    method = create_sample_method()
    method['method_id'] = 'ALG_001'  # 与existing_kb冲突

    # 只运行Layer 0
    result = pipeline.run(method, skip_layers=[1, 2, 3])

    assert result.passed == False

    print("✓ 测试Layer 0快速失败")


def test_disabled_layer():
    """测试禁用层级"""
    config = ValidationConfig(
        layer0=Layer0Config(enabled=False),
        layer1=Layer1Config(enabled=False)
    )

    pipeline = ValidationPipeline(config=config)

    assert pipeline.layer0 is None
    assert pipeline.layer1 is None

    print("✓ 测试禁用层级")


def test_ensemble_weights_in_config():
    """测试配置中的集成权重"""
    custom_weights = {0: 0.1, 1: 0.2, 2: 0.35, 3: 0.35}
    config = ValidationConfig(
        ensemble=EnsembleConfig(weights=custom_weights)
    )

    pipeline = ValidationPipeline(config=config)

    weights = pipeline.ensemble.get_weights()
    assert weights[0] == 0.1
    assert weights[1] == 0.2

    print("✓ 测试配置中的集成权重")


def test_budget_config():
    """测试预算配置"""
    config = ValidationConfig(
        budget=BudgetConfig(total=100.0, alert_threshold=0.9)
    )

    assert config.budget.total == 100.0
    assert config.budget.alert_threshold == 0.9

    print("✓ 测试预算配置")


def test_layer_result_collection():
    """测试层级结果收集"""
    config = ValidationConfig()
    mock_model = create_mock_model()
    model_clients = create_mock_model_clients()

    pipeline = ValidationPipeline(
        config=config,
        model=mock_model,
        model_clients=model_clients
    )

    method = create_sample_method()
    result = pipeline.run(method)

    # 检查result有details属性（可能包含各种信息）
    assert result.details is not None or result.layer == -1  # 集成层或有详情

    print("✓ 测试层级结果收集")


def test_model_clients_parameter():
    """测试model_clients参数"""
    config = ValidationConfig()
    model_clients = create_mock_model_clients()

    pipeline = ValidationPipeline(
        config=config,
        model_clients=model_clients
    )

    assert pipeline.model_clients == model_clients
    assert pipeline.layer2 is not None  # Layer 2应该被初始化

    print("✓ 测试model_clients参数")


def test_test_dataset_parameter():
    """测试test_dataset参数"""
    config = ValidationConfig()
    mock_model = create_mock_model()
    test_dataset = create_test_dataset()

    pipeline = ValidationPipeline(
        config=config,
        model=mock_model,
        test_dataset=test_dataset
    )

    assert pipeline.test_dataset == test_dataset
    assert pipeline.layer3 is not None  # Layer 3应该被初始化

    print("✓ 测试test_dataset参数")


def test_config_budget_used_in_layer2():
    """测试Layer2使用config中的预算"""
    config = ValidationConfig(
        budget=BudgetConfig(total=200.0)
    )
    model_clients = create_mock_model_clients()

    pipeline = ValidationPipeline(
        config=config,
        model_clients=model_clients
    )

    # Layer2应该使用config中的预算
    assert pipeline.layer2 is not None
    assert pipeline.layer2.budget == 200.0

    print("✓ 测试config预算用于Layer2")


if __name__ == '__main__':
    test_config_dataclasses()
    test_custom_config()
    test_pipeline_initialization_with_config()
    test_pipeline_with_all_layers()
    test_pipeline_run()
    test_skip_layers()
    test_layer0_fast_fail()
    test_disabled_layer()
    test_ensemble_weights_in_config()
    test_budget_config()
    test_layer_result_collection()
    test_model_clients_parameter()
    test_test_dataset_parameter()
    test_config_budget_used_in_layer2()
    print("\n所有流水线测试通过! ✓")