"""测试集成决策引擎"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.ensemble_decision import (
    EnsembleDecisionEngine,
    LayerResult,
    ValidationResult
)


def test_all_layers_pass():
    """测试所有层通过"""
    engine = EnsembleDecisionEngine()

    layer_results = [
        LayerResult(layer=0, passed=True, confidence=0.9, issues=[], weight=0.05),
        LayerResult(layer=1, passed=True, confidence=0.8, issues=[], weight=0.15),
        LayerResult(layer=2, passed=True, confidence=0.85, issues=[], weight=0.40),
        LayerResult(layer=3, passed=True, confidence=0.7, issues=[], weight=0.40)
    ]

    result = engine.decide(layer_results)

    assert result.passed == True, "所有层通过应返回True"
    assert result.confidence > 0.5
    assert result.layer == -1  # 集成层标记

    print("✓ 测试所有层通过")


def test_veto_from_core_layer():
    """测试核心层否决"""
    engine = EnsembleDecisionEngine()

    # Layer 2高置信度失败触发否决
    layer_results = [
        LayerResult(layer=0, passed=True, confidence=0.9, issues=[], weight=0.05),
        LayerResult(layer=1, passed=True, confidence=0.8, issues=[], weight=0.15),
        LayerResult(layer=2, passed=False, confidence=0.9, issues=['严重问题'], weight=0.40),  # 否决
        LayerResult(layer=3, passed=True, confidence=0.7, issues=[], weight=0.40)
    ]

    result = engine.decide(layer_results)

    assert result.passed == False, "核心层否决应失败"
    assert result.confidence >= 0.8, "否决置信度应高"
    assert '否决' in result.details.get('reason', '')

    print("✓ 测试核心层否决")


def test_layer3_veto():
    """测试Layer 3否决"""
    engine = EnsembleDecisionEngine()

    layer_results = [
        LayerResult(layer=0, passed=True, confidence=0.9, issues=[], weight=0.05),
        LayerResult(layer=1, passed=True, confidence=0.8, issues=[], weight=0.15),
        LayerResult(layer=2, passed=True, confidence=0.85, issues=[], weight=0.40),
        LayerResult(layer=3, passed=False, confidence=0.95, issues=['测试失败'], weight=0.40)  # Layer 3否决
    ]

    result = engine.decide(layer_results)

    assert result.passed == False
    veto_layer = engine._check_veto(layer_results)
    assert veto_layer == 3

    print("✓ 测试Layer 3否决")


def test_weighted_score_calculation():
    """测试加权得分计算"""
    engine = EnsembleDecisionEngine({
        0: 0.1,
        1: 0.2,
        2: 0.3,
        3: 0.4
    })

    layer_results = [
        LayerResult(layer=0, passed=True, confidence=0.9, issues=[], weight=0.1),
        LayerResult(layer=1, passed=False, confidence=0.6, issues=[], weight=0.2),
        LayerResult(layer=2, passed=True, confidence=0.8, issues=[], weight=0.3),
        LayerResult(layer=3, passed=False, confidence=0.7, issues=[], weight=0.4)
    ]

    result = engine.decide(layer_results)

    assert 'weighted_score' in result.details
    weighted_score = result.details['weighted_score']
    # 验证得分在合理范围
    assert 0 <= weighted_score <= 1

    print("✓ 测试加权得分计算")


def test_default_weights():
    """测试默认权重"""
    engine = EnsembleDecisionEngine()

    weights = engine.get_weights()

    assert weights[0] == 0.05
    assert weights[1] == 0.15
    assert weights[2] == 0.40
    assert weights[3] == 0.40

    print("✓ 测试默认权重")


def test_custom_weights():
    """测试自定义权重"""
    custom_weights = {
        0: 0.1,
        1: 0.1,
        2: 0.4,
        3: 0.4
    }

    engine = EnsembleDecisionEngine(custom_weights)
    weights = engine.get_weights()

    assert weights[0] == 0.1
    assert weights[1] == 0.1

    print("✓ 测试自定义权重")


def test_update_weights():
    """测试更新权重"""
    engine = EnsembleDecisionEngine()

    engine.update_weights({0: 0.2, 1: 0.3})
    weights = engine.get_weights()

    assert weights[0] == 0.2
    assert weights[1] == 0.3
    # 其他权重保持不变
    assert weights[2] == 0.40

    print("✓ 测试更新权重")


def test_issues_collection():
    """测试问题收集"""
    engine = EnsembleDecisionEngine()

    layer_results = [
        LayerResult(layer=0, passed=False, confidence=0.5, issues=['格式错误'], weight=0.05),
        LayerResult(layer=1, passed=False, confidence=0.6, issues=['逻辑问题'], weight=0.15),
        LayerResult(layer=2, passed=True, confidence=0.9, issues=[], weight=0.40),
        LayerResult(layer=3, passed=True, confidence=0.8, issues=[], weight=0.40)
    ]

    result = engine.decide(layer_results)

    assert len(result.issues) > 0
    # 问题应带有层级标记
    assert any('[L0]' in issue for issue in result.issues)
    assert any('[L1]' in issue for issue in result.issues)

    print("✓ 测试问题收集")


def test_empty_layer_results():
    """测试空层级结果"""
    engine = EnsembleDecisionEngine()

    result = engine.decide([])

    # 空结果的处理（取决于实现）
    assert result.layer == -1

    print("✓ 测试空层级结果")


def test_layer_result_dataclass():
    """测试LayerResult数据类"""
    result = LayerResult(
        layer=2,
        passed=True,
        confidence=0.85,
        issues=['问题1', '问题2'],
        weight=0.40
    )

    assert result.layer == 2
    assert result.passed == True
    assert result.confidence == 0.85
    assert len(result.issues) == 2
    assert result.weight == 0.40

    print("✓ 测试LayerResult数据类")


if __name__ == '__main__':
    test_all_layers_pass()
    test_veto_from_core_layer()
    test_layer3_veto()
    test_weighted_score_calculation()
    test_default_weights()
    test_custom_weights()
    test_update_weights()
    test_issues_collection()
    test_empty_layer_results()
    test_layer_result_dataclass()
    print("\n所有集成决策测试通过! ✓")