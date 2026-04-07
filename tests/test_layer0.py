"""测试Layer 0快速过滤器"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.layer0_fast_filter import Layer0FastFilter, ValidationResult


def test_valid_method():
    """测试有效方法"""
    filter = Layer0FastFilter()

    valid_method = {
        'method_id': 'ALG_001',
        'name': '变量替换法',
        'description': '通过引入新变量简化表达式结构，适用于存在重复或对称模式的问题',
        'applicability': [
            {'condition': '表达式中存在重复结构', 'keywords': ['对称', '重复']}
        ],
        'template': {
            'steps': ['识别模式', '选择替换变量', '变换表达式', '求解', '回代']
        },
        'difficulty': 3,
        'frequency': 0.8
    }

    result = filter.validate(valid_method)
    assert result.passed == True, f"有效方法验证失败: {result.issues}"
    print("✓ 测试有效方法通过")


def test_missing_field():
    """测试缺少字段"""
    filter = Layer0FastFilter()

    invalid_method = {
        'method_id': 'ALG_002',
        'name': '测试方法'
        # 缺少 description, applicability, template
    }

    result = filter.validate(invalid_method)
    assert result.passed == False, "应该检测到缺少字段"
    assert any('缺少必要字段' in issue for issue in result.issues)
    print("✓ 测试缺少字段通过")


def test_duplicate():
    """测试重复检测"""
    existing_kb = {
        'methods': {
            'ALG_001': {'name': '变量替换法', 'method_id': 'ALG_001'}
        }
    }

    filter = Layer0FastFilter(existing_kb)

    # 测试ID重复
    duplicate_method = {
        'method_id': 'ALG_001',  # 重复ID
        'name': '新方法',
        'description': '这是一个测试方法，描述需要足够长才能通过检查',
        'applicability': [{'condition': '测试', 'keywords': ['test']}],
        'template': {'steps': ['步骤1', '步骤2']}
    }

    result = filter.validate(duplicate_method)
    assert result.passed == False
    assert any('重复' in issue for issue in result.issues)
    print("✓ 测试重复检测通过")


def test_description_quality():
    """测试描述质量"""
    filter = Layer0FastFilter()

    vague_method = {
        'method_id': 'ALG_003',
        'name': '通用的方法',
        'description': '这是一个通用的方法',  # 过于泛化
        'applicability': [{'condition': '一般情况', 'keywords': ['通用']}],
        'template': {'steps': ['步骤1', '步骤2']}
    }

    result = filter.validate(vague_method)
    # 应该检测到描述过短和过于泛化
    assert any('过短' in issue or '泛化' in issue for issue in result.issues)
    print("✓ 测试描述质量通过")


if __name__ == '__main__':
    test_valid_method()
    test_missing_field()
    test_duplicate()
    test_description_quality()
    print("\n所有测试通过! ✓")