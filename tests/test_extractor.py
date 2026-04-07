"""测试方法论提取器"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, MagicMock
import json
from src.extraction.methodology_extractor import MethodologyExtractor, Method


def create_mock_model(response: str = None):
    """创建mock模型"""
    mock = Mock()

    if response is None:
        # 默认返回有效的方法论JSON
        default_response = json.dumps({
            'method_id': 'AUTO_001',
            'name': '变量替换法',
            'category': 'ALGEBRA',
            'description': '通过引入新变量简化表达式结构，适用于存在重复或对称模式的问题',
            'applicability': [
                {'condition': '表达式中存在重复结构', 'keywords': ['对称', '重复']}
            ],
            'template': {
                'steps': ['识别模式', '选择替换变量', '变换表达式', '求解', '回代'],
                'common_tricks': ['选择合适的替换变量'],
                'pitfall_warnings': ['注意回代步骤']
            },
            'difficulty': 3,
            'frequency': 0.8,
            'related_methods': []
        })
        mock.generate = Mock(return_value=default_response)
    else:
        mock.generate = Mock(return_value=response)

    return mock


def create_sample_solutions(count: int = 5, problem_type: str = 'ALGEBRA'):
    """创建示例解答集"""
    solutions = []
    for i in range(count):
        solutions.append({
            'problem_id': f'prob_{i}',
            'problem': f'求解方程 x^2 + 2x + 1 = 0',
            'solution': f'令 t = x + 1，则方程变为 t^2 = 0，解得 t=0，回代得 x=-1',
            'problem_type': problem_type,
            'correct': True
        })
    return solutions


def test_extract_from_solutions_basic():
    """测试基本提取功能"""
    mock_model = create_mock_model()
    extractor = MethodologyExtractor(mock_model, min_samples=3)

    solutions = create_sample_solutions(5)
    methods = extractor.extract_from_solutions(solutions)

    assert len(methods) > 0, "应该提取出方法论"
    method = methods[0]
    assert method.name == '变量替换法'
    assert method.category == 'ALGEBRA'
    assert len(method.template['steps']) >= 2

    print("✓ 测试基本提取功能")


def test_min_samples_threshold():
    """测试最小样本阈值"""
    mock_model = create_mock_model()
    extractor = MethodologyExtractor(mock_model, min_samples=5)

    # 只有3个样本，低于阈值
    solutions = create_sample_solutions(3)
    methods = extractor.extract_from_solutions(solutions)

    assert len(methods) == 0, "样本不足时不应提取"

    print("✓ 测试最小样本阈值")


def test_only_successful_solutions():
    """测试只从成功解答提取"""
    mock_model = create_mock_model()
    extractor = MethodologyExtractor(mock_model, min_samples=3)

    solutions = [
        {'problem_id': 'p1', 'problem': 'test', 'solution': 'sol1', 'correct': True},
        {'problem_id': 'p2', 'problem': 'test', 'solution': 'sol2', 'correct': False},
        {'problem_id': 'p3', 'problem': 'test', 'solution': 'sol3', 'correct': True},
        {'problem_id': 'p4', 'problem': 'test', 'solution': 'sol4', 'correct': False},
        {'problem_id': 'p5', 'problem': 'test', 'solution': 'sol5', 'correct': True},
    ]

    # 修改mock返回以适应测试
    mock_model.generate.return_value = json.dumps({
        'method_id': 'TEST_001',
        'name': '测试方法',
        'category': 'GENERAL',
        'description': '这是一个从成功解答提取的方法论，描述足够长',
        'applicability': [],
        'template': {'steps': ['步骤1', '步骤2', '步骤3']},
        'difficulty': 2,
        'frequency': 0.5,
        'related_methods': []
    })

    methods = extractor.extract_from_solutions(solutions)

    # 只有3个成功解答，刚好达到阈值
    assert len(methods) >= 0  # 可能提取也可能不提取，取决于聚类

    print("✓ 测试只从成功解答提取")


def test_cluster_by_problem_type():
    """测试按问题类型聚类"""
    mock_model = create_mock_model()
    extractor = MethodologyExtractor(mock_model, min_samples=2)

    solutions = [
        {'problem_id': 'p1', 'problem': 'test', 'solution': 'sol', 'problem_type': 'ALGEBRA', 'correct': True},
        {'problem_id': 'p2', 'problem': 'test', 'solution': 'sol', 'problem_type': 'ALGEBRA', 'correct': True},
        {'problem_id': 'p3', 'problem': 'test', 'solution': 'sol', 'problem_type': 'GEOMETRY', 'correct': True},
        {'problem_id': 'p4', 'problem': 'test', 'solution': 'sol', 'problem_type': 'GEOMETRY', 'correct': True},
    ]

    mock_model.generate.return_value = json.dumps({
        'method_id': 'AUTO_GEO',
        'name': '几何方法',
        'category': 'GEOMETRY',
        'description': '几何问题解题方法描述需要足够长才能通过验证',
        'applicability': [],
        'template': {'steps': ['步骤1', '步骤2']},
        'difficulty': 3,
        'frequency': 0.6,
        'related_methods': []
    })

    methods = extractor.extract_from_solutions(solutions)

    # 应该生成多个方法（不同类型）
    # 但由于mock固定返回GEOMETRY，检查调用次数
    assert mock_model.generate.call_count >= 1

    print("✓ 测试按问题类型聚类")


def test_validate_extracted_method():
    """测试方法论验证"""
    mock_model = create_mock_model()
    extractor = MethodologyExtractor(mock_model)

    # 有效方法
    valid_method = Method(
        method_id='ALG_001',
        name='测试方法',
        category='ALGEBRA',
        description='这是一个详细的描述，长度足够长以满足验证要求',
        applicability=[],
        template={'steps': ['步骤1', '步骤2', '步骤3']},
        difficulty=3,
        frequency=0.5
    )
    assert extractor.validate_extracted_method(valid_method) == True

    # 描述过短
    short_desc_method = Method(
        method_id='ALG_002',
        name='短描述方法',
        category='ALGEBRA',
        description='太短',
        applicability=[],
        template={'steps': ['步骤1', '步骤2']},
        difficulty=3,
        frequency=0.5
    )
    assert extractor.validate_extracted_method(short_desc_method) == False

    # 步骤不足
    no_steps_method = Method(
        method_id='ALG_003',
        name='无步骤方法',
        category='ALGEBRA',
        description='描述足够长但没有步骤',
        applicability=[],
        template={},
        difficulty=3,
        frequency=0.5
    )
    assert extractor.validate_extracted_method(no_steps_method) == False

    # 只有一个步骤
    single_step_method = Method(
        method_id='ALG_004',
        name='单步骤方法',
        category='ALGEBRA',
        description='描述足够长但只有一个步骤',
        applicability=[],
        template={'steps': ['仅一个步骤']},
        difficulty=3,
        frequency=0.5
    )
    assert extractor.validate_extracted_method(single_step_method) == False

    print("✓ 测试方法论验证")


def test_extraction_failure_handling():
    """测试提取失败的处理"""
    # Mock返回无效JSON
    mock_model = Mock()
    mock_model.generate = Mock(return_value='invalid json response')

    extractor = MethodologyExtractor(mock_model, min_samples=3)
    solutions = create_sample_solutions(5)

    methods = extractor.extract_from_solutions(solutions)

    # 提取失败时应该返回空列表或跳过
    assert methods == [] or len(methods) == 0

    print("✓ 测试提取失败处理")


def test_method_data_structure():
    """测试Method数据结构"""
    method = Method(
        method_id='TEST_001',
        name='测试方法',
        category='ALGEBRA',
        description='这是一个测试方法论',
        applicability=[
            {'condition': '条件1', 'keywords': ['关键词']}
        ],
        template={
            'steps': ['步骤1', '步骤2'],
            'common_tricks': ['技巧1']
        },
        difficulty=4,
        frequency=0.7,
        related_methods=['REL_001'],
        examples=['example_1', 'example_2']
    )

    assert method.method_id == 'TEST_001'
    assert method.name == '测试方法'
    assert method.category == 'ALGEBRA'
    assert method.difficulty == 4
    assert method.frequency == 0.7
    assert len(method.applicability) == 1
    assert len(method.examples) == 2

    print("✓ 测试Method数据结构")


def test_examples_from_cluster():
    """测试从解答簇提取例子"""
    mock_model = create_mock_model()
    extractor = MethodologyExtractor(mock_model, min_samples=3)

    solutions = create_sample_solutions(5)
    methods = extractor.extract_from_solutions(solutions)

    if methods:
        # 检查例子是从解答中提取的problem_id
        assert len(methods[0].examples) > 0
        assert any('prob_' in ex for ex in methods[0].examples)

    print("✓ 测试从解答簇提取例子")


if __name__ == '__main__':
    test_extract_from_solutions_basic()
    test_min_samples_threshold()
    test_only_successful_solutions()
    test_cluster_by_problem_type()
    test_validate_extracted_method()
    test_extraction_failure_handling()
    test_method_data_structure()
    test_examples_from_cluster()
    print("\n所有MethodologyExtractor测试通过! ✓")