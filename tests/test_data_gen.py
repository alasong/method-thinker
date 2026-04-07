"""测试数据生成模块"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_generator import DataGenerator, TrainingSample
from src.data.method_injector import MethodInjector, MethodAnnotation, create_annotated_dataset
from src.kb.knowledge_base import KnowledgeBase, Method


def test_data_generator_init():
    """测试数据生成器初始化"""
    kb = KnowledgeBase()

    # 添加测试方法
    method = Method(
        method_id='ALG_001',
        name='变量替换法',
        category='ALGEBRA',
        description='通过引入新变量简化表达式',
        applicability=[
            {
                'condition': '表达式中存在重复结构',
                'keywords': ['对称', '重复'],
                'problem_types': ['方程求解', '不等式证明']
            }
        ],
        template={
            'steps': ['识别模式', '选择替换变量', '变换表达式', '求解', '回代'],
            'common_tricks': ['设 t = x + y'],
            'pitfall_warnings': ['注意变量取值范围']
        },
        difficulty=3,
        frequency=0.8
    )
    kb.add_method(method)

    generator = DataGenerator(kb)
    assert generator.kb == kb
    assert generator.difficulty_range == (1, 5)
    print("✓ 测试数据生成器初始化通过")


def test_generate_single_sample():
    """测试生成单个样本"""
    kb = KnowledgeBase()

    method = Method(
        method_id='ALG_001',
        name='变量替换法',
        category='ALGEBRA',
        description='通过引入新变量简化表达式结构',
        applicability=[
            {
                'condition': '表达式中存在重复结构',
                'keywords': ['对称', '重复', '复合函数'],
                'problem_types': ['方程求解', '函数最值']
            }
        ],
        template={
            'steps': ['识别模式', '选择替换变量', '变换表达式', '求解', '回代'],
            'common_tricks': ['设 t = x + 1/x'],
            'pitfall_warnings': ['注意变量取值范围']
        },
        difficulty=3
    )
    kb.add_method(method)

    generator = DataGenerator(kb)

    # 生成样本
    sample = generator.generate_sample('ALG_001')

    assert sample is not None
    assert sample.method_used == 'ALG_001'
    assert sample.problem_id.startswith('ALG_001')
    assert sample.problem_type in ['方程求解', '函数最值']
    assert sample.difficulty >= 1 and sample.difficulty <= 5
    assert len(sample.method_steps) > 0
    assert len(sample.annotations) > 0

    print(f"生成的样本问题: {sample.problem[:50]}...")
    print(f"方法步骤数: {len(sample.method_steps)}")
    print(f"标注数: {len(sample.annotations)}")
    print("✓ 测试生成单个样本通过")


def test_generate_batch():
    """测试批量生成样本"""
    kb = KnowledgeBase()

    # 添加多个方法
    methods = [
        Method(
            method_id='ALG_001',
            name='变量替换法',
            category='ALGEBRA',
            description='通过引入新变量简化表达式',
            applicability=[{'condition': '有重复结构', 'keywords': ['对称'], 'problem_types': ['方程求解']}],
            template={'steps': ['识别', '替换', '求解']},
            difficulty=3
        ),
        Method(
            method_id='ALG_002',
            name='配方法',
            category='ALGEBRA',
            description='将二次式化为完全平方',
            applicability=[{'condition': '二次表达式', 'keywords': ['二次'], 'problem_types': ['函数最值']}],
            template={'steps': ['提取系数', '配方', '分析']},
            difficulty=2
        )
    ]

    for m in methods:
        kb.add_method(m)

    generator = DataGenerator(kb)

    # 批量生成
    samples = generator.generate_batch(count_per_method=2)

    assert len(samples) == 4  # 2方法 * 2样本
    assert all(s.method_used in ['ALG_001', 'ALG_002'] for s in samples)

    # 检查样本多样性
    difficulties = [s.difficulty for s in samples]
    problem_types = [s.problem_type for s in samples]

    print(f"生成样本数: {len(samples)}")
    print(f"难度分布: {difficulties}")
    print(f"问题类型分布: {problem_types}")
    print("✓ 测试批量生成通过")


def test_generate_diverse_dataset():
    """测试生成多样性数据集"""
    kb = KnowledgeBase()

    # 添加多类别方法
    methods = [
        Method(
            method_id='ALG_001',
            name='变量替换法',
            category='ALGEBRA',
            description='通过引入新变量简化表达式',
            applicability=[{'condition': '有重复结构', 'keywords': ['对称'], 'problem_types': ['方程求解']}],
            template={'steps': ['识别', '替换', '求解']},
            difficulty=3
        ),
        Method(
            method_id='GEO_001',
            name='坐标化方法',
            category='GEOMETRY',
            description='将几何问题转为代数问题',
            applicability=[{'condition': '距离角度问题', 'keywords': ['距离'], 'problem_types': ['几何证明']}],
            template={'steps': ['建系', '坐标化', '计算']},
            difficulty=3
        ),
        Method(
            method_id='NUM_001',
            name='模运算',
            category='NUMBER_THEORY',
            description='利用模运算研究整数性质',
            applicability=[{'condition': '整除余数', 'keywords': ['整除'], 'problem_types': ['整除性']}],
            template={'steps': ['选模', '取模', '推导']},
            difficulty=3
        )
    ]

    for m in methods:
        kb.add_method(m)

    generator = DataGenerator(kb)

    # 生成多样性数据集
    samples = generator.generate_diverse_dataset(
        total_samples=15,
        balance_by_category=True,
        balance_by_difficulty=True
    )

    assert len(samples) <= 15

    # 检查类别平衡
    categories_used = set()
    for s in samples:
        method = kb.get_method(s.method_used)
        if method:
            categories_used.add(method.category)

    print(f"生成样本数: {len(samples)}")
    print(f"覆盖类别: {categories_used}")

    # 检查难度分布
    difficulties = [s.difficulty for s in samples]
    print(f"难度范围: {min(difficulties)} - {max(difficulties)}")
    print("✓ 测试生成多样性数据集通过")


def test_method_injector_init():
    """测试方法注入器初始化"""
    kb = KnowledgeBase()

    injector = MethodInjector(kb)

    assert injector.kb == kb
    assert 'include_method_description' in injector.injection_config
    assert injector.injection_config['include_method_description'] == True

    print("✓ 测试方法注入器初始化通过")


def test_inject_method():
    """测试方法注入"""
    kb = KnowledgeBase()

    method = Method(
        method_id='ALG_001',
        name='变量替换法',
        category='ALGEBRA',
        description='通过引入新变量简化表达式结构，适用于存在重复或对称模式的问题',
        applicability=[
            {
                'condition': '表达式中存在重复结构',
                'keywords': ['对称', '重复'],
                'problem_types': ['方程求解']
            }
        ],
        template={
            'steps': ['识别重复模式', '选择替换变量', '变换表达式', '求解新方程', '回代并验证'],
            'common_tricks': ['设 t = x + y，利用韦达定理'],
            'pitfall_warnings': ['注意新变量的取值范围']
        },
        difficulty=3
    )
    kb.add_method(method)

    injector = MethodInjector(kb)

    problem = "求解方程 x + 1/x = 5"
    raw_solution = "观察到方程中x和1/x互为倒数，设t = x + 1/x = 5，则x * 1/x = 1，利用韦达定理得到x是方程t^2 - 5t + 1 = 0的根，解得x = (5 ± √21)/2"

    injected_solution, annotations = injector.inject_method(
        problem, raw_solution, 'ALG_001', '方程求解'
    )

    assert injected_solution is not None
    assert len(annotations) > 0

    # 检查标注类型
    annotation_types = [a.annotation_type for a in annotations]
    assert 'selection' in annotation_types
    assert 'application' in annotation_types

    print(f"注入后的解（前100字）:\n{injected_solution[:100]}...")
    print(f"标注数: {len(annotations)}")
    print(f"标注类型: {annotation_types}")
    print("✓ 测试方法注入通过")


def test_inject_methods_chain():
    """测试方法链注入"""
    kb = KnowledgeBase()

    methods = [
        Method(
            method_id='ALG_001',
            name='变量替换法',
            category='ALGEBRA',
            description='引入新变量简化',
            applicability=[{'condition': '重复结构', 'keywords': ['对称'], 'problem_types': ['方程求解']}],
            template={'steps': ['识别模式', '设变量', '变换']},
            difficulty=3
        ),
        Method(
            method_id='ALG_002',
            name='韦达定理应用',
            category='ALGEBRA',
            description='利用根与系数关系',
            applicability=[{'condition': '涉及方程根', 'keywords': ['根'], 'problem_types': ['方程求解']}],
            template={'steps': ['写韦达定理', '建立方程']},
            difficulty=3
        )
    ]

    for m in methods:
        kb.add_method(m)

    injector = MethodInjector(kb)

    problem = "已知x + 1/x = 5，求x的值"
    raw_solution = "设t = x + 1/x = 5，则由韦达定理，x满足方程t^2 - 5t + 1 = 0，解得x = (5 ± √21)/2"

    injected_solution, annotations = injector.inject_methods_chain(
        problem, raw_solution, ['ALG_001', 'ALG_002']
    )

    assert len(annotations) > 0

    # 检查方法链标注
    method_ids_in_annotations = [a.method_id for a in annotations]
    assert 'ALG_001' in method_ids_in_annotations
    assert 'ALG_002' in method_ids_in_annotations

    # 检查过渡标注
    transition_annotations = [a for a in annotations if a.annotation_type == 'transition']
    assert len(transition_annotations) > 0

    print(f"注入后的解（前150字）:\n{injected_solution[:150]}...")
    print(f"方法链标注数: {len(annotations)}")
    print(f"包含方法: {set(method_ids_in_annotations)}")
    print("✓ 测试方法链注入通过")


def test_annotation_dataclass():
    """测试标注数据类"""
    annotation = MethodAnnotation(
        method_id='ALG_001',
        method_name='变量替换法',
        step_index=0,
        step_description='识别重复模式',
        annotation_type='application',
        reasoning='观察到x和1/x互为倒数'
    )

    assert annotation.method_id == 'ALG_001'
    assert annotation.annotation_type == 'application'
    assert annotation.step_index == 0

    print("✓ 测试标注数据类通过")


def test_training_sample_dataclass():
    """测试训练样本数据类"""
    sample = TrainingSample(
        problem_id='TEST_001',
        problem='求解方程 x^2 - 5x + 6 = 0',
        problem_type='方程求解',
        solution='使用因式分解，得到(x-2)(x-3)=0，解得x=2或x=3',
        method_used='ALG_003',
        method_steps=['提取公因式', '十字相乘', '求解'],
        difficulty=3,
        annotations=['[方法选择] 使用因式分解法'],
        source='test'
    )

    assert sample.problem_id == 'TEST_001'
    assert sample.difficulty == 3
    assert len(sample.method_steps) == 3

    print("✓ 测试训练样本数据类通过")


def test_create_annotated_dataset():
    """测试创建标注数据集"""
    kb = KnowledgeBase()

    method = Method(
        method_id='ALG_001',
        name='变量替换法',
        category='ALGEBRA',
        description='引入新变量简化表达式',
        applicability=[{'condition': '重复结构', 'keywords': ['对称'], 'problem_types': ['方程求解']}],
        template={'steps': ['识别', '设变量', '变换']},
        difficulty=3
    )
    kb.add_method(method)

    problems = [
        {
            'problem_id': 'P001',
            'problem': '求解 x + 1/x = 3',
            'solution': '设t = x + 1/x，解方程',
            'method_used': 'ALG_001',
            'problem_type': '方程求解',
            'difficulty': 3
        },
        {
            'problem_id': 'P002',
            'problem': '求解 x + 1/x = 4',
            'solution': '设t = x + 1/x，解方程',
            'method_used': 'ALG_001',
            'problem_type': '方程求解',
            'difficulty': 3
        }
    ]

    annotated_data = create_annotated_dataset(problems, kb)

    assert len(annotated_data) == 2
    assert all('annotations' in item for item in annotated_data)
    assert all(item['method_used'] == 'ALG_001' for item in annotated_data)

    print(f"标注数据集大小: {len(annotated_data)}")
    print(f"每项标注数: {[len(item['annotations']) for item in annotated_data]}")
    print("✓ 测试创建标注数据集通过")


def test_save_and_load_samples():
    """测试样本保存和加载"""
    import tempfile

    kb = KnowledgeBase()

    method = Method(
        method_id='ALG_001',
        name='变量替换法',
        category='ALGEBRA',
        description='引入新变量简化表达式',
        applicability=[{'condition': '重复结构', 'keywords': ['对称'], 'problem_types': ['方程求解']}],
        template={'steps': ['识别', '设变量', '变换']},
        difficulty=3
    )
    kb.add_method(method)

    generator = DataGenerator(kb)
    samples = generator.generate_batch(['ALG_001'], count_per_method=3)

    # 保存
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name

    generator.save_samples(samples, temp_path)

    # 加载
    loaded_samples = DataGenerator.load_samples(temp_path)

    assert len(loaded_samples) == len(samples)
    assert loaded_samples[0].problem_id == samples[0].problem_id
    assert loaded_samples[0].method_used == samples[0].method_used

    # 清理
    os.unlink(temp_path)

    print(f"保存样本数: {len(samples)}")
    print(f"加载样本数: {len(loaded_samples)}")
    print("✓ 测试样本保存和加载通过")


def test_difficulty_distribution():
    """测试难度分布生成"""
    kb = KnowledgeBase()

    method = Method(
        method_id='ALG_001',
        name='变量替换法',
        category='ALGEBRA',
        description='引入新变量简化表达式',
        applicability=[{'condition': '重复结构', 'keywords': ['对称'], 'problem_types': ['方程求解']}],
        template={'steps': ['识别', '设变量', '变换']},
        difficulty=3
    )
    kb.add_method(method)

    generator = DataGenerator(kb)

    # 使用自定义难度分布
    custom_distribution = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.2, 5: 0.2}

    samples = generator.generate_batch(
        ['ALG_001'],
        count_per_method=20,
        difficulty_distribution=custom_distribution
    )

    difficulties = [s.difficulty for s in samples]

    # 检查难度范围
    assert min(difficulties) >= 1
    assert max(difficulties) <= 5

    # 统计分布
    from collections import Counter
    diff_counts = Counter(difficulties)

    print(f"生成样本数: {len(samples)}")
    print(f"难度分布: {dict(diff_counts)}")
    print("✓ 测试难度分布生成通过")


def test_integration():
    """集成测试：完整数据生成流程"""
    kb = KnowledgeBase()

    # 添加多个方法
    methods = [
        Method(
            method_id='ALG_001',
            name='变量替换法',
            category='ALGEBRA',
            description='引入新变量简化表达式结构',
            applicability=[{'condition': '重复结构', 'keywords': ['对称'], 'problem_types': ['方程求解', '函数最值']}],
            template={
                'steps': ['识别模式', '选择替换变量', '变换表达式', '求解', '回代'],
                'common_tricks': ['设 t = x + 1/x'],
                'pitfall_warnings': ['注意变量范围']
            },
            difficulty=3,
            frequency=0.85
        ),
        Method(
            method_id='ALG_002',
            name='配方法',
            category='ALGEBRA',
            description='将二次式化为完全平方',
            applicability=[{'condition': '二次表达式', 'keywords': ['二次'], 'problem_types': ['函数最值']}],
            template={
                'steps': ['提取系数', '配方', '分析'],
                'common_tricks': ['配方后用均值不等式'],
                'pitfall_warnings': ['注意二次项系数正负']
            },
            difficulty=2,
            frequency=0.92
        )
    ]

    for m in methods:
        kb.add_method(m)

    # 1. 使用数据生成器生成原始样本
    generator = DataGenerator(kb)
    raw_samples = generator.generate_batch(count_per_method=3)

    print(f"Step 1: 生成原始样本数: {len(raw_samples)}")

    # 2. 使用方法注入器增强样本
    injector = MethodInjector(kb)

    enhanced_samples = []
    for sample in raw_samples:
        injected_solution, annotations = injector.inject_method(
            sample.problem,
            sample.solution,
            sample.method_used,
            sample.problem_type
        )

        enhanced_sample = TrainingSample(
            problem_id=sample.problem_id,
            problem=sample.problem,
            problem_type=sample.problem_type,
            solution=injected_solution,
            method_used=sample.method_used,
            method_steps=sample.method_steps,
            difficulty=sample.difficulty,
            annotations=[a.reasoning for a in annotations],
            source='enhanced'
        )
        enhanced_samples.append(enhanced_sample)

    print(f"Step 2: 增强样本数: {len(enhanced_samples)}")
    print(f"增强后平均标注数: {sum(len(s.annotations) for s in enhanced_samples) / len(enhanced_samples):.1f}")

    # 3. 保存最终数据
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name

    generator.save_samples(enhanced_samples, temp_path)
    print(f"Step 3: 数据已保存到: {temp_path}")

    # 4. 验证加载
    loaded = DataGenerator.load_samples(temp_path)
    assert len(loaded) == len(enhanced_samples)

    print(f"Step 4: 加载验证成功，样本数: {len(loaded)}")

    # 清理
    os.unlink(temp_path)

    print("✓ 集成测试通过")


if __name__ == '__main__':
    print("=" * 60)
    print("开始测试数据生成模块")
    print("=" * 60)

    test_data_generator_init()
    test_generate_single_sample()
    test_generate_batch()
    test_generate_diverse_dataset()
    test_method_injector_init()
    test_inject_method()
    test_inject_methods_chain()
    test_annotation_dataclass()
    test_training_sample_dataclass()
    test_create_annotated_dataset()
    test_save_and_load_samples()
    test_difficulty_distribution()
    test_integration()

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)