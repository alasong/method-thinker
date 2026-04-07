"""测试训练样本生成器"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sample_generator import SampleGenerator, TrainingSampleV2, create_training_samples
from src.data.method_injector import MethodInjector
from src.kb.knowledge_base import KnowledgeBase, Method
import tempfile


def create_test_kb() -> KnowledgeBase:
    """创建测试知识库"""
    kb = KnowledgeBase()

    methods = [
        Method(
            method_id='ALG_001',
            name='变量替换法',
            category='ALGEBRA',
            description='通过引入新变量简化表达式结构，适用于存在重复或对称模式的问题',
            applicability=[
                {
                    'condition': '表达式中存在重复结构',
                    'keywords': ['对称', '重复', '倒数'],
                    'problem_types': ['方程求解', '函数最值']
                }
            ],
            template={
                'steps': ['识别重复模式', '选择替换变量', '变换表达式', '求解新方程', '回代并验证'],
                'common_tricks': ['设 t = x + 1/x', '利用韦达定理'],
                'pitfall_warnings': ['注意新变量的取值范围', '检验回代后的解']
            },
            difficulty=3,
            frequency=0.85
        ),
        Method(
            method_id='ALG_002',
            name='配方法',
            category='ALGEBRA',
            description='将二次式化为完全平方形式，便于分析最值或证明不等式',
            applicability=[
                {
                    'condition': '涉及二次表达式',
                    'keywords': ['二次', '平方', '最值'],
                    'problem_types': ['函数最值', '不等式证明']
                }
            ],
            template={
                'steps': ['提取二次项系数', '配方成完全平方', '分析最值或范围'],
                'common_tricks': ['配方后结合均值不等式'],
                'pitfall_warnings': ['注意二次项系数的正负']
            },
            difficulty=2,
            frequency=0.92
        ),
        Method(
            method_id='GEO_001',
            name='坐标化方法',
            category='GEOMETRY',
            description='建立坐标系，将几何问题转化为代数计算问题',
            applicability=[
                {
                    'condition': '涉及距离、角度等可量化元素',
                    'keywords': ['距离', '坐标', '长度'],
                    'problem_types': ['几何证明', '距离计算']
                }
            ],
            template={
                'steps': ['建立坐标系', '确定关键点坐标', '转化为代数计算'],
                'common_tricks': ['选择合适的坐标系简化计算'],
                'pitfall_warnings': ['验证几何与代数的对应关系']
            },
            difficulty=3
        )
    ]

    for m in methods:
        kb.add_method(m)

    return kb


def test_sample_generator_init():
    """测试样本生成器初始化"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    assert generator.kb == kb
    assert generator.injector is not None
    assert generator.config is not None
    assert 'include_method_description' in generator.config
    print("✓ 测试样本生成器初始化通过")


def test_training_sample_v2():
    """测试训练样本V2数据类"""
    sample = TrainingSampleV2(
        problem="求解方程 x + 1/x = 5",
        method_selection="观察到题目包含倒数、对称特征，选择变量替换法",
        solution_steps=["识别重复模式", "设t = x + 1/x", "求解方程"],
        final_answer="x = (5 ± √21)/2",
        method_id="ALG_001",
        method_name="变量替换法",
        problem_type="方程求解",
        difficulty=3,
        sample_id="ALG_001_001"
    )

    assert sample.method_id == "ALG_001"
    assert sample.method_name == "变量替换法"
    assert len(sample.solution_steps) == 3
    assert sample.final_answer != ""

    # 测试转换字典
    sample_dict = sample.to_dict()
    assert 'problem' in sample_dict
    assert 'method_selection' in sample_dict
    assert 'solution_steps' in sample_dict

    print("✓ 测试训练样本V2数据类通过")


def test_generate_single_sample():
    """测试生成单个样本"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    problem = "求解方程 x + 1/x = 5"
    sample = generator.generate_sample(
        problem=problem,
        method_id='ALG_001',
        problem_type='方程求解',
        difficulty=3
    )

    assert sample is not None
    assert sample.problem == problem
    assert sample.method_id == 'ALG_001'
    assert sample.method_name == '变量替换法'
    assert sample.method_selection != ""
    assert len(sample.solution_steps) > 0
    assert sample.final_answer != ""
    assert sample.sample_id is not None

    print(f"生成样本ID: {sample.sample_id}")
    print(f"方法选择: {sample.method_selection[:50]}...")
    print(f"步骤数: {len(sample.solution_steps)}")
    print(f"答案: {sample.final_answer}")
    print("✓ 测试生成单个样本通过")


def test_generate_pass_k_samples():
    """测试Pass@K多样本生成"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    problem = "求函数 f(x) = x^2 - 4x + 5 的最小值"

    # 测试方法多样性模式
    samples = generator.generate_pass_k_samples(
        problem=problem,
        k=4,
        problem_type='函数最值',
        difficulty=3,
        diversity_mode='method'
    )

    assert len(samples) == 4
    assert all(s.problem == problem for s in samples)

    # 检查方法多样性
    method_ids = [s.method_id for s in samples]
    print(f"生成样本数: {len(samples)}")
    print(f"使用方法: {method_ids}")

    # 测试步骤多样性模式
    step_samples = generator.generate_pass_k_samples(
        problem=problem,
        k=3,
        problem_type='函数最值',
        difficulty=3,
        diversity_mode='step'
    )

    assert len(step_samples) == 3
    assert all(s.method_id == step_samples[0].method_id for s in step_samples)

    print(f"步骤多样性样本数: {len(step_samples)}")
    print("✓ 测试Pass@K多样本生成通过")


def test_generate_batch():
    """测试批量生成样本"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    problems = [
        {'problem': '求解 x + 1/x = 3', 'problem_type': '方程求解', 'difficulty': 2},
        {'problem': '求 f(x) = x^2 + 4x + 1 的最小值', 'problem_type': '函数最值', 'difficulty': 3},
        {'problem': '证明三角形ABC中AB=AC时∠B=∠C', 'problem_type': '几何证明', 'difficulty': 3}
    ]

    samples = generator.generate_batch(
        problems=problems,
        samples_per_problem=2
    )

    assert len(samples) == 6  # 3 problems * 2 samples
    assert all(s.problem in [p['problem'] for p in problems] for s in samples)

    print(f"批量生成样本数: {len(samples)}")
    print(f"问题覆盖: {[s.problem[:20] + '...' for s in samples[:3]]}")
    print("✓ 测试批量生成通过")


def test_generate_batch_with_pass_k():
    """测试批量生成（Pass@K模式）"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    problems = [
        {'problem': '求解 x + 1/x = 4', 'problem_type': '方程求解'},
        {'problem': '求 x^2 - 6x + 9 的最值', 'problem_type': '函数最值'}
    ]

    samples = generator.generate_batch(
        problems=problems,
        pass_k=3
    )

    assert len(samples) == 6  # 2 problems * 3 samples each
    assert all(s.problem in [p['problem'] for p in problems] for s in samples)

    print(f"Pass@K批量生成: {len(samples)} 样本")
    print("✓ 测试批量生成（Pass@K模式）通过")


def test_generate_from_kb():
    """测试从KB自动生成"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    samples = generator.generate_from_kb(
        total_samples=20,
        balance_by_category=True,
        balance_by_difficulty=True
    )

    assert len(samples) <= 20

    # 检查类别覆盖
    categories = set()
    for s in samples:
        method = kb.get_method(s.method_id)
        if method:
            categories.add(method.category)

    print(f"从KB生成样本数: {len(samples)}")
    print(f"覆盖类别: {categories}")

    # 检查难度分布
    difficulties = [s.difficulty for s in samples]
    print(f"难度范围: {min(difficulties)} - {max(difficulties)}")
    print("✓ 测试从KB自动生成通过")


def test_method_injection_in_samples():
    """测试样本中的方法注入"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    problem = "求解方程 x + 1/x = 5"
    sample = generator.generate_sample(
        problem=problem,
        method_id='ALG_001',
        problem_type='方程求解',
        difficulty=3
    )

    # 检查方法选择包含描述
    assert '变量替换法' in sample.method_selection
    assert 'ALG_001' in sample.method_selection

    # 检查步骤包含标注
    annotated_steps = [s for s in sample.solution_steps if s.startswith('[')]
    print(f"标注步骤数: {len(annotated_steps)}")

    # 检查标注数据
    if sample.annotations:
        assert len(sample.annotations) > 0
        annotation_types = [a['annotation_type'] for a in sample.annotations]
        assert 'selection' in annotation_types
        print(f"标注数: {len(sample.annotations)}")
        print(f"标注类型: {annotation_types}")

    print("✓ 测试样本中的方法注入通过")


def test_save_and_load_samples():
    """测试样本保存和加载"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    # 生成样本
    problems = [
        {'problem': '求解 x + 1/x = 3', 'problem_type': '方程求解'},
        {'problem': '求 x^2 + 4x 的最值', 'problem_type': '函数最值'}
    ]
    samples = generator.generate_batch(problems, samples_per_problem=2)

    # 保存JSON
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_json_path = f.name

    generator.save_samples(samples, temp_json_path, format='json')

    # 加载
    loaded_samples = SampleGenerator.load_samples(temp_json_path)

    assert len(loaded_samples) == len(samples)
    assert loaded_samples[0].method_id == samples[0].method_id
    assert loaded_samples[0].problem == samples[0].problem

    print(f"JSON保存/加载: {len(samples)} -> {len(loaded_samples)}")

    # 保存JSONL
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        temp_jsonl_path = f.name

    generator.save_samples(samples, temp_jsonl_path, format='jsonl')

    loaded_jsonl = SampleGenerator.load_samples(temp_jsonl_path)
    assert len(loaded_jsonl) == len(samples)

    print(f"JSONL保存/加载: {len(samples)} -> {len(loaded_jsonl)}")

    # 清理
    os.unlink(temp_json_path)
    os.unlink(temp_jsonl_path)

    print("✓ 测试样本保存和加载通过")


def test_difficulty_distribution():
    """测试难度分布"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    problems = [{'problem': f'测试问题{i}', 'problem_type': '方程求解'} for i in range(10)]

    custom_distribution = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.2, 5: 0.2}

    samples = generator.generate_batch(
        problems=problems,
        samples_per_problem=1,
        difficulty_distribution=custom_distribution
    )

    difficulties = [s.difficulty for s in samples]
    assert min(difficulties) >= 1
    assert max(difficulties) <= 5

    # 统计分布
    from collections import Counter
    diff_counts = Counter(difficulties)

    print(f"生成样本数: {len(samples)}")
    print(f"难度分布: {dict(diff_counts)}")
    print("✓ 测试难度分布通过")


def test_create_training_samples_function():
    """测试便捷函数"""
    kb = create_test_kb()

    problems = [
        {'problem': '求解方程 x + 1/x = 5', 'problem_type': '方程求解'}
    ]

    # 单样本模式
    samples = create_training_samples(problems, kb)
    assert len(samples) == 1

    print(f"便捷函数生成: {len(samples)} 样本")

    # Pass@K模式
    pass_k_samples = create_training_samples(problems, kb, pass_k=4)
    assert len(pass_k_samples) == 4

    print(f"Pass@K模式: {len(pass_k_samples)} 样本")
    print("✓ 测试便捷函数通过")


def test_sample_id_generation():
    """测试样本ID生成"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    # 相同问题，不同方法
    problem = "求解方程 x + 1/x = 5"
    sample1 = generator.generate_sample(problem, 'ALG_001')
    sample2 = generator.generate_sample(problem, 'ALG_002')

    assert sample1.sample_id != sample2.sample_id
    assert 'ALG_001' in sample1.sample_id
    assert 'ALG_002' in sample2.sample_id

    print(f"样本1 ID: {sample1.sample_id}")
    print(f"样本2 ID: {sample2.sample_id}")
    print("✓ 测试样本ID生成通过")


def test_step_variant_generation():
    """测试步骤变体生成"""
    kb = create_test_kb()
    generator = SampleGenerator(kb)

    problem = "求函数 f(x) = x^2 - 4x + 5 的最小值"

    # 生成步骤变体
    variant_samples = generator.generate_pass_k_samples(
        problem=problem,
        k=3,
        problem_type='函数最值',
        diversity_mode='step'
    )

    assert len(variant_samples) == 3

    # 检查步骤表述不同
    steps_list = [s.solution_steps for s in variant_samples]
    # 至少应该有不同的前缀
    prefixes = [s[0] if s else '' for s in steps_list]
    print(f"步骤前缀: {prefixes[:2]}")

    print("✓ 测试步骤变体生成通过")


def test_integration():
    """集成测试：完整生成流程"""
    kb = create_test_kb()

    # 1. 创建生成器
    generator = SampleGenerator(kb, config={
        'include_method_description': True,
        'include_common_tricks': True,
        'include_pitfall_warnings': True,
        'answer_format': 'detailed'
    })

    print("Step 1: 创建生成器")

    # 2. 批量生成Pass@K样本
    problems = [
        {'problem': '求解 x + 1/x = 5', 'problem_type': '方程求解', 'difficulty': 3},
        {'problem': '求 f(x) = x^2 + 2x + 1 的最小值', 'problem_type': '函数最值', 'difficulty': 2}
    ]

    samples = generator.generate_batch(problems, pass_k=4)
    print(f"Step 2: Pass@K生成 {len(samples)} 样本")

    # 3. 检查样本完整性
    for sample in samples[:3]:
        assert sample.problem is not None
        assert sample.method_selection is not None
        assert len(sample.solution_steps) > 0
        assert sample.final_answer is not None
        assert sample.method_id is not None

    print("Step 3: 验证样本完整性")

    # 4. 保存样本
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name

    generator.save_samples(samples, temp_path)
    print(f"Step 4: 保存到 {temp_path}")

    # 5. 加载并验证
    loaded = SampleGenerator.load_samples(temp_path)
    assert len(loaded) == len(samples)

    print(f"Step 5: 加载验证成功，{len(loaded)} 样本")

    # 6. 输出样本示例
    example = samples[0]
    print(f"\n样本示例:")
    print(f"  问题: {example.problem[:40]}...")
    print(f"  方法: {example.method_name} ({example.method_id})")
    print(f"  选择推理: {example.method_selection[:60]}...")
    print(f"  步骤数: {len(example.solution_steps)}")
    print(f"  答案: {example.final_answer[:40]}")

    # 清理
    os.unlink(temp_path)

    print("✓ 集成测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试样本生成器模块")
    print("=" * 60)

    test_sample_generator_init()
    test_training_sample_v2()
    test_generate_single_sample()
    test_generate_pass_k_samples()
    test_generate_batch()
    test_generate_batch_with_pass_k()
    test_generate_from_kb()
    test_method_injection_in_samples()
    test_save_and_load_samples()
    test_difficulty_distribution()
    test_create_training_samples_function()
    test_sample_id_generation()
    test_step_variant_generation()
    test_integration()

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()