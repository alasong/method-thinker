#!/usr/bin/env python
"""训练数据生成CLI工具

MethodThinker训练样本生成器，支持批量生成和多种输出格式。

用法示例:
    # 从KB生成训练数据
    python scripts/generate_training_data.py --kb data/methodology_kb/v1/kb.yaml --output data/train.json

    # Pass@K多样本生成
    python scripts/generate_training_data.py --kb kb.yaml --mode pass-k --k 8 --output data/pass_k_train.jsonl

    # 批量生成（指定问题文件）
    python scripts/generate_training_data.py --problems problems.json --kb kb.yaml --output data/batch_train.json

    # 从KB自动生成（平衡类别和难度）
    python scripts/generate_training_data.py --kb kb.yaml --mode from-kb --total 100 --output data/auto_train.json

    # 使用自定义配置
    python scripts/generate_training_data.py --kb kb.yaml --config configs/generator_config.yaml
"""

import sys
import os
import json
import yaml
import argparse
from typing import List, Dict, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sample_generator import SampleGenerator, TrainingSampleV2, create_training_samples
from src.kb.knowledge_base import KnowledgeBase, Method


def load_kb(kb_path: str) -> KnowledgeBase:
    """加载知识库

    Args:
        kb_path: 知识库路径（支持.yaml, .yml, .json）

    Returns:
        KnowledgeBase实例
    """
    if kb_path.endswith('.yaml') or kb_path.endswith('.yml'):
        kb = KnowledgeBase.from_yaml(kb_path)
    elif kb_path.endswith('.json'):
        kb = KnowledgeBase.load(kb_path)
    else:
        raise ValueError(f"不支持的KB格式: {kb_path}")

    print(f"加载知识库: {kb_path}")
    print(f"  方法数: {len(kb.methods)}")
    print(f"  类别: {list(kb.category_index.keys())}")

    return kb


def load_problems(problems_path: str) -> List[Dict]:
    """加载问题列表

    Args:
        problems_path: 问题文件路径（支持.yaml, .yml, .json, .jsonl）

    Returns:
        问题列表
    """
    problems = []

    if problems_path.endswith('.yaml') or problems_path.endswith('.yml'):
        with open(problems_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'problems' in data:
                problems = data['problems']
            elif isinstance(data, list):
                problems = data
            else:
                problems = [data]
    elif problems_path.endswith('.jsonl'):
        with open(problems_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
    else:
        with open(problems_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                problems = data
            elif isinstance(data, dict) and 'problems' in data:
                problems = data['problems']
            else:
                problems = [data]

    print(f"加载问题: {problems_path}")
    print(f"  问题数: {len(problems)}")

    return problems


def load_config(config_path: str) -> Dict:
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        print(f"警告: 配置文件不存在: {config_path}")
        return {}

    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def create_sample_problems() -> List[Dict]:
    """创建示例问题列表"""
    return [
        {
            'problem': '求解方程 x + 1/x = 5',
            'problem_type': '方程求解',
            'difficulty': 3
        },
        {
            'problem': '求函数 f(x) = x^2 - 4x + 5 的最小值',
            'problem_type': '函数最值',
            'difficulty': 2
        },
        {
            'problem': '证明对于所有正数x，x + 1/x ≥ 2',
            'problem_type': '不等式证明',
            'difficulty': 2
        },
        {
            'problem': '求将5个不同元素排列的方法数',
            'problem_type': '计数问题',
            'difficulty': 2
        },
        {
            'problem': '在三角形ABC中，AB=AC，证明∠B=∠C',
            'problem_type': '几何证明',
            'difficulty': 3
        }
    ]


def run_single_mode(args, kb: KnowledgeBase, generator: SampleGenerator):
    """单样本生成模式"""
    problem = args.problem or "求解方程 x + 1/x = 5"
    method_id = args.method or generator._select_best_method(problem, args.problem_type)

    sample = generator.generate_sample(
        problem=problem,
        method_id=method_id,
        problem_type=args.problem_type,
        difficulty=args.difficulty
    )

    return [sample]


def run_batch_mode(args, kb: KnowledgeBase, generator: SampleGenerator):
    """批量生成模式"""
    problems = load_problems(args.problems) if args.problems else create_sample_problems()

    samples = generator.generate_batch(
        problems=problems,
        samples_per_problem=args.samples_per_problem,
        difficulty_distribution=args.difficulty_distribution
    )

    return samples


def run_pass_k_mode(args, kb: KnowledgeBase, generator: SampleGenerator):
    """Pass@K多样本生成模式"""
    problems = load_problems(args.problems) if args.problems else create_sample_problems()

    samples = generator.generate_batch(
        problems=problems,
        pass_k=args.k,
        difficulty_distribution=args.difficulty_distribution
    )

    return samples


def run_from_kb_mode(args, kb: KnowledgeBase, generator: SampleGenerator):
    """从KB自动生成模式"""
    samples = generator.generate_from_kb(
        total_samples=args.total,
        balance_by_category=args.balance_category,
        balance_by_difficulty=args.balance_difficulty
    )

    return samples


def save_samples(samples: List[TrainingSampleV2], output_path: str, format: str = 'json'):
    """保存样本到文件

    Args:
        samples: 样本列表
        output_path: 输出路径
        format: 输出格式
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([s.to_dict() for s in samples], f, ensure_ascii=False, indent=2)

    print(f"保存样本: {len(samples)} 条 -> {output_path}")


def generate_statistics(samples: List[TrainingSampleV2], kb: KnowledgeBase) -> Dict:
    """生成样本统计信息"""
    stats = {
        'total_samples': len(samples),
        'methods_used': {},
        'problem_types': {},
        'difficulty_distribution': {},
        'categories': {}
    }

    for sample in samples:
        # 方法统计
        method_id = sample.method_id
        stats['methods_used'][method_id] = stats['methods_used'].get(method_id, 0) + 1

        # 问题类型统计
        ptype = sample.problem_type or 'unknown'
        stats['problem_types'][ptype] = stats['problem_types'].get(ptype, 0) + 1

        # 难度分布
        diff = sample.difficulty
        stats['difficulty_distribution'][diff] = stats['difficulty_distribution'].get(diff, 0) + 1

        # 类别统计
        method = kb.get_method(method_id)
        if method:
            cat = method.category
            stats['categories'][cat] = stats['categories'].get(cat, 0) + 1

    return stats


def print_statistics(stats: Dict):
    """打印统计信息"""
    print("\n" + "=" * 50)
    print("样本统计信息")
    print("=" * 50)

    print(f"\n总样本数: {stats['total_samples']}")

    print(f"\n方法分布:")
    for method_id, count in stats['methods_used'].items():
        print(f"  {method_id}: {count} ({count/stats['total_samples']*100:.1f}%)")

    print(f"\n问题类型分布:")
    for ptype, count in stats['problem_types'].items():
        print(f"  {ptype}: {count} ({count/stats['total_samples']*100:.1f}%)")

    print(f"\n难度分布:")
    for diff, count in sorted(stats['difficulty_distribution'].items()):
        print(f"  难度{diff}: {count} ({count/stats['total_samples']*100:.1f}%)")

    print(f"\n类别分布:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count} ({count/stats['total_samples']*100:.1f}%)")


def save_report(stats: Dict, output_dir: str, samples_path: str):
    """保存生成报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'samples_path': samples_path,
        'statistics': stats
    }

    report_path = os.path.join(output_dir, 'generation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n生成报告已保存: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MethodThinker训练数据生成器 - 批量生成带方法论注入的训练样本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
生成模式:
  single:   单样本生成（指定问题和方法）
  batch:    批量生成（从问题文件）
  pass-k:   Pass@K多样本生成（每题K个不同解）
  from-kb:  从KB自动生成（平衡类别和难度）

示例:
  # 单样本生成
  %(prog)s --kb kb.yaml --mode single --problem "求解方程 x + 1/x = 5" --method ALG_001

  # 批量生成
  %(prog)s --kb kb.yaml --mode batch --problems problems.json --output data/train.json

  # Pass@K多样本生成
  %(prog)s --kb kb.yaml --mode pass-k --k 8 --problems problems.json --output data/pass_k.jsonl

  # 从KB自动生成
  %(prog)s --kb kb.yaml --mode from-kb --total 100 --output data/auto_train.json

  # 使用自定义配置
  %(prog)s --kb kb.yaml --config configs/generator_config.yaml
"""
    )

    # 必需参数
    parser.add_argument(
        '--kb',
        type=str,
        required=True,
        help='方法论知识库路径（.yaml/.yml/.json）'
    )

    # 生成模式
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'batch', 'pass-k', 'from-kb'],
        default='batch',
        help='生成模式 (default: batch)'
    )

    # 输出参数
    parser.add_argument(
        '--output',
        type=str,
        default='data/generated/train_samples.json',
        help='输出文件路径 (default: data/generated/train_samples.json)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'jsonl'],
        default='json',
        help='输出格式 (default: json)'
    )

    # 问题参数（single/batch/pass-k模式）
    parser.add_argument(
        '--problems',
        type=str,
        help='问题文件路径（.json/.jsonl）'
    )
    parser.add_argument(
        '--problem',
        type=str,
        help='单个问题描述（single模式）'
    )
    parser.add_argument(
        '--problem-type',
        type=str,
        help='问题类型'
    )
    parser.add_argument(
        '--method',
        type=str,
        help='指定方法ID（single模式）'
    )
    parser.add_argument(
        '--difficulty',
        type=int,
        default=3,
        help='难度等级 (default: 3)'
    )
    parser.add_argument(
        '--samples-per-problem',
        type=int,
        default=1,
        help='每题样本数（batch模式） (default: 1)'
    )

    # Pass@K参数
    parser.add_argument(
        '--k',
        type=int,
        default=8,
        help='Pass@K数量 (default: 8)'
    )
    parser.add_argument(
        '--diversity-mode',
        type=str,
        choices=['method', 'step', 'random'],
        default='method',
        help='Pass@K多样性模式 (default: method)'
    )

    # From-KB参数
    parser.add_argument(
        '--total',
        type=int,
        default=100,
        help='总样本数（from-kb模式） (default: 100)'
    )
    parser.add_argument(
        '--balance-category',
        action='store_true',
        default=True,
        help='按类别平衡样本'
    )
    parser.add_argument(
        '--balance-difficulty',
        action='store_true',
        default=True,
        help='按难度平衡样本'
    )

    # 难度分布
    parser.add_argument(
        '--difficulty-distribution',
        type=str,
        help='难度分布配置（JSON格式，如 {1:0.1,2:0.2,3:0.3,4:0.2,5:0.2}）'
    )

    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        help='生成器配置文件路径'
    )

    # 其他参数
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只显示配置，不实际生成'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='生成统计报告'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MethodThinker训练数据生成器")
    print("=" * 60)

    # 加载知识库
    try:
        kb = load_kb(args.kb)
    except Exception as e:
        print(f"错误: 无法加载知识库: {e}")
        return

    # 加载配置
    config = {}
    if args.config:
        config = load_config(args.config)

    # 解析难度分布
    difficulty_distribution = None
    if args.difficulty_distribution:
        try:
            difficulty_distribution = json.loads(args.difficulty_distribution)
        except json.JSONDecodeError:
            print(f"错误: 难度分布格式错误")
            return

    # 创建生成器
    generator = SampleGenerator(kb, config=config)

    print(f"\n生成配置:")
    print(f"  模式: {args.mode}")
    print(f"  输出: {args.output}")
    print(f"  格式: {args.format}")

    if args.dry_run:
        print("\n[Dry Run] 不执行实际生成")
        return

    # 执行生成
    print(f"\n开始生成...")
    start_time = datetime.now()

    if args.mode == 'single':
        samples = run_single_mode(args, kb, generator)
    elif args.mode == 'batch':
        samples = run_batch_mode(args, kb, generator)
    elif args.mode == 'pass-k':
        samples = run_pass_k_mode(args, kb, generator)
    elif args.mode == 'from-kb':
        samples = run_from_kb_mode(args, kb, generator)
    else:
        samples = run_batch_mode(args, kb, generator)

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n生成完成!")
    print(f"  样本数: {len(samples)}")
    print(f"  耗时: {elapsed:.2f} 秒")

    # 保存样本
    save_samples(samples, args.output, args.format)

    # 统计信息
    stats = generate_statistics(samples, kb)

    if args.report or args.verbose:
        print_statistics(stats)

        # 保存报告
        output_dir = os.path.dirname(args.output) or '.'
        save_report(stats, output_dir, args.output)

    # 显示样本示例（verbose模式）
    if args.verbose and len(samples) > 0:
        print("\n" + "=" * 50)
        print("样本示例")
        print("=" * 50)

        example = samples[0]
        print(f"\n问题: {example.problem[:80]}...")
        print(f"方法: {example.method_name} ({example.method_id})")
        print(f"方法选择: {example.method_selection[:100]}...")
        print(f"步骤数: {len(example.solution_steps)}")
        if example.solution_steps:
            print(f"首步: {example.solution_steps[0][:60]}...")
        print(f"答案: {example.final_answer[:60]}...")

    print("\n" + "=" * 60)
    print("生成完成! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()