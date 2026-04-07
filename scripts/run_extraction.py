#!/usr/bin/env python
"""运行方法论提取

MethodThinker方法论提炼CLI工具，从成功解答中提炼方法论模式。

用法示例:
    # 从解答文件提炼方法论
    python scripts/run_extraction.py --input data/solutions.json --output data/methodology_kb/v1/

    # 分析现有知识库模式
    python scripts/run_extraction.py --kb data/methodology_kb/v0/math_methods.yaml

    # 使用辅助模型提炼
    python scripts/run_extraction.py --input data/solutions.json --model deepseek_v3
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

from src.kb.knowledge_base import KnowledgeBase, Method
from src.extraction.methodology_extractor import MethodologyExtractor
from src.extraction.pattern_miner import PatternMiner
from src.validation.layer0_fast_filter import Layer0FastFilter


def load_solutions(path: str) -> List[Dict]:
    """加载解答数据

    Args:
        path: 解答文件路径

    Returns:
        解答列表
    """
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif path.endswith('.jsonl'):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    elif path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的文件格式: {path}")

    return data


def save_methods(methods: List[Method], output_dir: str, format: str = 'yaml'):
    """保存提炼出的方法论

    Args:
        methods: 方法论列表
        output_dir: 输出目录
        format: 输出格式 (yaml/json)
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if format == 'yaml':
        output_path = os.path.join(output_dir, f'extracted_methods_{timestamp}.yaml')

        data = {
            'methods': [
                {
                    'method_id': m.method_id,
                    'name': m.name,
                    'category': m.category,
                    'description': m.description,
                    'applicability': m.applicability,
                    'template': m.template,
                    'difficulty': m.difficulty,
                    'frequency': m.frequency,
                    'related_methods': m.related_methods,
                    'examples': m.examples
                }
                for m in methods
            ],
            'metadata': {
                'extracted_at': datetime.now().isoformat(),
                'count': len(methods),
                'source': 'run_extraction.py'
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    else:  # json
        output_path = os.path.join(output_dir, f'extracted_methods_{timestamp}.json')

        data = {
            'methods': {
                m.method_id: {
                    'method_id': m.method_id,
                    'name': m.name,
                    'category': m.category,
                    'description': m.description,
                    'applicability': m.applicability,
                    'template': m.template,
                    'difficulty': m.difficulty,
                    'frequency': m.frequency,
                    'related_methods': m.related_methods,
                    'examples': m.examples
                }
                for m in methods
            },
            'metadata': {
                'extracted_at': datetime.now().isoformat(),
                'count': len(methods)
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(methods)} 个方法论到: {output_path}")
    return output_path


def validate_extracted_methods(methods: List[Method], verbose: bool = False) -> List[Method]:
    """验证提炼出的方法论

    Args:
        methods: 方法论列表
        verbose: 是否显示详细信息

    Returns:
        通过验证的方法论列表
    """
    layer0 = Layer0FastFilter({'methods': {}})

    valid_methods = []
    invalid_count = 0

    print(f"\n验证提炼出的方法论...")

    for method in methods:
        method_dict = {
            'method_id': method.method_id,
            'name': method.name,
            'category': method.category,
            'description': method.description,
            'applicability': method.applicability,
            'template': method.template,
            'difficulty': method.difficulty,
            'frequency': method.frequency
        }

        result = layer0.validate(method_dict)

        if result.passed:
            valid_methods.append(method)
            if verbose:
                print(f"  [PASS] {method.method_id}: {method.name}")
        else:
            invalid_count += 1
            if verbose:
                print(f"  [FAIL] {method.method_id}: {method.name}")
                for issue in result.issues[:3]:
                    print(f"    - {issue}")

    print(f"\n验证结果: {len(valid_methods)}/{len(methods)} 通过")
    if invalid_count > 0:
        print(f"  失败: {invalid_count} 个方法未通过验证")

    return valid_methods


def analyze_kb_patterns(kb: KnowledgeBase, verbose: bool = False):
    """分析知识库模式

    Args:
        kb: 知识库
        verbose: 是否显示详细信息
    """
    miner = PatternMiner()

    print(f"\n分析方法论关键词模式...")

    # 按类别统计
    categories = ['ALGEBRA', 'GEOMETRY', 'NUMBER_THEORY', 'COMBINATORICS', 'GENERAL']

    for category in categories:
        methods = kb.get_methods_by_category(category)
        if methods:
            print(f"\n{category} ({len(methods)}个方法):")

            # 分析关键词频率
            keyword_freq = {}
            for m in methods:
                for app in m.applicability:
                    for kw in app.get('keywords', []):
                        keyword_freq[kw] = keyword_freq.get(kw, 0) + 1

            if verbose and keyword_freq:
                sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
                print(f"  关键词: {sorted_keywords[:10]}")

            # 分析难度分布
            difficulty_dist = {}
            for m in methods:
                d = m.difficulty
                difficulty_dist[d] = difficulty_dist.get(d, 0) + 1

            if verbose and difficulty_dist:
                print(f"  难度分布: {dict(sorted(difficulty_dist.items()))}")

            for m in methods[:3]:
                print(f"  - {m.name}: {m.description[:50]}...")


def extract_from_solutions(
    solutions: List[Dict],
    output_dir: str,
    min_samples: int = 3,
    model: Optional[str] = None,
    verbose: bool = False,
    validate: bool = True
) -> List[Method]:
    """从解答提炼方法论

    Args:
        solutions: 解答列表
        output_dir: 输出目录
        min_samples: 最小样本数
        model: 辅助模型名称
        verbose: 是否显示详细信息
        validate: 是否验证提炼结果

    Returns:
        提炼出的方法论列表
    """
    print(f"\n从解答提炼方法论...")
    print(f"  解答数量: {len(solutions)}")
    print(f"  最小样本数: {min_samples}")
    print(f"  辅助模型: {model or '无'}")

    # 统计成功解答
    successful = [s for s in solutions if s.get('correct', False)]
    print(f"  成功解答: {len(successful)} ({len(successful)/len(solutions)*100:.1f}%)")

    if len(successful) < min_samples:
        print(f"\n警告: 成功解答数量不足 ({len(successful)} < {min_samples})")
        return []

    # 创建提取器
    extractor = MethodologyExtractor(
        assistant_model=None,  # 实际使用时传入模型
        min_samples=min_samples
    )

    # 按问题类型分组
    type_groups = {}
    for s in successful:
        ptype = s.get('problem_type', 'unknown')
        if ptype not in type_groups:
            type_groups[ptype] = []
        type_groups[ptype].append(s)

    print(f"\n问题类型分布:")
    for ptype, sols in type_groups.items():
        print(f"  {ptype}: {len(sols)} 个解答")

    # 提炼方法论
    methods = []

    for ptype, sols in type_groups.items():
        if len(sols) >= min_samples:
            print(f"\n处理 {ptype} 类型...")

            # 调用提取器
            extracted = extractor.extract_from_solutions(sols)

            if extracted:
                methods.extend(extracted)
                print(f"  提炼出 {len(extracted)} 个方法论")

                if verbose:
                    for m in extracted:
                        print(f"    - {m.method_id}: {m.name}")
            else:
                print(f"  未提炼出方法论")

    print(f"\n总计提炼: {len(methods)} 个方法论")

    # 验证提炼结果
    if validate and methods:
        methods = validate_extracted_methods(methods, verbose)

    # 保存结果
    if methods:
        save_methods(methods, output_dir)

    return methods


def merge_with_kb(
    new_methods: List[Method],
    kb: KnowledgeBase,
    output_dir: str,
    verbose: bool = False
) -> KnowledgeBase:
    """将新方法合并到知识库

    Args:
        new_methods: 新方法列表
        kb: 现有知识库
        output_dir: 输出目录
        verbose: 是否显示详细信息

    Returns:
        更新后的知识库
    """
    print(f"\n合并新方法到知识库...")

    merged_count = 0
    skipped_count = 0

    for method in new_methods:
        # 检查是否存在相似方法
        similar = kb.find_similar_methods(method, threshold=0.8)

        if similar:
            skipped_count += 1
            if verbose:
                print(f"  跳过 {method.method_id}: 与 {similar[0].method_id} 相似")
        else:
            kb.add_method(method)
            merged_count += 1
            if verbose:
                print(f"  添加 {method.method_id}: {method.name}")

    print(f"\n合并结果:")
    print(f"  新添加: {merged_count}")
    print(f"  跳过(相似): {skipped_count}")
    print(f"  知识库总数: {len(kb.methods)}")

    # 保存更新后的知识库
    output_path = os.path.join(output_dir, 'updated_kb.json')
    kb.save(output_path)
    print(f"  已保存到: {output_path}")

    return kb


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MethodThinker方法论提炼 - 从成功解答中提炼方法论',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从解答文件提炼方法论
  %(prog)s --input data/solutions.json --output data/methodology_kb/v1/

  # 分析现有知识库模式
  %(prog)s --kb data/methodology_kb/v0/math_methods.yaml

  # 使用辅助模型提炼
  %(prog)s --input data/solutions.json --model deepseek_v3 --verbose

  # 合并到现有知识库
  %(prog)s --input data/solutions.json --merge-kb data/methodology_kb/v0/math_methods.yaml
"""
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='解答数据文件路径 (json/jsonl/yaml)'
    )
    parser.add_argument(
        '--kb',
        type=str,
        default='data/methodology_kb/v0/math_methods.yaml',
        help='现有知识库路径 (default: data/methodology_kb/v0/math_methods.yaml)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/methodology_kb/v1/',
        help='输出目录 (default: data/methodology_kb/v1/)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['deepseek_v3', 'qwen_math', 'gpt4o_mini', 'local'],
        help='辅助提炼模型'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=3,
        help='提炼所需的最小成功样本数 (default: 3)'
    )
    parser.add_argument(
        '--merge-kb',
        type=str,
        help='将提炼结果合并到指定知识库'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='验证提炼结果 (default: True)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='跳过验证'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['yaml', 'json'],
        default='yaml',
        help='输出格式 (default: yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='只分析现有知识库，不提炼'
    )

    args = parser.parse_args()

    # 处理验证选项
    validate = args.validate and not args.no_validate

    # 分析模式
    if args.analyze_only or not args.input:
        print(f"加载知识库: {args.kb}")
        try:
            kb = KnowledgeBase.from_yaml(args.kb)
            print(f"知识库包含 {len(kb.methods)} 个方法")
            analyze_kb_patterns(kb, args.verbose)
        except FileNotFoundError:
            print(f"错误: 知识库文件不存在: {args.kb}")
        return

    # 提炼模式
    if args.input:
        print(f"MethodThinker 方法论提炼")
        print(f"{'='*50}")

        # 加载解答数据
        print(f"\n加载解答数据: {args.input}")
        try:
            solutions = load_solutions(args.input)
            print(f"加载 {len(solutions)} 条解答")
        except Exception as e:
            print(f"错误: 无法加载解答数据: {e}")
            return

        # 提炼方法论
        methods = extract_from_solutions(
            solutions=solutions,
            output_dir=args.output,
            min_samples=args.min_samples,
            model=args.model,
            verbose=args.verbose,
            validate=validate
        )

        # 合并到知识库
        if args.merge_kb and methods:
            print(f"\n加载目标知识库: {args.merge_kb}")
            try:
                target_kb = KnowledgeBase.from_yaml(args.merge_kb)
                merge_with_kb(methods, target_kb, args.output, args.verbose)
            except FileNotFoundError:
                print(f"警告: 目标知识库不存在，创建新知识库")
                new_kb = KnowledgeBase()
                for m in methods:
                    new_kb.add_method(m)
                new_kb.save(os.path.join(args.output, 'new_kb.json'))

        print(f"\n{'='*50}")
        print(f"提取完成!")

        # 显示摘要
        if methods:
            print(f"\n提炼摘要:")
            for m in methods[:5]:
                print(f"  {m.method_id}: {m.name} ({m.category})")
            if len(methods) > 5:
                print(f"  ... 还有 {len(methods) - 5} 个方法")


if __name__ == '__main__':
    main()