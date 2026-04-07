#!/usr/bin/env python
"""数据集管理CLI工具

MethodThinker数据集管理命令行工具，支持多种数据集操作。

用法示例:
    # 加载并验证数据集
    python scripts/manage_datasets.py load data/train_data/train.json --validate

    # 显示数据集统计信息
    python scripts/manage_datasets.py stats data/train_data/train.json

    # 合并多个数据集
    python scripts/manage_datasets.py merge data/train_data/*.json -o data/merged.json

    # 分割数据集
    python scripts/manage_datasets.py split data/raw_data.json --ratio 0.8,0.1,0.1

    # 格式转换
    python scripts/manage_datasets.py convert data/train.json --to jsonl

    # 按难度筛选
    python scripts/manage_datasets.py filter data/train.json --difficulty 3-5

    # 检查类别平衡
    python scripts/manage_datasets.py balance data/train.json --check
"""

import sys
import os
import json
import yaml
import argparse
import random
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MethodologyDataset, MethodologySample


class DatasetManager:
    """数据集管理器

    提供数据集的加载、验证、统计、合并、分割、转换等功能。
    """

    def __init__(self, config_path: Optional[str] = None):
        """初始化管理器

        Args:
            config_path: 配置文件路径
        """
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, path: str) -> Dict:
        """加载配置文件

        Args:
            path: 配置文件路径

        Returns:
            配置字典
        """
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"不支持的配置格式: {path}")
        return self.config

    def load_dataset(self, path: str, validate: bool = False) -> MethodologyDataset:
        """加载数据集

        Args:
            path: 数据集路径
            validate: 是否验证

        Returns:
            MethodologyDataset实例
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据集文件不存在: {path}")

        dataset = MethodologyDataset(path)

        if validate:
            errors = self.validate_dataset(dataset)
            if errors:
                print(f"验证发现 {len(errors)} 个问题:")
                for err in errors[:10]:  # 显示前10个错误
                    print(f"  - {err}")
                if len(errors) > 10:
                    print(f"  ... 还有 {len(errors) - 10} 个问题")

        return dataset

    def validate_dataset(self, dataset: MethodologyDataset) -> List[str]:
        """验证数据集

        Args:
            dataset: 数据集

        Returns:
            错误列表
        """
        errors = []
        required_fields = ['problem_id', 'problem', 'problem_type', 'difficulty']

        for i, sample in enumerate(dataset.samples):
            # 检查必要字段
            for field in required_fields:
                if not getattr(sample, field, None):
                    errors.append(f"样本 {i}: 缺少必要字段 '{field}'")

            # 检查难度范围
            if sample.difficulty < 1 or sample.difficulty > 5:
                errors.append(f"样本 {i}: 难度值 {sample.difficulty} 超出范围 [1,5]")

            # 检查解答步骤
            if not sample.solution_steps:
                errors.append(f"样本 {i}: 缺少解答步骤")

            # 检查问题类型有效性
            valid_types = self.config.get('valid_problem_types', [
                'ALGEBRA', 'GEOMETRY', 'NUMBER_THEORY', 'COMBINATORICS',
                'CALCULUS', 'PROBABILITY', 'LOGIC', 'GENERAL'
            ])
            if sample.problem_type not in valid_types:
                errors.append(f"样本 {i}: 无效问题类型 '{sample.problem_type}'")

        return errors

    def get_statistics(self, dataset: MethodologyDataset) -> Dict:
        """获取数据集统计信息

        Args:
            dataset: 数据集

        Returns:
            统计信息字典
        """
        stats = {
            'total_samples': len(dataset),
            'problem_types': Counter(),
            'difficulty_distribution': Counter(),
            'verified_ratio': 0,
            'avg_solution_steps': 0,
            'sources': Counter(),
            'methods_used': Counter()
        }

        if len(dataset) == 0:
            return stats

        verified_count = 0
        total_steps = 0

        for sample in dataset.samples:
            stats['problem_types'][sample.problem_type] += 1
            stats['difficulty_distribution'][sample.difficulty] += 1
            stats['sources'][sample.source] += 1

            if sample.verified:
                verified_count += 1

            total_steps += len(sample.solution_steps)
            if sample.selected_method:
                stats['methods_used'][sample.selected_method] += 1

        stats['verified_ratio'] = verified_count / len(dataset)
        stats['avg_solution_steps'] = total_steps / len(dataset)

        return stats

    def print_statistics(self, stats: Dict, verbose: bool = False):
        """打印统计信息

        Args:
            stats: 统计信息
            verbose: 是否详细输出
        """
        print("\n数据集统计信息")
        print("=" * 50)
        print(f"总样本数: {stats['total_samples']}")
        print(f"验证比例: {stats['verified_ratio']:.2%}")
        print(f"平均解答步骤: {stats['avg_solution_steps']:.1f}")

        print("\n问题类型分布:")
        for type_name, count in stats['problem_types'].most_common():
            pct = count / stats['total_samples'] * 100
            print(f"  {type_name}: {count} ({pct:.1f}%)")

        print("\n难度分布:")
        for diff in sorted(stats['difficulty_distribution'].keys()):
            count = stats['difficulty_distribution'][diff]
            pct = count / stats['total_samples'] * 100
            print(f"  难度 {diff}: {count} ({pct:.1f}%)")

        if verbose:
            print("\n数据来源:")
            for source, count in stats['sources'].most_common(10):
                print(f"  {source}: {count}")

            print("\n常用方法 (前10):")
            for method, count in stats['methods_used'].most_common(10):
                print(f"  {method}: {count}")

    def merge_datasets(
        self,
        paths: List[str],
        output_path: str,
        deduplicate: bool = True,
        shuffle: bool = True
    ) -> MethodologyDataset:
        """合并多个数据集

        Args:
            paths: 数据集路径列表
            output_path: 输出路径
            deduplicate: 是否去重
            shuffle: 是否打乱

        Returns:
            合并后的数据集
        """
        merged = MethodologyDataset()
        seen_ids = set()

        for path in paths:
            if not os.path.exists(path):
                print(f"警告: 文件不存在，跳过: {path}")
                continue

            dataset = MethodologyDataset(path)
            for sample in dataset.samples:
                if deduplicate:
                    if sample.problem_id in seen_ids:
                        continue
                    seen_ids.add(sample.problem_id)
                merged.samples.append(sample)

            print(f"加载: {path} ({len(dataset)} 样本)")

        if shuffle:
            random.shuffle(merged.samples)

        merged.save(output_path)
        print(f"合并完成: {output_path} ({len(merged)} 样本)")

        return merged

    def split_dataset(
        self,
        path: str,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        output_dir: Optional[str] = None,
        stratify: bool = True
    ) -> Tuple[MethodologyDataset, MethodologyDataset, MethodologyDataset]:
        """分割数据集

        Args:
            path: 数据集路径
            ratios: 分割比例 (train, val, test)
            output_dir: 输出目录
            stratify: 是否按类型分层

        Returns:
            (train_set, val_set, test_set)
        """
        dataset = MethodologyDataset(path)

        if stratify:
            # 按问题类型分层分割
            type_groups = defaultdict(list)
            for sample in dataset.samples:
                type_groups[sample.problem_type].append(sample)

            train_samples = []
            val_samples = []
            test_samples = []

            for type_name, samples in type_groups.items():
                random.shuffle(samples)
                n = len(samples)
                train_end = int(n * ratios[0])
                val_end = train_end + int(n * ratios[1])

                train_samples.extend(samples[:train_end])
                val_samples.extend(samples[train_end:val_end])
                test_samples.extend(samples[val_end:])
        else:
            # 随机分割
            samples = dataset.samples.copy()
            random.shuffle(samples)
            n = len(samples)
            train_end = int(n * ratios[0])
            val_end = train_end + int(n * ratios[1])

            train_samples = samples[:train_end]
            val_samples = samples[train_end:val_end]
            test_samples = samples[val_end:]

        # 创建数据集
        train_set = MethodologyDataset()
        val_set = MethodologyDataset()
        test_set = MethodologyDataset()

        train_set.samples = train_samples
        val_set.samples = val_samples
        test_set.samples = test_samples

        # 保存
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            train_path = os.path.join(output_dir, 'train.json')
            val_path = os.path.join(output_dir, 'val.json')
            test_path = os.path.join(output_dir, 'test.json')

            train_set.save(train_path)
            val_set.save(val_path)
            test_set.save(test_path)

            print(f"\n分割完成:")
            print(f"  训练集: {train_path} ({len(train_set)} 样本, {len(train_set)/len(dataset)*100:.1f}%)")
            print(f"  验证集: {val_path} ({len(val_set)} 样本, {len(val_set)/len(dataset)*100:.1f}%)")
            print(f"  测试集: {test_path} ({len(test_set)} 样本, {len(test_set)/len(dataset)*100:.1f}%)")

        return train_set, val_set, test_set

    def convert_format(
        self,
        input_path: str,
        output_path: str,
        target_format: str = 'jsonl'
    ) -> bool:
        """转换数据集格式

        Args:
            input_path: 输入路径
            output_path: 输出路径
            target_format: 目标格式 (json, jsonl, parquet)

        Returns:
            是否成功
        """
        dataset = MethodologyDataset(input_path)

        if target_format == 'json':
            dataset.save(output_path)
        elif target_format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in dataset.samples:
                    data = {
                        'problem_id': sample.problem_id,
                        'problem': sample.problem,
                        'problem_type': sample.problem_type,
                        'difficulty': sample.difficulty,
                        'candidate_methods': sample.candidate_methods,
                        'selected_method': sample.selected_method,
                        'selection_reasoning': sample.selection_reasoning,
                        'solution_steps': sample.solution_steps,
                        'solution_annotations': sample.solution_annotations,
                        'reflection': sample.reflection,
                        'source': sample.source,
                        'verified': sample.verified
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
        elif target_format == 'parquet':
            try:
                import pandas as pd
                records = [
                    {
                        'problem_id': s.problem_id,
                        'problem': s.problem,
                        'problem_type': s.problem_type,
                        'difficulty': s.difficulty,
                        'selected_method': s.selected_method,
                        'solution_steps': json.dumps(s.solution_steps),
                        'reflection': s.reflection,
                        'source': s.source,
                        'verified': s.verified
                    }
                    for s in dataset.samples
                ]
                df = pd.DataFrame(records)
                df.to_parquet(output_path)
            except ImportError:
                print("错误: 需要安装 pandas 和 pyarrow: pip install pandas pyarrow")
                return False
        else:
            print(f"错误: 不支持的目标格式: {target_format}")
            return False

        print(f"转换完成: {output_path} ({len(dataset)} 样本)")
        return True

    def filter_dataset(
        self,
        path: str,
        output_path: str,
        filters: Dict[str, Any]
    ) -> MethodologyDataset:
        """筛选数据集

        Args:
            path: 数据集路径
            output_path: 输出路径
            filters: 筛选条件

        Returns:
            筛选后的数据集
        """
        dataset = MethodologyDataset(path)
        filtered = MethodologyDataset()

        for sample in dataset.samples:
            include = True

            # 按难度筛选
            if 'difficulty' in filters:
                diff_range = filters['difficulty']
                if '-' in diff_range:
                    min_diff, max_diff = map(int, diff_range.split('-'))
                else:
                    min_diff = max_diff = int(diff_range)
                if not (min_diff <= sample.difficulty <= max_diff):
                    include = False

            # 按类型筛选
            if 'problem_type' in filters:
                if sample.problem_type not in filters['problem_type']:
                    include = False

            # 按来源筛选
            if 'source' in filters:
                if sample.source not in filters['source']:
                    include = False

            # 按验证状态筛选
            if 'verified' in filters:
                if sample.verified != filters['verified']:
                    include = False

            if include:
                filtered.samples.append(sample)

        filtered.save(output_path)
        print(f"筛选完成: {output_path} ({len(filtered)} / {len(dataset)} 样本)")
        return filtered

    def check_balance(
        self,
        dataset: MethodologyDataset,
        target_distribution: Optional[Dict] = None
    ) -> Dict:
        """检查类别平衡

        Args:
            dataset: 数据集
            target_distribution: 目标分布

        Returns:
            平衡分析结果
        """
        stats = self.get_statistics(dataset)
        total = stats['total_samples']

        if total == 0:
            return {'balanced': False, 'reason': '数据集为空'}

        # 检查问题类型平衡
        type_counts = stats['problem_types']
        expected_ratio = 1.0 / len(type_counts) if len(type_counts) > 0 else 0

        type_balance = {}
        for type_name, count in type_counts.items():
            actual_ratio = count / total
            deviation = abs(actual_ratio - expected_ratio)
            type_balance[type_name] = {
                'count': count,
                'ratio': actual_ratio,
                'deviation': deviation,
                'balanced': deviation < 0.15  # 允许15%偏差
            }

        # 检查难度平衡
        diff_counts = stats['difficulty_distribution']
        diff_balance = {}
        for diff, count in diff_counts.items():
            actual_ratio = count / total
            diff_balance[diff] = {
                'count': count,
                'ratio': actual_ratio,
                'balanced': actual_ratio > 0.05  # 至少5%
            }

        # 综合判断
        type_balanced = all(t['balanced'] for t in type_balance.values())
        diff_balanced = all(d['balanced'] for d in diff_balance.values())

        result = {
            'balanced': type_balanced and diff_balanced,
            'type_balance': type_balance,
            'difficulty_balance': diff_balance,
            'recommendations': []
        }

        # 生成建议
        if not type_balanced:
            underrepresented = [
                t for t, v in type_balance.items()
                if not v['balanced'] and v['ratio'] < expected_ratio
            ]
            if underrepresented:
                result['recommendations'].append(
                    f"建议增加以下类型数据: {', '.join(underrepresented)}"
                )

        if not diff_balanced:
            low_diff = [d for d, v in diff_balance.items() if not v['balanced']]
            if low_diff:
                result['recommendations'].append(
                    f"难度分布不均衡，低比例难度: {low_diff}"
                )

        return result

    def print_balance_report(self, balance_result: Dict):
        """打印平衡报告

        Args:
            balance_result: 平衡分析结果
        """
        print("\n类别平衡分析")
        print("=" * 50)

        print(f"整体平衡状态: {'均衡' if balance_result['balanced'] else '不均衡'}")

        print("\n问题类型分布:")
        for type_name, info in balance_result['type_balance'].items():
            status = "均衡" if info['balanced'] else "不均衡"
            print(f"  {type_name}: {info['count']} ({info['ratio']:.1%}) [{status}]")

        print("\n难度分布:")
        for diff, info in balance_result['difficulty_balance'].items():
            status = "充足" if info['balanced'] else "不足"
            print(f"  难度{diff}: {info['count']} ({info['ratio']:.1%}) [{status}]")

        if balance_result['recommendations']:
            print("\n优化建议:")
            for rec in balance_result['recommendations']:
                print(f"  - {rec}")

    def balance_dataset(
        self,
        path: str,
        output_path: str,
        strategy: str = 'oversample',
        target_distribution: Optional[Dict] = None
    ) -> MethodologyDataset:
        """平衡数据集

        Args:
            path: 数据集路径
            output_path: 输出路径
            strategy: 平衡策略 (oversample, undersample, hybrid)
            target_distribution: 目标分布

        Returns:
            平衡后的数据集
        """
        dataset = MethodologyDataset(path)

        # 使用配置中的目标分布
        if target_distribution is None and self.config:
            target_distribution = self.config.get('balance', {}).get('target_distribution')

        # 按问题类型分组
        type_groups = defaultdict(list)
        for sample in dataset.samples:
            type_groups[sample.problem_type].append(sample)

        balanced = MethodologyDataset()

        if strategy == 'oversample':
            # 过采样：复制少数类样本
            max_count = max(len(g) for g in type_groups.values())

            for type_name, samples in type_groups.items():
                n = len(samples)
                if n < max_count:
                    # 复制样本以达到目标数量
                    needed = max_count - n
                    additional = random.choices(samples, k=needed)
                    samples = samples + additional
                balanced.samples.extend(samples)

            print(f"过采样平衡: {len(dataset)} -> {len(balanced)} 样本")

        elif strategy == 'undersample':
            # 欠采样：减少多数类样本
            min_count = min(len(g) for g in type_groups.values())

            for type_name, samples in type_groups.items():
                n = len(samples)
                if n > min_count:
                    # 随机抽取到目标数量
                    samples = random.sample(samples, min_count)
                balanced.samples.extend(samples)

            print(f"欠采样平衡: {len(dataset)} -> {len(balanced)} 样本")

        elif strategy == 'hybrid':
            # 混合策略：设定目标数量
            target_count = sum(len(g) for g in type_groups.values()) // len(type_groups)

            for type_name, samples in type_groups.items():
                n = len(samples)
                if n < target_count:
                    # 过采样
                    needed = target_count - n
                    additional = random.choices(samples, k=needed)
                    samples = samples + additional
                elif n > target_count:
                    # 欠采样
                    samples = random.sample(samples, target_count)
                balanced.samples.extend(samples)

            print(f"混合平衡: {len(dataset)} -> {len(balanced)} 样本")

        random.shuffle(balanced.samples)
        balanced.save(output_path)

        return balanced

    def export_summary(
        self,
        dataset: MethodologyDataset,
        output_path: str,
        format: str = 'markdown'
    ):
        """导出数据集摘要

        Args:
            dataset: 数据集
            output_path: 输出路径
            format: 格式 (markdown, json)
        """
        stats = self.get_statistics(dataset)
        balance = self.check_balance(dataset)

        if format == 'markdown':
            content = f"""# 数据集摘要

> 生成时间: {datetime.now().isoformat()}

## 基本信息

- 总样本数: {stats['total_samples']}
- 已验证比例: {stats['verified_ratio']:.2%}
- 平均解答步骤数: {stats['avg_solution_steps']:.1f}

## 问题类型分布

| 类型 | 数量 | 占比 |
|-----|------|------|
"""
            for type_name, count in stats['problem_types'].most_common():
                pct = count / stats['total_samples'] * 100
                content += f"| {type_name} | {count} | {pct:.1f}% |\n"

            content += "\n## 难度分布\n\n| 难度 | 数量 | 占比 |\n|-----|------|------|\n"
            for diff in sorted(stats['difficulty_distribution'].keys()):
                count = stats['difficulty_distribution'][diff]
                pct = count / stats['total_samples'] * 100
                content += f"| {diff} | {count} | {pct:.1f}% |\n"

            content += "\n## 平衡状态\n\n"
            content += f"- 整体平衡: {'均衡' if balance['balanced'] else '不均衡'}\n"
            if balance['recommendations']:
                content += "- 建议:\n"
                for rec in balance['recommendations']:
                    content += f"  - {rec}\n"

        elif format == 'json':
            content = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'statistics': stats,
                'balance': balance
            }, ensure_ascii=False, indent=2)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"摘要已导出: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MethodThinker数据集管理CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
操作命令:
  load      加载并验证数据集
  stats     显示数据集统计信息
  merge     合并多个数据集
  split     分割数据集
  convert   格式转换
  filter    筛选数据集
  balance   平衡数据集类别分布
  summary   导出数据集摘要

示例:
  %(prog)s load data/train.json --validate
  %(prog)s stats data/train.json --verbose
  %(prog)s merge data/*.json -o data/merged.json
  %(prog)s split data/raw.json --ratio 0.8,0.1,0.1
  %(prog)s convert data/train.json --to jsonl
  %(prog)s filter data/train.json --difficulty 3-5 -o filtered.json
  %(prog)s balance data/train.json --strategy oversample -o balanced.json
  %(prog)s summary data/train.json -o SUMMARY.md
"""
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/dataset_config.yaml',
        help='配置文件路径'
    )

    subparsers = parser.add_subparsers(dest='command', help='操作命令')

    # load 命令
    load_parser = subparsers.add_parser('load', help='加载并验证数据集')
    load_parser.add_argument('path', type=str, help='数据集路径')
    load_parser.add_argument('--validate', action='store_true', help='验证数据集')
    load_parser.add_argument('--stats', action='store_true', help='显示统计信息')

    # stats 命令
    stats_parser = subparsers.add_parser('stats', help='显示统计信息')
    stats_parser.add_argument('path', type=str, help='数据集路径')
    stats_parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')

    # merge 命令
    merge_parser = subparsers.add_parser('merge', help='合并数据集')
    merge_parser.add_argument('paths', type=str, nargs='+', help='数据集路径')
    merge_parser.add_argument('-o', '--output', type=str, required=True, help='输出路径')
    merge_parser.add_argument('--no-dedup', action='store_true', help='不去重')
    merge_parser.add_argument('--no-shuffle', action='store_true', help='不打乱')

    # split 命令
    split_parser = subparsers.add_parser('split', help='分割数据集')
    split_parser.add_argument('path', type=str, help='数据集路径')
    split_parser.add_argument('--ratio', type=str, default='0.8,0.1,0.1', help='分割比例')
    split_parser.add_argument('-o', '--output-dir', type=str, help='输出目录')
    split_parser.add_argument('--no-stratify', action='store_true', help='不分层分割')

    # convert 命令
    convert_parser = subparsers.add_parser('convert', help='格式转换')
    convert_parser.add_argument('path', type=str, help='输入路径')
    convert_parser.add_argument('--to', type=str, choices=['json', 'jsonl', 'parquet'],
                                default='jsonl', help='目标格式')
    convert_parser.add_argument('-o', '--output', type=str, help='输出路径')

    # filter 命令
    filter_parser = subparsers.add_parser('filter', help='筛选数据集')
    filter_parser.add_argument('path', type=str, help='数据集路径')
    filter_parser.add_argument('--difficulty', type=str, help='难度范围 (如 3-5)')
    filter_parser.add_argument('--type', type=str, nargs='+', help='问题类型')
    filter_parser.add_argument('--source', type=str, nargs='+', help='数据来源')
    filter_parser.add_argument('--verified', type=bool, help='验证状态')
    filter_parser.add_argument('-o', '--output', type=str, required=True, help='输出路径')

    # balance 命令
    balance_parser = subparsers.add_parser('balance', help='平衡数据集')
    balance_parser.add_argument('path', type=str, help='数据集路径')
    balance_parser.add_argument('--strategy', type=str,
                                choices=['oversample', 'undersample', 'hybrid'],
                                default='oversample', help='平衡策略')
    balance_parser.add_argument('--check', action='store_true', help='只检查不平衡')
    balance_parser.add_argument('-o', '--output', type=str, help='输出路径')

    # summary 命令
    summary_parser = subparsers.add_parser('summary', help='导出摘要')
    summary_parser.add_argument('path', type=str, help='数据集路径')
    summary_parser.add_argument('-o', '--output', type=str, required=True, help='输出路径')
    summary_parser.add_argument('--format', type=str, choices=['markdown', 'json'],
                                default='markdown', help='输出格式')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 创建管理器
    manager = DatasetManager(args.config)

    # 执行命令
    if args.command == 'load':
        dataset = manager.load_dataset(args.path, validate=args.validate)
        if args.stats:
            stats = manager.get_statistics(dataset)
            manager.print_statistics(stats)

    elif args.command == 'stats':
        dataset = manager.load_dataset(args.path)
        stats = manager.get_statistics(dataset)
        manager.print_statistics(stats, verbose=args.verbose)

    elif args.command == 'merge':
        ratios = [float(r) for r in args.ratio.split(',')]
        if len(ratios) != 3:
            print("错误: ratio 必须是三个值，如 0.8,0.1,0.1")
            return
        manager.merge_datasets(
            args.paths,
            args.output,
            deduplicate=not args.no_dedup,
            shuffle=not args.no_shuffle
        )

    elif args.command == 'split':
        ratios = tuple(float(r) for r in args.ratio.split(','))
        if len(ratios) != 3:
            print("错误: ratio 必须是三个值，如 0.8,0.1,0.1")
            return
        manager.split_dataset(
            args.path,
            ratios=ratios,
            output_dir=args.output_dir,
            stratify=not args.no_stratify
        )

    elif args.command == 'convert':
        output_path = args.output
        if not output_path:
            base = os.path.splitext(args.path)[0]
            output_path = f"{base}.{args.to}"

        manager.convert_format(args.path, output_path, args.to)

    elif args.command == 'filter':
        filters = {}
        if args.difficulty:
            filters['difficulty'] = args.difficulty
        if args.type:
            filters['problem_type'] = args.type
        if args.source:
            filters['source'] = args.source
        if args.verified is not None:
            filters['verified'] = args.verified

        manager.filter_dataset(args.path, args.output, filters)

    elif args.command == 'balance':
        dataset = manager.load_dataset(args.path)

        if args.check:
            balance_result = manager.check_balance(dataset)
            manager.print_balance_report(balance_result)
        else:
            if not args.output:
                print("错误: 平衡操作需要指定输出路径 -o")
                return
            manager.balance_dataset(args.path, args.output, args.strategy)

    elif args.command == 'summary':
        dataset = manager.load_dataset(args.path)
        manager.export_summary(dataset, args.output, args.format)


if __name__ == '__main__':
    main()