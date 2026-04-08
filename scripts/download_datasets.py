#!/usr/bin/env python
"""数据集下载脚本

从HuggingFace下载公开数学竞赛数据集并转换为项目格式。

用法:
    python scripts/download_datasets.py --dataset aime --output data/train_data/
    python scripts/download_datasets.py --dataset math --output data/train_data/
    python scripts/download_datasets.py --dataset all --output data/train_data/
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

# HuggingFace数据集配置
DATASETS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "config": "main",
        "name": "GSM8K",
        "description": "小学数学应用题",
        "expected_samples": 8500,
    },
    "math": {
        "path": "lighteval/MATH",
        "name": "MATH",
        "description": "数学竞赛题目集合",
        "expected_samples": 12500,
    },
    "aime": {
        "path": "yukuai/AIME",
        "name": "AIME",
        "description": "美国数学邀请赛历年真题",
        "expected_samples": 500,
    },
    "amc": {
        "path": "digitous/AMC_AIME_Accumulated",
        "name": "AMC/AIME Accumulated",
        "description": "AMC/AIME累积题目",
        "expected_samples": 3000,
    },
    "omni-math": {
        "path": "KbsdJames/Omni-MATH",
        "name": "Omni-MATH",
        "description": "综合数学推理数据集",
        "expected_samples": 5000,
    },
}


def check_huggingface_connection() -> bool:
    """检查HuggingFace连接"""
    try:
        from datasets import load_dataset
        return True
    except ImportError:
        print("错误: 未安装datasets库")
        print("请运行: pip install datasets")
        return False


def download_dataset(dataset_name: str, cache_dir: str = None) -> Any:
    """下载数据集

    Args:
        dataset_name: 数据集名称 (aime/math/amc/gsm8k)
        cache_dir: 缓存目录

    Returns:
        数据集对象
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"未知数据集: {dataset_name}")

    from datasets import load_dataset

    config = DATASETS[dataset_name]
    print(f"下载数据集: {config['name']}")
    print(f"  路径: {config['path']}")
    print(f"  预期样本数: {config['expected_samples']}")

    try:
        # 某些数据集需要config参数
        dataset_config = config.get('config', None)
        if dataset_config:
            dataset = load_dataset(config['path'], dataset_config, cache_dir=cache_dir)
        else:
            dataset = load_dataset(config['path'], cache_dir=cache_dir)
        print(f"  下载成功!")
        return dataset
    except Exception as e:
        print(f"  下载失败: {e}")
        return None


def convert_aime_to_project_format(dataset: Any) -> List[Dict]:
    """转换AIME数据集为项目格式

    Args:
        dataset: HuggingFace数据集对象

    Returns:
        转换后的样本列表
    """
    samples = []

    for split in dataset:
        for item in dataset[split]:
            # 提取字段
            problem = item.get('problem', item.get('question', ''))
            solution = item.get('solution', item.get('answer', ''))
            answer = item.get('answer', '')

            # 构建项目格式
            sample = {
                "problem": problem,
                "method_selection": "",  # 需要后续填充
                "solution_steps": [solution] if solution else [],
                "final_answer": str(answer),
                "method_id": "UNKNOWN",
                "method_name": "待标注",
                "problem_type": "UNKNOWN",
                "difficulty": 3,
                "annotations": [],
                "sample_id": f"aime_{len(samples)}",
                "source": "aime",
                "verified": False
            }
            samples.append(sample)

    print(f"  转换完成: {len(samples)} 样本")
    return samples


def convert_math_to_project_format(dataset: Any) -> List[Dict]:
    """转换MATH数据集为项目格式

    Args:
        dataset: HuggingFace数据集对象

    Returns:
        转换后的样本列表
    """
    samples = []

    for split in dataset:
        for item in dataset[split]:
            problem = item.get('problem', '')
            solution = item.get('solution', '')
            answer = item.get('answer', '')
            level = item.get('level', 3)
            problem_type = item.get('type', 'unknown')

            sample = {
                "problem": problem,
                "method_selection": "",
                "solution_steps": [solution] if solution else [],
                "final_answer": str(answer),
                "method_id": "UNKNOWN",
                "method_name": "待标注",
                "problem_type": problem_type.upper(),
                "difficulty": int(level) if isinstance(level, (int, str)) else 3,
                "annotations": [],
                "sample_id": f"math_{len(samples)}",
                "source": "math",
                "verified": False
            }
            samples.append(sample)

    print(f"  转换完成: {len(samples)} 样本")
    return samples


def convert_gsm8k_to_project_format(dataset: Any) -> List[Dict]:
    """转换GSM8K数据集为项目格式

    Args:
        dataset: HuggingFace数据集对象

    Returns:
        转换后的样本列表
    """
    samples = []

    for split in dataset:
        for item in dataset[split]:
            problem = item.get('question', '')
            solution = item.get('answer', '')

            sample = {
                "problem": problem,
                "method_selection": "",
                "solution_steps": [solution] if solution else [],
                "final_answer": "",
                "method_id": "UNKNOWN",
                "method_name": "待标注",
                "problem_type": "ARITHMETIC",
                "difficulty": 2,
                "annotations": [],
                "sample_id": f"gsm8k_{len(samples)}",
                "source": "gsm8k",
                "verified": False
            }
            samples.append(sample)

    print(f"  转换完成: {len(samples)} 样本")
    return samples


def convert_omni_math_to_project_format(dataset: Any) -> List[Dict]:
    """转换Omni-MATH数据集为项目格式

    Args:
        dataset: HuggingFace数据集对象

    Returns:
        转换后的样本列表
    """
    samples = []

    for split in dataset:
        for item in dataset[split]:
            problem = item.get('problem', item.get('question', ''))
            solution = item.get('solution', '')

            sample = {
                "problem": problem,
                "method_selection": "",
                "solution_steps": [solution] if solution else [],
                "final_answer": "",
                "method_id": "UNKNOWN",
                "method_name": "待标注",
                "problem_type": "UNKNOWN",
                "difficulty": 3,
                "annotations": [],
                "sample_id": f"omni_math_{len(samples)}",
                "source": "omni_math",
                "verified": False
            }
            samples.append(sample)

    print(f"  转换完成: {len(samples)} 样本")
    return samples


def save_samples(samples: List[Dict], output_path: str, dataset_name: str):
    """保存样本到文件

    Args:
        samples: 样本列表
        output_path: 输出目录
        dataset_name: 数据集名称
    """
    os.makedirs(output_path, exist_ok=True)

    filename = f"{dataset_name}_downloaded_{datetime.now().strftime('%Y%m%d')}.json"
    filepath = os.path.join(output_path, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"  保存到: {filepath}")
    print(f"  总样本数: {len(samples)}")


def main():
    parser = argparse.ArgumentParser(description='下载公开数学数据集')

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['aime', 'math', 'amc', 'gsm8k', 'omni-math', 'all'],
        default='aime',
        help='要下载的数据集 (default: aime)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/train_data/',
        help='输出目录 (default: data/train_data/)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='HuggingFace缓存目录'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大样本数限制'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只显示信息，不实际下载'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MethodThinker 数据集下载工具")
    print("=" * 60)

    if not check_huggingface_connection():
        return

    if args.dry_run:
        print("\n[Dry Run] 可用数据集:")
        for name, config in DATASETS.items():
            print(f"  - {name}: {config['name']} (~{config['expected_samples']}样本)")
        return

    datasets_to_download = ['gsm8k', 'math', 'aime', 'amc', 'omni-math'] if args.dataset == 'all' else [args.dataset]

    for dataset_name in datasets_to_download:
        print(f"\n处理数据集: {dataset_name}")
        print("-" * 40)

        dataset = download_dataset(dataset_name, args.cache_dir)
        if dataset is None:
            continue

        # 转换格式
        if dataset_name == 'aime':
            samples = convert_aime_to_project_format(dataset)
        elif dataset_name == 'math':
            samples = convert_math_to_project_format(dataset)
        elif dataset_name == 'gsm8k':
            samples = convert_gsm8k_to_project_format(dataset)
        elif dataset_name == 'omni-math':
            samples = convert_omni_math_to_project_format(dataset)
        else:
            print(f"  警告: {dataset_name} 的转换器未实现")
            continue

        # 限制样本数
        if args.max_samples and len(samples) > args.max_samples:
            samples = samples[:args.max_samples]
            print(f"  限制样本数: {args.max_samples}")

        # 保存
        save_samples(samples, args.output, dataset_name)

    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()