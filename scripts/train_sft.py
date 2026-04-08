#!/usr/bin/env python
"""SFT训练脚本

MethodThinker方法论注入训练CLI工具，支持多种训练模式。

用法示例:
    # 基础SFT训练
    python scripts/train_sft.py --config configs/training_config.yaml

    # 指定基座模型
    python scripts/train_sft.py --base-model Qwen/Qwen2.5-Math-1.5B

    # 方法论注入训练
    python scripts/train_sft.py --mode methodology-injection --kb data/methodology_kb/v1/

    # 多样性训练
    python scripts/train_sft.py --mode diversity --methods-per-problem 4

    # 使用LoRA节省显存
    python scripts/train_sft.py --use-lora --lora-r 16
"""

import sys
import os

# 设置HuggingFace镜像（国内加速）- 必须在import transformers之前
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import yaml
import argparse
import time
from typing import List, Dict, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import MethodThinkerTrainer, TrainingConfig
from src.data.dataset import MethodologyDataset, MethodologySample
from src.kb.knowledge_base import KnowledgeBase


def load_config(config_path: str) -> Dict:
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"不支持的配置格式: {config_path}")


def create_training_config(config_dict: Dict, args: argparse.Namespace) -> TrainingConfig:
    """创建训练配置

    Args:
        config_dict: 配置字典
        args: 命令行参数

    Returns:
        TrainingConfig实例
    """
    # 提取模型配置
    model_config = config_dict.get('model', {})
    training_params = config_dict.get('training', {})

    # 命令行参数覆盖配置文件
    config = TrainingConfig(
        base_model=args.base_model or model_config.get('base_model', 'Qwen/Qwen2.5-Math-1.5B'),
        output_dir=args.output_dir or config_dict.get('output', {}).get('output_dir', 'outputs/checkpoints'),

        # 训练参数 - 强制类型转换
        num_epochs=int(args.epochs or training_params.get('num_epochs', 3)),
        batch_size=int(args.batch_size or training_params.get('batch_size', 8)),
        learning_rate=float(args.learning_rate or training_params.get('learning_rate', 5e-5)),
        warmup_ratio=float(training_params.get('warmup_ratio', 0.1)),
        weight_decay=float(training_params.get('weight_decay', 0.01)),

        # 方法论训练参数
        method_selection_weight=float(config_dict.get('methodology_injection', {}).get('weights', {}).get('method_selection', 0.3)),
        solution_generation_weight=float(config_dict.get('methodology_injection', {}).get('weights', {}).get('solution_generation', 0.4)),
        reflection_weight=float(config_dict.get('methodology_injection', {}).get('weights', {}).get('reflection', 0.3)),

        # 序列长度
        max_length=int(args.max_length or training_params.get('max_length', 4096))
    )

    return config


def load_training_data(data_config: Dict, kb_path: Optional[str] = None,
                        train_path: Optional[str] = None,
                        val_path: Optional[str] = None,
                        test_path: Optional[str] = None,
                        auto_generate: bool = False) -> tuple:
    """加载训练数据

    Args:
        data_config: 数据配置
        kb_path: 知识库路径
        train_path: 训练数据路径（命令行参数）
        val_path: 验证数据路径（命令行参数）
        test_path: 测试数据路径（命令行参数）
        auto_generate: 是否从KB自动生成

    Returns:
        (train_data, val_data, test_data, kb)
    """
    # 优先使用命令行参数，否则使用配置文件
    train_data_path = train_path or data_config.get('train_path', 'data/train_data/train.json')
    val_data_path = val_path or data_config.get('val_path', 'data/train_data/val.json')
    test_data_path = test_path or data_config.get('test_path', 'data/train_data/test.json')

    # 加载知识库
    kb = None
    if kb_path:
        try:
            if kb_path.endswith('.yaml') or kb_path.endswith('.yml'):
                kb = KnowledgeBase.from_yaml(kb_path)
            else:
                kb = KnowledgeBase.load(kb_path)
            print(f"加载知识库: {kb_path} ({len(kb.methods)} 个方法)")
        except FileNotFoundError:
            print(f"警告: 知识库文件不存在: {kb_path}")

    # 加载训练数据
    train_data = None
    val_data = None
    test_data = None

    if os.path.exists(train_data_path):
        train_data = MethodologyDataset(train_data_path, kb)
        print(f"加载训练数据: {train_data_path} ({len(train_data)} 样本)")
    else:
        print(f"警告: 训练数据不存在: {train_data_path}")

        # 自动从KB生成训练数据
        if auto_generate and kb:
            print("从知识库自动生成训练样本...")
            train_data = generate_samples_from_kb(kb, num_samples=100)
            print(f"生成训练样本: {len(train_data)} 个")
        elif kb and not auto_generate:
            print("提示: 使用 --auto-generate 可从KB自动生成训练数据")

    if os.path.exists(val_data_path):
        val_data = MethodologyDataset(val_data_path, kb)
        print(f"加载验证数据: {val_data_path} ({len(val_data)} 样本)")

    if os.path.exists(test_data_path):
        test_data = MethodologyDataset(test_data_path, kb)
        print(f"加载测试数据: {test_data_path} ({len(test_data)} 样本)")

    return train_data, val_data, test_data, kb


def generate_samples_from_kb(kb: KnowledgeBase, num_samples: int = 100) -> MethodologyDataset:
    """从知识库生成训练样本

    Args:
        kb: 知识库
        num_samples: 生成样本数

    Returns:
        MethodologyDataset: 生成的数据集
    """
    import random
    from src.data.sample_generator import SampleGenerator

    dataset = MethodologyDataset()
    generator = SampleGenerator(kb)

    # 为每个方法生成示例
    methods = list(kb.methods.values())
    samples_per_method = max(1, num_samples // len(methods))

    for method in methods:
        # 生成模拟问题
        for i in range(samples_per_method):
            sample = MethodologySample(
                problem_id=f"{method.method_id}_sample_{i}",
                problem=f"使用{method.name}解决的问题示例 {i+1}",
                problem_type=method.category,
                difficulty=method.difficulty,
                candidate_methods=[{'method_id': method.method_id, 'name': method.name}],
                selected_method=method.method_id,
                selection_reasoning=f"该问题涉及{method.name}的适用场景",
                solution_steps=method.template.get('steps', ['分析问题', '应用方法', '得出结论']),
                solution_annotations=[],
                reflection="",
                source="auto_generated",
                verified=False
            )
            dataset.samples.append(sample)

    return dataset


def setup_lora(config: TrainingConfig, lora_r: int = 16, lora_alpha: int = 32):
    """设置LoRA配置

    Args:
        config: 训练配置
        lora_r: LoRA秩
        lora_alpha: LoRA alpha
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )

        print(f"使用LoRA: r={lora_r}, alpha={lora_alpha}")
        return lora_config

    except ImportError:
        print("警告: PEFT库未安装，跳过LoRA设置")
        return None


def run_methodology_injection_training(
    trainer: MethodThinkerTrainer,
    train_data: MethodologyDataset,
    val_data: Optional[MethodologyDataset],
    config: TrainingConfig,
    args: argparse.Namespace
) -> Dict:
    """运行方法论注入训练

    Args:
        trainer: 训练器
        train_data: 训练数据
        val_data: 验证数据
        config: 训练配置
        args: 命令行参数

    Returns:
        训练结果
    """
    print("\n" + "="*50)
    print("方法论注入训练")
    print("="*50)

    # 转换数据格式
    train_samples = [
        {
            'problem_id': s.problem_id,
            'problem': s.problem,
            'problem_type': s.problem_type,
            'difficulty': s.difficulty,
            'candidate_methods': s.candidate_methods,
            'selected_method': s.selected_method,
            'solution_steps': s.solution_steps,
            'reflection': s.reflection
        }
        for s in train_data.samples
    ]

    val_samples = None
    if val_data:
        val_samples = [
            {
                'problem_id': s.problem_id,
                'problem': s.problem,
                'selected_method': s.selected_method,
                'solution_steps': s.solution_steps
            }
            for s in val_data.samples
        ]

    # 运行训练
    results = trainer.train_methodology_injection(train_samples, val_samples)

    print(f"\n训练结果:")
    print(f"  状态: {results.get('status', 'unknown')}")
    if 'final_loss' in results:
        print(f"  最终损失: {results['final_loss']:.4f}")

    return results


def run_diversity_training(
    trainer: MethodThinkerTrainer,
    train_data: MethodologyDataset,
    config: TrainingConfig,
    args: argparse.Namespace
) -> Dict:
    """运行多样性训练

    Args:
        trainer: 训练器
        train_data: 训练数据
        config: 训练配置
        args: 命令行参数

    Returns:
        训练结果
    """
    print("\n" + "="*50)
    print("多样性训练")
    print("="*50)

    methods_per_problem = args.methods_per_problem or 4

    # 转换数据格式
    train_samples = [
        {
            'problem_id': s.problem_id,
            'problem': s.problem,
            'problem_type': s.problem_type,
            'candidate_methods': s.candidate_methods,
            'solution_steps': s.solution_steps
        }
        for s in train_data.samples
    ]

    # 运行训练
    results = trainer.train_diversity(train_samples, methods_per_problem)

    print(f"\n训练结果:")
    print(f"  状态: {results.get('status', 'unknown')}")
    print(f"  每题方法数: {methods_per_problem}")

    return results


def run_reflection_training(
    trainer: MethodThinkerTrainer,
    train_data: MethodologyDataset,
    config: TrainingConfig,
    args: argparse.Namespace
) -> Dict:
    """运行反思强化训练

    Args:
        trainer: 训练器
        train_data: 训练数据
        config: 训练配置
        args: 命令行参数

    Returns:
        训练结果
    """
    print("\n" + "="*50)
    print("反思强化训练")
    print("="*50)

    # 转换数据格式
    train_samples = [
        {
            'problem_id': s.problem_id,
            'problem': s.problem,
            'solution_steps': s.solution_steps,
            'reflection': s.reflection
        }
        for s in train_data.samples
    ]

    # 运行训练
    results = trainer.train_reflection(train_samples)

    print(f"\n训练结果:")
    print(f"  状态: {results.get('status', 'unknown')}")

    return results


def run_full_training(
    trainer: MethodThinkerTrainer,
    train_data: MethodologyDataset,
    val_data: Optional[MethodologyDataset],
    test_data: Optional[MethodologyDataset],
    config: TrainingConfig,
    args: argparse.Namespace
) -> Dict:
    """运行完整训练流程

    Args:
        trainer: 训练器
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
        config: 训练配置
        args: 命令行参数

    Returns:
        训练结果
    """
    results = {}

    # 1. 方法论注入训练
    print("\n阶段 1: 方法论注入训练")
    injection_results = run_methodology_injection_training(
        trainer, train_data, val_data, config, args
    )
    results['methodology_injection'] = injection_results

    # 保存检查点
    checkpoint_path = os.path.join(config.output_dir, 'step1_injection')
    trainer.save_checkpoint(checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")

    # 2. 多样性训练
    print("\n阶段 2: 多样性训练")
    diversity_results = run_diversity_training(
        trainer, train_data, config, args
    )
    results['diversity'] = diversity_results

    checkpoint_path = os.path.join(config.output_dir, 'step2_diversity')
    trainer.save_checkpoint(checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")

    # 3. 反思强化训练
    print("\n阶段 3: 反思强化训练")
    reflection_results = run_reflection_training(
        trainer, train_data, config, args
    )
    results['reflection'] = reflection_results

    checkpoint_path = os.path.join(config.output_dir, 'step3_reflection')
    trainer.save_checkpoint(checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")

    # 4. 最终评估
    if test_data:
        print("\n阶段 4: 最终评估")
        test_samples = [
            {'problem': s.problem, 'problem_type': s.problem_type}
            for s in test_data.samples
        ]
        eval_results = trainer.evaluate(test_samples)
        results['evaluation'] = eval_results
        print(f"评估结果:")
        print(f"  Pass@1: {eval_results.get('pass@1', 0):.2%}")

    # 保存最终模型
    final_path = os.path.join(config.output_dir, 'final')
    trainer.save_checkpoint(final_path)
    print(f"\n最终模型已保存: {final_path}")

    return results


def save_training_report(results: Dict, output_dir: str, config: TrainingConfig):
    """保存训练报告

    Args:
        results: 训练结果
        output_dir: 输出目录
        config: 训练配置
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'base_model': config.base_model,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'max_length': config.max_length
        },
        'results': results
    }

    report_path = os.path.join(output_dir, 'training_report.json')
    os.makedirs(output_dir, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n训练报告已保存: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MethodThinker SFT训练 - 方法论注入训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
训练模式:
  methodology-injection: 方法论注入训练（学习选择和应用方法）
  diversity: 多样性训练（学习使用多种方法解决同一问题）
  reflection: 反思强化训练（强化自我反思能力）
  full: 完整训练流程（依次执行所有模式）

示例:
  # 基础SFT训练
  %(prog)s --config configs/training_config.yaml

  # 方法论注入训练
  %(prog)s --mode methodology-injection --kb data/methodology_kb/v1/

  # 多样性训练
  %(prog)s --mode diversity --methods-per-problem 4

  # 使用LoRA节省显存
  %(prog)s --use-lora --lora-r 16

  # 完整训练流程
  %(prog)s --mode full --kb data/methodology_kb/v1/
"""
    )

    # 基础参数
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='配置文件路径 (default: configs/training_config.yaml)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['methodology-injection', 'diversity', 'reflection', 'full'],
        default='methodology-injection',
        help='训练模式 (default: methodology-injection)'
    )

    # 模型参数
    parser.add_argument(
        '--base-model',
        type=str,
        help='基座模型路径'
    )
    parser.add_argument(
        '--kb',
        type=str,
        help='方法论知识库路径'
    )

    # 输出参数
    parser.add_argument(
        '--output-dir',
        type=str,
        help='输出目录'
    )

    # 训练数据参数
    parser.add_argument(
        '--train-data',
        type=str,
        help='训练数据路径 (支持 .json, .jsonl, .yaml)'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        help='验证数据路径'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        help='测试数据路径'
    )
    parser.add_argument(
        '--auto-generate',
        action='store_true',
        help='当没有训练数据时，从KB自动生成样本'
    )

    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='批大小'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='学习率'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        help='最大序列长度'
    )
    parser.add_argument(
        '--methods-per-problem',
        type=int,
        default=4,
        help='多样性训练时每题使用的方法数 (default: 4)'
    )

    # LoRA参数
    parser.add_argument(
        '--use-lora',
        action='store_true',
        help='使用LoRA节省显存'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=16,
        help='LoRA秩 (default: 16)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha (default: 32)'
    )

    # 其他参数
    parser.add_argument(
        '--resume',
        type=str,
        help='从检查点恢复训练'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只显示配置，不实际训练'
    )

    args = parser.parse_args()

    print("MethodThinker SFT训练")
    print("="*50)

    # 加载配置
    try:
        config_dict = load_config(args.config)
        print(f"加载配置: {args.config}")
    except FileNotFoundError:
        print(f"警告: 配置文件不存在: {args.config}")
        print("使用默认配置")
        config_dict = {}

    # 创建训练配置
    config = create_training_config(config_dict, args)

    print(f"\n训练配置:")
    print(f"  基座模型: {config.base_model}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  训练轮数: {config.num_epochs}")
    print(f"  批大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  最大长度: {config.max_length}")
    print(f"  训练模式: {args.mode}")

    if args.dry_run:
        print("\n[Dry Run] 不执行实际训练")
        return

    # 加载训练数据
    print(f"\n加载训练数据...")
    data_config = config_dict.get('data', {})
    train_data, val_data, test_data, kb = load_training_data(
        data_config,
        kb_path=args.kb,
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        auto_generate=args.auto_generate
    )

    if train_data is None:
        print("\n错误: 缺少训练数据")
        print("解决方案:")
        print("  1. 使用 --train-data 指定训练数据路径")
        print("  2. 使用 --kb 配合 --auto-generate 从KB生成数据")
        print("  3. 先运行 generate_training_data.py 生成数据")
        print("\n示例:")
        print("  python scripts/generate_training_data.py \\")
        print("    --kb data/methodology_kb/v0/math_methods.yaml \\")
        print("    --problems data/test_sets/aime_samples.yaml \\")
        print("    --output data/train_data/train.json")
        return

    # 创建训练器
    print(f"\n初始化训练器...")
    trainer = MethodThinkerTrainer(config)

    # 设置LoRA
    if args.use_lora:
        lora_config = setup_lora(config, args.lora_r, args.lora_alpha)
        if lora_config:
            # LoRA配置需要在setup后应用
            pass

    # 恢复训练
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 设置环境
    print(f"\n设置训练环境...")
    trainer.setup()

    # 运行训练
    start_time = time.time()

    if args.mode == 'methodology-injection':
        results = run_methodology_injection_training(
            trainer, train_data, val_data, config, args
        )
    elif args.mode == 'diversity':
        results = run_diversity_training(trainer, train_data, config, args)
    elif args.mode == 'reflection':
        results = run_reflection_training(trainer, train_data, config, args)
    elif args.mode == 'full':
        results = run_full_training(
            trainer, train_data, val_data, test_data, config, args
        )

    elapsed_time = time.time() - start_time

    # 保存检查点
    checkpoint_path = os.path.join(config.output_dir, 'checkpoint_final')
    trainer.save_checkpoint(checkpoint_path)
    print(f"\n检查点已保存: {checkpoint_path}")

    # 保存训练报告
    save_training_report(results, config.output_dir, config)

    print(f"\n训练完成!")
    print(f"总耗时: {elapsed_time:.1f} 秒")


if __name__ == '__main__':
    main()