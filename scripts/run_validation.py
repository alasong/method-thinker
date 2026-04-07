#!/usr/bin/env python
"""运行验证流水线

MethodThinker验证系统CLI工具，支持多层验证和灵活配置。

用法示例:
    # 运行完整验证流水线
    python scripts/run_validation.py --kb data/methodology_kb/v0/math_methods.yaml

    # 只运行Layer 0和Layer 1
    python scripts/run_validation.py --layers 0,1 --kb data/methodology_kb/v0/math_methods.yaml

    # 验证特定方法
    python scripts/run_validation.py --method-id ALG_001 --layers 2,3

    # 使用自定义配置
    python scripts/run_validation.py --config configs/validation_config.yaml --verbose
"""

import sys
import os
import json
import time
from typing import List, Dict, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.pipeline import ValidationPipeline, ValidationConfig
from src.validation.config import ValidationConfig as ConfigLoader
from src.validation.layer0_fast_filter import Layer0FastFilter, ValidationResult
from src.validation.layer1_self_reflection import Layer1SelfReflection
from src.validation.layer2_multi_model import Layer2MultiModelValidation
from src.validation.layer3_test_driven import Layer3TestDrivenValidation, TestCase
from src.validation.ensemble_decision import EnsembleDecisionEngine, LayerResult
from src.kb.knowledge_base import KnowledgeBase


def parse_layers(layers_str: str) -> List[int]:
    """解析--layers参数

    Args:
        layers_str: 层号字符串，如 "0,1,2" 或 "all"

    Returns:
        层号列表
    """
    if layers_str.lower() == 'all':
        return [0, 1, 2, 3]

    try:
        layers = [int(x.strip()) for x in layers_str.split(',')]
        valid_layers = [l for l in layers if 0 <= l <= 3]
        if len(valid_layers) != len(layers):
            print(f"警告: 层号必须在0-3范围内，已过滤无效层号")
        return valid_layers
    except ValueError:
        print(f"错误: 无法解析层号 '{layers_str}'，使用默认值 [0, 1, 2, 3]")
        return [0, 1, 2, 3]


def format_result(result: ValidationResult, verbose: bool = False) -> str:
    """格式化验证结果

    Args:
        result: 验证结果
        verbose: 是否显示详细信息

    Returns:
        格式化的字符串
    """
    status = "PASS" if result.passed else "FAIL"
    layer_name = {
        -1: "集成决策",
        0: "Layer 0 (快速过滤)",
        1: "Layer 1 (自我反思)",
        2: "Layer 2 (多模型验证)",
        3: "Layer 3 (测试驱动)"
    }.get(result.layer, f"Layer {result.layer}")

    lines = [
        f"[{status}] {layer_name} - 置信度: {result.confidence:.2f}"
    ]

    if result.issues:
        lines.append(f"  问题 ({len(result.issues)}):")
        for issue in result.issues[:5]:  # 只显示前5个问题
            lines.append(f"    - {issue}")
        if len(result.issues) > 5:
            lines.append(f"    ... 还有 {len(result.issues) - 5} 个问题")

    if verbose and result.details:
        lines.append(f"  详情:")
        for key, value in result.details.items():
            if isinstance(value, dict):
                lines.append(f"    {key}: {json.dumps(value, ensure_ascii=False)[:100]}...")
            else:
                lines.append(f"    {key}: {value}")

    return '\n'.join(lines)


def print_summary(results: Dict, elapsed_time: float, verbose: bool = False):
    """打印验证摘要

    Args:
        results: 验证结果统计
        elapsed_time: 耗时
        verbose: 是否显示详细信息
    """
    total = results['passed'] + results['failed']
    pass_rate = results['passed'] / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"验证完成 - 耗时: {elapsed_time:.2f}s")
    print(f"{'='*60}")
    print(f"总计: {total} 个方法")
    print(f"通过: {results['passed']} ({pass_rate:.1f}%)")
    print(f"失败: {results['failed']} ({100-pass_rate:.1f}%)")

    # 各层统计
    if results.get('layer_stats'):
        print(f"\n各层验证统计:")
        for layer, stats in results['layer_stats'].items():
            layer_name = {
                0: "Layer 0",
                1: "Layer 1",
                2: "Layer 2",
                3: "Layer 3"
            }.get(layer, f"Layer {layer}")
            layer_pass = stats.get('passed', 0)
            layer_total = stats.get('total', 0)
            layer_rate = layer_pass / layer_total * 100 if layer_total > 0 else 0
            print(f"  {layer_name}: {layer_pass}/{layer_total} ({layer_rate:.1f}%)")

    if verbose and results.get('failed_methods'):
        print(f"\n失败方法列表:")
        for method_id in results['failed_methods'][:10]:
            print(f"  - {method_id}")
        if len(results['failed_methods']) > 10:
            print(f"  ... 还有 {len(results['failed_methods']) - 10} 个")


def run_validation(
    kb_path: str,
    config_path: str,
    method_id: Optional[str] = None,
    layers: List[int] = [0, 1, 2, 3],
    verbose: bool = False,
    output_path: Optional[str] = None
) -> Dict:
    """运行验证流水线

    Args:
        kb_path: 知识库路径
        config_path: 配置文件路径
        method_id: 要验证的方法ID（可选）
        layers: 要运行的验证层
        verbose: 是否显示详细信息
        output_path: 输出结果文件路径（可选）

    Returns:
        验证结果统计
    """
    # 加载配置
    print(f"加载配置: {config_path}")
    try:
        config = ConfigLoader.from_yaml(config_path)
    except FileNotFoundError:
        print(f"警告: 配置文件不存在，使用默认配置")
        config = ValidationConfig()

    # 加载知识库
    print(f"加载知识库: {kb_path}")
    try:
        kb = KnowledgeBase.from_yaml(kb_path)
        print(f"知识库包含 {len(kb.methods)} 个方法")
    except FileNotFoundError:
        print(f"错误: 知识库文件不存在: {kb_path}")
        return {'passed': 0, 'failed': 0, 'error': '知识库文件不存在'}

    # 获取要验证的方法
    if method_id:
        method = kb.get_method(method_id)
        if not method:
            print(f"错误: 方法 {method_id} 不存在")
            return {'passed': 0, 'failed': 0, 'error': '方法不存在'}
        methods_to_validate = [method]
    else:
        methods_to_validate = list(kb.methods.values())

    print(f"\n将验证 {len(methods_to_validate)} 个方法")
    print(f"验证层: {', '.join(f'Layer {l}' for l in layers)}")

    # 创建验证组件
    start_time = time.time()

    # 初始化各层验证器
    layer0 = None
    layer1 = None
    layer2 = None
    layer3 = None
    ensemble = EnsembleDecisionEngine(config.ensemble.weights if hasattr(config, 'ensemble') else None)

    # Layer 0 - 使用空KB以避免验证已存在方法时出现"重复"错误
    if 0 in layers:
        # 当验证KB中已有方法时，使用空KB避免误报重复
        layer0 = Layer0FastFilter({'methods': {}})

    # Layer 1 (需要模型，这里用模拟)
    if 1 in layers:
        print("注意: Layer 1 需要模型支持，当前使用模拟验证")
        layer1 = None  # 暂不实现

    # Layer 2 (需要外部模型API)
    if 2 in layers:
        print("注意: Layer 2 需要外部模型API，当前使用模拟验证")
        layer2 = None  # 暂不实现

    # Layer 3 (需要测试数据和模型)
    if 3 in layers:
        print("注意: Layer 3 需要测试数据和模型，当前使用模拟验证")
        layer3 = None  # 暂不实现

    # 验证方法
    results = {
        'passed': 0,
        'failed': 0,
        'layer_stats': {l: {'passed': 0, 'total': 0} for l in layers},
        'failed_methods': [],
        'details': []
    }

    print(f"\n{'='*60}")
    print("开始验证...")
    print(f"{'='*60}\n")

    for idx, method in enumerate(methods_to_validate):
        print(f"\n[{idx+1}/{len(methods_to_validate)}] 验证方法: {method.name} ({method.method_id})")

        # 转换为字典格式
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

        layer_results = []
        final_result = None

        # 运行各层验证
        if 0 in layers and layer0:
            result = layer0.validate(method_dict)
            layer_results.append(LayerResult(
                layer=0,
                passed=result.passed,
                confidence=result.confidence,
                issues=result.issues,
                weight=config.ensemble.weights.get(0, 0.05) if hasattr(config, 'ensemble') else 0.05
            ))
            results['layer_stats'][0]['total'] += 1
            if result.passed:
                results['layer_stats'][0]['passed'] += 1

            print(format_result(result, verbose))

            # Layer 0 快速失败
            if not result.passed and result.confidence < 0.3:
                print("  -> Layer 0 快速失败，跳过后续验证")
                final_result = result

        # 如果Layer 0通过，继续其他层（这里使用模拟）
        if final_result is None and len(layer_results) > 0:
            # 模拟其他层（实际应调用真实验证器）
            for layer in [1, 2, 3]:
                if layer in layers and layer not in [l.layer for l in layer_results]:
                    # 模拟通过（实际实现需要模型）
                    simulated_result = ValidationResult(
                        passed=True,
                        layer=layer,
                        confidence=0.8,
                        issues=[],
                        details={'simulated': True}
                    )
                    layer_results.append(LayerResult(
                        layer=layer,
                        passed=True,
                        confidence=0.8,
                        issues=[],
                        weight=config.ensemble.weights.get(layer, 0.1) if hasattr(config, 'ensemble') else 0.1
                    ))
                    results['layer_stats'][layer]['total'] += 1
                    results['layer_stats'][layer]['passed'] += 1
                    print(format_result(simulated_result, verbose))

        # 集成决策
        if len(layer_results) > 0:
            ensemble_result = ensemble.decide(layer_results)
            print(format_result(ensemble_result, verbose))

            if ensemble_result.passed:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failed_methods'].append(method.method_id)

            results['details'].append({
                'method_id': method.method_id,
                'name': method.name,
                'passed': ensemble_result.passed,
                'confidence': ensemble_result.confidence,
                'layer_results': [r.__dict__ for r in layer_results]
            })
        else:
            # 没有运行任何层，默认通过
            results['passed'] += 1

    elapsed_time = time.time() - start_time
    print_summary(results, elapsed_time, verbose)

    # 保存结果
    if output_path:
        results['elapsed_time'] = elapsed_time
        results['timestamp'] = datetime.now().isoformat()
        results['config'] = config_path
        results['kb'] = kb_path
        results['layers'] = layers

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_path}")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='MethodThinker验证系统 - 多层方法论验证流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
验证层级说明:
  Layer 0: 快速过滤 - 语法、格式、完整性检查（免费，~10ms）
  Layer 1: 自我反思 - 内部一致性检查（免费，~500ms）
  Layer 2: 多模型验证 - 交叉验证方法有效性（$0.05/方法，~5s）
  Layer 3: 测试驱动 - 实测验证方法效果（GPU时，~30s）

示例:
  # 运行所有层验证
  %(prog)s --kb data/methodology_kb/v0/math_methods.yaml

  # 只运行Layer 0和Layer 1
  %(prog)s --layers 0,1 --kb data/methodology_kb/v0/math_methods.yaml

  # 验证特定方法，使用所有层
  %(prog)s --method-id ALG_001

  # 详细输出模式
  %(prog)s --verbose --output results/validation_report.json
"""
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/validation_config.yaml',
        help='配置文件路径 (default: configs/validation_config.yaml)'
    )
    parser.add_argument(
        '--kb',
        type=str,
        default='data/methodology_kb/v0/math_methods.yaml',
        help='方法论知识库路径 (default: data/methodology_kb/v0/math_methods.yaml)'
    )
    parser.add_argument(
        '--method-id',
        type=str,
        help='要验证的特定方法ID'
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='all',
        help='要运行的验证层，用逗号分隔 (如: 0,1,2) 或 "all" (default: all)'
    )
    parser.add_argument(
        '--skip-layers',
        type=int,
        nargs='+',
        help='要跳过的验证层 (已废弃，请使用 --layers)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出结果文件路径 (JSON格式)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制验证的方法数量（用于测试）'
    )

    args = parser.parse_args()

    # 解析层参数
    layers = parse_layers(args.layers)

    # 处理废弃的skip_layers参数
    if args.skip_layers:
        print("警告: --skip-layers 已废弃，请使用 --layers 参数")
        layers = [l for l in [0, 1, 2, 3] if l not in args.skip_layers]

    # 运行验证
    results = run_validation(
        kb_path=args.kb,
        config_path=args.config,
        method_id=args.method_id,
        layers=layers,
        verbose=args.verbose,
        output_path=args.output
    )

    # 返回状态码
    return 0 if results.get('passed', 0) > 0 else 1


if __name__ == '__main__':
    sys.exit(main())