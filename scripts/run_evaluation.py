#!/usr/bin/env python
"""评估脚本

MethodThinker模型评估CLI工具，支持AIME/HMMT等数学竞赛数据集。

用法示例:
    # 基础评估
    python scripts/run_evaluation.py --model outputs/checkpoints/final

    # 使用AIME数据集
    python scripts/run_evaluation.py --dataset aime --year 2024

    # 使用HMMT数据集
    python scripts/run_evaluation.py --dataset hmmt --round february

    # Pass@K评估
    python scripts/run_evaluation.py --pass-at-k 1,2,5,10

    # 详细报告输出
    python scripts/run_evaluation.py --output-report reports/eval_report.json
"""

import sys
import os
import json
import argparse
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict
import math

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import MethodThinkerTrainer, TrainingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============ 数据集加载器 ============

class DatasetLoader:
    """数学竞赛数据集加载器"""

    DATASET_INFO = {
        'aime': {
            'name': 'AIME (American Invitational Mathematics Examination)',
            'years': ['2024', '2023', '2022', '2021', '2020'],
            'difficulty_range': (3, 5),
            'time_limit': 180,  # 分钟
            'problems_per_test': 15,
        },
        'hmmt': {
            'name': 'HMMT (Harvard-MIT Mathematics Tournament)',
            'rounds': ['february', 'november'],
            'difficulty_range': (2, 5),
            'time_limit': 120,
        },
        'amo': {
            'name': 'AMO (American Mathematics Olympiad)',
            'difficulty_range': (4, 5),
        },
        'custom': {
            'name': '自定义数据集',
        }
    }

    @staticmethod
    def load_aime(year: str = '2024', part: str = 'both') -> List[Dict]:
        """加载AIME数据集

        Args:
            year: 年份
            part: 部分 ('I', 'II', 'both')

        Returns:
            问题列表
        """
        # 模拟数据 - 实际应从文件加载
        aime_dir = Path('data/evaluation/aime')

        problems = []

        # 如果目录存在，加载实际数据
        if aime_dir.exists():
            for file_path in aime_dir.glob(f'*{year}*.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        problems.extend(data)
                    elif isinstance(data, dict) and 'problems' in data:
                        problems.extend(data['problems'])
        else:
            # 使用模拟数据
            logger.warning(f"AIME数据目录不存在: {aime_dir}")
            logger.info("使用模拟数据进行评估演示")
            problems = DatasetLoader._generate_mock_aime(year, part)

        return problems

    @staticmethod
    def load_hmmt(year: str = '2024', round_name: str = 'february') -> List[Dict]:
        """加载HMMT数据集

        Args:
            year: 年份
            round_name: 赔轮 ('february', 'november')

        Returns:
            问题列表
        """
        hmmt_dir = Path('data/evaluation/hmmt')

        problems = []

        if hmmt_dir.exists():
            for file_path in hmmt_dir.glob(f'*{year}*{round_name}*.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        problems.extend(data)
                    elif isinstance(data, dict) and 'problems' in data:
                        problems.extend(data['problems'])
        else:
            logger.warning(f"HMMT数据目录不存在: {hmmt_dir}")
            logger.info("使用模拟数据进行评估演示")
            problems = DatasetLoader._generate_mock_hmmt(year, round_name)

        return problems

    @staticmethod
    def load_custom(path: str) -> List[Dict]:
        """加载自定义数据集

        Args:
            path: 数据文件路径

        Returns:
            问题列表
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'problems' in data:
            return data['problems']
        else:
            raise ValueError(f"不支持的数据格式: {path}")

    @staticmethod
    def _generate_mock_aime(year: str, part: str) -> List[Dict]:
        """生成模拟AIME数据"""
        mock_problems = [
            {
                'problem_id': f'AIME_{year}_I_1',
                'problem': 'Find the number of positive integers less than 1000 that are divisible by 7.',
                'problem_type': '数论',
                'answer': '142',
                'difficulty': 3,
                'year': year,
                'source': 'AIME',
            },
            {
                'problem_id': f'AIME_{year}_I_2',
                'problem': 'Let $x$ and $y$ be positive integers such that $xy - 2x - 3y = 1$. Find the minimum value of $x + y$.',
                'problem_type': '代数',
                'answer': '8',
                'difficulty': 3,
                'year': year,
                'source': 'AIME',
            },
            {
                'problem_id': f'AIME_{year}_I_3',
                'problem': 'A triangle has side lengths 7, 24, and 25. Find its area.',
                'problem_type': '几何',
                'answer': '84',
                'difficulty': 3,
                'year': year,
                'source': 'AIME',
            },
            {
                'problem_id': f'AIME_{year}_I_4',
                'problem': 'Find the remainder when $2^{100}$ is divided by 7.',
                'problem_type': '数论',
                'answer': '2',
                'difficulty': 4,
                'year': year,
                'source': 'AIME',
            },
            {
                'problem_id': f'AIME_{year}_I_5',
                'problem': 'Find the sum of all positive integers $n$ such that $n^2 + 19n + 99$ is a perfect square.',
                'problem_type': '代数',
                'answer': '40',
                'difficulty': 4,
                'year': year,
                'source': 'AIME',
            },
        ]
        return mock_problems

    @staticmethod
    def _generate_mock_hmmt(year: str, round_name: str) -> List[Dict]:
        """生成模拟HMMT数据"""
        mock_problems = [
            {
                'problem_id': f'HMMT_{year}_{round_name}_1',
                'problem': 'Compute $\\sum_{k=1}^{100} k^2$.',
                'problem_type': '代数',
                'answer': '338350',
                'difficulty': 2,
                'year': year,
                'source': 'HMMT',
            },
            {
                'problem_id': f'HMMT_{year}_{round_name}_2',
                'problem': 'Find the number of ways to arrange the letters in "MATHEMATICS".',
                'problem_type': '组合',
                'answer': '453600',
                'difficulty': 3,
                'year': year,
                'source': 'HMMT',
            },
        ]
        return mock_problems


# ============ Pass@K计算器 ============

class PassAtKCalculator:
    """Pass@K计算器

    计算模型在多次尝试中的通过率。

    Pass@K公式: Pass@K = 1 - C(n-c, K) / C(n, K)
    其中 n = 总尝试次数，c = 正确次数，K = 取的样本数
    """

    @staticmethod
    def compute_pass_at_k(n: int, c: int, k: int) -> float:
        """计算Pass@K

        Args:
            n: 总尝试次数（每个问题的采样数）
            c: 正确次数
            k: Pass@K的K值

        Returns:
            Pass@K值
        """
        if n - c < k:
            return 1.0

        # 使用稳定的计算方式避免数值溢出
        # Pass@K = 1 - prod((n - c - i) / (n - i)) for i in 0..k-1

        result = 1.0
        for i in range(k):
            result *= (n - c - i) / (n - i)

        return 1.0 - result

    @staticmethod
    def compute_pass_at_k_batch(
        results: List[Dict],
        k_values: List[int],
        samples_per_problem: int = 1
    ) -> Dict[str, float]:
        """批量计算Pass@K

        Args:
            results: 每个问题的结果列表
            k_values: 要计算的K值列表
            samples_per_problem: 每个问题的采样数

        Returns:
            各K值对应的Pass@K
        """
        # 统计正确数
        correct_count = sum(1 for r in results if r.get('correct', False))
        total = len(results)

        pass_at_k_results = {}

        if samples_per_problem == 1:
            # 单次采样，Pass@1就是准确率
            pass_at_k_results['pass@1'] = correct_count / total if total > 0 else 0.0

            # 对于更高的K，使用概率估计
            for k in k_values:
                if k == 1:
                    continue
                # 假设多次采样时的概率分布
                # 使用Beta分布估计
                pass_at_k_results[f'pass@{k}'] = min(1.0, pass_at_k_results['pass@1'] * k)
        else:
            # 多次采样，使用精确公式
            for k in k_values:
                if k > samples_per_problem:
                    k = samples_per_problem
                pass_at_k_results[f'pass@{k}'] = PassAtKCalculator.compute_pass_at_k(
                    samples_per_problem * total,
                    correct_count,
                    k
                )

        return pass_at_k_results


# ============ 评估报告生成器 ============

class EvaluationReportGenerator:
    """评估报告生成器"""

    @staticmethod
    def generate_report(
        results: Dict,
        dataset_info: Dict,
        model_info: Dict,
        output_path: Optional[str] = None
    ) -> Dict:
        """生成评估报告

        Args:
            results: 评估结果
            dataset_info: 数据集信息
            model_info: 模型信息
            output_path: 输出路径

        Returns:
            完整报告字典
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_summary': {
                'dataset': dataset_info.get('name', 'Unknown'),
                'total_problems': results.get('total', 0),
                'correct_count': results.get('correct', 0),
                'accuracy': results.get('pass@1', 0),
            },
            'pass_at_k': results.get('pass@k', {}),
            'per_category_results': EvaluationReportGenerator._compute_category_breakdown(
                results.get('per_problem_results', [])
            ),
            'per_difficulty_results': EvaluationReportGenerator._compute_difficulty_breakdown(
                results.get('per_problem_results', [])
            ),
            'model_info': model_info,
            'dataset_info': dataset_info,
            'detailed_results': results.get('per_problem_results', [])[:50],  # 限制详细结果数量
            'full_results': results.get('per_problem_results', []),  # 保存完整结果供提取使用
        }

        # 计算统计信息
        report['statistics'] = {
            'mean_time_per_problem': results.get('mean_time', 0),
            'total_time': results.get('total_time', 0),
            'error_count': sum(1 for r in results.get('per_problem_results', []) if 'error' in r),
        }

        # 添加分析建议
        report['analysis'] = EvaluationReportGenerator._generate_analysis(report)

        # 保存报告
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"报告已保存: {output_path}")

        return report

    @staticmethod
    def _compute_category_breakdown(results: List[Dict]) -> Dict:
        """计算按类别的统计"""
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

        for r in results:
            category = r.get('problem_type', 'unknown')
            category_stats[category]['total'] += 1
            if r.get('correct', False):
                category_stats[category]['correct'] += 1

        breakdown = {}
        for cat, stats in category_stats.items():
            breakdown[cat] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            }

        return breakdown

    @staticmethod
    def _compute_difficulty_breakdown(results: List[Dict]) -> Dict:
        """计算按难度的统计"""
        difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

        for r in results:
            difficulty = r.get('difficulty', 0)
            difficulty_stats[difficulty]['total'] += 1
            if r.get('correct', False):
                difficulty_stats[difficulty]['correct'] += 1

        breakdown = {}
        for diff, stats in sorted(difficulty_stats.items()):
            breakdown[f'difficulty_{diff}'] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            }

        return breakdown

    @staticmethod
    def _generate_analysis(report: Dict) -> Dict:
        """生成分析建议"""
        analysis = {}

        accuracy = report['evaluation_summary'].get('accuracy', 0)

        # 性能评级
        if accuracy >= 0.8:
            analysis['performance_level'] = '优秀'
        elif accuracy >= 0.6:
            analysis['performance_level'] = '良好'
        elif accuracy >= 0.4:
            analysis['performance_level'] = '中等'
        else:
            analysis['performance_level'] = '需改进'

        # 类别分析
        category_results = report.get('per_category_results', {})
        if category_results:
            weak_categories = [
                cat for cat, stats in category_results.items()
                if stats.get('accuracy', 1) < accuracy * 0.8
            ]
            if weak_categories:
                analysis['weak_categories'] = weak_categories
                analysis['suggestions'] = [
                    f"建议加强对 {cat} 类型问题的训练" for cat in weak_categories
                ]

        # 难度分析
        difficulty_results = report.get('per_difficulty_results', {})
        if difficulty_results:
            max_difficulty = max(
                int(k.replace('difficulty_', '')) for k in difficulty_results.keys()
            ) if difficulty_results else 0
            if max_difficulty >= 4:
                high_diff_accuracy = difficulty_results.get(f'difficulty_{max_difficulty}', {}).get('accuracy', 0)
                if high_diff_accuracy < accuracy * 0.7:
                    analysis['high_difficulty_gap'] = True
                    analysis['suggestions'].append(
                        f"高难度问题准确率较低（{high_diff_accuracy:.1%}），建议增加高难度样本训练"
                    )

        return analysis


# ============ 评估器 ============

class ModelEvaluator:
    """模型评估器"""

    def __init__(
        self,
        model_path: str,
        k_values: List[int] = [1, 2, 5, 10],
        samples_per_problem: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """初始化评估器

        Args:
            model_path: 模型路径
            k_values: Pass@K的K值列表
            samples_per_problem: 每个问题的采样数
            max_new_tokens: 最大生成token数
            temperature: 生成温度
        """
        self.model_path = model_path
        self.k_values = k_values
        self.samples_per_problem = samples_per_problem
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.trainer = None
        self._setup_done = False

    def setup(self) -> bool:
        """设置评估环境"""
        try:
            config = TrainingConfig(
                base_model=self.model_path,
                max_length=4096,
            )
            self.trainer = MethodThinkerTrainer(config)

            # 尝试加载模型
            if self.trainer.load_checkpoint(self.model_path):
                self._setup_done = True
                logger.info(f"模型加载成功: {self.model_path}")
                return True
            else:
                # 如果加载失败，尝试setup（可能是基座模型路径）
                if self.trainer.setup():
                    self._setup_done = True
                    logger.info(f"模型设置成功: {self.model_path}")
                    return True
                else:
                    logger.error(f"模型加载失败: {self.model_path}")
                    return False

        except Exception as e:
            logger.error(f"设置评估环境失败: {e}")
            return False

    def evaluate(
        self,
        problems: List[Dict],
        show_progress: bool = True
    ) -> Dict:
        """评估模型

        Args:
            problems: 问题列表
            show_progress: 是否显示进度

        Returns:
            评估结果
        """
        if not self._setup_done:
            if not self.setup():
                return {'error': '模型未加载', 'status': 'failed'}

        results = {
            'total': len(problems),
            'correct': 0,
            'per_problem_results': [],
            'total_time': 0,
            'mean_time': 0,
        }

        start_time = time.time()
        times = []

        for i, problem in enumerate(problems):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"进度: {i + 1}/{len(problems)}")

            problem_start = time.time()

            # 准备数据格式
            eval_sample = {
                'problem_id': problem.get('problem_id', f'problem_{i}'),
                'problem': problem.get('problem', ''),
                'problem_type': problem.get('problem_type', ''),
                'answer': problem.get('answer', ''),
                'expected': problem.get('answer', ''),
                'candidate_methods': [],
            }

            # 生成解答
            generated = self.trainer._generate_solution(
                eval_sample['problem'],
                eval_sample
            )

            # 验证答案
            is_correct = self.trainer._verify_answer(
                generated,
                eval_sample['answer']
            )

            problem_time = time.time() - problem_start
            times.append(problem_time)

            problem_result = {
                'problem_id': eval_sample['problem_id'],
                'problem_type': problem.get('problem_type', ''),
                'difficulty': problem.get('difficulty', 0),
                'correct': is_correct,
                'generated': generated[:500] if generated else '',  # 截断
                'expected': eval_sample['answer'],
                'time': problem_time,
            }

            if is_correct:
                results['correct'] += 1

            results['per_problem_results'].append(problem_result)

        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['mean_time'] = sum(times) / len(times) if times else 0

        # 计算Pass@K
        results['pass@k'] = PassAtKCalculator.compute_pass_at_k_batch(
            results['per_problem_results'],
            self.k_values,
            self.samples_per_problem
        )
        results['pass@1'] = results['correct'] / results['total'] if results['total'] > 0 else 0
        results['status'] = 'completed'

        return results


# ============ 主函数 ============

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MethodThinker模型评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据集选项:
  --dataset aime       使用AIME数据集
  --dataset hmmt       使用HMMT数据集
  --dataset custom     使用自定义数据集（需指定--data-path）

评估选项:
  --pass-at-k          计算Pass@K，如 --pass-at-k 1,2,5,10
  --samples            每个问题的采样数（用于Pass@K）
  --temperature        生成温度

示例:
  # AIME 2024评估
  %(prog)s --model outputs/final --dataset aime --year 2024

  # HMMT评估
  %(prog)s --model outputs/final --dataset hmmt --round february

  # 自定义数据集
  %(prog)s --model outputs/final --dataset custom --data-path data/test.json

  # Pass@K评估
  %(prog)s --model outputs/final --pass-at-k 1,2,5,10 --samples 10
"""
    )

    # 模型参数
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型路径'
    )

    # 数据集参数
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['aime', 'hmmt', 'custom'],
        default='aime',
        help='数据集类型'
    )
    parser.add_argument(
        '--year',
        type=str,
        default='2024',
        help='数据集年份'
    )
    parser.add_argument(
        '--round',
        type=str,
        choices=['february', 'november'],
        default='february',
        help='HMMT赛轮'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        help='自定义数据集路径'
    )

    # 评估参数
    parser.add_argument(
        '--pass-at-k',
        type=str,
        default='1,2,5',
        help='Pass@K的K值列表（逗号分隔）'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1,
        help='每个问题的采样数'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='生成温度'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=1024,
        help='最大生成token数'
    )

    # 输出参数
    parser.add_argument(
        '--output-report',
        type=str,
        help='报告输出路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='报告输出目录'
    )
    parser.add_argument(
        '--show-details',
        action='store_true',
        help='显示详细结果'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )

    args = parser.parse_args()

    print("MethodThinker 模型评估")
    print("="*50)

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载数据集
    print(f"\n加载数据集: {args.dataset}")
    dataset_info = DatasetLoader.DATASET_INFO.get(args.dataset, {})

    if args.dataset == 'aime':
        problems = DatasetLoader.load_aime(args.year)
        dataset_info['year'] = args.year
    elif args.dataset == 'hmmt':
        problems = DatasetLoader.load_hmmt(args.year, args.round)
        dataset_info['year'] = args.year
        dataset_info['round'] = args.round
    elif args.dataset == 'custom':
        if not args.data_path:
            print("错误: 自定义数据集需要指定 --data-path")
            return
        problems = DatasetLoader.load_custom(args.data_path)
        dataset_info['path'] = args.data_path

    print(f"加载问题数: {len(problems)}")

    # 解析K值
    k_values = [int(k) for k in args.pass_at_k.split(',')]
    print(f"Pass@K值: {k_values}")

    # 创建评估器
    evaluator = ModelEvaluator(
        model_path=args.model,
        k_values=k_values,
        samples_per_problem=args.samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # 设置评估环境
    print(f"\n设置评估环境...")
    if not evaluator.setup():
        print("错误: 无法设置评估环境")
        return

    # 开始评估
    print(f"\n开始评估...")
    start_time = time.time()

    results = evaluator.evaluate(problems, show_progress=True)

    elapsed = time.time() - start_time
    print(f"\n评估耗时: {elapsed:.1f}秒")

    # 显示结果
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)

    print(f"\n总体准确率:")
    print(f"  Pass@1: {results.get('pass@1', 0):.2%}")

    if 'pass@k' in results:
        print(f"\nPass@K 结果:")
        for k_name, k_value in results['pass@k'].items():
            print(f"  {k_name}: {k_value:.2%}")

    # 类别统计
    category_results = EvaluationReportGenerator._compute_category_breakdown(
        results.get('per_problem_results', [])
    )
    if category_results:
        print(f"\n按类别统计:")
        for cat, stats in category_results.items():
            print(f"  {cat}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # 难度统计
    difficulty_results = EvaluationReportGenerator._compute_difficulty_breakdown(
        results.get('per_problem_results', [])
    )
    if difficulty_results:
        print(f"\n按难度统计:")
        for diff, stats in sorted(difficulty_results.items()):
            print(f"  {diff}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # 生成报告
    model_info = {
        'path': args.model,
        'samples_per_problem': args.samples,
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
    }

    report_path = args.output_report
    if not report_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(
            args.output_dir,
            f'eval_{args.dataset}_{timestamp}.json'
        )

    report = EvaluationReportGenerator.generate_report(
        results,
        dataset_info,
        model_info,
        report_path
    )

    print(f"\n报告已生成: {report_path}")

    # 显示详细结果
    if args.show_details:
        print("\n" + "="*50)
        print("详细结果")
        print("="*50)

        for r in results.get('per_problem_results', [])[:10]:
            status = '✓' if r.get('correct') else '✗'
            print(f"\n{status} {r.get('problem_id')}")
            print(f"  类型: {r.get('problem_type')}")
            print(f"  难度: {r.get('difficulty')}")
            print(f"  生成: {r.get('generated', '')[:100]}...")
            print(f"  期望: {r.get('expected')}")
            print(f"  时间: {r.get('time', 0):.2f}秒")

    print("\n" + "="*50)
    print("评估完成!")
    print("="*50)


if __name__ == '__main__':
    main()