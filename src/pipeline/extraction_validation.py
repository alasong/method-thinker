"""提炼-验证一体化流水线

整合方法论提取、验证和知识库更新的完整流程。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import json
import os
from datetime import datetime

from ..extraction.methodology_extractor import MethodologyExtractor, Method
from ..extraction.pattern_miner import PatternMiner, Pattern
from ..kb.knowledge_base import KnowledgeBase
from ..kb.incremental_updater import IncrementalKBUpdater
from ..iteration.iteration_controller import IterationController
from ..iteration.convergence_detector import ConvergenceDetector, ConvergenceResult
from ..validation.pipeline import ValidationPipeline
from ..validation.config import ValidationConfig
from ..validation.pipeline import ValidationPipeline


@dataclass
class ExtractionValidationConfig:
    """提炼-验证流水线配置"""
    # 提取配置
    min_samples: int = 3
    extraction_batch_size: int = 10

    # 迭代配置
    max_iterations: int = 5
    convergence_threshold: float = 0.02
    degradation_threshold: float = -0.05

    # 验证配置
    validation_config: Optional[ValidationConfig] = None

    # KB更新配置
    prune_threshold: float = 0.3

    # 输出配置
    output_dir: str = ".omc/extraction"
    save_intermediate: bool = True


@dataclass
class PipelineResult:
    """流水线运行结果"""
    iteration: int
    extracted_methods: List[Method]
    validated_methods: List[Method]
    kb_update_stats: Dict
    convergence: ConvergenceResult
    metrics: Dict
    elapsed_time: float


class ExtractionValidationPipeline:
    """提炼-验证一体化流水线

    整合以下组件形成完整的迭代提炼流程：
    1. MethodologyExtractor - 方法论提取
    2. PatternMiner - 模式挖掘
    3. ValidationPipeline - 方法验证
    4. IncrementalKBUpdater - 知识库增量更新
    5. IterationController - 迭代控制
    6. ConvergenceDetector - 收敛检测

    Attributes:
        config: 流水线配置
        extractor: 方法论提取器
        pattern_miner: 模式挖掘器
        validator: 验证流水线
        kb_updater: 知识库更新器
        controller: 迭代控制器
        convergence_detector: 收敛检测器
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        model,
        model_clients: Optional[Dict] = None,
        test_dataset: Optional[List] = None,
        config: Optional[ExtractionValidationConfig] = None
    ):
        """初始化提炼-验证流水线

        Args:
            kb: 知识库实例
            model: 用于提取和验证的主模型
            model_clients: 多模型验证客户端
            test_dataset: 测试驱动验证数据集
            config: 流水线配置
        """
        self.kb = kb
        self.model = model
        self.config = config or ExtractionValidationConfig()

        # 初始化组件
        self.extractor = MethodologyExtractor(
            model,
            min_samples=self.config.min_samples
        )

        self.pattern_miner = PatternMiner()

        self.validator = ValidationPipeline(
            config=self.config.validation_config,
            model=model,
            model_clients=model_clients,
            test_dataset=test_dataset,
            existing_kb={'methods': kb.methods}
        )

        self.kb_updater = IncrementalKBUpdater(kb)

        self.controller = IterationController(
            max_iterations=self.config.max_iterations,
            state_dir=self.config.output_dir
        )

        self.convergence_detector = ConvergenceDetector(
            improvement_threshold=self.config.convergence_threshold,
            degradation_threshold=self.config.degradation_threshold
        )

        # 回调函数
        self.on_extraction_complete: Optional[Callable] = None
        self.on_validation_complete: Optional[Callable] = None
        self.on_kb_update: Optional[Callable] = None
        self.on_iteration_complete: Optional[Callable] = None

    def run(
        self,
        solutions: List[Dict],
        stop_on_convergence: bool = True
    ) -> List[PipelineResult]:
        """运行完整提炼-验证流程

        Args:
            solutions: 解答数据集
            stop_on_convergence: 是否在收敛时停止

        Returns:
            List[PipelineResult]: 各迭代的结果列表
        """
        results = []

        while self.controller.start_iteration():
            iteration_result = self._run_iteration(solutions)
            results.append(iteration_result)

            # 更新收敛检测器
            score = iteration_result.metrics.get('validation_pass_rate', 0)
            self.convergence_detector.add_metric(score)

            # 检查收敛
            convergence = self.convergence_detector.check_convergence()
            iteration_result.convergence = convergence

            # 回调
            if self.on_iteration_complete:
                self.on_iteration_complete(iteration_result)

            # 保存中间结果
            if self.config.save_intermediate:
                self._save_iteration_result(iteration_result)

            # 收敛检查
            if stop_on_convergence and convergence.converged:
                break

            # 早停检查
            early_stop = self.convergence_detector.detect_early_stopping_needed()
            if early_stop['stop']:
                break

            self.controller.complete_iteration()

        return results

    def _run_iteration(self, solutions: List[Dict]) -> PipelineResult:
        """执行单次迭代"""
        start_time = datetime.now()
        self.controller.set_phase("extract")

        # 1. 提取方法论
        extracted_methods = self.extractor.extract_from_solutions(solutions)
        self.controller.update_metrics({
            'extracted_count': len(extracted_methods)
        })

        if self.on_extraction_complete:
            self.on_extraction_complete(extracted_methods)

        # 2. 挖掘模式
        patterns = self.pattern_miner.mine_patterns(solutions)
        self.controller.update_metrics({
            'patterns_found': len(patterns)
        })

        self.controller.set_phase("validate")

        # 3. 验证提取的方法
        validated_methods = []
        for method in extracted_methods:
            method_dict = self._method_to_dict(method)
            result = self.validator.run(method_dict)

            if result.passed:
                validated_methods.append(method)

        pass_rate = len(validated_methods) / len(extracted_methods) if extracted_methods else 0
        self.controller.update_metrics({
            'validation_pass_rate': pass_rate,
            'validated_count': len(validated_methods)
        })

        if self.on_validation_complete:
            self.on_validation_complete(validated_methods)

        self.controller.set_phase("update")

        # 4. 更新知识库
        update_stats = self.kb_updater.update(validated_methods)
        self.controller.update_metrics({
            'kb_added': update_stats['added'],
            'kb_replaced': update_stats['replaced'],
            'kb_merged': update_stats['merged']
        })

        if self.on_kb_update:
            self.on_kb_update(update_stats)

        # 5. 清理低质量方法
        removed = self.kb_updater.prune_low_quality(self.config.prune_threshold)
        self.controller.update_metrics({'kb_pruned': removed})

        self.controller.set_phase("evaluate")

        # 6. 计算综合指标
        metrics = self._compute_metrics(
            extracted_methods, validated_methods, update_stats
        )
        self.controller.update_metrics(metrics)

        elapsed = (datetime.now() - start_time).total_seconds()

        return PipelineResult(
            iteration=self.controller.state.iteration,
            extracted_methods=extracted_methods,
            validated_methods=validated_methods,
            kb_update_stats=update_stats,
            convergence=self.convergence_detector.check_convergence(),
            metrics=self.controller.state.metrics,
            elapsed_time=elapsed
        )

    def _method_to_dict(self, method: Method) -> Dict:
        """将Method对象转换为字典"""
        return {
            'method_id': method.method_id,
            'name': method.name,
            'category': method.category,
            'description': method.description,
            'applicability': method.applicability,
            'template': method.template,
            'difficulty': method.difficulty,
            'frequency': method.frequency,
            'related_methods': method.related_methods,
            'examples': method.examples
        }

    def _compute_metrics(
        self,
        extracted: List[Method],
        validated: List[Method],
        update_stats: Dict
    ) -> Dict:
        """计算综合评估指标"""
        kb_size = len(self.kb.methods)

        return {
            'kb_total_size': kb_size,
            'extraction_efficiency': len(validated) / len(extracted) if extracted else 0,
            'kb_growth_rate': update_stats['added'] / kb_size if kb_size > 0 else 0,
            'avg_method_quality': self._compute_avg_quality(validated)
        }

    def _compute_avg_quality(self, methods: List[Method]) -> float:
        """计算平均方法质量"""
        if not methods:
            return 0.0

        scores = []
        for m in methods:
            # 综合评估：频率 + 描述完整性 + 步骤数量
            desc_score = min(1.0, len(m.description) / 100)
            steps_score = min(1.0, len(m.template.get('steps', [])) / 5)
            scores.append(m.frequency * 0.5 + desc_score * 0.3 + steps_score * 0.2)

        return sum(scores) / len(scores)

    def _save_iteration_result(self, result: PipelineResult):
        """保存迭代结果"""
        os.makedirs(self.config.output_dir, exist_ok=True)

        output_file = os.path.join(
            self.config.output_dir,
            f"iteration_{result.iteration}.json"
        )

        data = {
            'iteration': result.iteration,
            'extracted_methods': [
                self._method_to_dict(m) for m in result.extracted_methods
            ],
            'validated_methods': [
                self._method_to_dict(m) for m in result.validated_methods
            ],
            'kb_update_stats': result.kb_update_stats,
            'convergence': {
                'converged': result.convergence.converged,
                'reason': result.convergence.reason,
                'confidence': result.convergence.confidence
            },
            'metrics': result.metrics,
            'elapsed_time': result.elapsed_time
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run_single_batch(self, solutions: List[Dict]) -> PipelineResult:
        """运行单次提炼-验证（不迭代）

        Args:
            solutions: 解答数据集

        Returns:
            PipelineResult: 运行结果
        """
        self.controller.start_iteration()
        result = self._run_iteration(solutions)
        self.controller.complete_iteration()
        return result

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'iteration': self.controller.state.iteration,
            'phase': self.controller.state.phase,
            'kb_size': len(self.kb.methods),
            'kb_categories': {
                cat: len(methods)
                for cat, methods in self.kb.category_index.items()
            },
            'convergence': self.convergence_detector.get_trend(),
            'history_length': len(self.controller.history)
        }

    def reset(self):
        """重置流水线状态"""
        self.controller = IterationController(
            max_iterations=self.config.max_iterations,
            state_dir=self.config.output_dir
        )
        self.convergence_detector.reset()

    def export_kb(self, path: str):
        """导出知识库"""
        self.kb.save(path)


def create_default_pipeline(
    kb: KnowledgeBase,
    model,
    **kwargs
) -> ExtractionValidationPipeline:
    """创建默认配置的提炼-验证流水线

    Args:
        kb: 知识库实例
        model: 主模型
        **kwargs: 其他参数传递给ExtractionValidationConfig

    Returns:
        ExtractionValidationPipeline: 配置好的流水线实例
    """
    config = ExtractionValidationConfig(**kwargs)
    return ExtractionValidationPipeline(kb, model, config=config)