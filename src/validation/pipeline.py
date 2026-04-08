"""验证流水线

整合各层验证形成完整流程。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
import os

from .layer0_fast_filter import Layer0FastFilter, ValidationResult
from .layer1_self_reflection import Layer1SelfReflection
from .layer2_multi_model import Layer2MultiModelValidation
from .layer3_test_driven import Layer3TestDrivenValidation, MethodTestCase
from .ensemble_decision import EnsembleDecisionEngine, LayerResult


@dataclass
class ValidationConfig:
    """验证配置"""
    # Layer 0
    layer0_enabled: bool = True

    # Layer 1
    layer1_enabled: bool = True
    layer1_max_iterations: int = 3

    # Layer 2
    layer2_enabled: bool = True
    layer2_models: List[str] = field(default_factory=lambda: ['deepseek_v3', 'qwen_math', 'gpt4o_mini'])
    layer2_budget: float = 500.0
    layer2_approval_threshold: float = 0.6

    # Layer 3
    layer3_enabled: bool = True
    layer3_pass_threshold: float = 0.6

    # Ensemble
    ensemble_weights: Dict[int, float] = field(
        default_factory=lambda: {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.40}
    )


class ValidationPipeline:
    """验证流水线

    整合Layer 0-3和集成决策引擎，形成完整的验证流程。

    Attributes:
        config: 验证配置
        layer0: Layer 0快速过滤器
        layer1: Layer 1自我反思验证器
        layer2: Layer 2多模型验证器
        layer3: Layer 3测试驱动验证器
        ensemble: 集成决策引擎
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        model=None,
        model_clients: Optional[Dict] = None,
        test_dataset: Optional[List[MethodTestCase]] = None,
        existing_kb: Optional[Dict] = None
    ):
        """初始化验证流水线

        Args:
            config: 验证配置
            model: 用于Layer 1和Layer 3的模型
            model_clients: Layer 2使用的模型客户端
            test_dataset: Layer 3使用的测试集
            existing_kb: 已有的知识库（用于Layer 0去重）
        """
        self.config = config or ValidationConfig()
        self.model = model
        self.model_clients = model_clients or {}
        self.test_dataset = test_dataset or []

        # 初始化各层
        self.layer0 = Layer0FastFilter(existing_kb) if (self.config.layer0 and self.config.layer0.enabled) else None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.ensemble = EnsembleDecisionEngine(self.config.ensemble.weights if self.config.ensemble else None)

        if (self.config.layer1 and self.config.layer1.enabled) and model:
            self.layer1 = Layer1SelfReflection(model, self.config.layer1.max_iterations)

        if (self.config.layer2 and self.config.layer2.enabled) and model_clients:
            self.layer2 = Layer2MultiModelValidation(
                model_clients,
                self.config.budget.total if self.config.budget else 500,
                self.config.layer2.approval_threshold
            )

        if (self.config.layer3 and self.config.layer3.enabled) and model and test_dataset:
            self.layer3 = Layer3TestDrivenValidation(
                model,
                test_dataset,
                self.config.layer3.pass_threshold
            )

    def run(self, method: Dict, skip_layers: Optional[List[int]] = None) -> ValidationResult:
        """运行验证流水线

        Args:
            method: 待验证的方法
            skip_layers: 要跳过的层列表

        Returns:
            ValidationResult: 最终验证结果
        """
        skip_layers = skip_layers or []
        layer_results = []

        # Layer 0: 快速过滤
        if 0 not in skip_layers and self.layer0:
            result = self.layer0.validate(method)
            weights = self.config.ensemble.weights if self.config.ensemble else {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.40}
            layer_results.append(LayerResult(
                layer=0,
                passed=result.passed,
                confidence=result.confidence,
                issues=result.issues,
                weight=weights.get(0, 0.05)
            ))

            # 快速失败
            if not result.passed and result.confidence < 0.3:
                return result

        # Layer 1: 自我反思
        if 1 not in skip_layers and self.layer1:
            result = self.layer1.validate(method)
            weights = self.config.ensemble.weights if self.config.ensemble else {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.40}
            layer_results.append(LayerResult(
                layer=1,
                passed=result.passed,
                confidence=result.confidence,
                issues=result.issues,
                weight=weights.get(1, 0.15)
            ))

        # Layer 2: 多模型验证
        if 2 not in skip_layers and self.layer2:
            result = self.layer2.validate(method)
            layer_results.append(LayerResult(
                layer=2,
                passed=result.passed,
                confidence=result.confidence,
                issues=result.issues,
                weight=weights.get(2, 0.40)
            ))

        # Layer 3: 测试验证
        if 3 not in skip_layers and self.layer3:
            result = self.layer3.validate(method)
            layer_results.append(LayerResult(
                layer=3,
                passed=result.passed,
                confidence=result.confidence,
                issues=result.issues,
                weight=weights.get(3, 0.40)
            ))

        # 集成决策
        return self.ensemble.decide(layer_results)

    @classmethod
    def from_config_file(cls, config_path: str, **kwargs):
        """从配置文件创建流水线

        Args:
            config_path: 配置文件路径
            **kwargs: 其他参数

        Returns:
            ValidationPipeline: 验证流水线实例
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        config = ValidationConfig(
            layer0_enabled=config_dict.get('layer0', {}).get('enabled', True),
            layer1_enabled=config_dict.get('layer1', {}).get('enabled', True),
            layer1_max_iterations=config_dict.get('layer1', {}).get('max_iterations', 3),
            layer2_enabled=config_dict.get('layer2', {}).get('enabled', True),
            layer2_budget=config_dict.get('budget', {}).get('total', 500),
            layer3_enabled=config_dict.get('layer3', {}).get('enabled', True),
            layer3_pass_threshold=config_dict.get('layer3', {}).get('pass_threshold', 0.6),
            ensemble_weights=config_dict.get('ensemble', {}).get('weights', cls.DEFAULT_WEIGHTS)
        )

        return cls(config=config, **kwargs)