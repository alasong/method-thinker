"""配置管理

支持从YAML文件加载验证系统配置。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import os


@dataclass
class Layer0Config:
    """Layer 0配置"""
    enabled: bool = True
    strict_mode: bool = True
    check_required_fields: bool = True
    check_field_formats: bool = True
    check_value_constraints: bool = True
    check_duplicates: bool = True
    check_description_quality: bool = True
    required_fields: List[str] = field(default_factory=lambda: [
        'method_id', 'name', 'description', 'applicability', 'template'
    ])
    fast_fail_threshold: float = 0.3


@dataclass
class PerspectiveConfig:
    """反思角度配置"""
    role: str = "初学者"
    focus: str = ""
    temperature: float = 0.5


@dataclass
class Layer1Config:
    """Layer 1配置"""
    enabled: bool = True
    max_iterations: int = 3
    perspectives: List[Dict] = field(default_factory=lambda: [
        {'role': '初学者', 'focus': '这个描述我能理解吗？', 'temperature': 0.7},
        {'role': '专家', 'focus': '这个方法在数学上严谨吗？', 'temperature': 0.2},
        {'role': '教师', 'focus': '这个方法适合教学吗？', 'temperature': 0.5}
    ])
    pass_threshold: float = 0.7


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = ""
    provider: str = ""
    api_key_env: str = ""
    strength: List[str] = field(default_factory=list)
    cost_per_call: float = 0.01
    latency: float = 2.0
    max_tokens: int = 500
    temperature: float = 0.3


@dataclass
class Layer2Config:
    """Layer 2配置"""
    enabled: bool = True
    models: List[str] = field(default_factory=lambda: ['deepseek_v3', 'qwen_math', 'gpt4o_mini'])
    approval_threshold: float = 0.6
    veto_threshold: float = 0.3
    model_selection: Dict[str, List[str]] = field(default_factory=lambda: {
        'ALGEBRA': ['deepseek_v3', 'qwen_math', 'gpt4o_mini'],
        'GEOMETRY': ['qwen_math', 'deepseek_v3', 'gpt4o_mini'],
        'GENERAL': ['deepseek_v3', 'gpt4o_mini']
    })


@dataclass
class Layer3Config:
    """Layer 3配置"""
    enabled: bool = True
    min_test_count: int = 20
    max_test_count: int = 50
    pass_threshold: float = 0.6
    numeric_tolerance: float = 1e-6
    per_test_timeout: int = 60
    total_timeout: int = 600


@dataclass
class VetoConfig:
    """否决配置"""
    enabled: bool = True
    core_layers: List[int] = field(default_factory=lambda: [2, 3])
    confidence_threshold: float = 0.8


@dataclass
class EnsembleConfig:
    """集成决策配置"""
    weights: Dict[int, float] = field(
        default_factory=lambda: {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.40}
    )
    veto: Optional[VetoConfig] = None

    def __post_init__(self):
        if self.veto is None:
            self.veto = VetoConfig()


@dataclass
class BudgetConfig:
    """预算配置"""
    total: float = 500.0
    alert_threshold: float = 0.8
    hard_limit: float = 0.95


@dataclass
class OutputConfig:
    """输出配置"""
    verbose: bool = False
    show_progress: bool = True
    show_statistics: bool = True
    save_to_file: bool = False
    output_path: str = "results/"


@dataclass
class ValidationConfig:
    """完整验证配置"""
    layer0: Layer0Config = field(default_factory=Layer0Config)
    layer1: Layer1Config = field(default_factory=Layer1Config)
    layer2: Layer2Config = field(default_factory=Layer2Config)
    layer3: Layer3Config = field(default_factory=Layer3Config)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'ValidationConfig':
        """从YAML文件加载配置

        Args:
            path: 配置文件路径

        Returns:
            ValidationConfig: 配置实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 解析Layer 0配置
        layer0_data = data.get('layer0', {})
        layer0 = Layer0Config(
            enabled=layer0_data.get('enabled', True),
            strict_mode=layer0_data.get('strict_mode', True),
            check_required_fields=layer0_data.get('check_required_fields', True),
            check_field_formats=layer0_data.get('check_field_formats', True),
            check_value_constraints=layer0_data.get('check_value_constraints', True),
            check_duplicates=layer0_data.get('check_duplicates', True),
            check_description_quality=layer0_data.get('check_description_quality', True),
            required_fields=layer0_data.get('required_fields', [
                'method_id', 'name', 'description', 'applicability', 'template'
            ]),
            fast_fail_threshold=layer0_data.get('fast_fail_threshold', 0.3)
        )

        # 解析Layer 1配置
        layer1_data = data.get('layer1', {})
        layer1 = Layer1Config(
            enabled=layer1_data.get('enabled', True),
            max_iterations=layer1_data.get('max_iterations', 3),
            perspectives=layer1_data.get('perspectives', []),
            pass_threshold=layer1_data.get('pass_threshold', 0.7)
        )

        # 解析Layer 2配置
        layer2_data = data.get('layer2', {})
        layer2 = Layer2Config(
            enabled=layer2_data.get('enabled', True),
            models=layer2_data.get('models', ['deepseek_v3', 'qwen_math', 'gpt4o_mini']),
            approval_threshold=layer2_data.get('approval_threshold', 0.6),
            veto_threshold=layer2_data.get('veto_threshold', 0.3),
            model_selection=layer2_data.get('model_selection', {})
        )

        # 解析Layer 3配置
        layer3_data = data.get('layer3', {})
        layer3 = Layer3Config(
            enabled=layer3_data.get('enabled', True),
            min_test_count=layer3_data.get('min_test_count', 20),
            max_test_count=layer3_data.get('max_test_count', 50),
            pass_threshold=layer3_data.get('pass_threshold', 0.6),
            numeric_tolerance=layer3_data.get('answer_verification', {}).get('numeric_tolerance', 1e-6),
            per_test_timeout=layer3_data.get('timeout', {}).get('per_test', 60),
            total_timeout=layer3_data.get('timeout', {}).get('total', 600)
        )

        # 解析集成决策配置
        ensemble_data = data.get('ensemble', {})
        veto_data = ensemble_data.get('veto', {})
        veto = VetoConfig(
            enabled=veto_data.get('enabled', True),
            core_layers=veto_data.get('core_layers', [2, 3]),
            confidence_threshold=veto_data.get('confidence_threshold', 0.8)
        )
        ensemble = EnsembleConfig(
            weights=ensemble_data.get('weights', {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.40}),
            veto=veto
        )

        # 解析预算配置
        budget_data = data.get('budget', {})
        budget = BudgetConfig(
            total=budget_data.get('total', 500.0),
            alert_threshold=budget_data.get('alert_threshold', 0.8),
            hard_limit=budget_data.get('hard_limit', 0.95)
        )

        # 解析输出配置
        output_data = data.get('output', {})
        console_data = output_data.get('console', {})
        file_data = output_data.get('file', {})
        output = OutputConfig(
            verbose=console_data.get('verbose', False),
            show_progress=console_data.get('show_progress', True),
            show_statistics=console_data.get('show_statistics', True),
            save_to_file=file_data.get('enabled', False),
            output_path=file_data.get('path', 'results/')
        )

        return cls(
            layer0=layer0,
            layer1=layer1,
            layer2=layer2,
            layer3=layer3,
            ensemble=ensemble,
            budget=budget,
            output=output
        )

    def to_yaml(self, path: str):
        """保存配置到YAML文件

        Args:
            path: 配置文件路径
        """
        # 构建否决配置
        veto_dict = {
            'enabled': self.ensemble.veto.enabled,
            'core_layers': self.ensemble.veto.core_layers,
            'confidence_threshold': self.ensemble.veto.confidence_threshold
        }

        data = {
            'layer0': {
                'enabled': self.layer0.enabled,
                'strict_mode': self.layer0.strict_mode,
                'check_required_fields': self.layer0.check_required_fields,
                'check_field_formats': self.layer0.check_field_formats,
                'check_value_constraints': self.layer0.check_value_constraints,
                'check_duplicates': self.layer0.check_duplicates,
                'check_description_quality': self.layer0.check_description_quality,
                'required_fields': self.layer0.required_fields,
                'fast_fail_threshold': self.layer0.fast_fail_threshold
            },
            'layer1': {
                'enabled': self.layer1.enabled,
                'max_iterations': self.layer1.max_iterations,
                'perspectives': self.layer1.perspectives,
                'pass_threshold': self.layer1.pass_threshold
            },
            'layer2': {
                'enabled': self.layer2.enabled,
                'models': self.layer2.models,
                'approval_threshold': self.layer2.approval_threshold,
                'veto_threshold': self.layer2.veto_threshold,
                'model_selection': self.layer2.model_selection
            },
            'layer3': {
                'enabled': self.layer3.enabled,
                'min_test_count': self.layer3.min_test_count,
                'max_test_count': self.layer3.max_test_count,
                'pass_threshold': self.layer3.pass_threshold
            },
            'ensemble': {
                'weights': self.ensemble.weights,
                'veto': veto_dict
            },
            'budget': {
                'total': self.budget.total,
                'alert_threshold': self.budget.alert_threshold,
                'hard_limit': self.budget.hard_limit
            },
            'output': {
                'console': {
                    'verbose': self.output.verbose,
                    'show_progress': self.output.show_progress,
                    'show_statistics': self.output.show_statistics
                },
                'file': {
                    'enabled': self.output.save_to_file,
                    'path': self.output.output_path
                }
            }
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)