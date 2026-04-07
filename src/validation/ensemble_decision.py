"""集成决策引擎

综合各层验证结果做出最终决策。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class LayerResult:
    """单层验证结果"""
    layer: int
    passed: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    weight: float = 0.0


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    layer: int
    confidence: float
    issues: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


class EnsembleDecisionEngine:
    """集成决策引擎

    综合多层验证结果，通过加权投票和否决机制做出最终决策。

    Attributes:
        layer_weights: 各层权重
        veto_threshold: 否决阈值
    """

    DEFAULT_WEIGHTS = {
        0: 0.05,  # Layer 0: 快速过滤
        1: 0.15,  # Layer 1: 自我反思
        2: 0.40,  # Layer 2: 多模型验证
        3: 0.40   # Layer 3: 测试验证
    }

    def __init__(self, layer_weights: Optional[Dict[int, float]] = None):
        """初始化集成决策引擎

        Args:
            layer_weights: 各层权重，默认使用DEFAULT_WEIGHTS
        """
        self.layer_weights = layer_weights or self.DEFAULT_WEIGHTS.copy()

    def decide(self, layer_results: List[LayerResult]) -> ValidationResult:
        """综合决策

        Args:
            layer_results: 各层验证结果

        Returns:
            ValidationResult: 最终验证结果
        """
        # 计算加权得分
        weighted_score = sum(
            (r.confidence if r.passed else 1 - r.confidence) * self.layer_weights.get(r.layer, 0.1)
            for r in layer_results
        )

        # 检查否决情况
        veto_layer = self._check_veto(layer_results)

        # 检查一致通过
        all_passed = all(r.passed for r in layer_results)

        # 检查核心层失败
        core_layers_failed = any(
            not r.passed and r.layer in [2, 3]
            for r in layer_results
        )

        # 综合判断
        if veto_layer is not None:
            passed = False
            confidence = 0.9
            reason = f"Layer {veto_layer} 否决"
        elif all_passed:
            passed = True
            confidence = weighted_score
            reason = "所有层级验证通过"
        elif core_layers_failed:
            passed = False
            confidence = 0.8
            reason = "核心验证层失败"
        else:
            passed = weighted_score > 0.5
            confidence = weighted_score
            reason = f"加权得分: {weighted_score:.2f}"

        # 收集问题
        all_issues = []
        for r in layer_results:
            all_issues.extend([f"[L{r.layer}] {issue}" for issue in r.issues])

        return ValidationResult(
            passed=passed,
            layer=-1,  # 表示集成层
            confidence=confidence,
            issues=list(set(all_issues)),
            details={
                'weighted_score': weighted_score,
                'layer_results': [r.__dict__ for r in layer_results],
                'reason': reason
            }
        )

    def _check_veto(self, results: List[LayerResult]) -> Optional[int]:
        """检查否决

        如果核心层高置信度失败，则否决

        Args:
            results: 各层结果

        Returns:
            Optional[int]: 否决的层号，无否决返回None
        """
        for r in results:
            if r.layer in [2, 3] and not r.passed and r.confidence > 0.8:
                return r.layer
        return None

    def update_weights(self, new_weights: Dict[int, float]):
        """更新权重

        Args:
            new_weights: 新的权重字典
        """
        self.layer_weights.update(new_weights)

    def get_weights(self) -> Dict[int, float]:
        """获取当前权重"""
        return self.layer_weights.copy()