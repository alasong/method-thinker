"""收敛检测器

检测迭代是否收敛或退化。
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ConvergenceResult:
    """收敛检测结果"""
    converged: bool
    reason: str
    confidence: float
    metrics: Dict


class ConvergenceDetector:
    """收敛检测器

    分析迭代过程中的指标变化，判断是否收敛。

    Attributes:
        improvement_threshold: 改进阈值
        degradation_threshold: 退化阈值
        window_size: 观察窗口大小
    """

    def __init__(
        self,
        improvement_threshold: float = 0.02,
        degradation_threshold: float = -0.05,
        window_size: int = 3
    ):
        """初始化收敛检测器

        Args:
            improvement_threshold: 认为有改进的最小提升
            degradation_threshold: 认为退化的阈值
            window_size: 观察窗口大小
        """
        self.improvement_threshold = improvement_threshold
        self.degradation_threshold = degradation_threshold
        self.window_size = window_size
        self.metrics_history: List[float] = []

    def add_metric(self, value: float):
        """添加指标值"""
        self.metrics_history.append(value)

    def check_convergence(self) -> ConvergenceResult:
        """检查是否收敛

        Returns:
            ConvergenceResult: 收敛检测结果
        """
        if len(self.metrics_history) < 2:
            return ConvergenceResult(
                converged=False,
                reason="数据不足",
                confidence=0.0,
                metrics={"history_length": len(self.metrics_history)}
            )

        # 计算最近几轮的改进
        recent = self.metrics_history[-self.window_size:]

        if len(recent) < 2:
            return ConvergenceResult(
                converged=False,
                reason="窗口数据不足",
                confidence=0.0,
                metrics={"recent_values": recent}
            )

        # 计算平均改进率
        improvements = []
        for i in range(1, len(recent)):
            if recent[i-1] != 0:
                imp = (recent[i] - recent[i-1]) / recent[i-1]
                improvements.append(imp)

        if not improvements:
            return ConvergenceResult(
                converged=False,
                reason="无法计算改进率",
                confidence=0.0,
                metrics={}
            )

        avg_improvement = sum(improvements) / len(improvements)

        # 判断退化（优先检查，因为退化值也低于改进阈值）
        if avg_improvement < self.degradation_threshold:
            return ConvergenceResult(
                converged=False,
                reason=f"检测到退化({avg_improvement:.2%})",
                confidence=0.8,
                metrics={
                    "avg_improvement": avg_improvement,
                    "degraded": True
                }
            )

        # 判断收敛（改进太小）
        if avg_improvement < self.improvement_threshold:
            return ConvergenceResult(
                converged=True,
                reason=f"改进率({avg_improvement:.2%})低于阈值",
                confidence=min(1.0, len(recent) / self.window_size),
                metrics={
                    "avg_improvement": avg_improvement,
                    "recent_values": recent
                }
            )

        return ConvergenceResult(
            converged=False,
            reason="仍在改进中",
            confidence=0.5,
            metrics={"avg_improvement": avg_improvement}
        )

    def detect_oscillation(self) -> bool:
        """检测是否振荡"""
        if len(self.metrics_history) < 4:
            return False

        recent = self.metrics_history[-4:]
        # 检查是否有明显的上升下降交替
        direction_changes = 0
        for i in range(1, len(recent)):
            if (recent[i] - recent[i-1]) * (recent[i-1] - recent[i-2] if i > 1 else 1) < 0:
                direction_changes += 1

        return direction_changes >= 2

    def detect_early_stopping_needed(self) -> Dict:
        """检测是否需要早停"""
        if len(self.metrics_history) < 3:
            return {"stop": False, "reason": "数据不足"}

        # 连续无改进
        no_improvement_count = 0
        for i in range(len(self.metrics_history) - 1, 0, -1):
            if self.metrics_history[i] <= self.metrics_history[i-1]:
                no_improvement_count += 1
            else:
                break

        if no_improvement_count >= 3:
            return {
                "stop": True,
                "reason": f"连续{no_improvement_count}轮无改进"
            }

        return {"stop": False, "reason": ""}

    def get_trend(self) -> str:
        """获取趋势"""
        if len(self.metrics_history) < 2:
            return "unknown"

        recent = self.metrics_history[-3:]
        if len(recent) < 2:
            return "unknown"

        total_change = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0

        if total_change > self.improvement_threshold:
            return "improving"
        elif total_change < -self.improvement_threshold:
            return "degrading"
        else:
            return "stable"

    def reset(self):
        """重置历史"""
        self.metrics_history = []