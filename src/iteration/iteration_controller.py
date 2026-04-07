"""迭代控制器

管理方法论迭代提炼的整个生命周期。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os
from datetime import datetime


@dataclass
class IterationState:
    """迭代状态"""
    iteration: int = 0
    phase: str = "init"  # init, extract, validate, train, evaluate
    started_at: str = ""
    completed_at: str = ""
    metrics: Dict = field(default_factory=dict)
    kb_version: str = ""
    model_version: str = ""


class IterationController:
    """迭代控制器

    管理迭代提炼的整个流程，包括状态管理和收敛检测。

    Attributes:
        max_iterations: 最大迭代次数
        state: 当前迭代状态
        history: 迭代历史
    """

    def __init__(self, max_iterations: int = 5, state_dir: str = ".omc/iteration"):
        """初始化迭代控制器

        Args:
            max_iterations: 最大迭代次数
            state_dir: 状态保存目录
        """
        self.max_iterations = max_iterations
        self.state_dir = state_dir
        self.state = IterationState()
        self.history: List[IterationState] = []

    def start_iteration(self) -> bool:
        """开始新迭代

        Returns:
            bool: 是否可以开始新迭代
        """
        if self.state.iteration >= self.max_iterations:
            return False

        self.state.iteration += 1
        self.state.phase = "init"
        self.state.started_at = datetime.now().isoformat()
        self.state.completed_at = ""

        return True

    def set_phase(self, phase: str):
        """设置当前阶段"""
        valid_phases = ["init", "extract", "validate", "train", "evaluate"]
        if phase in valid_phases:
            self.state.phase = phase

    def update_metrics(self, metrics: Dict):
        """更新指标"""
        self.state.metrics.update(metrics)

    def complete_iteration(self):
        """完成当前迭代"""
        self.state.completed_at = datetime.now().isoformat()
        self.history.append(self._copy_state())
        self._save_state()

    def _copy_state(self) -> IterationState:
        """复制状态"""
        return IterationState(
            iteration=self.state.iteration,
            phase=self.state.phase,
            started_at=self.state.started_at,
            completed_at=self.state.completed_at,
            metrics=self.state.metrics.copy(),
            kb_version=self.state.kb_version,
            model_version=self.state.model_version
        )

    def _save_state(self):
        """保存状态到文件"""
        os.makedirs(self.state_dir, exist_ok=True)
        state_file = os.path.join(self.state_dir, "iteration_state.json")

        data = {
            "current": {
                "iteration": self.state.iteration,
                "phase": self.state.phase,
                "started_at": self.state.started_at,
                "completed_at": self.state.completed_at,
                "metrics": self.state.metrics,
                "kb_version": self.state.kb_version,
                "model_version": self.state.model_version
            },
            "history": [
                {
                    "iteration": h.iteration,
                    "phase": h.phase,
                    "started_at": h.started_at,
                    "completed_at": h.completed_at,
                    "metrics": h.metrics
                }
                for h in self.history
            ]
        }

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_state(self):
        """加载状态"""
        state_file = os.path.join(self.state_dir, "iteration_state.json")

        if not os.path.exists(state_file):
            return

        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        current = data.get("current", {})
        self.state = IterationState(
            iteration=current.get("iteration", 0),
            phase=current.get("phase", "init"),
            started_at=current.get("started_at", ""),
            completed_at=current.get("completed_at", ""),
            metrics=current.get("metrics", {}),
            kb_version=current.get("kb_version", ""),
            model_version=current.get("model_version", "")
        )

    def get_summary(self) -> Dict:
        """获取迭代摘要"""
        return {
            "current_iteration": self.state.iteration,
            "current_phase": self.state.phase,
            "max_iterations": self.max_iterations,
            "total_completed": len(self.history),
            "metrics_history": [h.metrics for h in self.history]
        }