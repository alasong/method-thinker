"""测试迭代控制器和收敛检测器"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import shutil
from src.iteration.iteration_controller import IterationController, IterationState
from src.iteration.convergence_detector import ConvergenceDetector, ConvergenceResult


def test_iteration_controller_init():
    """测试迭代控制器初始化"""
    controller = IterationController(max_iterations=5)

    assert controller.max_iterations == 5
    assert controller.state.iteration == 0
    assert controller.state.phase == "init"
    assert len(controller.history) == 0

    print("✓ 测试迭代控制器初始化")


def test_start_iteration():
    """测试开始迭代"""
    controller = IterationController(max_iterations=3)

    # 第一次迭代
    can_start = controller.start_iteration()
    assert can_start == True
    assert controller.state.iteration == 1
    assert controller.state.phase == "init"
    assert controller.state.started_at != ""

    # 第二次迭代
    can_start = controller.start_iteration()
    assert can_start == True
    assert controller.state.iteration == 2

    print("✓ 测试开始迭代")


def test_max_iterations_limit():
    """测试最大迭代限制"""
    controller = IterationController(max_iterations=2)

    controller.start_iteration()  # iteration 1
    controller.start_iteration()  # iteration 2

    # 第三次不应该允许
    can_start = controller.start_iteration()
    assert can_start == False
    assert controller.state.iteration == 2  # 保持在上一次

    print("✓ 测试最大迭代限制")


def test_set_phase():
    """测试设置阶段"""
    controller = IterationController()

    controller.start_iteration()

    controller.set_phase("extract")
    assert controller.state.phase == "extract"

    controller.set_phase("validate")
    assert controller.state.phase == "validate"

    controller.set_phase("train")
    assert controller.state.phase == "train"

    controller.set_phase("evaluate")
    assert controller.state.phase == "evaluate"

    # 无效阶段不应改变
    controller.set_phase("invalid_phase")
    assert controller.state.phase == "evaluate"

    print("✓ 测试设置阶段")


def test_update_metrics():
    """测试更新指标"""
    controller = IterationController()

    controller.start_iteration()

    controller.update_metrics({'accuracy': 0.8, 'loss': 0.2})
    assert controller.state.metrics['accuracy'] == 0.8
    assert controller.state.metrics['loss'] == 0.2

    controller.update_metrics({'accuracy': 0.85})
    assert controller.state.metrics['accuracy'] == 0.85
    assert controller.state.metrics['loss'] == 0.2  # 保留旧值

    print("✓ 测试更新指标")


def test_complete_iteration():
    """测试完成迭代"""
    controller = IterationController(max_iterations=3)

    controller.start_iteration()
    controller.set_phase("evaluate")
    controller.update_metrics({'accuracy': 0.75})

    controller.complete_iteration()

    assert controller.state.completed_at != ""
    assert len(controller.history) == 1
    assert controller.history[0].iteration == 1
    assert controller.history[0].metrics['accuracy'] == 0.75

    print("✓ 测试完成迭代")


def test_iteration_history():
    """测试迭代历史"""
    controller = IterationController(max_iterations=5)

    for i in range(3):
        controller.start_iteration()
        controller.update_metrics({'score': 0.7 + i * 0.05})
        controller.complete_iteration()

    assert len(controller.history) == 3

    # 检查历史顺序
    for i, h in enumerate(controller.history):
        assert h.iteration == i + 1

    print("✓ 测试迭代历史")


def test_state_persistence():
    """测试状态持久化"""
    temp_dir = tempfile.mkdtemp()

    try:
        controller = IterationController(max_iterations=5, state_dir=temp_dir)

        controller.start_iteration()
        controller.update_metrics({'accuracy': 0.8})
        controller.complete_iteration()

        # 创建新控制器加载状态
        controller2 = IterationController(max_iterations=5, state_dir=temp_dir)
        controller2.load_state()

        assert controller2.state.iteration == 1
        assert controller2.state.metrics['accuracy'] == 0.8

    finally:
        shutil.rmtree(temp_dir)

    print("✓ 测试状态持久化")


def test_get_summary():
    """测试获取摘要"""
    controller = IterationController(max_iterations=5)

    controller.start_iteration()
    controller.update_metrics({'score': 0.7})
    controller.complete_iteration()

    summary = controller.get_summary()

    assert summary['current_iteration'] == 1
    assert summary['max_iterations'] == 5
    assert summary['total_completed'] == 1
    assert len(summary['metrics_history']) == 1

    print("✓ 测试获取摘要")


def test_iteration_state_dataclass():
    """测试迭代状态数据类"""
    state = IterationState(
        iteration=3,
        phase="evaluate",
        started_at="2024-01-01",
        completed_at="2024-01-02",
        metrics={'accuracy': 0.9},
        kb_version="v1.0",
        model_version="m1.0"
    )

    assert state.iteration == 3
    assert state.phase == "evaluate"
    assert state.metrics['accuracy'] == 0.9
    assert state.kb_version == "v1.0"

    print("✓ 测试迭代状态数据类")


# ========== 收敛检测器测试 ==========

def test_convergence_detector_init():
    """测试收敛检测器初始化"""
    detector = ConvergenceDetector()

    assert detector.improvement_threshold == 0.02
    assert detector.degradation_threshold == -0.05
    assert detector.window_size == 3
    assert len(detector.metrics_history) == 0

    print("✓ 测试收敛检测器初始化")


def test_add_metric():
    """测试添加指标"""
    detector = ConvergenceDetector()

    detector.add_metric(0.7)
    detector.add_metric(0.75)
    detector.add_metric(0.8)

    assert len(detector.metrics_history) == 3
    assert detector.metrics_history == [0.7, 0.75, 0.8]

    print("✓ 测试添加指标")


def test_check_convergence_insufficient_data():
    """测试数据不足时的收敛检测"""
    detector = ConvergenceDetector()

    # 无数据
    result = detector.check_convergence()
    assert result.converged == False
    assert result.reason == "数据不足"

    # 只有一个数据点
    detector.add_metric(0.8)
    result = detector.check_convergence()
    assert result.converged == False

    print("✓ 测试数据不足时的收敛检测")


def test_check_convergence_improving():
    """测试仍在改进时的收敛检测"""
    detector = ConvergenceDetector()

    # 持续改进
    for val in [0.6, 0.7, 0.8, 0.85]:
        detector.add_metric(val)

    result = detector.check_convergence()

    # 改进率大于阈值
    assert result.converged == False
    assert "改进" in result.reason or "仍在改进" in result.reason

    print("✓ 测试仍在改进时的收敛检测")


def test_check_convergence_converged():
    """测试已收敛时的检测"""
    detector = ConvergenceDetector()

    # 改进率很小
    for val in [0.80, 0.81, 0.815, 0.818]:
        detector.add_metric(val)

    result = detector.check_convergence()

    # 改进率低于阈值，应该收敛
    assert result.converged == True
    assert "低于阈值" in result.reason

    print("✓ 测试已收敛时的检测")


def test_check_convergence_degrading():
    """测试退化时的检测"""
    detector = ConvergenceDetector()

    # 性能下降
    for val in [0.85, 0.70, 0.55]:
        detector.add_metric(val)

    result = detector.check_convergence()

    assert result.converged == False
    assert "退化" in result.reason

    print("✓ 测试退化时的检测")


def test_detect_oscillation():
    """测试振荡检测"""
    detector = ConvergenceDetector()

    # 无振荡 - 稳定上升
    for val in [0.6, 0.7, 0.8, 0.85]:
        detector.add_metric(val)
    assert detector.detect_oscillation() == False

    # 振荡模式
    detector.reset()
    for val in [0.7, 0.8, 0.7, 0.8]:  # 上下波动
        detector.add_metric(val)
    assert detector.detect_oscillation() == True

    print("✓ 测试振荡检测")


def test_detect_early_stopping():
    """测试早停检测"""
    detector = ConvergenceDetector()

    # 数据不足
    detector.add_metric(0.7)
    result = detector.detect_early_stopping_needed()
    assert result['stop'] == False

    # 有改进
    detector.add_metric(0.75)
    detector.add_metric(0.8)
    result = detector.detect_early_stopping_needed()
    assert result['stop'] == False

    # 连续无改进
    detector.add_metric(0.8)
    detector.add_metric(0.8)
    detector.add_metric(0.78)  # 连续下降/持平
    result = detector.detect_early_stopping_needed()
    assert result['stop'] == True

    print("✓ 测试早停检测")


def test_get_trend():
    """测试获取趋势"""
    detector = ConvergenceDetector()

    # 数据不足
    detector.add_metric(0.7)
    assert detector.get_trend() == "unknown"

    # 上升趋势
    detector.add_metric(0.75)
    detector.add_metric(0.8)
    assert detector.get_trend() == "improving"

    # 下降趋势
    detector.reset()
    detector.add_metric(0.8)
    detector.add_metric(0.7)
    detector.add_metric(0.6)
    assert detector.get_trend() == "degrading"

    # 稳定趋势
    detector.reset()
    detector.add_metric(0.75)
    detector.add_metric(0.755)
    detector.add_metric(0.758)
    assert detector.get_trend() == "stable"

    print("✓ 测试获取趋势")


def test_reset():
    """测试重置"""
    detector = ConvergenceDetector()

    detector.add_metric(0.7)
    detector.add_metric(0.8)

    detector.reset()

    assert len(detector.metrics_history) == 0

    print("✓ 测试重置")


def test_convergence_result_dataclass():
    """测试收敛结果数据类"""
    result = ConvergenceResult(
        converged=True,
        reason="改进率低于阈值",
        confidence=0.9,
        metrics={'avg_improvement': 0.01}
    )

    assert result.converged == True
    assert result.reason == "改进率低于阈值"
    assert result.confidence == 0.9
    assert result.metrics['avg_improvement'] == 0.01

    print("✓ 测试收敛结果数据类")


def test_custom_thresholds():
    """测试自定义阈值"""
    detector = ConvergenceDetector(
        improvement_threshold=0.05,
        degradation_threshold=-0.1,
        window_size=5
    )

    assert detector.improvement_threshold == 0.05
    assert detector.degradation_threshold == -0.1
    assert detector.window_size == 5

    print("✓ 测试自定义阈值")


def test_window_size_effect():
    """测试窗口大小效果"""
    detector = ConvergenceDetector(window_size=2)

    # 较小的窗口
    detector.add_metric(0.7)
    detector.add_metric(0.71)
    detector.add_metric(0.72)

    result = detector.check_convergence()
    # 使用最近2个数据点计算

    assert result is not None

    print("✓ 测试窗口大小效果")


if __name__ == '__main__':
    # IterationController tests
    test_iteration_controller_init()
    test_start_iteration()
    test_max_iterations_limit()
    test_set_phase()
    test_update_metrics()
    test_complete_iteration()
    test_iteration_history()
    test_state_persistence()
    test_get_summary()
    test_iteration_state_dataclass()

    # ConvergenceDetector tests
    test_convergence_detector_init()
    test_add_metric()
    test_check_convergence_insufficient_data()
    test_check_convergence_improving()
    test_check_convergence_converged()
    test_check_convergence_degrading()
    test_detect_oscillation()
    test_detect_early_stopping()
    test_get_trend()
    test_reset()
    test_convergence_result_dataclass()
    test_custom_thresholds()
    test_window_size_effect()

    print("\n所有IterationController和ConvergenceDetector测试通过! ✓")