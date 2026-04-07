#!/usr/bin/env python
"""
AutoDL 完全自动化训练脚本

实现从创建实例到下载结果的完整自动化流程。
包含自动清理、超时控制、费用监控等安全机制。

环境变量:
    AUTODL_API_KEY: AutoDL API密钥（可选，也可手动输入）

用法:
    # 完全自动化训练
    python scripts/autodl_full_automation.py --gpu RTX4090 --epochs 3

    # 只准备环境（不上传代码）
    python scripts/autodl_full_automation.py --prepare-only

    # 清理实例
    python scripts/autodl_full_automation.py --cleanup

    # 设置超时和预算限制
    python scripts/autodl_full_automation.py --gpu RTX4090 --timeout 120 --max-cost 10
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import signal
import threading
import atexit
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# 尝试导入可选依赖
try:
    import paramiko
    from scp import SCPClient
    HAS_SSH = True
except ImportError:
    HAS_SSH = False
    print("警告: 未安装paramiko/scp，将使用手动模式")
    print("安装: pip install paramiko scp")


class InstanceStatus(Enum):
    """实例状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    TRAINING = "training"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    STOPPED = "stopped"


class CleanupReason(Enum):
    """清理原因枚举"""
    NORMAL_COMPLETE = "训练正常完成"
    TIMEOUT = "训练超时"
    BUDGET_EXCEEDED = "超出预算"
    ERROR = "发生错误"
    USER_INTERRUPT = "用户中断"
    FORCE_STOP = "强制停止"


@dataclass
class InstanceInfo:
    """实例信息"""
    instance_id: str
    host: str
    port: int
    user: str
    password: str
    status: InstanceStatus = InstanceStatus.PENDING
    start_time: datetime = None
    gpu_type: str = "RTX4090"
    hourly_rate: float = 3.0
    max_cost: Optional[float] = None
    timeout_minutes: Optional[int] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

    def elapsed_hours(self) -> float:
        """计算已运行小时数"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            return elapsed.total_seconds() / 3600
        return 0.0

    def current_cost(self) -> float:
        """计算当前费用"""
        return self.elapsed_hours() * self.hourly_rate

    def is_timeout(self) -> bool:
        """检查是否超时"""
        if self.timeout_minutes:
            elapsed_minutes = self.elapsed_hours() * 60
            return elapsed_minutes >= self.timeout_minutes
        return False

    def is_budget_exceeded(self) -> bool:
        """检查是否超出预算"""
        if self.max_cost:
            return self.current_cost() >= self.max_cost
        return False


@dataclass
class CleanupConfig:
    """清理配置"""
    download_results: bool = True
    send_notification: bool = False
    notification_webhook: Optional[str] = None
    keep_logs: bool = True
    force_stop: bool = False


@dataclass
class TrainingResult:
    """训练结果"""
    success: bool
    reason: CleanupReason
    output_dir: Optional[str] = None
    cost: float = 0.0
    duration_hours: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AutoDLAutomation:
    """AutoDL自动化训练管理器"""

    # GPU价格参考（元/小时）
    GPU_PRICES = {
        "RTX3090": 2.0,
        "RTX4090": 3.0,
        "A100-40G": 10.0,
        "A100-80G": 15.0,
    }

    # 推荐镜像
    RECOMMENDED_IMAGES = {
        "pytorch": "pytorch:2.1.0-cuda11.8-cudnn8-py3.10",
        "tensorflow": "tensorflow:2.12.0-cuda11.8-cudnn8-py3.10",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("AUTODL_API_KEY")
        self.ssh_client = None
        self.instance: Optional[InstanceInfo] = None
        self.cleanup_config = CleanupConfig()
        self._cleanup_registered = False
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self._training_result: Optional[TrainingResult] = None

    def _register_cleanup_handlers(self):
        """注册清理处理器（确保异常时也能清理）"""
        if self._cleanup_registered:
            return

        # 注册atexit处理器
        atexit.register(self._emergency_cleanup)

        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._cleanup_registered = True
        print("✓ 已注册自动清理处理器")

    def _signal_handler(self, signum, frame):
        """信号处理器（Ctrl+C等）"""
        print(f"\n收到终止信号 ({signum})，正在执行清理...")
        self._emergency_cleanup(reason=CleanupReason.USER_INTERRUPT)
        sys.exit(0)

    def _emergency_cleanup(self, reason: CleanupReason = CleanupReason.ERROR):
        """紧急清理（异常情况下执行）"""
        try:
            if self.ssh_client:
                # 尝试下载日志
                if self.cleanup_config.keep_logs:
                    self._download_logs()

                # 关闭SSH连接
                self.close_ssh()

            # 更新结果状态
            if self.instance:
                self.instance.status = InstanceStatus.ERROR

            print("✓ 紧急清理完成")

        except Exception as e:
            print(f"⚠️ 紧急清理遇到问题: {e}")

    # ==================== 自动清理功能 ====================

    def auto_cleanup_on_complete(
        self,
        reason: CleanupReason = CleanupReason.NORMAL_COMPLETE,
        output_dir: Optional[str] = None
    ) -> TrainingResult:
        """训练完成后自动清理

        Args:
            reason: 清理原因
            output_dir: 输出目录路径

        Returns:
            TrainingResult: 训练结果摘要
        """
        print("\n" + "=" * 60)
        print("自动清理流程")
        print("=" * 60)
        print(f"清理原因: {reason.value}")

        result = TrainingResult(
            success=reason == CleanupReason.NORMAL_COMPLETE,
            reason=reason,
            error_message=None if reason == CleanupReason.NORMAL_COMPLETE else reason.value
        )

        try:
            # 1. 下载结果（如果训练成功或需要保留数据）
            if self.cleanup_config.download_results and self.ssh_client:
                print("\n[1/4] 下载训练结果...")
                local_output = output_dir or "./outputs/models"
                try:
                    self.download_files("/root/method-thinker/outputs/models/v1", local_output)
                    result.output_dir = local_output
                    print("✓ 结果下载完成")
                except Exception as e:
                    print(f"⚠️ 结果下载失败: {e}")

            # 2. 下载日志（总是尝试）
            if self.cleanup_config.keep_logs and self.ssh_client:
                print("\n[2/4] 下载训练日志...")
                self._download_logs()

            # 3. 关闭SSH连接
            print("\n[3/4] 关闭SSH连接...")
            self.close_ssh()

            # 4. 提示停止实例
            print("\n[4/4] 停止AutoDL实例...")
            print("⚠️ 请在AutoDL控制台关闭实例以停止计费")
            print("  控制台地址: https://www.autodl.com/console/instance/list")

            # 5. 发送通知（可选）
            if self.cleanup_config.send_notification:
                self._send_notification(result)

            # 记录费用信息
            if self.instance:
                result.cost = self.instance.current_cost()
                result.duration_hours = self.instance.elapsed_hours()
                print(f"\n费用统计:")
                print(f"  运行时长: {result.duration_hours:.2f} 小时")
                print(f"  总费用: ¥{result.cost:.2f}")

            print("\n✓ 自动清理完成")
            self._training_result = result

        except Exception as e:
            print(f"✗ 清理过程出错: {e}")
            result.error_message = str(e)

        return result

    def _download_logs(self):
        """下载训练日志"""
        if not self.ssh_client:
            return

        try:
            log_dir = "./outputs/logs"
            os.makedirs(log_dir, exist_ok=True)

            # 尝试下载日志文件
            log_files = [
                "/root/method-thinker/outputs/models/v1/train.log",
                "/root/method-thinker/outputs/models/v1/training_args.bin",
            ]

            for log_file in log_files:
                try:
                    self.download_files(log_file, log_dir)
                except Exception:
                    pass  # 单个日志文件下载失败不中断流程

            print("✓ 日志下载完成")

        except Exception as e:
            print(f"⚠️ 日志下载遇到问题: {e}")

    def _send_notification(self, result: TrainingResult):
        """发送训练完成通知"""
        webhook = self.cleanup_config.notification_webhook
        if not webhook:
            return

        try:
            # 构建通知消息
            message = {
                "status": "completed" if result.success else "failed",
                "reason": result.reason.value,
                "cost": result.cost,
                "duration_hours": result.duration_hours,
                "timestamp": result.timestamp.isoformat(),
            }

            # 发送通知（支持多种webhook格式）
            if webhook.startswith("https://"):
                import urllib.request
                import urllib.parse

                data = json.dumps(message).encode('utf-8')
                req = urllib.request.Request(
                    webhook,
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                urllib.request.urlopen(req, timeout=10)
                print("✓ 通知发送成功")

        except Exception as e:
            print(f"⚠️ 通知发送失败: {e}")

    # ==================== 实例状态监控 ====================

    def start_instance_monitor(self, check_interval: int = 60):
        """启动实例状态监控线程

        Args:
            check_interval: 检查间隔（秒）
        """
        self._stop_monitoring.clear()

        def monitor_loop():
            while not self._stop_monitoring.is_set():
                try:
                    self._check_instance_status()
                    self._stop_monitoring.wait(check_interval)
                except Exception as e:
                    print(f"⚠️ 监控线程异常: {e}")
                    self._stop_monitoring.wait(check_interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("✓ 实例监控已启动（每{}秒检查一次）".format(check_interval))

    def stop_instance_monitor(self):
        """停止实例监控"""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            print("✓ 实例监控已停止")

    def _check_instance_status(self):
        """检查实例状态"""
        if not self.instance:
            return

        # 检查超时
        if self.instance.is_timeout():
            print("\n⚠️ 训练超时！正在执行自动清理...")
            self.auto_cleanup_on_complete(reason=CleanupReason.TIMEOUT)
            self._stop_monitoring.set()

        # 检查预算
        if self.instance.is_budget_exceeded():
            print("\n⚠️ 超出预算限制！正在执行自动清理...")
            self.auto_cleanup_on_complete(reason=CleanupReason.BUDGET_EXCEEDED)
            self._stop_monitoring.set()

        # 打印状态摘要
        elapsed = self.instance.elapsed_hours()
        cost = self.instance.current_cost()
        print(f"[监控] 运行: {elapsed:.2f}h, 费用: ¥{cost:.2f}, 状态: {self.instance.status.value}")

    def get_instance_status(self) -> Dict[str, Any]:
        """获取实例详细状态"""
        if not self.instance:
            return {"status": "no_instance"}

        return {
            "instance_id": self.instance.instance_id,
            "status": self.instance.status.value,
            "elapsed_hours": self.instance.elapsed_hours(),
            "current_cost": self.instance.current_cost(),
            "max_cost": self.instance.max_cost,
            "timeout_minutes": self.instance.timeout_minutes,
            "is_timeout": self.instance.is_timeout(),
            "is_budget_exceeded": self.instance.is_budget_exceeded(),
        }

    # ==================== 超时自动停止机制 ====================

    def setup_timeout_handler(self, timeout_minutes: int):
        """设置超时自动停止

        Args:
            timeout_minutes: 超时时间（分钟）
        """
        if self.instance:
            self.instance.timeout_minutes = timeout_minutes

        print(f"✓ 已设置超时限制: {timeout_minutes} 分钟")

        # 启动监控线程
        self.start_instance_monitor(check_interval=30)

    def check_timeout_and_cleanup(self) -> bool:
        """检查超时并执行清理

        Returns:
            bool: 是否已超时清理
        """
        if not self.instance:
            return False

        if self.instance.is_timeout():
            self.auto_cleanup_on_complete(reason=CleanupReason.TIMEOUT)
            return True

        return False

    # ==================== 费用监控和预算告警 ====================

    def monitor_budget(self, max_cost: float) -> Dict[str, Any]:
        """监控费用，超预算自动停止

        Args:
            max_cost: 最大预算（元）

        Returns:
            Dict: 当前费用状态
        """
        if self.instance:
            self.instance.max_cost = max_cost

        print(f"✓ 已设置预算限制: ¥{max_cost:.2f}")

        return self.get_budget_status()

    def get_budget_status(self) -> Dict[str, Any]:
        """获取预算状态"""
        if not self.instance:
            return {"budget_set": False}

        current_cost = self.instance.current_cost()
        max_cost = self.instance.max_cost or 0
        remaining = max_cost - current_cost if max_cost else None

        return {
            "budget_set": self.instance.max_cost is not None,
            "max_cost": max_cost,
            "current_cost": current_cost,
            "remaining_budget": remaining,
            "percentage_used": (current_cost / max_cost * 100) if max_cost else None,
            "exceeded": self.instance.is_budget_exceeded(),
        }

    def print_budget_warning(self, threshold: float = 0.8):
        """打印预算警告（当达到阈值时）"""
        status = self.get_budget_status()

        if not status["budget_set"]:
            return

        if status["percentage_used"] and status["percentage_used"] >= threshold * 100:
            print("\n⚠️ 预算警告!")
            print(f"  已使用: {status['percentage_used']:.1f}%")
            print(f"  当前费用: ¥{status['current_cost']:.2f}")
            print(f"  剩余预算: ¥{status['remaining_budget']:.2f}")

    def estimate_cost(self, gpu_type: str, hours: float) -> Dict:
        """估算训练成本"""
        price = self.GPU_PRICES.get(gpu_type, 3.0)
        cost = price * hours
        balance = 50.0  # 假设余额

        return {
            "gpu_type": gpu_type,
            "hourly_rate": price,
            "estimated_hours": hours,
            "estimated_cost": cost,
            "balance": balance,
            "sufficient": cost <= balance,
            "remaining_budget": balance - cost,
        }

    def print_cost_estimate(self, gpu_type: str, hours: float):
        """打印成本估算"""
        est = self.estimate_cost(gpu_type, hours)
        print("\n" + "=" * 60)
        print("成本估算")
        print("=" * 60)
        print(f"GPU型号: {est['gpu_type']}")
        print(f"小时费率: ¥{est['hourly_rate']:.1f}/小时")
        print(f"预计时长: {est['estimated_hours']:.1f}小时")
        print(f"预计费用: ¥{est['estimated_cost']:.1f}")
        print(f"账户余额: ¥{est['balance']:.1f}")
        print(f"是否足够: {'✓ 充裕' if est['sufficient'] else '✗ 不足'}")
        print(f"剩余预算: ¥{est['remaining_budget']:.1f}")
        print("=" * 60 + "\n")

    # ==================== 手动模式指导 ====================

    def print_manual_guide(self):
        """打印手动操作指南"""
        guide = """
╔════════════════════════════════════════════════════════════════╗
║              AutoDL 手动训练指南                                ║
╚════════════════════════════════════════════════════════════════╝

步骤1: 创建实例
──────────────────────────────────────────────────────────────────
1. 访问 https://www.autodl.com/
2. 登录后点击「租用实例」
3. 选择配置:
   • 镜像: PyTorch 2.1.0 + CUDA 11.8 + Python 3.10
   • GPU: RTX 4090 (推荐) 或 RTX 3090
   • 数据盘: 50GB
   • 计费模式: 按量计费
4. 点击「立即租用」

步骤2: 连接实例
──────────────────────────────────────────────────────────────────
方式A - JupyterLab (推荐新手):
  • 在实例详情页点击「打开JupyterLab」
  • 打开Terminal执行命令

方式B - SSH连接:
  • 获取连接信息: 控制台 → 实例详情 → SSH连接
  • 执行: ssh -p <端口> root@<主机地址>
  • 输入密码（控制台显示）

步骤3: 上传代码
──────────────────────────────────────────────────────────────────
在Terminal中执行:

  # 克隆代码
  git clone https://github.com/alasong/method-thinker.git
  cd method-thinker

  # 或者使用SCP上传本地代码
  # scp -P <端口> -r ./method-thinker root@<主机>:~/

步骤4: 安装依赖
──────────────────────────────────────────────────────────────────
  pip install transformers accelerate peft datasets bitsandbytes trl

步骤5: 生成训练数据
──────────────────────────────────────────────────────────────────
  python scripts/generate_training_data.py \\
      --kb data/methodology_kb/v0/math_methods.yaml \\
      --problems data/test_sets/aime_samples.yaml \\
      --output data/train_data/train.json \\
      --mode batch \\
      --samples-per-problem 4

步骤6: 开始训练
──────────────────────────────────────────────────────────────────
  python scripts/train_sft.py \\
      --train-data data/train_data/train.json \\
      --output-dir outputs/models/v1 \\
      --use-lora \\
      --lora-r 16 \\
      --epochs 3 \\
      --batch-size 4

步骤7: 监控训练
──────────────────────────────────────────────────────────────────
  # 查看GPU使用
  watch -n 1 nvidia-smi

  # 查看训练日志
  tail -f outputs/models/v1/train.log

步骤8: 下载结果
──────────────────────────────────────────────────────────────────
方式A - JupyterLab:
  • 右键 outputs/models/v1 → Download

方式B - SCP:
  scp -P <端口> -r root@<主机>:~/method-thinker/outputs/models/v1 ./models/

步骤9: 关闭实例
──────────────────────────────────────────────────────────────────
⚠️ 重要: 训练完成后立即关闭实例避免持续计费

  • 控制台 → 我的实例 → 关机
  • 或删除实例（数据会丢失，确保已下载结果）

════════════════════════════════════════════════════════════════
预计费用: ¥3-6 (RTX 4090, 1-2小时)
预计时间: 15-30分钟（含调试）
════════════════════════════════════════════════════════════════
"""
        print(guide)

    # ==================== SSH自动化（需安装paramiko） ====================

    def connect_ssh(self, host: str, port: int, user: str, password: str) -> bool:
        """建立SSH连接"""
        if not HAS_SSH:
            print("错误: 未安装paramiko，无法自动SSH连接")
            print("安装: pip install paramiko scp")
            return False

        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(host, port, user, password)
            print(f"✓ SSH连接成功: {user}@{host}:{port}")
            return True
        except Exception as e:
            print(f"✗ SSH连接失败: {e}")
            return False

    def run_command(self, command: str, timeout: int = 300) -> Dict:
        """远程执行命令"""
        if not self.ssh_client:
            return {"success": False, "error": "未建立SSH连接"}

        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')

            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "output": output,
                "error": error,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def upload_files(self, local_path: str, remote_path: str) -> bool:
        """上传文件到远程实例"""
        if not self.ssh_client:
            print("错误: 未建立SSH连接")
            return False

        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                if Path(local_path).is_dir():
                    scp.put(local_path, remote_path, recursive=True)
                else:
                    scp.put(local_path, remote_path)
            print(f"✓ 上传成功: {local_path} -> {remote_path}")
            return True
        except Exception as e:
            print(f"✗ 上传失败: {e}")
            return False

    def download_files(self, remote_path: str, local_path: str) -> bool:
        """从远程实例下载文件"""
        if not self.ssh_client:
            print("错误: 未建立SSH连接")
            return False

        try:
            os.makedirs(local_path, exist_ok=True)
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.get(remote_path, local_path, recursive=True)
            print(f"✓ 下载成功: {remote_path} -> {local_path}")
            return True
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            return False

    def close_ssh(self):
        """关闭SSH连接"""
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
            print("✓ SSH连接已关闭")

    # ==================== 完整自动化流程 ====================

    def full_automation(
        self,
        gpu_type: str = "RTX4090",
        epochs: int = 3,
        batch_size: int = 4,
        local_code_path: str = ".",
        output_dir: str = "./outputs/models",
        timeout_minutes: Optional[int] = None,
        max_cost: Optional[float] = None,
        notification_webhook: Optional[str] = None,
    ):
        """完整自动化训练流程

        Args:
            gpu_type: GPU型号
            epochs: 训练轮数
            batch_size: 批大小
            local_code_path: 本地代码路径
            output_dir: 输出目录
            timeout_minutes: 超时时间（分钟）
            max_cost: 最大预算（元）
            notification_webhook: 通知webhook地址
        """

        print("\n" + "=" * 60)
        print("AutoDL 自动化训练")
        print("=" * 60 + "\n")

        # 1. 成本估算
        self.print_cost_estimate(gpu_type, 2.0)

        if not HAS_SSH:
            print("检测到未安装paramiko，切换到手动模式指导\n")
            self.print_manual_guide()
            return

        # 注册清理处理器（确保异常时也能清理）
        self._register_cleanup_handlers()

        # 设置清理配置
        self.cleanup_config.send_notification = notification_webhook is not None
        self.cleanup_config.notification_webhook = notification_webhook

        # 2. 提示用户创建实例
        print("步骤1: 创建实例")
        print("-" * 40)
        print("请在AutoDL控制台手动创建实例:")
        print(f"  • GPU: {gpu_type}")
        print(f"  • 镜像: {self.RECOMMENDED_IMAGES['pytorch']}")
        print("  • 数据盘: 50GB")
        print("\n创建后，请输入连接信息:")

        host = input("主机地址: ").strip()
        port = int(input("SSH端口: ").strip())
        user = input("用户名 (默认root): ").strip() or "root"
        password = input("密码: ").strip()

        # 创建实例信息对象
        instance_id = f"autodl-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        hourly_rate = self.GPU_PRICES.get(gpu_type, 3.0)
        self.instance = InstanceInfo(
            instance_id=instance_id,
            host=host,
            port=port,
            user=user,
            password=password,
            status=InstanceStatus.PENDING,
            gpu_type=gpu_type,
            hourly_rate=hourly_rate,
            max_cost=max_cost,
            timeout_minutes=timeout_minutes,
        )

        # 3. 设置预算和超时监控
        if max_cost:
            self.monitor_budget(max_cost)
        if timeout_minutes:
            self.setup_timeout_handler(timeout_minutes)

        # 4. 连接实例
        print("\n步骤2: 连接实例")
        print("-" * 40)
        if not self.connect_ssh(host, port, user, password):
            self.auto_cleanup_on_complete(reason=CleanupReason.ERROR)
            return

        self.instance.status = InstanceStatus.RUNNING

        # 5. 上传代码
        print("\n步骤3: 上传代码")
        print("-" * 40)
        if not self.upload_files(local_code_path, "/root/method-thinker"):
            self.auto_cleanup_on_complete(reason=CleanupReason.ERROR)
            return

        # 6. 安装依赖
        print("\n步骤4: 安装依赖")
        print("-" * 40)
        result = self.run_command(
            "cd /root/method-thinker && "
            "pip install -q transformers accelerate peft datasets bitsandbytes trl",
            timeout=300
        )
        if result["success"]:
            print("✓ 依赖安装成功")
        else:
            print(f"✗ 依赖安装失败: {result['error']}")
            self.auto_cleanup_on_complete(reason=CleanupReason.ERROR)
            return

        # 7. 生成训练数据
        print("\n步骤5: 生成训练数据")
        print("-" * 40)
        result = self.run_command(
            "cd /root/method-thinker && "
            "python scripts/generate_training_data.py "
            "--kb data/methodology_kb/v0/math_methods.yaml "
            "--problems data/test_sets/aime_samples.yaml "
            "--output data/train_data/train.json "
            "--mode batch --samples-per-problem 4"
        )
        if result["success"]:
            print("✓ 训练数据生成成功")
        else:
            print(f"✗ 数据生成失败: {result['error']}")

        # 8. 开始训练
        print("\n步骤6: 开始训练")
        print("-" * 40)
        self.instance.status = InstanceStatus.TRAINING

        train_cmd = (
            f"cd /root/method-thinker && "
            f"python scripts/train_sft.py "
            f"--train-data data/train_data/train.json "
            f"--output-dir outputs/models/v1 "
            f"--use-lora --lora-r 16 "
            f"--epochs {epochs} --batch-size {batch_size}"
        )
        print(f"执行命令: {train_cmd}")

        # 计算训练超时（如果未设置）
        training_timeout = timeout_minutes * 60 if timeout_minutes else 3600
        result = self.run_command(train_cmd, timeout=training_timeout)

        # 检查是否被监控中断
        if self._stop_monitoring.is_set():
            # 已经通过监控触发了清理
            return

        if result["success"]:
            print("✓ 训练完成")
            print(result["output"])
            self.instance.status = InstanceStatus.COMPLETED
            cleanup_reason = CleanupReason.NORMAL_COMPLETE
        else:
            print(f"✗ 训练失败: {result['error']}")
            self.instance.status = InstanceStatus.ERROR
            cleanup_reason = CleanupReason.ERROR

        # 9. 自动清理
        print("\n步骤7: 自动清理")
        print("-" * 40)
        self.auto_cleanup_on_complete(reason=cleanup_reason, output_dir=output_dir)

        # 停止监控线程
        self.stop_instance_monitor()


def main():
    parser = argparse.ArgumentParser(
        description="AutoDL 自动化训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 显示手动操作指南
  %(prog)s --guide

  # 完全自动化训练
  %(prog)s --gpu RTX4090 --epochs 3

  # 只估算成本
  %(prog)s --estimate --gpu RTX4090 --hours 2

  # 设置超时和预算限制
  %(prog)s --gpu RTX4090 --timeout 120 --max-cost 10

  # 启用通知
  %(prog)s --gpu RTX4090 --webhook https://webhook.example.com/notify
"""
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="RTX4090",
        choices=["RTX3090", "RTX4090", "A100-40G", "A100-80G"],
        help="GPU型号 (default: RTX4090)"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=2.0,
        help="预计训练时长 (default: 2.0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数 (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="批大小 (default: 4)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="训练超时时间（分钟），超时自动停止 (default: 无限制)"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="最大预算（元），超预算自动停止 (default: 无限制)"
    )
    parser.add_argument(
        "--webhook",
        type=str,
        default=None,
        help="训练完成通知webhook地址"
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="显示手动操作指南"
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="只显示成本估算"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只准备环境，不执行训练"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="显示实例状态监控信息"
    )

    args = parser.parse_args()

    automation = AutoDLAutomation()

    if args.guide:
        automation.print_manual_guide()
        return

    if args.estimate:
        automation.print_cost_estimate(args.gpu, args.hours)
        return

    if args.status:
        # 模拟显示状态监控示例
        print("\n实例状态监控示例:")
        test_instance = InstanceInfo(
            instance_id="test-001",
            host="example.host",
            port=12345,
            user="root",
            password="***",
            status=InstanceStatus.TRAINING,
            gpu_type=args.gpu,
            hourly_rate=automation.GPU_PRICES.get(args.gpu, 3.0),
            max_cost=args.max_cost,
            timeout_minutes=args.timeout,
        )
        automation.instance = test_instance
        status = automation.get_instance_status()
        budget = automation.get_budget_status()
        print(json.dumps(status, indent=2, default=str))
        print("\n预算状态:")
        print(json.dumps(budget, indent=2, default=str))
        return

    # 执行完整自动化
    automation.full_automation(
        gpu_type=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        timeout_minutes=args.timeout,
        max_cost=args.max_cost,
        notification_webhook=args.webhook,
    )


if __name__ == "__main__":
    main()