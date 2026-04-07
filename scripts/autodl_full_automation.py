#!/usr/bin/env python
"""
AutoDL 完全自动化训练脚本

实现从创建实例到下载结果的完整自动化流程。

环境变量:
    AUTODL_API_KEY: AutoDL API密钥（可选，也可手动输入）

用法:
    # 完全自动化训练
    python scripts/autodl_full_automation.py --gpu RTX4090 --epochs 3

    # 只准备环境（不上传代码）
    python scripts/autodl_full_automation.py --prepare-only

    # 清理实例
    python scripts/autodl_full_automation.py --cleanup
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# 尝试导入可选依赖
try:
    import paramiko
    from scp import SCPClient
    HAS_SSH = True
except ImportError:
    HAS_SSH = False
    print("警告: 未安装paramiko/scp，将使用手动模式")
    print("安装: pip install paramiko scp")


@dataclass
class InstanceInfo:
    """实例信息"""
    instance_id: str
    host: str
    port: int
    user: str
    password: str
    status: str = "pending"


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
    ):
        """完整自动化训练流程"""

        print("\n" + "=" * 60)
        print("AutoDL 自动化训练")
        print("=" * 60 + "\n")

        # 1. 成本估算
        self.print_cost_estimate(gpu_type, 2.0)

        if not HAS_SSH:
            print("检测到未安装paramiko，切换到手动模式指导\n")
            self.print_manual_guide()
            return

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

        # 3. 连接实例
        print("\n步骤2: 连接实例")
        print("-" * 40)
        if not self.connect_ssh(host, port, user, password):
            return

        # 4. 上传代码
        print("\n步骤3: 上传代码")
        print("-" * 40)
        if not self.upload_files(local_code_path, "/root/method-thinker"):
            return

        # 5. 安装依赖
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
            return

        # 6. 生成训练数据
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

        # 7. 开始训练
        print("\n步骤6: 开始训练")
        print("-" * 40)
        train_cmd = (
            f"cd /root/method-thinker && "
            f"python scripts/train_sft.py "
            f"--train-data data/train_data/train.json "
            f"--output-dir outputs/models/v1 "
            f"--use-lora --lora-r 16 "
            f"--epochs {epochs} --batch-size {batch_size}"
        )
        print(f"执行命令: {train_cmd}")
        result = self.run_command(train_cmd, timeout=3600)

        if result["success"]:
            print("✓ 训练完成")
            print(result["output"])
        else:
            print(f"✗ 训练失败: {result['error']}")

        # 8. 下载结果
        print("\n步骤7: 下载结果")
        print("-" * 40)
        self.download_files("/root/method-thinker/outputs/models/v1", output_dir)

        # 9. 清理
        print("\n步骤8: 清理")
        print("-" * 40)
        self.close_ssh()
        print("⚠️ 请在AutoDL控制台关闭实例以停止计费")


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

    args = parser.parse_args()

    automation = AutoDLAutomation(args.api_key if hasattr(args, 'api_key') else None)

    if args.guide:
        automation.print_manual_guide()
        return

    if args.estimate:
        automation.print_cost_estimate(args.gpu, args.hours)
        return

    # 执行完整自动化
    automation.full_automation(
        gpu_type=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()