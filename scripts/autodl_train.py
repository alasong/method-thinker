#!/usr/bin/env python
"""
AutoDL GPU训练脚本

使用AutoDL API进行云端GPU训练。

环境变量:
    AUTODL_API_KEY: AutoDL API密钥

用法:
    python scripts/autodl_train.py --gpu-type RTX4090 --hours 2
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# AutoDL API配置
AUTODL_API_BASE = "https://api.autodl.com/api/v1"


class AutoDLClient:
    """AutoDL API客户端"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("AUTODL_API_KEY")
        if not self.api_key:
            raise ValueError("请设置AUTODL_API_KEY环境变量")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def list_instances(self) -> Dict:
        """列出所有实例"""
        import requests
        resp = requests.get(
            f"{AUTODL_API_BASE}/instances",
            headers=self.headers
        )
        return resp.json()

    def create_instance(
        self,
        gpu_type: str = "RTX4090",
        gpu_count: int = 1,
        image_name: str = "pytorch:2.0-py3.10-cu118",
        disk_size: int = 50,
        name: str = "method-thinker-train"
    ) -> Dict:
        """创建GPU实例

        Args:
            gpu_type: GPU型号 (RTX3090, RTX4090, A100等)
            gpu_count: GPU数量
            image_name: 镜像名称
            disk_size: 数据盘大小(GB)
            name: 实例名称
        """
        import requests

        payload = {
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "image_name": image_name,
            "disk_size": disk_size,
            "name": name
        }

        resp = requests.post(
            f"{AUTODL_API_BASE}/instances",
            headers=self.headers,
            json=payload
        )
        return resp.json()

    def get_instance(self, instance_id: str) -> Dict:
        """获取实例详情"""
        import requests
        resp = requests.get(
            f"{AUTODL_API_BASE}/instances/{instance_id}",
            headers=self.headers
        )
        return resp.json()

    def start_instance(self, instance_id: str) -> Dict:
        """启动实例"""
        import requests
        resp = requests.post(
            f"{AUTODL_API_BASE}/instances/{instance_id}/start",
            headers=self.headers
        )
        return resp.json()

    def stop_instance(self, instance_id: str) -> Dict:
        """停止实例"""
        import requests
        resp = requests.post(
            f"{AUTODL_API_BASE}/instances/{instance_id}/stop",
            headers=self.headers
        )
        return resp.json()

    def delete_instance(self, instance_id: str) -> Dict:
        """删除实例"""
        import requests
        resp = requests.delete(
            f"{AUTODL_API_BASE}/instances/{instance_id}",
            headers=self.headers
        )
        return resp.json()

    def get_ssh_info(self, instance_id: str) -> Dict:
        """获取SSH连接信息"""
        import requests
        resp = requests.get(
            f"{AUTODL_API_BASE}/instances/{instance_id}/ssh",
            headers=self.headers
        )
        return resp.json()


class RemoteTrainer:
    """远程训练管理器"""

    def __init__(self, autodl_client: AutoDLClient):
        self.client = autodl_client
        self.instance_id = None
        self.ssh_host = None
        self.ssh_port = None
        self.ssh_user = None
        self.ssh_password = None

    def setup_instance(
        self,
        gpu_type: str = "RTX4090",
        wait_ready: bool = True,
        timeout: int = 600
    ) -> str:
        """设置训练实例

        Args:
            gpu_type: GPU型号
            wait_ready: 是否等待就绪
            timeout: 超时时间(秒)

        Returns:
            实例ID
        """
        print(f"创建GPU实例: {gpu_type}")

        # 创建实例
        result = self.client.create_instance(
            gpu_type=gpu_type,
            name=f"method-thinker-{int(time.time())}"
        )

        self.instance_id = result.get("data", {}).get("instance_id")
        if not self.instance_id:
            raise RuntimeError(f"创建实例失败: {result}")

        print(f"实例ID: {self.instance_id}")

        if wait_ready:
            print("等待实例就绪...")
            start_time = time.time()

            while time.time() - start_time < timeout:
                status = self.client.get_instance(self.instance_id)
                state = status.get("data", {}).get("status")

                if state == "running":
                    print("实例已就绪!")
                    break
                elif state == "failed":
                    raise RuntimeError("实例启动失败")

                print(f"  状态: {state}, 等待中...")
                time.sleep(10)
            else:
                raise TimeoutError("等待实例超时")

        # 获取SSH信息
        ssh_info = self.client.get_ssh_info(self.instance_id)
        self.ssh_host = ssh_info.get("data", {}).get("host")
        self.ssh_port = ssh_info.get("data", {}).get("port")
        self.ssh_user = ssh_info.get("data", {}).get("user")
        self.ssh_password = ssh_info.get("data", {}).get("password")

        return self.instance_id

    def upload_code(self, local_path: str = "."):
        """上传代码到远程实例"""
        print("上传代码...")

        # 使用rsync或scp上传
        # 实际实现需要SSH连接
        print(f"  本地路径: {local_path}")
        print(f"  远程主机: {self.ssh_host}:{self.ssh_port}")

        # TODO: 实现实际的上传逻辑
        print("  请手动使用以下命令上传:")
        print(f"  rsync -avz -e 'ssh -p {self.ssh_port}' {local_path}/ {self.ssh_user}@{self.ssh_host}:~/method-thinker/")

    def run_training(
        self,
        config: Dict[str, Any],
        wait: bool = True
    ):
        """运行训练任务

        Args:
            config: 训练配置
            wait: 是否等待完成
        """
        print("启动训练...")

        # 构建训练命令
        cmd = self._build_train_command(config)
        print(f"  命令: {cmd}")

        # TODO: 通过SSH执行命令
        print("  请手动SSH登录执行:")
        print(f"  ssh -p {self.ssh_port} {self.ssh_user}@{self.ssh_host}")
        print(f"  密码: {self.ssh_password}")
        print(f"  然后执行: {cmd}")

    def _build_train_command(self, config: Dict) -> str:
        """构建训练命令"""
        parts = [
            "cd ~/method-thinker",
            "source venv/bin/activate" if Path("venv").exists() else "",
            "pip install -q transformers accelerate peft datasets bitsandbytes trl",
            "python scripts/train_sft.py",
            f"--base-model {config.get('model', 'Qwen/Qwen2.5-Math-1.5B')}",
            f"--train-data {config.get('train_data', 'data/train_data/train.json')}",
            f"--output-dir {config.get('output_dir', 'outputs/models/v1')}",
            "--use-lora",
            f"--lora-r {config.get('lora_r', 16)}",
            f"--epochs {config.get('epochs', 3)}",
            f"--batch-size {config.get('batch_size', 4)}",
            f"--max-length {config.get('max_length', 2048)}",
        ]
        return " && ".join([p for p in parts if p])

    def download_results(self, remote_path: str, local_path: str):
        """下载训练结果"""
        print("下载结果...")
        print(f"  远程: {remote_path}")
        print(f"  本地: {local_path}")

        print("  请手动使用以下命令下载:")
        print(f"  rsync -avz -e 'ssh -p {self.ssh_port}' {self.ssh_user}@{self.ssh_host}:{remote_path}/ {local_path}/")

    def cleanup(self):
        """清理资源"""
        if self.instance_id:
            print(f"停止实例: {self.instance_id}")
            self.client.stop_instance(self.instance_id)
            print("  实例已停止，建议在控制台确认删除以避免持续计费")


def main():
    parser = argparse.ArgumentParser(
        description="AutoDL GPU训练 - 云端GPU训练管理"
    )

    parser.add_argument(
        "--gpu-type",
        type=str,
        default="RTX4090",
        choices=["RTX3090", "RTX4090", "A100"],
        help="GPU型号 (default: RTX4090)"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=2,
        help="预计训练时长(小时) (default: 2)"
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
        "--lora-r",
        type=int,
        default=16,
        help="LoRA秩 (default: 16)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示操作步骤，不实际执行"
    )

    args = parser.parse_args()

    # 成本估算
    prices = {
        "RTX3090": 2.0,
        "RTX4090": 3.0,
        "A100": 10.0
    }

    estimated_cost = prices.get(args.gpu_type, 3.0) * args.hours

    print("=" * 60)
    print("AutoDL GPU训练 - 成本估算")
    print("=" * 60)
    print(f"GPU型号: {args.gpu_type}")
    print(f"预计时长: {args.hours}小时")
    print(f"单价: ¥{prices.get(args.gpu_type, 3.0):.1f}/小时")
    print(f"预估费用: ¥{estimated_cost:.1f}")
    print(f"账户余额: ¥50.0")
    print(f"是否足够: {'✓ 充裕' if estimated_cost <= 50 else '✗ 不足'}")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry Run] 操作步骤:")
        print("1. 登录AutoDL控制台: https://www.autodl.com/")
        print("2. 创建GPU实例:")
        print(f"   - GPU: {args.gpu_type}")
        print(f"   - 镜像: PyTorch 2.0 + Python 3.10")
        print(f"   - 数据盘: 50GB")
        print("3. 通过SSH/Jupyter连接实例")
        print("4. 上传代码和数据")
        print("5. 安装依赖并运行训练")
        print("6. 下载训练结果")
        print("7. 关闭/删除实例")
        return

    # 实际执行
    try:
        client = AutoDLClient()
        trainer = RemoteTrainer(client)

        # 创建实例
        trainer.setup_instance(gpu_type=args.gpu_type)

        # 训练配置
        config = {
            "model": "Qwen/Qwen2.5-Math-1.5B",
            "train_data": "data/train_data/train.json",
            "output_dir": "outputs/models/v1",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lora_r": args.lora_r,
            "max_length": 2048
        }

        # 运行训练
        trainer.run_training(config)

    except Exception as e:
        print(f"错误: {e}")
        raise


if __name__ == "__main__":
    main()