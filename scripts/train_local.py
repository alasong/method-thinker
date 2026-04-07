#!/usr/bin/env python
"""本地训练启动脚本

针对NVIDIA Quadro T1000 4GB等低显存GPU优化的训练脚本。
自动检测GPU显存并选择合适配置。

用法示例:
    # 自动检测并训练
    python scripts/train_local.py

    # 指定配置文件
    python scripts/train_local.py --config configs/training_config_local.yaml

    # 指定基座模型
    python scripts/train_local.py --base-model Qwen/Qwen2.5-Math-1.5B

    # 快速测试模式（减少训练样本）
    python scripts/train_local.py --quick-test

    # 监控显存模式（不训练，只监控GPU）
    python scripts/train_local.py --monitor-only
"""

import sys
import os
import json
import yaml
import argparse
import time
import signal
import subprocess
import threading
from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# GPU检测和内存监控
# =============================================================================

class GPUMonitor:
    """GPU显存监控器

    实时监控GPU显存使用情况，支持告警和自动降级。
    """

    def __init__(self, gpu_id: int = 0, alert_threshold: float = 85.0):
        """初始化GPU监控器

        Args:
            gpu_id: GPU设备ID
            alert_threshold: 显存告警阈值（百分比）
        """
        self.gpu_id = gpu_id
        self.alert_threshold = alert_threshold
        self._stop_flag = False
        self._monitor_thread = None
        self._callbacks: List[callable] = []

    def get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息

        Returns:
            GPU信息字典，包含显存总量、使用量、使用率等
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return {"available": False, "error": "CUDA不可用"}

            # 获取GPU属性
            props = torch.cuda.get_device_properties(self.gpu_id)
            total_memory = props.total_memory

            # 获取当前显存使用
            allocated = torch.cuda.memory_allocated(self.gpu_id)
            reserved = torch.cuda.memory_reserved(self.gpu_id)

            # 计算使用率
            used_percent = (allocated / total_memory) * 100 if total_memory > 0 else 0

            return {
                "available": True,
                "gpu_id": self.gpu_id,
                "name": props.name,
                "total_memory_mb": total_memory / (1024 * 1024),
                "total_memory_gb": total_memory / (1024 * 1024 * 1024),
                "allocated_mb": allocated / (1024 * 1024),
                "reserved_mb": reserved / (1024 * 1024),
                "used_percent": used_percent,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }

        except ImportError:
            return {"available": False, "error": "torch未安装"}
        except Exception as e:
            return {"available": False, "error": str(e)}

    def is_low_memory_gpu(self) -> bool:
        """判断是否为低显存GPU

        Returns:
            True如果显存小于等于8GB
        """
        info = self.get_gpu_info()
        if not info.get("available", False):
            return True  # 无GPU时默认使用低显存配置

        total_gb = info.get("total_memory_gb", 0)
        return total_gb <= 8.0

    def get_recommended_config(self) -> Dict[str, Any]:
        """根据GPU显存推荐训练配置

        Returns:
            推荐配置字典
        """
        info = self.get_gpu_info()

        if not info.get("available", False):
            # 无GPU，使用CPU配置（不建议训练）
            return self._get_cpu_config()

        total_gb = info.get("total_memory_gb", 0)
        gpu_name = info.get("name", "")

        # T1000 4GB或类似的低显存GPU
        if total_gb <= 4.5:
            return self._get_4gb_config()

        # 6-8GB显存
        elif total_gb <= 8.0:
            return self._get_8gb_config()

        # 8-16GB显存
        elif total_gb <= 16.0:
            return self._get_16gb_config()

        # 16GB+显存
        else:
            return self._get_high_memory_config()

    def _get_cpu_config(self) -> Dict[str, Any]:
        """CPU配置（仅用于测试）"""
        return {
            "device": "cpu",
            "quantization": {"enabled": False},
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_length": 512,
            "fp16": False,
            "bf16": False,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0,
        }

    def _get_4gb_config(self) -> Dict[str, Any]:
        """4GB显存配置（如Quadro T1000）"""
        return {
            "device": "cuda",
            "quantization": {
                "enabled": True,
                "bits": 4,
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
            "lora": {
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            },
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "max_length": 2048,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "optim": "adamw_8bit",
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
        }

    def _get_8gb_config(self) -> Dict[str, Any]:
        """8GB显存配置"""
        return {
            "device": "cuda",
            "quantization": {
                "enabled": True,
                "bits": 4,
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
            "lora": {
                "enabled": True,
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            },
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_length": 3072,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "optim": "adamw_8bit",
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
        }

    def _get_16gb_config(self) -> Dict[str, Any]:
        """16GB显存配置"""
        return {
            "device": "cuda",
            "quantization": {
                "enabled": True,
                "bits": 4,
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
            "lora": {
                "enabled": True,
                "r": 32,
                "alpha": 64,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
            },
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_length": 4096,
            "fp16": False,
            "bf16": True,
            "gradient_checkpointing": True,
            "optim": "adamw_8bit",
            "dataloader_num_workers": 2,
            "dataloader_pin_memory": True,
        }

    def _get_high_memory_config(self) -> Dict[str, Any]:
        """高显存配置（24GB+）"""
        return {
            "device": "cuda",
            "quantization": {
                "enabled": False,  # 高显存可以不量化
            },
            "lora": {
                "enabled": True,
                "r": 64,
                "alpha": 128,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
            },
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "max_length": 4096,
            "fp16": False,
            "bf16": True,
            "gradient_checkpointing": False,
            "optim": "adamw_torch",
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
        }

    def start_monitoring(self, interval: float = 10.0):
        """启动显存监控

        Args:
            interval: 监控间隔（秒）
        """
        self._stop_flag = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        """停止显存监控"""
        self._stop_flag = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def register_callback(self, callback: callable):
        """注册告警回调函数

        Args:
            callback: 回调函数，接收gpu_info参数
        """
        self._callbacks.append(callback)

    def _monitor_loop(self, interval: float):
        """监控循环"""
        while not self._stop_flag:
            info = self.get_gpu_info()
            if info.get("available", False):
                used_percent = info.get("used_percent", 0)

                # 超过阈值时触发回调
                if used_percent >= self.alert_threshold:
                    for callback in self._callbacks:
                        try:
                            callback(info)
                        except Exception:
                            pass

                # 打印监控信息
                allocated_mb = info.get("allocated_mb", 0)
                total_mb = info.get("total_memory_mb", 0)
                print(f"[GPU监控] 显存: {allocated_mb:.0f}/{total_mb:.0f} MB ({used_percent:.1f}%)")

            time.sleep(interval)

    def clear_cache(self):
        """清空GPU缓存"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[GPU监控] 缓存已清空")
        except Exception:
            pass


def check_gpu_compatibility() -> Dict[str, Any]:
    """检查GPU兼容性

    Returns:
        兼容性检查结果
    """
    result = {
        "cuda_available": False,
        "bf16_supported": False,
        "flash_attention_supported": False,
        "gpu_name": None,
        "total_memory_gb": 0,
        "warnings": [],
        "recommendations": [],
    }

    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()

        if result["cuda_available"]:
            # GPU基本信息
            props = torch.cuda.get_device_properties(0)
            result["gpu_name"] = props.name
            result["total_memory_gb"] = props.total_memory / (1024**3)

            # BF16支持检查
            result["bf16_supported"] = torch.cuda.is_bf16_supported()

            # T1000等老架构GPU不支持BF16
            if not result["bf16_supported"]:
                result["warnings"].append(
                    f"{props.name} 不支持BF16，将使用FP16训练"
                )
                result["recommendations"].append(
                    "建议在配置中设置 fp16: true, bf16: false"
                )

            # Flash Attention检查（需要特定架构）
            if props.major < 8:
                result["flash_attention_supported"] = False
                result["warnings"].append(
                    f"{props.name} 可能不支持Flash Attention 2"
                )
                result["recommendations"].append(
                    "建议在配置中设置 use_flash_attention: false"
                )

            # 低显存警告
            if result["total_memory_gb"] <= 4.0:
                result["warnings"].append(
                    f"显存仅 {result['total_memory_gb']:.1f}GB，强烈建议使用QLoRA"
                )
                result["recommendations"].append(
                    "使用 4-bit量化 + LoRA (r=8) + gradient_checkpointing"
                )

            # bitsandbytes兼容性
            try:
                import bitsandbytes as bnb
                result["bitsandbytes_available"] = True
            except ImportError:
                result["bitsandbytes_available"] = False
                result["warnings"].append(
                    "bitsandbytes未安装，无法使用4-bit量化"
                )
                result["recommendations"].append(
                    "安装: pip install bitsandbytes>=0.41.0"
                )

        else:
            result["warnings"].append("CUDA不可用，无法使用GPU训练")
            result["recommendations"].append("检查CUDA驱动和PyTorch安装")

    except ImportError:
        result["warnings"].append("torch未安装")
        result["recommendations"].append("安装: pip install torch>=2.1.0")

    return result


def print_gpu_info():
    """打印GPU详细信息"""
    monitor = GPUMonitor()
    info = monitor.get_gpu_info()

    print("\n" + "="*60)
    print("GPU 信息")
    print("="*60)

    if info.get("available", False):
        print(f"  设备名称: {info.get('name', '未知')}")
        print(f"  显存总量: {info.get('total_memory_gb', 0):.2f} GB")
        print(f"  计算能力: {info.get('compute_capability', '未知')}")
        print(f"  处理器数: {info.get('multi_processor_count', 0)}")

        allocated_mb = info.get("allocated_mb", 0)
        reserved_mb = info.get("reserved_mb", 0)
        print(f"\n  当前状态:")
        print(f"    已分配: {allocated_mb:.1f} MB")
        print(f"    已保留: {reserved_mb:.1f} MB")
        print(f"    使用率: {info.get('used_percent', 0):.1f}%")

        # 推荐配置
        recommended = monitor.get_recommended_config()
        print(f"\n  推荐训练配置:")
        print(f"    量化: {recommended['quantization']['enabled']}")
        if recommended['quantization']['enabled']:
            print(f"    量化位数: {recommended['quantization']['bits']}-bit")
        print(f"    LoRA: {recommended['lora']['enabled']}")
        if recommended['lora']['enabled']:
            print(f"    LoRA秩: r={recommended['lora']['r']}")
        print(f"    Batch: {recommended['batch_size']}")
        print(f"    梯度累积: {recommended['gradient_accumulation_steps']}")
        print(f"    序列长度: {recommended['max_length']}")
        print(f"    FP16: {recommended['fp16']}")
        print(f"    BF16: {recommended['bf16']}")
        print(f"    梯度检查点: {recommended['gradient_checkpointing']}")
        print(f"    优化器: {recommended['optim']}")

    else:
        print(f"  GPU不可用: {info.get('error', '未知错误')}")

    print("="*60 + "\n")


def print_compatibility_check():
    """打印GPU兼容性检查结果"""
    compat = check_gpu_compatibility()

    print("\n" + "="*60)
    print("GPU 兼容性检查")
    print("="*60)

    print(f"\n  CUDA可用: {compat['cuda_available']}")
    if compat['gpu_name']:
        print(f"  GPU名称: {compat['gpu_name']}")
        print(f"  显存大小: {compat['total_memory_gb']:.2f} GB")

    print(f"\n  功能支持:")
    print(f"    BF16: {compat['bf16_supported']}")
    print(f"    Flash Attention: {compat.get('flash_attention_supported', '未知')}")
    print(f"    bitsandbytes: {compat.get('bitsandbytes_available', False)}")

    if compat['warnings']:
        print(f"\n  警告:")
        for warning in compat['warnings']:
            print(f"    - {warning}")

    if compat['recommendations']:
        print(f"\n  建议:")
        for rec in compat['recommendations']:
            print(f"    - {rec}")

    print("="*60 + "\n")


# =============================================================================
# 配置加载和合并
# =============================================================================

def load_config(config_path: str) -> Dict:
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        print(f"警告: 配置文件不存在: {config_path}")
        return {}

    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"不支持的配置格式: {config_path}")


def merge_configs(base_config: Dict, gpu_config: Dict) -> Dict:
    """合并GPU推荐配置到基础配置

    Args:
        base_config: 基础配置（从yaml文件加载）
        gpu_config: GPU推荐配置

    Returns:
        合并后的配置
    """
    merged = base_config.copy()

    # 合并量化配置
    if 'quantization' not in merged:
        merged['quantization'] = {}
    merged['quantization'].update(gpu_config.get('quantization', {}))

    # 合并LoRA配置
    if gpu_config.get('lora', {}).get('enabled', False):
        if 'quantization' not in merged:
            merged['quantization'] = {}
        if 'lora' not in merged['quantization']:
            merged['quantization']['lora'] = {}
        merged['quantization']['lora'].update(gpu_config['lora'])

    # 合并训练参数
    if 'training' not in merged:
        merged['training'] = {}

    training_updates = {
        'batch_size': gpu_config.get('batch_size', 1),
        'gradient_accumulation_steps': gpu_config.get('gradient_accumulation_steps', 32),
        'max_length': gpu_config.get('max_length', 2048),
        'fp16': gpu_config.get('fp16', True),
        'bf16': gpu_config.get('bf16', False),
        'gradient_checkpointing': gpu_config.get('gradient_checkpointing', True),
        'optim': gpu_config.get('optim', 'adamw_8bit'),
        'dataloader_num_workers': gpu_config.get('dataloader_num_workers', 0),
        'dataloader_pin_memory': gpu_config.get('dataloader_pin_memory', False),
    }
    merged['training'].update(training_updates)

    return merged


# =============================================================================
# 训练执行
# =============================================================================

def setup_training_environment(config: Dict) -> bool:
    """设置训练环境

    Args:
        config: 训练配置

    Returns:
        是否成功设置
    """
    # 检查必要依赖
    required_packages = ['torch', 'transformers', 'peft', 'trl', 'accelerate']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"错误: 缺少必要依赖: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False

    # 检查bitsandbytes（量化需要）
    if config.get('quantization', {}).get('enabled', False):
        try:
            import bitsandbytes
        except ImportError:
            print("警告: bitsandbytes未安装，将禁用4-bit量化")
            config['quantization']['enabled'] = False

    # 检查CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("警告: CUDA不可用，训练将非常缓慢")
    except ImportError:
        print("错误: PyTorch未安装")
        return False

    return True


def run_training(config: Dict, args: argparse.Namespace) -> Dict:
    """执行训练

    Args:
        config: 训练配置
        args: 命令行参数

    Returns:
        训练结果
    """
    # 动态导入训练模块
    try:
        from src.training.trainer import MethodThinkerTrainer, TrainingConfig
    except ImportError as e:
        print(f"错误: 无法导入训练模块: {e}")
        return {"status": "failed", "error": str(e)}

    # 构建TrainingConfig
    training_params = config.get('training', {})
    quant_config = config.get('quantization', {})
    lora_config = quant_config.get('lora', {})

    training_config = TrainingConfig(
        base_model=config.get('model', {}).get('base_model', 'Qwen/Qwen2.5-Math-1.5B'),
        output_dir=config.get('output', {}).get('output_dir', 'outputs/local_checkpoints'),

        # 训练参数
        num_epochs=training_params.get('num_epochs', 3),
        batch_size=training_params.get('batch_size', 1),
        learning_rate=training_params.get('learning_rate', 2e-4),
        warmup_ratio=training_params.get('warmup_ratio', 0.05),
        weight_decay=training_params.get('weight_decay', 0.01),
        gradient_accumulation_steps=training_params.get('gradient_accumulation_steps', 32),

        # 序列长度
        max_length=training_params.get('max_length', 2048),

        # LoRA配置
        use_lora=lora_config.get('enabled', False),
        lora_r=lora_config.get('lora_r', lora_config.get('r', 8)),
        lora_alpha=lora_config.get('lora_alpha', lora_config.get('alpha', 16)),
        lora_dropout=lora_config.get('lora_dropout', lora_config.get('dropout', 0.05)),

        # 其他参数
        save_steps=config.get('output', {}).get('save_steps', 200),
        eval_steps=config.get('output', {}).get('evaluation', {}).get('eval_steps', 400),
        logging_steps=config.get('output', {}).get('logging', {}).get('logging_steps', 50),
        seed=training_params.get('seed', 42),
    )

    # 创建训练器
    trainer = MethodThinkerTrainer(training_config)

    # 设置环境
    print("\n初始化训练环境...")
    if not trainer.setup():
        return {"status": "failed", "error": "训练环境设置失败"}

    # 加载训练数据
    # 这里简化处理，实际应该加载真实数据
    train_data = []

    # 快速测试模式使用少量样本
    if args.quick_test:
        print("\n[快速测试模式] 使用少量训练样本")
        train_data = [
            {
                "problem_id": "test_001",
                "problem": "求解方程 x^2 - 5x + 6 = 0",
                "problem_type": "代数方程",
                "candidate_methods": [
                    {"method_name": "配方法", "applicability_score": 0.8},
                    {"method_name": "公式法", "applicability_score": 0.9},
                ],
                "selected_method": "公式法",
                "selection_reasoning": "二次方程直接使用求根公式最高效",
                "solution_steps": ["使用求根公式: x = (5 ± sqrt(25-24))/2", "x = (5 ± 1)/2", "x1 = 3, x2 = 2"],
                "reflection": "公式法适用于所有二次方程，是通用解法。",
            }
        ]

    # 运行训练
    print("\n开始训练...")
    start_time = time.time()

    results = trainer.train_methodology_injection(train_data)

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed

    # 保存模型
    checkpoint_path = os.path.join(training_config.output_dir, 'final')
    trainer.save_checkpoint(checkpoint_path)
    print(f"\n模型已保存: {checkpoint_path}")

    return results


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MethodThinker 本地训练脚本 (Quadro T1000 4GB优化)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测GPU并训练
  %(prog)s

  # 查看GPU信息
  %(prog)s --show-gpu

  # 检查兼容性
  %(prog)s --check-compat

  # 快速测试模式
  %(prog)s --quick-test

  # 监控显存（不训练）
  %(prog)s --monitor-only

  # 使用指定配置
  %(prog)s --config configs/training_config_local.yaml
"""
    )

    # 基础参数
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config_local.yaml',
        help='配置文件路径 (default: configs/training_config_local.yaml)'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='Qwen/Qwen2.5-Math-1.5B',
        help='基座模型路径 (default: Qwen/Qwen2.5-Math-1.5B)'
    )

    # GPU参数
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU设备ID (default: 0)'
    )
    parser.add_argument(
        '--show-gpu',
        action='store_true',
        help='显示GPU信息'
    )
    parser.add_argument(
        '--check-compat',
        action='store_true',
        help='检查GPU兼容性'
    )
    parser.add_argument(
        '--monitor-only',
        action='store_true',
        help='仅监控显存，不训练'
    )
    parser.add_argument(
        '--monitor-interval',
        type=float,
        default=10.0,
        help='显存监控间隔（秒） (default: 10.0)'
    )

    # 训练参数
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试模式（少量样本，快速验证配置）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='训练轮数'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='最大训练样本数'
    )

    # 输出参数
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/local_checkpoints',
        help='输出目录 (default: outputs/local_checkpoints)'
    )

    # 其他参数
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只显示配置，不实际训练'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='详细输出'
    )

    args = parser.parse_args()

    print("="*60)
    print("MethodThinker 本地训练")
    print("针对 Quadro T1000 4GB 等低显存GPU优化")
    print("="*60)

    # 显示GPU信息
    if args.show_gpu:
        print_gpu_info()
        return

    # 检查兼容性
    if args.check_compat:
        print_compatibility_check()
        return

    # 仅监控模式
    if args.monitor_only:
        print("\n启动GPU显存监控...")
        monitor = GPUMonitor(args.gpu_id)

        def on_alert(info):
            print(f"\n[警告] 显存使用超过阈值: {info.get('used_percent', 0):.1f}%")
            print("建议: 清空缓存或降低batch_size")

        monitor.register_callback(on_alert)
        monitor.start_monitoring(args.monitor_interval)

        print("监控运行中，按Ctrl+C停止...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\n监控已停止")
        return

    # 初始化GPU监控
    monitor = GPUMonitor(args.gpu_id)

    # 显示GPU信息
    print_gpu_info()

    # 检查兼容性
    compat = check_gpu_compatibility()
    if compat['warnings']:
        print("兼容性警告:")
        for warning in compat['warnings']:
            print(f"  - {warning}")

    # 获取GPU推荐配置
    gpu_config = monitor.get_recommended_config()

    # 加载基础配置
    print(f"\n加载配置文件: {args.config}")
    base_config = load_config(args.config)

    # 合并配置
    config = merge_configs(base_config, gpu_config)

    # 应用命令行参数覆盖
    if args.base_model:
        config['model']['base_model'] = args.base_model
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.max_samples:
        config['data']['preprocessing']['max_samples'] = args.max_samples

    # 打印最终配置
    print("\n" + "="*60)
    print("最终训练配置")
    print("="*60)
    print(f"  基座模型: {config.get('model', {}).get('base_model', '未知')}")
    print(f"  输出目录: {config.get('output', {}).get('output_dir', '未知')}")

    training_params = config.get('training', {})
    print(f"\n  训练参数:")
    print(f"    轮数: {training_params.get('num_epochs', 3)}")
    print(f"    Batch: {training_params.get('batch_size', 1)}")
    print(f"    梯度累积: {training_params.get('gradient_accumulation_steps', 32)}")
    print(f"    序列长度: {training_params.get('max_length', 2048)}")
    print(f"    FP16: {training_params.get('fp16', True)}")
    print(f"    BF16: {training_params.get('bf16', False)}")
    print(f"    梯度检查点: {training_params.get('gradient_checkpointing', True)}")
    print(f"    优化器: {training_params.get('optim', 'adamw_8bit')}")

    quant_config = config.get('quantization', {})
    print(f"\n  量化配置:")
    print(f"    启用: {quant_config.get('enabled', False)}")
    if quant_config.get('enabled', False):
        print(f"    位数: {quant_config.get('bits', 4)}-bit")
        print(f"    计算类型: {quant_config.get('bnb_4bit_compute_dtype', 'float16')}")
        print(f"    量化类型: {quant_config.get('bnb_4bit_quant_type', 'nf4')}")
        print(f"    双重量化: {quant_config.get('bnb_4bit_use_double_quant', True)}")

    lora_config = quant_config.get('lora', {})
    print(f"\n  LoRA配置:")
    print(f"    启用: {lora_config.get('enabled', False)}")
    if lora_config.get('enabled', False):
        print(f"    r: {lora_config.get('r', lora_config.get('lora_r', 8))}")
        print(f"    alpha: {lora_config.get('alpha', lora_config.get('lora_alpha', 16))}")
        print(f"    dropout: {lora_config.get('dropout', lora_config.get('lora_dropout', 0.05))}")

    print("="*60 + "\n")

    if args.dry_run:
        print("[Dry Run] 配置检查完成，不执行训练")
        return

    # 设置训练环境
    if not setup_training_environment(config):
        print("错误: 训练环境设置失败")
        return

    # 启动显存监控（训练时）
    monitor.start_monitoring(args.monitor_interval)

    try:
        # 运行训练
        results = run_training(config, args)

        # 打印结果
        print("\n" + "="*60)
        print("训练结果")
        print("="*60)
        print(f"  状态: {results.get('status', '未知')}")
        if 'final_loss' in results:
            print(f"  最终损失: {results['final_loss']:.4f}")
        if 'elapsed_time' in results:
            print(f"  耗时: {results['elapsed_time']:.1f} 秒")

        if results.get('status') == 'failed':
            print(f"  错误: {results.get('error', '未知')}")

        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n训练被中断")

    finally:
        monitor.stop_monitoring()


if __name__ == '__main__':
    main()