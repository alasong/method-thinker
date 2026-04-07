#!/usr/bin/env python
"""训练问题调试脚本"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import traceback

def test_data_types():
    """测试数据类型"""
    print("=" * 60)
    print("测试1: 数据类型检查")
    print("=" * 60)

    with open('data/train_data/train.json', 'r') as f:
        data = json.load(f)

    print(f"样本数: {len(data)}")

    # 检查所有数值字段
    numeric_issues = []
    for i, s in enumerate(data):
        for key, value in s.items():
            if key in ['difficulty'] and not isinstance(value, (int, float)):
                numeric_issues.append(f"样本{i}.{key}: {type(value).__name__} = {value!r}")

        # 检查annotations
        for j, ann in enumerate(s.get('annotations', [])):
            for key, value in ann.items():
                if key in ['step_index', 'score'] and not isinstance(value, (int, float, type(None))):
                    numeric_issues.append(f"样本{i}.annotations[{j}].{key}: {type(value).__name__} = {value!r}")

    if numeric_issues:
        print("发现问题:")
        for issue in numeric_issues[:20]:
            print(f"  {issue}")
    else:
        print("✓ 数据类型正常")

    return len(numeric_issues) == 0


def test_training_args():
    """测试TrainingArguments"""
    print("\n" + "=" * 60)
    print("测试2: TrainingArguments参数检查")
    print("=" * 60)

    try:
        from transformers import TrainingArguments
        import torch

        # 测试所有可能的数值参数
        test_params = {
            'output_dir': '/tmp/test',
            'num_train_epochs': 1,
            'per_device_train_batch_size': 1,
            'learning_rate': 5e-5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'save_steps': 500,
            'eval_steps': 100,
            'logging_steps': 10,
            'save_total_limit': 3,
            'seed': 42,
            'bf16': False,
            'fp16': False,
        }

        # 检查参数类型
        for key, value in test_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value} (float)")
            elif isinstance(value, int):
                print(f"  {key}: {value} (int)")
            elif isinstance(value, str):
                print(f"  {key}: {value!r} (str) <- 可能问题!")
            else:
                print(f"  {key}: {value} ({type(value).__name__})")

        args = TrainingArguments(**test_params)
        print("✓ TrainingArguments创建成功")
        return True

    except Exception as e:
        print(f"✗ TrainingArguments失败: {e}")
        traceback.print_exc()
        return False


def test_sft_trainer():
    """测试SFTTrainer"""
    print("\n" + "=" * 60)
    print("测试3: SFTTrainer完整测试")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from datasets import Dataset
        from trl import SFTTrainer
        import torch

        # 加载数据
        with open('data/train_data/train.json', 'r') as f:
            data = json.load(f)

        texts = [{'text': s.get('problem', '') + ' ' + s.get('method_selection', '')} for s in data[:5]]
        dataset = Dataset.from_list(texts)
        print(f"✓ 数据集创建: {len(dataset)}样本")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer加载成功")

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Math-1.5B",
            torch_dtype=torch.float32,
            device_map=None
        )
        print("✓ 模型加载成功")

        # 训练参数
        training_args = TrainingArguments(
            output_dir="/tmp/test",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=5e-5,
            report_to="none",
        )
        print("✓ 训练参数创建成功")

        # 创建SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        print("✓ SFTTrainer创建成功")

        # 尝试训练
        print("\n开始训练测试...")
        result = trainer.train()
        print(f"✓ 训练成功! loss: {result.training_loss:.4f}")
        return True

    except Exception as e:
        print(f"✗ 失败: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = {
        "数据类型": test_data_types(),
        "训练参数": test_training_args(),
        "完整训练": test_sft_trainer(),
    }

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")