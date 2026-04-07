#!/usr/bin/env python
"""训练问题调试脚本 - 全面类型检查"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import yaml
import traceback

def test_data_types():
    """测试数据类型"""
    print("=" * 60)
    print("测试1: 数据类型检查")
    print("=" * 60)

    train_path = 'data/train_data/train.json'
    if not os.path.exists(train_path):
        print(f"✗ 训练数据不存在: {train_path}")
        return False

    with open(train_path, 'r') as f:
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


def test_yaml_config_types():
    """测试YAML配置文件类型"""
    print("\n" + "=" * 60)
    print("测试2: YAML配置类型检查")
    print("=" * 60)

    config_path = 'configs/training_config.yaml'
    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    type_issues = []

    # 检查training参数
    training = config.get('training', {})
    numeric_keys = ['num_epochs', 'batch_size', 'gradient_accumulation_steps',
                    'learning_rate', 'weight_decay', 'warmup_ratio', 'warmup_steps',
                    'max_grad_norm', 'max_length', 'max_prompt_length', 'seed',
                    'dataloader_num_workers', 'save_steps', 'logging_steps']

    for key in numeric_keys:
        if key in training:
            value = training[key]
            if key in ['learning_rate', 'weight_decay', 'warmup_ratio', 'max_grad_norm']:
                expected_type = float
            else:
                expected_type = int

            # YAML可能将数字解析为字符串
            if isinstance(value, str):
                type_issues.append(f"training.{key}: str -> {value!r} (应为{expected_type.__name__})")
            elif not isinstance(value, (int, float)) and value is not None:
                type_issues.append(f"training.{key}: {type(value).__name__} -> {value!r}")
            else:
                print(f"  training.{key}: {value} ({type(value).__name__}) ✓")

    # 检查methodology_injection.weights
    weights = config.get('methodology_injection', {}).get('weights', {})
    for key in ['method_selection', 'solution_generation', 'reflection']:
        if key in weights:
            value = weights[key]
            if isinstance(value, str):
                type_issues.append(f"weights.{key}: str -> {value!r} (应为float)")
            elif not isinstance(value, (int, float)):
                type_issues.append(f"weights.{key}: {type(value).__name__} -> {value!r}")
            else:
                print(f"  weights.{key}: {value} ({type(value).__name__}) ✓")

    if type_issues:
        print("\n发现问题:")
        for issue in type_issues:
            print(f"  {issue}")
        return False
    else:
        print("✓ YAML配置类型正常")
        return True


def test_training_args():
    """测试TrainingArguments"""
    print("\n" + "=" * 60)
    print("测试3: TrainingArguments参数检查")
    print("=" * 60)

    try:
        from transformers import TrainingArguments
        import torch

        # 测试所有可能的数值参数 - 强制类型转换
        test_params = {
            'output_dir': '/tmp/test',
            'num_train_epochs': int(3),
            'per_device_train_batch_size': int(8),
            'learning_rate': float(5e-5),
            'warmup_ratio': float(0.1),
            'weight_decay': float(0.01),
            'gradient_accumulation_steps': int(4),
            'save_steps': int(500),
            'logging_steps': int(100),
            'save_total_limit': int(3),
            'seed': int(42),
            'bf16': bool(False),
            'fp16': bool(False),
        }

        # 检查参数类型
        for key, value in test_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value} (float) ✓")
            elif isinstance(value, int):
                print(f"  {key}: {value} (int) ✓")
            elif isinstance(value, str):
                print(f"  {key}: {value!r} (str) <- 可能问题!")
            else:
                print(f"  {key}: {value} ({type(value).__name__}) ✓")

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
    print("测试4: SFTTrainer完整测试")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
        from datasets import Dataset
        from trl import SFTTrainer
        import torch

        # 加载数据
        train_path = 'data/train_data/train.json'
        if not os.path.exists(train_path):
            print(f"✗ 训练数据不存在: {train_path}")
            return False

        with open(train_path, 'r') as f:
            data = json.load(f)

        texts = [{'text': s.get('problem', '') + ' ' + s.get('method_selection', '')} for s in data[:5]]
        dataset = Dataset.from_list(texts)
        print(f"✓ 数据集创建: {len(dataset)}样本")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer加载成功")

        # 加载模型（使用4-bit量化节省显存）
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Math-1.5B",
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("✓ 模型加载成功（4-bit量化）")

        # 应用LoRA
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(8),
            lora_alpha=int(16),
            lora_dropout=float(0.05),
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        print("✓ LoRA配置完成")

        # 训练参数 - 强制类型转换，使用warmup_steps替代deprecated的warmup_ratio
        warmup_steps = int(5 // 1 * float(0.1))  # 简化计算
        training_args = TrainingArguments(
            output_dir="/tmp/test",
            num_train_epochs=int(1),
            per_device_train_batch_size=int(1),
            learning_rate=float(5e-5),
            warmup_steps=warmup_steps,
            weight_decay=float(0.01),
            save_steps=int(500),
            logging_steps=int(10),
            seed=int(42),
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=torch.cuda.is_available(),
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
        "YAML配置类型": test_yaml_config_types(),
        "训练参数": test_training_args(),
        "完整训练": test_sft_trainer(),
    }

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ 所有测试通过，可以开始正式训练")
    else:
        print("\n✗ 存在问题，请修复后再训练")