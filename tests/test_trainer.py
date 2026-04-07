"""测试训练器模块

使用mock测试MethodThinkerTrainer，避免真实模型加载和训练。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import json

# 先导入trainer，然后处理torch mock
from src.training.trainer import (
    MethodThinkerTrainer,
    TrainingConfig,
    HAS_TRANSFORMERS,
    HAS_TRL,
    HAS_DATASETS,
)

# 创建一个mock torch对象用于测试
_mock_torch = MagicMock()
_mock_torch.cuda.is_available = Mock(return_value=False)
_mock_torch.cuda.is_bf16_supported = Mock(return_value=False)
_mock_torch.bfloat16 = "bfloat16"
_mock_torch.float32 = "float32"

# 创建一个上下文管理器mock
class _MockNoGrad:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

_mock_torch.no_grad = _MockNoGrad


# ============ 配置测试 ============

def test_training_config_defaults():
    """测试默认训练配置"""
    config = TrainingConfig()

    assert config.base_model == "Qwen/Qwen2.5-Math-1.5B"
    assert config.output_dir == "outputs/checkpoints"
    assert config.num_epochs == 3
    assert config.batch_size == 8
    assert config.learning_rate == 5e-5
    assert config.warmup_ratio == 0.1
    assert config.weight_decay == 0.01
    assert config.max_length == 4096
    assert config.method_selection_weight == 0.3
    assert config.solution_generation_weight == 0.4
    assert config.reflection_weight == 0.3
    assert config.use_lora == False
    assert config.seed == 42
    print("✓ 测试默认配置通过")


def test_training_config_custom():
    """测试自定义训练配置"""
    config = TrainingConfig(
        base_model="custom/model",
        output_dir="custom/output",
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-4,
        use_lora=True,
        lora_r=32,
    )

    assert config.base_model == "custom/model"
    assert config.output_dir == "custom/output"
    assert config.num_epochs == 5
    assert config.batch_size == 16
    assert config.learning_rate == 1e-4
    assert config.use_lora == True
    assert config.lora_r == 32
    print("✓ 测试自定义配置通过")


# ============ 训练器初始化测试 ============

def test_trainer_init_default():
    """测试训练器默认初始化"""
    trainer = MethodThinkerTrainer()

    assert trainer.config is not None
    assert trainer.config.base_model == "Qwen/Qwen2.5-Math-1.5B"
    assert trainer.model is None
    assert trainer.tokenizer is None
    print("✓ 测试训练器默认初始化通过")


def test_trainer_init_with_config():
    """测试带配置的训练器初始化"""
    config = TrainingConfig(base_model="test/model", num_epochs=10)
    trainer = MethodThinkerTrainer(config)

    assert trainer.config.base_model == "test/model"
    assert trainer.config.num_epochs == 10
    print("✓ 测试带配置初始化通过")


def test_trainer_init_without_deps():
    """测试无依赖库时的初始化"""
    with patch('src.training.trainer.HAS_TRANSFORMERS', False):
        with patch('src.training.trainer.HAS_TRL', False):
            trainer = MethodThinkerTrainer()
            assert trainer._has_deps == False
    print("✓ 测试无依赖库初始化通过")


# ============ setup方法测试 ============

@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="需要transformers库")
def test_setup_with_mocked_model():
    """测试setup方法（mock模型）"""
    trainer = MethodThinkerTrainer()

    # Mock transformers
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<|eos|>"

    with patch('src.training.trainer.AutoModelForCausalLM.from_pretrained', return_value=mock_model):
        with patch('src.training.trainer.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            with patch('src.training.trainer.torch.cuda.is_available', return_value=False):
                result = trainer.setup()

    assert result == True
    assert trainer.model == mock_model
    assert trainer.tokenizer == mock_tokenizer
    assert trainer.tokenizer.pad_token == trainer.tokenizer.eos_token
    print("✓ 测试setup方法通过")


def test_setup_without_deps():
    """测试无依赖库时的setup"""
    trainer = MethodThinkerTrainer()
    trainer._has_deps = False

    result = trainer.setup()
    assert result == False
    print("✓ 测试无依赖setup失败通过")


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="需要transformers库")
def test_setup_with_lora():
    """测试带LoRA的setup"""
    config = TrainingConfig(use_lora=True, lora_r=16, lora_alpha=32)
    trainer = MethodThinkerTrainer(config)

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None

    mock_lora_config = MagicMock()
    mock_peft_model = MagicMock()

    with patch('src.training.trainer.AutoModelForCausalLM.from_pretrained', return_value=mock_model):
        with patch('src.training.trainer.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            with patch('src.training.trainer.torch.cuda.is_available', return_value=False):
                with patch('src.training.trainer.LoraConfig', return_value=mock_lora_config):
                    with patch('src.training.trainer.get_peft_model', return_value=mock_peft_model):
                        result = trainer.setup()

    assert result == True
    print("✓ 测试带LoRA setup通过")


# ============ train_methodology_injection测试 ============

def test_train_methodology_injection_without_deps():
    """测试无依赖库时的方法论注入训练"""
    trainer = MethodThinkerTrainer()
    trainer._has_deps = False

    train_data = [{'problem': 'test', 'solution': 'answer'}]
    result = trainer.train_methodology_injection(train_data)

    assert result['status'] == 'failed'
    assert 'error' in result
    print("✓ 测试无依赖训练失败通过")


def test_train_methodology_injection_with_mock():
    """测试方法论注入训练（mock trainer）"""
    trainer = MethodThinkerTrainer()

    # Mock setup成功
    trainer.model = MagicMock()
    trainer.tokenizer = MagicMock()
    trainer._has_deps = True

    # Mock训练数据构建
    mock_dataset = MagicMock()
    mock_dataset.__len__ = Mock(return_value=10)

    # Mock SFTTrainer
    mock_trainer_instance = MagicMock()
    mock_train_result = MagicMock()
    mock_train_result.training_loss = 0.5
    mock_train_result.metrics = {
        'train_runtime': 100,
        'train_samples_per_second': 10,
        'train_steps_per_second': 1,
    }
    mock_trainer_instance.train = Mock(return_value=mock_train_result)
    mock_trainer_instance.save_model = Mock()

    train_data = [
        {
            'problem_id': 'test_001',
            'problem': '求解方程 x^2 = 4',
            'problem_type': '方程求解',
            'candidate_methods': [{'method_name': '因式分解', 'applicability_score': 0.9}],
            'selected_method': '因式分解',
            'selection_reasoning': '适合二次方程',
            'solution_steps': ['识别方程类型', '分解求解'],
            'reflection': '方法选择正确',
        }
    ]

    with patch('src.training.trainer.TrainingArguments'):
        with patch('src.training.trainer.SFTTrainer', return_value=mock_trainer_instance):
            with patch('src.training.trainer.torch', _mock_torch):
                with patch.object(trainer, '_build_methodology_dataset', return_value=mock_dataset):
                    result = trainer.train_methodology_injection(train_data)

    assert result['status'] == 'completed'
    assert result['final_loss'] == 0.5
    assert 'metrics' in result
    print("✓ 测试mock方法论注入训练通过")


# ============ train_diversity测试 ============

def test_train_diversity_without_deps():
    """测试无依赖库时的多样性训练"""
    trainer = MethodThinkerTrainer()
    trainer._has_deps = False

    result = trainer.train_diversity([], methods_per_problem=4)
    assert result['status'] == 'failed'
    print("✓ 测试无依赖多样性训练失败通过")


def test_train_diversity_with_mock():
    """测试多样性训练（mock）"""
    trainer = MethodThinkerTrainer()
    trainer.model = MagicMock()
    trainer.tokenizer = MagicMock()
    trainer._has_deps = True

    train_data = [
        {
            'problem_id': 'div_001',
            'problem': '求解方程',
            'problem_type': '方程求解',
            'candidate_methods': [
                {'method_name': '方法1', 'score': 0.9},
                {'method_name': '方法2', 'score': 0.8},
            ],
            'solution_steps': ['步骤1'],
        }
    ]

    mock_dataset = MagicMock()
    mock_trainer_instance = MagicMock()
    mock_train_result = MagicMock()
    mock_train_result.training_loss = 0.3
    mock_trainer_instance.train = Mock(return_value=mock_train_result)
    mock_trainer_instance.save_model = Mock()

    with patch('src.training.trainer.TrainingArguments'):
        with patch('src.training.trainer.SFTTrainer', return_value=mock_trainer_instance):
            with patch('src.training.trainer.torch', _mock_torch):
                with patch.object(trainer, '_build_methodology_dataset', return_value=mock_dataset):
                    with patch.object(trainer, '_build_diversity_dataset', return_value=train_data):
                        result = trainer.train_diversity(train_data, methods_per_problem=2)

    assert result['status'] == 'completed'
    assert result['methods_per_problem'] == 2
    print("✓ 测试mock多样性训练通过")


# ============ train_reflection测试 ============

def test_train_reflection_without_deps():
    """测试无依赖库时的反思训练"""
    trainer = MethodThinkerTrainer()
    trainer._has_deps = False

    result = trainer.train_reflection([])
    assert result['status'] == 'failed'
    print("✓ 测试无依赖反思训练失败通过")


def test_train_reflection_with_mock():
    """测试反思训练（mock）"""
    trainer = MethodThinkerTrainer()
    trainer.model = MagicMock()
    trainer.tokenizer = MagicMock()
    trainer._has_deps = True

    train_data = [
        {
            'problem_id': 'ref_001',
            'problem': '测试问题',
            'solution_steps': ['步骤'],
            'reflection': '这是反思内容',
        }
    ]

    mock_dataset = MagicMock()
    mock_trainer_instance = MagicMock()
    mock_train_result = MagicMock()
    mock_train_result.training_loss = 0.2
    mock_trainer_instance.train = Mock(return_value=mock_train_result)
    mock_trainer_instance.save_model = Mock()

    with patch('src.training.trainer.TrainingArguments'):
        with patch('src.training.trainer.SFTTrainer', return_value=mock_trainer_instance):
            with patch('src.training.trainer.torch', _mock_torch):
                with patch.object(trainer, '_build_methodology_dataset', return_value=mock_dataset):
                    with patch.object(trainer, '_build_reflection_dataset', return_value=train_data):
                        result = trainer.train_reflection(train_data)

    assert result['status'] == 'completed'
    assert 'reflection_metrics' in result
    print("✓ 测试mock反思训练通过")


# ============ evaluate测试 ============

def test_evaluate_without_model():
    """测试无模型时的评估"""
    trainer = MethodThinkerTrainer()
    trainer.model = None

    result = trainer.evaluate([])
    assert 'error' in result
    assert result['status'] == 'failed'
    print("✓ 测试无模型评估失败通过")


def test_evaluate_with_mock():
    """测试评估（mock模型）"""
    trainer = MethodThinkerTrainer()

    # Mock模型和tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock生成结果
    mock_outputs = MagicMock()
    mock_outputs.__getitem__ = Mock(return_value=MagicMock())
    mock_model.generate = Mock(return_value=mock_outputs)
    mock_tokenizer.decode = Mock(return_value="答案：42\n解释...")
    mock_tokenizer.return_tensors = "pt"
    mock_tokenizer.pad_token_id = 1
    mock_tokenizer.eos_token_id = 2

    trainer.model = mock_model
    trainer.tokenizer = mock_tokenizer
    trainer._has_deps = True

    test_data = [
        {
            'problem_id': 'eval_001',
            'problem': '问题1',
            'answer': '42',
            'candidate_methods': [],
        }
    ]

    with patch('src.training.trainer.torch', _mock_torch):
        with patch.object(trainer, '_verify_answer', return_value=True):
            result = trainer.evaluate(test_data)

    assert result['status'] == 'completed'
    assert result['total'] == 1
    assert result['correct'] == 1
    assert result['pass@1'] == 1.0
    assert 'pass@k' in result
    print("✓ 测试mock评估通过")


def test_evaluate_pass_at_k():
    """测试Pass@K计算"""
    trainer = MethodThinkerTrainer()
    trainer.model = MagicMock()
    trainer.tokenizer = MagicMock()
    trainer._has_deps = True

    # Mock _generate_solution和_verify_answer
    trainer._generate_solution = Mock(return_value="答案")
    trainer._verify_answer = Mock(side_effect=[True, False, True, False, True])

    test_data = [
        {'problem_id': 'p1', 'problem': 'q1', 'answer': 'a1'},
        {'problem_id': 'p2', 'problem': 'q2', 'answer': 'a2'},
        {'problem_id': 'p3', 'problem': 'q3', 'answer': 'a3'},
        {'problem_id': 'p4', 'problem': 'q4', 'answer': 'a4'},
        {'problem_id': 'p5', 'problem': 'q5', 'answer': 'a5'},
    ]

    with patch('src.training.trainer.torch', _mock_torch):
        result = trainer.evaluate(test_data, k_values=[1, 2, 5])

    assert result['total'] == 5
    assert result['correct'] == 3
    assert result['pass@1'] == 0.6
    assert 'pass@1' in result['pass@k']
    assert 'pass@2' in result['pass@k']
    assert 'pass@5' in result['pass@k']
    print("✓ 测试Pass@K计算通过")


# ============ 数据集构建测试 ============

def test_format_input():
    """测试输入格式化"""
    trainer = MethodThinkerTrainer()

    sample = {
        'problem': '求解方程 x^2 = 4',
        'problem_type': '方程求解',
        'candidate_methods': [
            {'method_name': '因式分解', 'applicability_score': 0.9},
            {'method_name': '直接开方', 'applicability_score': 0.8},
        ],
    }

    input_text = trainer._format_input(sample)

    assert '【问题】' in input_text
    assert '求解方程 x^2 = 4' in input_text
    assert '【题型】' in input_text
    assert '方程求解' in input_text
    assert '【候选方法】' in input_text
    assert '因式分解' in input_text
    assert '直接开方' in input_text
    print("✓ 测试输入格式化通过")


def test_format_output():
    """测试输出格式化"""
    trainer = MethodThinkerTrainer()

    sample = {
        'selected_method': '因式分解',
        'selection_reasoning': '方程为二次方程，适合因式分解',
        'solution_steps': ['识别为二次方程', '分解为(x-2)(x+2)=0', '解得x=±2'],
        'reflection': '方法选择正确，结果需要验证',
    }

    output_text = trainer._format_output(sample)

    assert '【方法选择】' in output_text
    assert '因式分解' in output_text
    assert '【选择理由】' in output_text
    assert '【解答过程】' in output_text
    assert '【反思与验证】' in output_text
    print("✓ 测试输出格式化通过")


def test_build_diversity_dataset():
    """测试构建多样性数据集"""
    trainer = MethodThinkerTrainer()

    train_data = [
        {
            'problem_id': 'd1',
            'problem': '问题',
            'problem_type': '类型',
            'candidate_methods': [
                {'method_name': '方法A', 'score': 0.9},
                {'method_name': '方法B', 'score': 0.8},
                {'method_name': '方法C', 'score': 0.7},
            ],
            'solution_steps': ['步骤'],
        }
    ]

    diversity_data = trainer._build_diversity_dataset(train_data, methods_per_problem=2)

    assert len(diversity_data) == 2
    assert diversity_data[0]['selected_method'] == '方法A'
    assert diversity_data[1]['selected_method'] == '方法B'
    print("✓ 测试构建多样性数据集通过")


def test_build_reflection_dataset():
    """测试构建反思数据集"""
    trainer = MethodThinkerTrainer()

    train_data = [
        {
            'problem_id': 'r1',
            'problem': '问题',
            'solution_steps': ['步骤'],
            'reflection': '已有反思',
        },
        {
            'problem_id': 'r2',
            'problem': '问题2',
            'solution_steps': ['步骤2'],
            # 无reflection，需要生成默认
        }
    ]

    reflection_data = trainer._build_reflection_dataset(train_data)

    assert len(reflection_data) == 2
    assert reflection_data[0]['reflection'] == '已有反思'
    assert reflection_data[1]['reflection'] != ''  # 生成了默认反思
    print("✓ 测试构建反思数据集通过")


def test_generate_default_reflection():
    """测试生成默认反思"""
    trainer = MethodThinkerTrainer()

    sample = {
        'selected_method': '变量替换法',
        'problem': '求解方程',
    }

    reflection = trainer._generate_default_reflection(sample)

    assert '变量替换法' in reflection
    assert '方法选择' in reflection
    assert '关键点' in reflection
    assert '陷阱' in reflection
    assert '改进空间' in reflection
    print("✓ 测试生成默认反思通过")


# ============ 答案验证测试 ============

def test_verify_answer_exact_match():
    """测试答案精确匹配"""
    trainer = MethodThinkerTrainer()

    result = trainer._verify_answer("答案是42", "42")
    assert result == True
    print("✓ 测试精确匹配通过")


def test_verify_answer_in_text():
    """测试答案包含在文本中"""
    trainer = MethodThinkerTrainer()

    result = trainer._verify_answer("计算结果为 x = 5", "5")
    assert result == True
    print("✓ 测试包含匹配通过")


def test_verify_answer_pattern():
    """测试答案模式匹配"""
    trainer = MethodThinkerTrainer()

    result = trainer._verify_answer("最终答案为：(5, 3)", "(5, 3)")
    assert result == True
    print("✓ 测试模式匹配通过")


def test_verify_answer_empty():
    """测试空答案"""
    trainer = MethodThinkerTrainer()

    result = trainer._verify_answer("", "42")
    assert result == False

    result = trainer._verify_answer("答案", "")
    assert result == False
    print("✓ 测试空答案验证通过")


# ============ 检查点测试 ============

def test_save_checkpoint_without_model():
    """测试无模型保存"""
    trainer = MethodThinkerTrainer()
    trainer.model = None

    # 应该不抛异常，只是警告
    trainer.save_checkpoint("/tmp/test_checkpoint")
    print("✓ 测试无模型保存通过")


def test_save_checkpoint_with_mock():
    """测试检查点保存（mock）"""
    trainer = MethodThinkerTrainer()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    trainer.model = mock_model
    trainer.tokenizer = mock_tokenizer

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(tmpdir)

        # 验证调用
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    print("✓ 测试检查点保存通过")


def test_load_checkpoint_without_deps():
    """测试无依赖时加载"""
    trainer = MethodThinkerTrainer()
    trainer._has_deps = False

    result = trainer.load_checkpoint("/tmp/test")
    assert result == False
    print("✓ 测试无依赖加载失败通过")


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="需要transformers库")
def test_load_checkpoint_with_mock():
    """测试检查点加载（mock）"""
    trainer = MethodThinkerTrainer()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    with patch('src.training.trainer.AutoModelForCausalLM.from_pretrained', return_value=mock_model):
        with patch('src.training.trainer.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            with patch('src.training.trainer.torch', _mock_torch):
                result = trainer.load_checkpoint("/tmp/test_checkpoint")

    assert result == True
    assert trainer.model == mock_model
    print("✓ 测试检查点加载通过")


# ============ 集成测试 ============

def test_full_workflow_mock():
    """测试完整训练流程（mock）"""
    trainer = MethodThinkerTrainer(TrainingConfig(num_epochs=1, batch_size=1))

    # Mock所有依赖
    trainer._has_deps = True
    trainer.model = MagicMock()
    trainer.tokenizer = MagicMock()

    mock_dataset = MagicMock()
    mock_trainer = MagicMock()
    mock_result = MagicMock()
    mock_result.training_loss = 0.5
    mock_trainer.train = Mock(return_value=mock_result)
    mock_trainer.save_model = Mock()

    train_data = [{
        'problem_id': 't1',
        'problem': '问题',
        'problem_type': '类型',
        'candidate_methods': [{'method_name': '方法', 'score': 0.9}],
        'selected_method': '方法',
        'solution_steps': ['步骤'],
        'reflection': '反思',
    }]

    with patch('src.training.trainer.TrainingArguments'):
        with patch('src.training.trainer.SFTTrainer', return_value=mock_trainer):
            with patch('src.training.trainer.torch', _mock_torch):
                with patch.object(trainer, '_build_methodology_dataset', return_value=mock_dataset):
                    # 1. 方法论注入
                    result1 = trainer.train_methodology_injection(train_data)
                    assert result1['status'] == 'completed'

                    # 2. 多样性训练
                    with patch.object(trainer, '_build_diversity_dataset', return_value=train_data):
                        result2 = trainer.train_diversity(train_data)
                        assert result2['status'] == 'completed'

                    # 3. 反思训练
                    with patch.object(trainer, '_build_reflection_dataset', return_value=train_data):
                        result3 = trainer.train_reflection(train_data)
                        assert result3['status'] == 'completed'

    print("✓ 测试完整流程通过")


if __name__ == '__main__':
    print("=" * 60)
    print("开始测试训练器模块")
    print("=" * 60)

    # 配置测试
    test_training_config_defaults()
    test_training_config_custom()

    # 初始化测试
    test_trainer_init_default()
    test_trainer_init_with_config()
    test_trainer_init_without_deps()

    # setup测试
    test_setup_without_deps()

    # 训练测试
    test_train_methodology_injection_without_deps()
    test_train_methodology_injection_with_mock()
    test_train_diversity_without_deps()
    test_train_diversity_with_mock()
    test_train_reflection_without_deps()
    test_train_reflection_with_mock()

    # 评估测试
    test_evaluate_without_model()
    test_evaluate_with_mock()
    test_evaluate_pass_at_k()

    # 数据构建测试
    test_format_input()
    test_format_output()
    test_build_diversity_dataset()
    test_build_reflection_dataset()
    test_generate_default_reflection()

    # 答案验证测试
    test_verify_answer_exact_match()
    test_verify_answer_in_text()
    test_verify_answer_pattern()
    test_verify_answer_empty()

    # 检查点测试
    test_save_checkpoint_without_model()
    test_save_checkpoint_with_mock()
    test_load_checkpoint_without_deps()
    test_load_checkpoint_with_mock()

    # 集成测试
    test_full_workflow_mock()

    print("\n" + "=" * 60)
    print("所有训练器测试通过! ✓")
    print("=" * 60)