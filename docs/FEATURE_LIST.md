# MethodThinker 功能清单

> 版本: v1.0 | 更新日期: 2026-04-08

---

## 一、核心模块功能

### 1. 验证系统 (src/validation/)

| 模块 | 功能 | 关键类/函数 |
|-----|------|-----------|
| layer0_fast_filter.py | 快速过滤层 | Layer0FastFilter |
| layer1_self_reflection.py | 自我反思层 | Layer1SelfReflection |
| layer2_multi_model.py | 多模型验证层 | Layer2MultiModelValidation |
| layer3_test_driven.py | 测试驱动层 | Layer3TestDrivenValidation |
| ensemble_decision.py | 集成决策引擎 | EnsembleDecisionEngine |
| pipeline.py | 验证流水线 | ValidationPipeline |
| config.py | 配置管理 | ValidationConfig |

**验证流程:** Layer0 → Layer1 → Layer2 → Layer3 → 集成决策

### 2. 训练系统 (src/training/)

| 模块 | 功能 | 关键类/函数 |
|-----|------|-----------|
| trainer.py | SFT训练器 | MethodThinkerTrainer |

**训练模式:**
- methodology-injection: 方法论注入训练
- diversity: 多样性训练
- reflection: 反思强化训练
- full: 完整训练流程

### 3. 数据处理 (src/data/)

| 模块 | 功能 | 关键类/函数 |
|-----|------|-----------|
| dataset.py | 数据集类 | MethodologyDataset |
| sample_generator.py | 样本生成器 | SampleGenerator |
| method_injector.py | 方法注入器 | MethodInjector |
| aime_loader.py | AIME数据加载 | AIMELoader |
| collator.py | 数据整理器 | MethodologyCollator |

### 4. 知识库 (src/kb/)

| 模块 | 功能 | 关键类/函数 |
|-----|------|-----------|
| knowledge_base.py | 知识库管理 | KnowledgeBase |
| incremental_updater.py | KB增量更新 | IncrementalUpdater |

### 5. 提取系统 (src/extraction/)

| 模块 | 功能 | 关键类/函数 |
|-----|------|-----------|
| methodology_extractor.py | 方法论提取 | MethodologyExtractor |
| pattern_miner.py | 模式挖掘 | PatternMiner |

### 6. 迭代控制 (src/iteration/)

| 模块 | 功能 | 关键类/函数 |
|-----|------|-----------|
| iteration_controller.py | 迭代控制器 | IterationController |
| convergence_detector.py | 收敛检测 | ConvergenceDetector |

---

## 二、脚本功能

### 训练脚本

| 脚本 | 功能 | 用法 |
|-----|------|------|
| train_sft.py | SFT训练主脚本 | `python scripts/train_sft.py --use-lora` |
| train_local.py | 本地GPU训练 | `python scripts/train_local.py` |

### 自动化脚本

| 脚本 | 功能 | 用法 |
|-----|------|------|
| autodl_full_automation.py | AutoDL完全自动化 | `python scripts/autodl_full_automation.py` |
| colab_automation.py | Colab自动化 | `python scripts/colab_automation.py` |

### 数据脚本

| 脚本 | 功能 | 用法 |
|-----|------|------|
| generate_training_data.py | 生成训练数据 | `python scripts/generate_training_data.py` |
| download_datasets.py | 下载数据集 | `python scripts/download_datasets.py --dataset all` |
| manage_datasets.py | 数据集管理 | `python scripts/manage_datasets.py` |

### 运行脚本

| 脚本 | 功能 | 用法 |
|-----|------|------|
| run_validation.py | 运行验证 | `python scripts/run_validation.py` |
| run_evaluation.py | 运行评估 | `python scripts/run_evaluation.py` |
| run_extraction.py | 运行提取 | `python scripts/run_extraction.py |

### 调试脚本

| 脚本 | 功能 | 用法 |
|-----|------|------|
| debug_training.py | 训练调试 | `python scripts/debug_training.py` |

---

## 三、配置文件

| 配置文件 | 功能 | 路径 |
|---------|------|------|
| training_config.yaml | 训练配置 | configs/training_config.yaml |
| training_gpu_presets.yaml | GPU预设 | configs/training_gpu_presets.yaml |
| validation_config.yaml | 验证配置 | configs/validation_config.yaml |
| evaluation_config.yaml | 评估配置 | configs/evaluation_config.yaml |
| dataset_config.yaml | 数据集配置 | configs/dataset_config.yaml |
| colab_automation.yaml | Colab自动化配置 | configs/colab_automation.yaml |

---

## 四、训练数据

| 数据集 | 样本数 | 来源 | 难度 |
|-------|--------|------|------|
| train.json | 120 | 原始数据 | 高 |
| gsm8k_downloaded.json | 8,792 | GSM8K | 低 |
| omni-math_downloaded.json | 4,428 | Omni-MATH | 中高 |
| **all_merged.json** | **13,340** | **合并数据** | **混合** |

---

## 五、使用场景

### 场景1: 方法论验证

```bash
# 验证单个方法论
python scripts/run_validation.py --method "因式分解法"

# 验证整个知识库
python scripts/run_validation.py --kb data/methodology_kb/v0/
```

### 场景2: 模型训练

```bash
# 快速训练（Colab T4）
python scripts/train_sft.py --train-data data/train_data/all_merged.json --use-lora --epochs 3

# 完整训练流程
python scripts/train_sft.py --mode full --use-lora --epochs 3
```

### 场景3: 模型评估

```bash
# 评估AIME基准
python scripts/run_evaluation.py --benchmark aime25 --model outputs/checkpoints/final
```

### 场景4: 数据下载

```bash
# 下载所有数据集
python scripts/download_datasets.py --dataset all

# 下载单个数据集
python scripts/download_datasets.py --dataset gsm8k
```

---

## 六、GPU训练预设

| GPU | batch_size | max_length | lora_r | 预计时间 |
|-----|------------|------------|--------|---------|
| T4 16GB | 4 | 2048 | 16 | 30分钟 |
| RTX 3090 24GB | 8 | 4096 | 32 | 15分钟 |
| RTX 4090 24GB | 8 | 4096 | 32 | 10分钟 |
| A100 40GB | 16 | 4096 | 64 | 5分钟 |

---

## 七、测试覆盖

| 模块 | 测试数 | 通过率 |
|-----|-------|--------|
| Layer 0-3 | 54 | 100% |
| Ensemble | 13 | 100% |
| Pipeline | 10 | 100% |
| Data | 15 | 100% |
| Extraction | 9 | 100% |
| Iteration | 10 | 100% |
| **总计** | **151** | **100%** |

---

*文档生成时间: 2026-04-08*