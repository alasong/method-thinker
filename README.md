# MethodThinker

一个基于方法论显式教学的小模型推理能力提升项目。

## 项目目标

通过迭代提炼方法论的自动化流程，实现无需人工干预的模型能力提升。

核心指标：
- AIME25 Pass@64 ≥ 75%
- 验证准确率 80-85%
- 自动化程度 100%

## 项目结构

```
MethodThinker/
├── src/
│   ├── validation/      # 多层验证系统
│   ├── extraction/      # 方法论提炼
│   ├── kb/              # 知识库管理
│   ├── training/        # 训练相关
│   ├── data/            # 数据处理
│   └── iteration/       # 迭代控制
├── scripts/             # 运行脚本
├── configs/             # 配置文件
├── tests/               # 测试代码
├── data/                # 数据目录
├── docs/                # 文档
└── outputs/             # 输出目录
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行验证
python scripts/run_validation.py

# 训练模型
python scripts/train_sft.py
```

## 文档

- [项目计划](MethodThinker-项目计划详细分解-2026-04-07.md)
- [验证系统设计](推荐方案详细设计-2026-04-07.md)
- [可行性分析](MethodThinker-可行性分析报告-2026-04-07.md)