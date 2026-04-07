# AutoDL云端GPU训练指南

> 更新时间: 2026-04-07 | 预算: ¥50 | 预计费用: ¥2-6

## 一、成本评估

### 1.1 GPU价格对比

| GPU型号 | 显存 | 小时价格 | 推荐场景 |
|---------|------|----------|----------|
| RTX 3090 | 24GB | ¥1.5-2.0 | ✓ 推荐，性价比高 |
| RTX 4090 | 24GB | ¥2.5-3.5 | ✓ 推荐，速度快 |
| A100 40GB | 40GB | ¥8-12 | 大模型训练 |
| A100 80GB | 80GB | ¥15-20 | 全量微调 |

### 1.2 训练成本估算

**MethodThinker训练需求：**
- 模型: Qwen2.5-Math-1.5B (1.5B参数)
- 训练数据: 120样本
- 方式: QLoRA (4-bit量化)
- 目标: 3 epochs

**时间估算:**
```
RTX 4090: ~5-10分钟
RTX 3090: ~8-15分钟
含调试时间: 预留1-2小时
```

**费用估算:**

| 方案 | GPU | 时长 | 费用 | 50元够吗 |
|------|-----|------|------|----------|
| 经济 | RTX 3090 | 1小时 | ¥2 | ✓ 充裕 |
| 标准 | RTX 4090 | 1小时 | ¥3 | ✓ 充裕 |
| 调试 | RTX 3090 | 2小时 | ¥4 | ✓ 充裕 |
| 多轮 | RTX 4090 | 5小时 | ¥15 | ✓ 充裕 |
| 完整 | RTX 4090 | 10小时 | ¥30 | ✓ 足够 |

**结论: 50元完全足够！** 可支持5-25小时训练。

---

## 二、快速开始

### 2.1 准备工作

1. **注册AutoDL账号**: https://www.autodl.com/
2. **充值**: 最低充值¥10，建议¥50
3. **获取API Key**: 控制台 → API管理 → 创建密钥

2.2 创建实例（Web控制台）

1. 登录 https://www.autodl.com/
2. 点击 **租用实例**
3. 选择配置:
   - 镜像: `PyTorch 2.0 / Python 3.10`
   - GPU: `RTX 4090` 或 `RTX 3090`
   - 数据盘: `50GB`
4. 点击 **立即租用**

### 2.3 连接实例

**方式A: JupyterLab（推荐新手）**
- 控制台点击 **打开JupyterLab**
- 在Terminal中执行命令

**方式B: SSH连接**
```bash
# 控制台获取连接信息后
ssh -p <端口> root@<主机地址>
# 密码在控制台显示
```

### 2.4 运行训练

```bash
# 1. 克隆代码
git clone https://github.com/alasong/method-thinker.git
cd method-thinker

# 2. 安装依赖
pip install -q transformers accelerate peft datasets bitsandbytes trl

# 3. 生成训练数据
python scripts/generate_training_data.py \
    --kb data/methodology_kb/v0/math_methods.yaml \
    --problems data/test_sets/aime_samples.yaml \
    --output data/train_data/train.json \
    --mode batch \
    --samples-per-problem 4

# 4. 开始训练
python scripts/train_sft.py \
    --train-data data/train_data/train.json \
    --output-dir outputs/models/v1 \
    --use-lora \
    --epochs 3 \
    --batch-size 4
```

### 2.5 下载结果

**方式A: JupyterLab下载**
- 右键文件 → Download

**方式B: SCP下载**
```bash
scp -P <端口> root@<主机>:~/method-thinker/outputs/models/v1 ./models/
```

### 2.6 关闭实例

**重要! 训练完成后务必关闭实例！**
- 控制台 → 我的实例 → 关机
- 或删除实例（数据会丢失，请先下载）

---

## 三、推荐配置

### 3.1 最小配置（经济）

```bash
GPU: RTX 3090
时长: 1小时
费用: ~¥2
batch_size: 4
max_length: 2048
```

### 3.2 推荐配置（标准）

```bash
GPU: RTX 4090
时长: 1-2小时
费用: ~¥3-6
batch_size: 8
max_length: 4096
```

### 3.3 高性能配置

```bash
GPU: A100 40GB
时长: 1小时
费用: ~¥10
batch_size: 16
max_length: 4096
```

---

## 四、常见问题

### Q1: 训练中断怎么办？
- 使用`--resume`参数继续训练
- 定期保存检查点

### Q2: 如何查看GPU使用情况？
```bash
nvidia-smi
watch -n 1 nvidia-smi  # 实时监控
```

### Q3: 如何避免费用超支？
- 设置AutoDL预算提醒
- 训练完成后立即关机
- 使用脚本自动监控费用

### Q4: 数据如何持久化？
- 使用数据盘（/root/autodl-tmp）
- 定期同步到云盘或GitHub

---

## 五、成本优化建议

1. **选择合适GPU**: RTX 3090性价比最高
2. **预装镜像**: 选择包含PyTorch的镜像
3. **批量测试**: 先用少量数据测试
4. **及时关机**: 训练完成立即关机
5. **使用监控**: 设置费用告警

---

## 六、一键脚本

```bash
# 使用AutoDL训练脚本
python scripts/autodl_train.py --gpu-type RTX4090 --hours 1 --dry-run

# 实际执行（需要AUTODL_API_KEY）
export AUTODL_API_KEY="your-api-key"
python scripts/autodl_train.py --gpu-type RTX4090 --hours 1
```