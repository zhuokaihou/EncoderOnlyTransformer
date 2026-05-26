# EncoderOnlyTransformer

课程期末项目：使用 PyTorch 从零实现一个 Encoder-only Transformer，并在字符级语言建模任务上训练、评估和生成文本。

## 项目目标

本项目实现了一个只由 Transformer Encoder block 堆叠而成的自回归字符级语言模型。虽然结构名称是 Encoder-only，但为了完成 next-token prediction，注意力层使用了 causal mask，保证当前位置只能看到当前位置及之前的 token。

模型训练目标是最小化下一个字符预测的交叉熵损失，并使用 Perplexity 作为辅助评估指标。

## 当前状态

- 实现字符级数据读取、编码、解码和 batch 采样
- 实现 sinusoidal positional encoding
- 实现 masked multi-head self-attention
- 实现 feed-forward network、LayerNorm、残差连接和 Encoder block
- 实现训练脚本、生成脚本、模型保存和曲线保存
- 提供最小 smoke test，验证数据、前向传播、loss 和生成流程能跑通

## 项目结构

```text
.
├── config.py                 # 全局超参数配置
├── data/input.txt            # 默认小型训练语料
├── generate.py               # 文本生成入口
├── models/
│   ├── __init__.py
│   └── transformer.py        # Encoder-only Transformer 模型
├── train.py                  # 训练入口
├── utils/
│   ├── __init__.py
│   └── dataset.py            # 数据加载、编码、batch 采样
├── tests/test_smoke.py       # 最小可运行测试
└── requirements.txt
```

## 环境安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如需运行测试，还需要安装 pytest：

```bash
pip install pytest
```

## 训练模型

默认配置适合快速验证流程：

```bash
python train.py
```

训练完成后会在 `outputs/` 下生成：

- `model.pt`：训练后的模型参数
- `loss_ppl_curve.png`：训练集/验证集 loss 和 PPL 曲线
- `generated_text.txt`：训练结束后的文本生成样例

可以通过环境变量快速覆盖配置，例如只训练 20 步：

```bash
MAX_ITERS=20 EVAL_INTERVAL=5 EVAL_ITERS=2 python train.py
```

## 生成文本

先训练得到 `outputs/model.pt`，然后运行：

```bash
python generate.py
```

生成结果会保存到：

```text
outputs/generated_result.txt
```

如果没有训练好的 `outputs/model.pt`，脚本也能运行，但会使用随机初始化模型，生成结果没有语言质量。

## 测试

```bash
pytest -q
```

测试会检查：

- 数据能正确加载并转换成字符级 token
- batch 的形状符合配置
- 模型 forward 能输出 logits 和 loss
- generate 方法能生成指定长度的新 token

## 模型说明

模型的主要组件包括：

1. Token Embedding：把字符 id 映射到向量空间
2. Positional Encoding：使用正弦/余弦位置编码注入顺序信息
3. Masked Multi-Head Self-Attention：多头自注意力，并用下三角 mask 防止看到未来 token
4. Feed-Forward Network：两层 MLP 提升非线性表达能力
5. Residual Connection + LayerNorm：稳定训练
6. Linear Head：输出每个位置对下一个字符的 logits

## 已修复的问题

原始项目缺少运行说明和依赖文件。当前版本补齐了：

- `requirements.txt`
- `train.py`
- `generate.py`
- `models/__init__.py`
- `utils/__init__.py`
- `tests/test_smoke.py`

同时将训练脚本改为无界面绘图后端，避免在服务器环境中因为 `plt.show()` 阻塞。
