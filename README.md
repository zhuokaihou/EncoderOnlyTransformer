# EncoderOnlyTransformer

课程期末项目：使用 PyTorch 从零实现一个字符级 Transformer 语言模型，并完成训练、评估、保存 checkpoint 和文本生成。

## 项目定位

当前模型使用 encoder-style 的 Pre-Norm block、残差连接、前馈网络和多头自注意力，但为了完成 next-token prediction，注意力层使用 causal mask，保证当前位置只能看到当前位置及之前的 token。

模型说明中明确标注了 causal 语言建模设定，避免和标准双向 Encoder-only 模型混淆。

训练目标是最小化下一个字符预测的交叉熵损失，并使用 Perplexity 作为辅助评估指标。

**分词器功能更新（期末口头报告后）：**
- 新增轻量级单词级分词器支持，可选择使用空格分词替代原有的字符级处理
- 保持向后兼容，默认仍使用字符级分词，通过参数可切换到单词级分词模式

## 当前状态

- 字符级数据读取、编码、解码和 batch 采样
- **新增：轻量级单词级分词器（utils/tokenizer.py）**
- 支持单文件 `data/input.txt` 和多文件目录 `data/corpus/*.txt`
- 提供 `scripts/build_corpus.py` 清洗、去重并合并外部纯文本语料
- Sinusoidal positional encoding
- Causal masked multi-head self-attention
- Feed-forward network、LayerNorm、残差连接和 Transformer block
- 参数化训练脚本、生成脚本、模型保存和曲线保存
- 完整 checkpoint：模型参数、优化器状态、模型配置、词表、训练历史
- smoke test、采样参数校验、causal mask 行为测试和语料构建测试
- GitHub Actions 自动测试和 1-step smoke training

## 项目结构
```
text
.
├── .github/workflows/tests.yml  # GitHub Actions 测试流程
├── DATA.md                      # 数据集扩展说明
├── config.py                    # 环境变量默认配置
├── data/
│   ├── corpus/                  # 可放置多份 raw .txt 语料
│   └── input.txt                # 默认训练语料
├── generate.py                  # 文本生成入口
├── models/
│   ├── __init__.py
│   └── transformer.py           # Transformer 语言模型
├── pyproject.toml               # 项目和测试/格式配置
├── requirements.txt
├── scripts/
│   └── build_corpus.py          # 清洗并合并多文件语料
├── tests/test_build_corpus.py
├── tests/test_smoke.py
├── train.py                     # 训练入口
└── utils/
    ├── __init__.py
    ├── dataset.py               # 数据加载、编码、batch 采样
    └── tokenizer.py             # 轻量级单词分词器
```

## 环境安装

建议使用 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 扩大数据集

项目现在支持两种数据组织方式。

第一种是继续使用单个合并后的训练文件：

```bash
python3 train.py --data-path data/input.txt
```

第二种是把多份纯文本放在 `data/corpus/` 下直接训练：

```text
data/corpus/
├── alice.txt
├── shakespeare.txt
└── stories.txt
```

```bash
python3 train.py --data-path data/corpus
```

也可以先清洗并合并外部文本，再生成新的 `data/input.txt`：

```bash
python3 scripts/build_corpus.py data/corpus --output data/input.txt --min-chars 200
```

`build_corpus.py` 会做几件事：

- 递归读取输入目录下的 `.txt` 文件
- 去掉 Project Gutenberg 常见头尾声明
- 统一换行和空白字符
- 跳过太短的文档
- 对完全重复的文档去重
- 输出一个 UTF-8 纯文本训练文件

推荐语料来源：

- public-domain 纯文本书籍，例如 Project Gutenberg
- TinyStories 这类适合小语言模型的短故事语料
- 自己整理的课程文本或英文短篇文本

不建议直接混入大量网页噪声或多语言文本。当前模型较小，干净、风格一致的语料通常比杂乱的大语料更有帮助。

## 训练模型

默认配置适合快速验证流程：

```bash
python3 train.py
```

训练完成后会在 `outputs/` 下生成：

- `model.pt`：最终 checkpoint
- `best_model.pt`：验证集 loss 最优 checkpoint
- `loss_ppl_curve.png`：训练集/验证集 loss 和 PPL 曲线
- `generated_text.txt`：训练结束后的文本生成样例

可以通过命令行参数覆盖配置，例如只训练 20 步：

```bash
python3 train.py --max-iters 20 --eval-interval 5 --eval-iters 2
```

也可以继续使用环境变量：

```bash
MAX_ITERS=20 EVAL_INTERVAL=5 EVAL_ITERS=2 python3 train.py
```

常用参数：

```bash
python3 train.py \
  --data-path data/input.txt \
  --batch-size 32 \
  --block-size 64 \
  --n-embd 128 \
  --n-head 4 \
  --n-layer 3 \
  --learning-rate 0.001 \
  --grad-clip 1.0
```

目录语料训练示例：

```bash
python3 train.py \
  --data-path data/corpus \
  --batch-size 32 \
  --block-size 64
```

## 生成文本

先训练得到 `outputs/model.pt`，然后运行：

```bash
python3 generate.py
```

生成结果会保存到：

```text
outputs/generated_result.txt
```

可以指定 prompt 和采样参数：

```bash
python3 generate.py \
  --prompt "the " \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9
```

如果 checkpoint 中包含词表和模型配置，生成脚本会直接从 checkpoint 恢复；如果只加载旧版 `state_dict`，则会回退到从 `data/input.txt` 重建词表。

## 测试

```bash
pytest -q
```

测试会检查：

- 单文件和目录语料能正确加载并转换成字符级 token
- batch 的形状符合配置
- 模型 forward 能输出 logits 和 loss
- generate 方法能生成指定长度的新 token
- causal mask 不会让当前位置读取未来 token
- 无效 split、未知字符和非法采样参数会报错
- 语料构建脚本能清洗 Gutenberg 头尾、去重并合并文本

## 维护说明

- 仓库已移除重复模型模块，主实现集中在 `models/transformer.py`
- `.gitignore` 会继续忽略本地缓存、虚拟环境和训练输出
- GitHub Actions 会在 push 和 pull request 时运行测试，并执行一次最小训练流程
- **新增分词器模块**：`utils/tokenizer.py` 提供轻量级单词分词功能，默认使用字符级分词以保持兼容性
