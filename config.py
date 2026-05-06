# config.py - 项目全局超参数配置
import torch

# 设备配置（自动检测GPU/CPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据与模型超参数
batch_size = 32          # 批次大小
block_size = 64          # 序列长度（上下文窗口）
vocab_size = None        # 由数据集自动赋值，无需修改

# Transformer模型结构
n_embd = 128             # 词嵌入维度
n_head = 4               # 注意力头数
n_layer = 3              # Encoder堆叠层数
dropout = 0.1            # Dropout正则化率

# 训练超参数
learning_rate = 1e-3     # 学习率
max_iters = 1000         # 最大训练迭代次数
eval_interval = 100      # 每N步评估一次训练/验证集