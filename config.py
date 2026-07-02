import os

import torch


device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

# 模型架构配置 - 提升模型容量和表达能力
batch_size = int(os.getenv("BATCH_SIZE", 64))  # 增大批次以提高训练稳定性
block_size = int(os.getenv("BLOCK_SIZE", 128))  # 增加上下文窗口长度
vocab_size = None

n_embd = int(os.getenv("N_EMBD", 192))  # 增大嵌入维度以捕获更丰富的语义
n_head = int(os.getenv("N_HEAD", 6))  # 增加注意力头数以学习多角度的依赖关系
n_layer = int(os.getenv("N_LAYER", 4))  # 增加层数以提升模型深度和表达能力
dropout = float(os.getenv("DROPOUT", 0.3))  # 适度提高dropout防止过拟合

# 优化器配置 - 更保守的学习策略以获得更好的收敛
learning_rate = float(os.getenv("LEARNING_RATE", 5e-4))  # 降低学习率以提高训练稳定性
weight_decay = float(os.getenv("WEIGHT_DECAY", 0.05))
grad_clip = float(os.getenv("GRAD_CLIP", 1.0))
max_iters = int(os.getenv("MAX_ITERS", 2000))  # 大幅增加训练迭代次数
eval_interval = int(os.getenv("EVAL_INTERVAL", 200))  # 调整评估间隔以适应更多迭代
eval_iters = int(os.getenv("EVAL_ITERS", 30))  # 增加评估迭代数以获得更准确的指标

data_path = os.getenv("DATA_PATH", "data/input.txt")
output_dir = os.getenv("OUTPUT_DIR", "outputs")
checkpoint_path = os.getenv("CHECKPOINT_PATH", os.path.join(output_dir, "model.pt"))
generated_text_path = os.getenv(
    "GENERATED_TEXT_PATH", os.path.join(output_dir, "generated_text.txt")
)
generated_result_path = os.getenv(
    "GENERATED_RESULT_PATH", os.path.join(output_dir, "generated_result.txt")
)

seed = int(os.getenv("SEED", 1337))
