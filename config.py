import os

import torch


device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

batch_size = int(os.getenv("BATCH_SIZE", 32))
block_size = int(os.getenv("BLOCK_SIZE", 64))
vocab_size = None

n_embd = int(os.getenv("N_EMBD", 128))
n_head = int(os.getenv("N_HEAD", 4))
n_layer = int(os.getenv("N_LAYER", 3))
dropout = float(os.getenv("DROPOUT", 0.1))

learning_rate = float(os.getenv("LEARNING_RATE", 1e-3))
weight_decay = float(os.getenv("WEIGHT_DECAY", 0.01))
grad_clip = float(os.getenv("GRAD_CLIP", 1.0))
max_iters = int(os.getenv("MAX_ITERS", 1000))
eval_interval = int(os.getenv("EVAL_INTERVAL", 100))
eval_iters = int(os.getenv("EVAL_ITERS", 20))

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
