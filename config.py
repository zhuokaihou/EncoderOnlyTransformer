# config.py - project-wide hyperparameters
import os

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = int(os.getenv("BATCH_SIZE", 32))
block_size = int(os.getenv("BLOCK_SIZE", 64))
vocab_size = None

n_embd = int(os.getenv("N_EMBD", 128))
n_head = int(os.getenv("N_HEAD", 4))
n_layer = int(os.getenv("N_LAYER", 3))
dropout = float(os.getenv("DROPOUT", 0.1))

learning_rate = float(os.getenv("LEARNING_RATE", 1e-3))
max_iters = int(os.getenv("MAX_ITERS", 1000))
eval_interval = int(os.getenv("EVAL_INTERVAL", 100))
eval_iters = int(os.getenv("EVAL_ITERS", 20))

seed = int(os.getenv("SEED", 1337))
