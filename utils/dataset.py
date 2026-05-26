from pathlib import Path
from urllib.request import urlretrieve

import torch

import config


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("data/input.txt")

train_data = None
val_data = None
encode = None
decode = None
vocab_size = None


def load_data(data_path=DATA_PATH, download_if_missing=False):
    global train_data, val_data, encode, decode, vocab_size

    data_path = Path(data_path)
    if not data_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"{data_path} not found. Add a text corpus or call load_data(download_if_missing=True)."
            )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, data_path)

    text = data_path.read_text(encoding="utf-8")
    if len(text) <= config.block_size + 1:
        raise ValueError(
            f"Dataset must contain more than {config.block_size + 1} characters."
        )

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode_func(s):
        unknown = sorted(set(s) - set(stoi))
        if unknown:
            raise ValueError(f"Input contains unknown characters: {unknown}")
        return [stoi[c] for c in s]

    def decode_func(indices):
        return "".join(itos[int(i)] for i in indices)

    encode = encode_func
    decode = decode_func

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, encode, decode, vocab_size


def get_batch(split):
    if train_data is None or val_data is None:
        load_data()

    data = train_data if split == "train" else val_data
    if len(data) <= config.block_size:
        raise ValueError(f"{split} split is too small for block_size={config.block_size}")

    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
    return x.to(config.device), y.to(config.device)
