from dataclasses import dataclass
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
_dataset = None


@dataclass
class TextDataset:
    train_data: torch.Tensor
    val_data: torch.Tensor
    stoi: dict
    itos: dict
    data_path: Path

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"Input contains unknown characters: {unknown}")
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        unknown = sorted({int(i) for i in indices if int(i) not in self.itos})
        if unknown:
            raise ValueError(f"Token ids are outside the vocabulary: {unknown}")
        return "".join(self.itos[int(i)] for i in indices)

    def get_batch(self, split, batch_size=None, block_size=None, device=None):
        if split not in {"train", "val"}:
            raise ValueError("split must be either 'train' or 'val'")

        batch_size = config.batch_size if batch_size is None else batch_size
        block_size = config.block_size if block_size is None else block_size
        device = config.device if device is None else device

        data = self.train_data if split == "train" else self.val_data
        if len(data) <= block_size:
            raise ValueError(f"{split} split is too small for block_size={block_size}")

        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


def load_data(data_path=None, download_if_missing=False):
    global train_data, val_data, encode, decode, vocab_size, _dataset

    data_path = Path(config.data_path if data_path is None else data_path)
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
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    _dataset = TextDataset(train_data, val_data, stoi, itos, data_path)

    encode = _dataset.encode
    decode = _dataset.decode
    vocab_size = _dataset.vocab_size

    return train_data, val_data, encode, decode, vocab_size


def get_dataset(data_path=None, download_if_missing=False):
    requested_path = Path(config.data_path if data_path is None else data_path)
    if _dataset is None or _dataset.data_path != requested_path:
        load_data(data_path=data_path, download_if_missing=download_if_missing)
    return _dataset


def get_batch(split):
    return get_dataset().get_batch(split)
