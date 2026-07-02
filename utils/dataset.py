from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import torch

import config
from .tokenizer import WordTokenizer


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("data/input.txt")
DEFAULT_CORPUS_DIR = Path("data/corpus")

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
    source_paths: tuple[Path, ...]
    tokenizer: WordTokenizer = None

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def num_documents(self):
        return len(self.source_paths)

    @property
    def num_tokens(self):
        return len(self.train_data) + len(self.val_data)

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


def _find_default_data_path(requested_path):
    if requested_path.exists():
        return requested_path
    if requested_path == DATA_PATH and DEFAULT_CORPUS_DIR.exists():
        return DEFAULT_CORPUS_DIR
    return requested_path


def _read_text_sources(data_path):
    data_path = Path(data_path)
    if data_path.is_file():
        return [(data_path, data_path.read_text(encoding="utf-8"))]

    if data_path.is_dir():
        paths = tuple(sorted(path for path in data_path.rglob("*.txt") if path.is_file()))
        if not paths:
            raise FileNotFoundError(f"No .txt files found under corpus directory: {data_path}")
        return [(path, path.read_text(encoding="utf-8")) for path in paths]

    raise FileNotFoundError(f"{data_path} does not exist.")


def _split_sources(sources):
    if len(sources) == 1:
        text = sources[0][1]
        n = int(0.9 * len(text))
        return text[:n], text[n:]

    n_train = int(0.9 * len(sources))
    n_train = min(max(1, n_train), len(sources) - 1)
    train_text = "\n\n".join(text for _, text in sources[:n_train])
    val_text = "\n\n".join(text for _, text in sources[n_train:])
    return train_text, val_text


def _validate_split_sizes(train_text, val_text):
    min_chars = config.block_size + 1
    if len(train_text) <= min_chars or len(val_text) <= min_chars:
        raise ValueError(
            "Train and validation splits must each contain more than "
            f"{min_chars} characters. Add more corpus text or lower block_size."
        )


def _can_download_to_path(data_path):
    return data_path.suffix.lower() == ".txt"


def _build_word_tokenizer(train_text: str, val_text: str) -> WordTokenizer:
    """Build a word-level tokenizer from training and validation texts."""
    tokenizer = WordTokenizer()
    # Build vocabulary from both train and val texts
    tokenizer.build_vocab([train_text, val_text])
    return tokenizer


def load_data(data_path=None, download_if_missing=False, use_word_tokenizer=False):
    global train_data, val_data, encode, decode, vocab_size, _dataset

    requested_path = Path(config.data_path if data_path is None else data_path)
    requested_path = _find_default_data_path(requested_path)
    if not requested_path.exists():
        if not download_if_missing or not _can_download_to_path(requested_path):
            raise FileNotFoundError(
                f"{requested_path} not found. Add a text corpus or call "
                "load_data(download_if_missing=True)."
            )
        requested_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, requested_path)

    sources = _read_text_sources(requested_path)
    train_text, val_text = _split_sources(sources)
    _validate_split_sizes(train_text, val_text)

    if use_word_tokenizer:
        # Use word-level tokenizer
        tokenizer = _build_word_tokenizer(train_text, val_text)
        
        # Encode texts using word tokenizer
        train_tokens = tokenizer.encode(train_text)
        val_tokens = tokenizer.encode(val_text)
        
        train_data = torch.tensor(train_tokens, dtype=torch.long)
        val_data = torch.tensor(val_tokens, dtype=torch.long)
        
        _dataset = TextDataset(
            train_data=train_data,
            val_data=val_data,
            stoi=tokenizer.vocab,
            itos=tokenizer.inv_vocab,
            data_path=requested_path,
            source_paths=tuple(path for path, _ in sources),
            tokenizer=tokenizer,
        )
    else:
        # Use character-level tokenizer (original behavior)
        text = train_text + val_text
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        train_data = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
        val_data = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)
        _dataset = TextDataset(
            train_data=train_data,
            val_data=val_data,
            stoi=stoi,
            itos=itos,
            data_path=requested_path,
            source_paths=tuple(path for path, _ in sources),
        )

    encode = _dataset.encode
    decode = _dataset.decode
    vocab_size = _dataset.vocab_size

    return train_data, val_data, encode, decode, vocab_size


def get_dataset(data_path=None, download_if_missing=False):
    requested_path = Path(config.data_path if data_path is None else data_path)
    requested_path = _find_default_data_path(requested_path)
    if _dataset is None or _dataset.data_path != requested_path:
        load_data(data_path=requested_path, download_if_missing=download_if_missing)
    return _dataset


def get_batch(split):
    return get_dataset().get_batch(split)
